"""
融合核心：Engram × DSA —— 超长上下文记忆稀疏路由。

模块组成：
1. EngramMemory — 外置条件记忆表（哈希槽 → 记忆嵌入，可写入/更新）。
   与论文静态 embedding table 的区别：本表是「外置可写」的——由 Node
   插件把会话蒸馏出的记忆条目写入，训练时不可知、运行时增长。
2. EngramGateModule — 论文 Engram 的 gate 融合：hidden_states 作 query，
   与记忆表取出的 key 打分 → sigmoid gate → gate × value 残差注入。
3. DsaEngramIndexer — DSA Lightning Indexer 的融合版：**KV 参与 engram
   寻址后的路由**——当前 token 的 hidden states 投影成 indexer query，
   对 engram 候选（哈希寻址结果）打分（ReLU + 多头加权聚合），top-K
   稀疏选择：只有被选中的 engram 才参与 gate 融合。

融合语义（为什么比向量索引强）：
- 哈希寻址是确定性的（精确匹配模式，非近似相似度）；
- KV 路由是学习过的打分器（query 表示经投影对候选打分），比纯哈希
  碰撞更准——哈希粗筛 + KV 精筛 + top-K 超稀疏。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class EngramMemory(nn.Module):
    """外置条件记忆表：哈希槽位 → 记忆嵌入（记忆池 + 稀疏槽索引）。

    容量设计（1M token 小目标的核心）：
    - 嵌入集中存在**记忆池**（pool：[MAX_ENTRIES, D]），每条记忆一个嵌入；
    - 槽位只存**索引**（slot_index：[num_slots, per_slot_capacity] 的 int64
      表，存记忆池下标，-1 = 空）——这是论文「massive embedding tables
      offload」思想的落地：槽位稀疏，容量由池子决定，内存可控；
    - 写入：哈希寻址到槽 → 槽内追加记忆池下标（round-robin 找空位）；
    - 查找：槽 → 索引 → index_select 取嵌入。

    容量测算（bf16，1024 维）：
    - 池 65536 条 × 1024 × 2B ≈ 134MB 权重 + 槽索引 65652×32×4B ≈ 8MB；
    - 按「1 条记忆 ≈ 1 回合 ≈ 500 token」，65536 条 ≈ **3270 万 token**
      历史可寻址容量——1M token 目标富余 32 倍。
    """

    def __init__(
        self,
        num_slots: int,
        embed_dim: int,
        per_slot_capacity: int = 32,
        max_entries: int = 65536,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.per_slot_capacity = per_slot_capacity
        self.max_entries = max_entries
        self.embed_dim = embed_dim

        # 记忆池：全部记忆嵌入（集中存储）
        self.pool = nn.Parameter(torch.zeros(max_entries, embed_dim, dtype=dtype))
        nn.init.normal_(self.pool, std=0.02)
        # 槽索引：slot -> [capacity] 个记忆池下标（-1 = 空）；容量 32 可寻址
        # 65536×32 ≈ 210 万条（3270 万 token），远超 1M 目标。
        self.register_buffer(
            "slot_index",
            torch.full((num_slots, per_slot_capacity), -1, dtype=torch.long),
        )
        # 计数：每槽已用位置数 + 池已用数
        self.register_buffer("slot_used", torch.zeros(num_slots, dtype=torch.long))
        self.register_buffer("pool_used", torch.zeros((), dtype=torch.long))
        # 记忆有效性（供 num_entries 统计）
        self.register_buffer("active", torch.zeros(max_entries, dtype=torch.bool))

    def write(self, slot_ids: torch.Tensor, embeds: torch.Tensor):
        """写入记忆：slot_ids [N]，embeds [N, D]。逐槽追加到空位（槽满则丢弃并计数）。"""
        for i in range(slot_ids.shape[0]):
            s = int(slot_ids[i])
            used = int(self.slot_used[s])
            if used >= self.per_slot_capacity:
                continue  # 槽满：该记忆不可寻址（不丢弃文本存储，只是不入表）
            if int(self.pool_used) >= self.max_entries:
                return  # 池满
            pid = int(self.pool_used)
            self.pool.data[pid] = embeds[i].detach()
            self.slot_index[s, used] = pid
            self.slot_used[s] = used + 1
            self.pool_used += 1
            self.active[pid] = True

    def erase(self, slot_ids: torch.Tensor, positions: torch.Tensor):
        for s, p in zip(slot_ids.tolist(), positions.tolist()):
            pid = int(self.slot_index[s, p])
            if pid >= 0:
                self.active[pid] = False
                self.slot_index[s, p] = -1
                self.slot_used[s] -= 1

    def lookup(self, slot_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """按槽位取回全部有效记忆：返回 (embeds [B, ..., slots, cap, D], valid [B, ..., slots, cap])。"""
        # slot_ids 可为 [B, S] 或 [B, S, n_slots]（多头哈希）
        idx = self.slot_index[slot_ids]  # [B, ..., cap]
        valid = idx >= 0  # [B, ..., cap]
        safe = idx.clamp(min=0)
        # 展平 index_select（保证梯度路径一致）
        flat = safe.reshape(-1)
        pool_embeds = self.pool[flat].reshape(*idx.shape, self.embed_dim)
        # 无效位置嵌入置零（gate 处按 valid 掩码为 0，不影响）
        pool_embeds = pool_embeds * valid.unsqueeze(-1)
        return pool_embeds, valid

    def num_entries(self) -> int:
        return int(self.active.sum().item())

    def slot_usage(self) -> dict:
        """槽位占用统计（仿真/诊断用）。"""
        used = self.slot_used.tolist()
        full = sum(1 for u in used if u >= self.per_slot_capacity)
        nonempty = sum(1 for u in used if u > 0)
        return {
            "entries": self.num_entries(),
            "slots_nonempty": nonempty,
            "slots_full": full,
            "pool_used": int(self.pool_used),
            "pool_capacity": self.max_entries,
        }


class DsaEngramIndexer(nn.Module):
    """DSA Lightning Indexer 融合版：KV 负责 engram 寻址后的路由打分。

    结构移植自 huggingface transformers deepseek_v32 的 DeepseekV32Indexer：
      - wq_b：query 投影（从 hidden_states 或 q_resid 派生）
      - wk：engram 候选的 key 投影（候选 = 哈希寻址命中的记忆嵌入）
      - weights_proj：多头加权聚合（ReLU 门控 + 头间加权求和）
      - top-k 选择：只有得分最高的 k 条记忆参与后续 gate 融合

    与原生 DSA 的差异：原生 indexer 对「历史 token 的 KV cache」打分
    （稀疏注意力）；这里对「外置 engram 记忆表」打分（稀疏记忆路由）。
    """

    def __init__(self, hidden_size: int, n_heads: int, head_dim: int, topk: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.topk = topk
        self.wq = nn.Linear(hidden_size, n_heads * head_dim, bias=False, dtype=dtype)
        self.wk = nn.Linear(hidden_size, head_dim, bias=False, dtype=dtype)
        self.k_norm = nn.LayerNorm(head_dim, eps=1e-6, dtype=dtype)
        self.weights_proj = nn.Linear(hidden_size, n_heads, bias=False, dtype=dtype)
        self.softmax_scale = head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, S, H]
        engram_keys: torch.Tensor,  # [B, S, N_cand, D_cand]（候选记忆嵌入）
        engram_valid: torch.Tensor,  # [B, S, N_cand] bool
        topk: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (topk_indices, scores)：
        topk_indices [B, S, k]：被选中的候选下标（-1 = 无效候选占位）
        scores [B, S, N_cand]：全部候选的索引得分（路由分数）。
        """
        B, S, _ = hidden_states.shape
        k = topk or self.topk

        q = self.wq(hidden_states).view(B, S, self.n_heads, self.head_dim)
        # 候选 key：把每个候选嵌入过 wk 投影成 indexer key
        # engram_keys [B, S, N, D] -> [B, S, N, D_cand]
        key_flat = engram_keys.reshape(B * S, -1, self.hidden_size) if engram_keys.shape[-1] == self.hidden_size else engram_keys.reshape(B * S, -1, engram_keys.shape[-1])
        # 若候选嵌入维度 != hidden_size，先投影对齐（论文 value_proj 语义）
        if engram_keys.shape[-1] != self.hidden_size:
            if not hasattr(self, "_proj_cand"):
                self._proj_cand = nn.Linear(engram_keys.shape[-1], self.hidden_size, bias=False).to(engram_keys.device)
            key_flat = self._proj_cand(key_flat)
        kk = self.k_norm(self.wk(key_flat))  # [B*S, N, head_dim]
        N = kk.shape[1]
        kk = kk.view(B, S, N, self.head_dim)

        # 打分：q [B,S,H,D] x k [B,S,N,D] -> [B,S,H,N]
        scores = torch.matmul(q, kk.transpose(-1, -2)) * self.softmax_scale
        scores = torch.relu(scores)

        # 多头加权聚合：weights [B,S,H] -> [B,S,N]
        weights = self.weights_proj(hidden_states) * (self.n_heads**-0.5)
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)  # [B,S,N]

        # 无效候选置 -inf
        index_scores = index_scores.masked_fill(~engram_valid, float("-inf"))

        # top-k 选择（-inf 候选会被 topk 返回，但无效 → 替换为 -1 占位，
        # gate 处 -1 → gate 0 不注入）
        topk = min(k, N)
        top_indices = index_scores.topk(topk, dim=-1).indices  # [B, S, k]
        invalid = ~engram_valid.gather(-1, top_indices.clamp(min=0))  # [B,S,k] 选中项是否无效
        top_indices = top_indices.masked_fill(invalid, -1)
        return top_indices, index_scores


class EngramGateModule(nn.Module):
    """Engram 论文的 gate 融合：gate × value 残差注入。

    gate 由 indexer 的路由分数派生（sigmoid 归一），value 是选中记忆
    嵌入经 value_proj 投影到 hidden_size。
    """

    def __init__(self, engram_hidden: int, hidden_size: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.value_proj = nn.Linear(engram_hidden, hidden_size, bias=False, dtype=dtype)
        self.gate_scale = nn.Parameter(torch.tensor(1.0, dtype=dtype))

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, S, H]
        selected_embeds: torch.Tensor,  # [B, S, k, D]
        route_scores: torch.Tensor,  # [B, S, N]（indexer 路由分数）
        top_indices: torch.Tensor,  # [B, S, k]
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape
        k = selected_embeds.shape[2]
        # gate：选中候选的路由分数 → sigmoid（论文 gate 的转接）
        # 无效候选（-1 或 -inf 得分）gate 应为 0，不注入
        safe_idx = top_indices.clamp(min=0)
        gathered = route_scores.gather(-1, safe_idx)  # [B,S,k]
        # -inf 得分（无效候选）→ gate 0
        gate = torch.sigmoid(gathered * self.gate_scale)  # [B,S,k]
        gate = gate.masked_fill(top_indices < 0, 0.0)
        gate = gate.masked_fill(torch.isinf(gathered), 0.0)
        value = self.value_proj(selected_embeds)  # [B,S,k,H]
        # 加权求和注入
        fused = (gate.unsqueeze(-1) * value).sum(dim=2)  # [B,S,H]
        return hidden_states + fused


class EngramFusedModule(nn.Module):
    """完整融合模块：哈希寻址 → KV 路由（indexer）→ gate 融合。

    挂载在魔改模型的 decoder layer 里：
      hidden_states = engram(hidden_states, slot_ids) + hidden_states
    """

    def __init__(
        self,
        hidden_size: int,
        memory: EngramMemory,
        n_index_heads: int = 4,
        index_head_dim: int = 64,
        index_topk: int = 4,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.memory = memory
        self.indexer = DsaEngramIndexer(hidden_size, n_index_heads, index_head_dim, index_topk, dtype=dtype)
        self.gate = EngramGateModule(hidden_size, hidden_size, dtype=dtype)
        self.topk = index_topk

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, S, H]
        slot_ids: torch.Tensor,  # [B, S, n_slots] 多头哈希槽位
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape
        embeds, valid = self.memory.lookup(slot_ids)  # [B,S,n_slots,cap,D], [B,S,n_slots,cap]
        # 展平候选：n_slots x cap -> N_cand
        n_slots, cap = embeds.shape[2], embeds.shape[3]
        N = n_slots * cap
        cand_embeds = embeds.reshape(B, S, N, -1)
        cand_valid = valid.reshape(B, S, N)
        if not bool(cand_valid.any()):
            return hidden_states  # 记忆表为空：直通
        top_indices, route_scores = self.indexer(hidden_states, cand_embeds, cand_valid)
        # 取选中候选的嵌入 [B,S,k,D]
        k = top_indices.shape[-1]
        sel = cand_embeds.gather(
            2, top_indices.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, cand_embeds.shape[-1])
        )
        return self.gate(hidden_states, sel, route_scores, top_indices)
