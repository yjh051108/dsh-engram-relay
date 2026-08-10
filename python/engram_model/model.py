"""
Qwen3-0.6B 魔改模型：在 decoder layer 注入 EngramFusedModule。

魔改方式（零源码修改，用包装类替换 decoder layer）：
  1. 加载 Qwen3ForCausalLM（torch 原版）；
  2. 把每层 self_attn 前插入 EngramFusedModule（论文 layer_ids 风格，
     可选插入层：默认第 1 层与中段层，可配置）；
  3. 模型 forward 前计算 input_ids 的 N-gram 哈希槽位，随 hidden_states
     一起传给注入层——即「KV 参与 engram 寻址后路由」的完整链路。

注入后的 forward 语义：
  hidden_states = self_attn(...)          # 主注意力（KV cache 只存工作记忆）
  hidden_states = engram(hidden_states) + hidden_states   # engram 稀疏记忆注入
  hidden_states = mlp(hidden_states) + hidden_states

记忆表（EngramMemory）外置可写：插件侧蒸馏出的记忆经
memory.write(slot_ids, embeds) 写入，推理时自动参与哈希寻址 + KV 路由。
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .engram_module import EngramMemory, EngramFusedModule
from .hash import NgramHashMapping

# 论文默认插入层（EngramConfig.layer_ids = [1, 15]）
DEFAULT_ENGRAM_LAYERS = (1, 7)


class EngramQwen3(nn.Module):
    """Qwen3 + Engram 融合模型包装。"""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-0.6B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
        engram_layers: tuple[int, ...] = DEFAULT_ENGRAM_LAYERS,
        num_memory_slots: int = 65536,
        per_slot_capacity: int = 32,
        index_n_heads: int = 4,
        index_head_dim: int = 64,
        index_topk: int = 4,
        hash_layer_ids: tuple[int, ...] = (1,),
        seed: int = 0,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.hidden_size = self.model.config.hidden_size
        self.engram_layers = engram_layers

        # 哈希寻址（论文 NgramHashMapping，多头素数取模）。
        # 记忆表槽位数 = 哈希模数空间（论文 engram_vocab_size 语义）：
        # 65536 槽 × 1024 维 × bf16 ≈ 134MB，8GB VRAM 可容纳；会话记忆
        # 场景不需要论文的 646k 槽（那是预训练静态知识表）。
        self.hasher = NgramHashMapping(
            self.tokenizer,
            layer_ids=hash_layer_ids,
            engram_vocab_size=(num_memory_slots, num_memory_slots),
            seed=seed,
        )
        # 记忆表：槽位空间 = 最大哈希模数 + 1（多头素数递增，必须覆盖全部）。
        lid = self.hasher.layer_ids[0]
        all_mods = [m for heads in self.hasher.vocab_size_across_layers[lid] for m in heads]
        table_slots = max(all_mods) + 1
        self.memory = EngramMemory(table_slots, self.hidden_size, per_slot_capacity, dtype=dtype).to(device)
        # 每注入层一个融合模块（共享同一记忆表）
        self.engram_modules = nn.ModuleDict()
        for lid in engram_layers:
            mod = EngramFusedModule(
                hidden_size=self.hidden_size,
                memory=self.memory,
                n_index_heads=index_n_heads,
                index_head_dim=index_head_dim,
                index_topk=index_topk,
                dtype=dtype,
            ).to(device)
            self.engram_modules[str(lid)] = mod
            self._inject(lid, mod)

    def _inject(self, layer_idx: int, module: EngramFusedModule):
        """把融合模块注入 decoder layer 的 forward（包装 self_attn 输出后）。

        engram_slots 不依赖 kwargs 传递链（transformers 不会把自定义 kwargs
        传到每层），而是由包装后的 model.forward 计算并暂存到实例，
        layer 包装从实例读取。
        """
        layer = self.model.model.layers[layer_idx]
        orig_forward = layer.forward
        owner = self

        def wrapped_forward(*args, **kwargs):
            hidden_states = orig_forward(*args, **kwargs)
            slots = owner._current_engram_slots
            if slots is not None and hidden_states is not None:
                hidden_states = module(hidden_states, slots)
            return hidden_states

        layer.forward = wrapped_forward

        # 包装 model.forward：每次调用（含 generate 内部每步）都重算槽位
        orig_model_forward = self.model.forward

        def wrapped_model_forward(*args, **kwargs):
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            if input_ids is not None:
                try:
                    owner._current_engram_slots = owner.engram_slots_for(input_ids)
                except Exception:
                    owner._current_engram_slots = None
            else:
                owner._current_engram_slots = None
            return orig_model_forward(*args, **kwargs)

        self.model.forward = wrapped_model_forward

    # 当前步的 engram 槽位（由包装的 model.forward 填充）
    _current_engram_slots: torch.Tensor | None = None

    def engram_slots_for(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        """计算 input_ids 的哈希槽位 [B, S, n_slots]（多头哈希，展平）。"""
        if len(self.hasher.layer_ids) == 0:
            return None
        hashes = self.hasher.hash(input_ids.cpu().numpy())
        # 论文 demo 用 layer_ids[0] 的哈希（每层独立乘子；这里取第一个）
        lid = self.hasher.layer_ids[0]
        h = hashes[lid]  # [B, S, n_heads*(max_ngram-1)]
        return torch.from_numpy(h).to(self.device)

    def write_memory(self, slot_ids: list[int], embeds: torch.Tensor):
        """外置记忆写入：slot_ids [N] 与 embeds [N, D]（插件蒸馏结果）。"""
        t = torch.tensor(slot_ids, dtype=torch.long, device=self.device)
        self.memory.write(t, embeds.to(self.device))

    def memory_stats(self) -> dict:
        return {
            "entries": self.memory.num_entries(),
            "slots": self.memory.num_slots,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        repetition_penalty: float = 1.1,
        **kwargs,
    ):
        """带 engram 槽位的生成（槽位由包装的 model.forward 每步计算）。"""
        return self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )


def load_engram_qwen3(model_id: str = "Qwen/Qwen3-0.6B", **kwargs) -> EngramQwen3:
    return EngramQwen3(model_id=model_id, **kwargs)
