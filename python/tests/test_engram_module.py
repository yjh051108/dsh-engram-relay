"""
融合模块纯逻辑测试（不加载大模型）：
- EngramMemory 读写/查找
- DsaEngramIndexer 打分 + top-K 稀疏选择
- EngramGateModule gate 融合
- EngramFusedModule 全链路（哈希槽位 → 路由 → 融合）
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from engram_model.engram_module import (
    EngramMemory,
    DsaEngramIndexer,
    EngramGateModule,
    EngramFusedModule,
)


def test_memory_write_lookup():
    mem = EngramMemory(num_slots=16, embed_dim=8, per_slot_capacity=4, dtype=torch.float32)
    # 写入槽 2、5
    embeds = torch.randn(2, 8)
    mem.write(torch.tensor([2, 5]), embeds)
    out, valid = mem.lookup(torch.tensor([[2, 5, 9]]))
    assert valid[0, 0, 0].item() is True, "槽 2 应有记忆"
    assert valid[0, 1, 0].item() is True, "槽 5 应有记忆"
    assert valid[0, 2, 0].item() is False, "槽 9 应为空"
    torch.testing.assert_close(out[0, 0, 0], embeds[0])
    print("✓ test_memory_write_lookup")


def test_memory_capacity():
    mem = EngramMemory(num_slots=4, embed_dim=4, per_slot_capacity=2, dtype=torch.float32)
    mem.write(torch.tensor([0, 0, 0]), torch.randn(3, 4))  # 槽 0 写 3 条 > cap 2
    _, valid = mem.lookup(torch.tensor([[0]]))
    # 只能容纳 2 条（第 3 条写不进去）
    assert valid[0, 0, 0].item() is True
    assert valid[0, 0, 1].item() is True
    assert valid.shape[-1] == 2
    print("✓ test_memory_capacity")


def test_indexer_route_topk():
    torch.manual_seed(0)
    indexer = DsaEngramIndexer(hidden_size=16, n_heads=2, head_dim=8, topk=2, dtype=torch.float32)
    B, S, N = 1, 2, 6
    hidden = torch.randn(B, S, 16)
    cand = torch.randn(B, S, N, 16)
    valid = torch.ones(B, S, N, dtype=torch.bool)
    # 构造强相关候选 3：让 wk(cand3) 与 wq(hidden) 对齐（在 key 空间直接操作）
    with torch.no_grad():
        q = indexer.wq(hidden)  # [B,S,H*D]
        q = q.view(B, S, indexer.n_heads, indexer.head_dim)
        # 目标：候选 3 经 wk 后等于每个 head 的 query 方向
        k3_target = q.permute(0, 1, 3, 2)  # [B,S,D,H]（转置对齐）
        # 用 wk 的伪逆近似构造候选嵌入：cand3 = pinv(wk) * target
        wk = indexer.wk.weight  # [head_dim, 16]
        wk_pinv = torch.linalg.pinv(wk)  # [16, head_dim]
        # 对每个 head 分别构造，然后平均（简化：用 head 0 的 query）
        target_vec = q[..., 0, :]  # [B,S,D] head 0 的 query
        cand3 = target_vec @ wk_pinv.T  # [B,S,16]
        cand.data[:, :, 3, :] = cand3 + 0.0
        # 其他候选保持随机（低相关）
    top_indices, scores = indexer(hidden, cand, valid)
    assert top_indices.shape[-1] == 2, "应选 top-2"
    # 候选 3 应被选中（得分最高）
    assert 3 in top_indices[0, 0].tolist() or 3 in top_indices[0, 1].tolist(), \
        f"强相关候选应入选，实际 {top_indices.tolist()}"
    # 选中候选得分应有效
    gathered = scores.gather(-1, top_indices.clamp(min=0))
    assert bool((gathered > -1e6).all()), "选中候选得分应有效"
    print("✓ test_indexer_route_topk", top_indices.tolist())


def test_gate_fusion():
    torch.manual_seed(0)
    gate = EngramGateModule(engram_hidden=8, hidden_size=16, dtype=torch.float32)
    B, S, k = 1, 2, 3
    hidden = torch.randn(B, S, 16)
    sel = torch.randn(B, S, k, 8)
    scores = torch.randn(B, S, 8)  # N=8 候选
    top_idx = torch.tensor([[[0, 1, 2], [0, 1, 2]]])  # [B=1, S=2, k=3]
    out = gate(hidden, sel, scores, top_idx)
    assert out.shape == hidden.shape, "输出形状不变"
    assert not torch.allclose(out, hidden), "gate 融合应改变 hidden states"
    print("✓ test_gate_fusion")


def test_fused_module_empty_memory_passthrough():
    torch.manual_seed(0)
    memory = EngramMemory(num_slots=8, embed_dim=16, per_slot_capacity=4, dtype=torch.float32)
    fused = EngramFusedModule(hidden_size=16, memory=memory, index_topk=2, dtype=torch.float32)
    B, S = 1, 3
    hidden = torch.randn(B, S, 16)
    slots = torch.randint(0, 8, (B, S, 8))
    out = fused(hidden, slots)
    torch.testing.assert_close(out, hidden, msg="空记忆表应直通")
    print("✓ test_fused_module_empty_memory_passthrough")


def test_fused_module_full_chain():
    torch.manual_seed(1)
    memory = EngramMemory(num_slots=8, embed_dim=16, per_slot_capacity=4, dtype=torch.float32)
    fused = EngramFusedModule(hidden_size=16, memory=memory, n_index_heads=2, index_head_dim=8, index_topk=2, dtype=torch.float32)
    B, S = 1, 2
    # 写入槽 3：一条记忆
    memory.write(torch.tensor([3]), torch.randn(1, 16))
    hidden = torch.randn(B, S, 16)
    # 槽位命中 3（形状与 hidden 的 B/S 一致）
    slots = torch.tensor([[[3, 0, 0, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0, 0, 0]]])
    out = fused(hidden, slots)
    assert out.shape == hidden.shape
    # 记忆非空时应产生融合（不要求必变——gate 可能接近 0，但路径可执行）
    print("✓ test_fused_module_full_chain")


def test_indexer_masks_invalid():
    torch.manual_seed(2)
    indexer = DsaEngramIndexer(hidden_size=16, n_heads=2, head_dim=8, topk=4, dtype=torch.float32)
    B, S, N = 1, 1, 5
    hidden = torch.randn(B, S, 16)
    cand = torch.randn(B, S, N, 16)
    valid = torch.tensor([[[True, False, True, False, True]]])
    top_indices, scores = indexer(hidden, cand, valid)
    # 无效候选得分应为 -inf
    assert torch.isinf(scores[0, 0, 1]).item() and scores[0, 0, 1] < 0
    assert torch.isinf(scores[0, 0, 3]).item() and scores[0, 0, 3] < 0
    # 选中的必须是有效候选
    for idx in top_indices[0, 0].tolist():
        assert valid[0, 0, idx].item() is True
    print("✓ test_indexer_masks_invalid")


if __name__ == "__main__":
    test_memory_write_lookup()
    test_memory_capacity()
    test_indexer_route_topk()
    test_gate_fusion()
    test_fused_module_empty_memory_passthrough()
    test_fused_module_full_chain()
    test_indexer_masks_invalid()
    print("\n=== all engram module tests PASS ===")
