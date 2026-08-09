"""
哈希寻址测试：验证 NgramHashMapping 的确定性 + 归一化语义。

用轻量 tokenizer 跑（无需下载 Qwen3）——用 transformers 自带的小
tokenizer 或手工 stub。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from engram_model.hash import NgramHashMapping, CompressedTokenizer


class StubTokenizer:
    """极简 tokenizer stub：ASCII 字符级 token，验证哈希数学。"""

    def __init__(self):
        self.vocab = [chr(c) for c in range(32, 127)] + ["<|pad|>"]
        self._map = {t: i for i, t in enumerate(self.vocab)}

    def __len__(self):
        return len(self.vocab)

    def decode(self, ids, skip_special_tokens=False):
        if isinstance(ids, int):
            return self.vocab[ids] if 0 <= ids < len(self.vocab) else ""
        return "".join(self.vocab[i] for i in ids if 0 <= i < len(self.vocab))

    def convert_ids_to_tokens(self, tid):
        return self.vocab[tid] if 0 <= tid < len(self.vocab) else ""

    def encode(self, text):
        return [self._map[c] for c in text if c in self._map]


def test_hash_deterministic():
    tok = StubTokenizer()
    h = NgramHashMapping(tok, max_ngram_size=3, n_head_per_ngram=4, layer_ids=(1,), seed=0)
    ids = np.array([[tok.encode("hello world")[0] for _ in range(5)]])  # [1,5]
    a = h.hash(ids)
    b = h.hash(ids.copy())
    for lid in h.layer_ids:
        assert np.array_equal(a[lid], b[lid]), "相同输入必须相同哈希"
    print("✓ test_hash_deterministic")


def test_hash_multihead_shape():
    tok = StubTokenizer()
    h = NgramHashMapping(tok, max_ngram_size=3, n_head_per_ngram=4, layer_ids=(1,), seed=0)
    ids = np.array([[1, 2, 3, 4, 5]])
    result = h.hash(ids)
    hh = result[1]
    # n=2 与 n=3 两级 × 4 头 = 8 通道
    assert hh.shape[2] == 8, f"期望 8 通道，实际 {hh.shape[2]}"
    print("✓ test_hash_multihead_shape")


def test_compressed_tokenizer_normalizes():
    tok = StubTokenizer()
    ct = CompressedTokenizer(tok)
    assert ct._normalize_str("Hello  World") == "hello world"
    assert ct._normalize_str("  ABC\tDEF  ") == "abc def"
    print("✓ test_compressed_tokenizer_normalizes")


def test_hash_different_inputs_diverge():
    tok = StubTokenizer()
    h = NgramHashMapping(tok, max_ngram_size=3, n_head_per_ngram=4, layer_ids=(1,), seed=0)
    a = h.hash(np.array([[tok.encode("deploy port")[0] for _ in range(6)]]))
    b = h.hash(np.array([[tok.encode("write poem")[0] for _ in range(6)]]))
    hh_a, hh_b = a[1], b[1]
    # 8 个通道中不应全部相同
    assert not np.array_equal(hh_a, hh_b), "不同输入不应全通道相同"
    print("✓ test_hash_different_inputs_diverge")


if __name__ == "__main__":
    test_hash_deterministic()
    test_hash_multihead_shape()
    test_compressed_tokenizer_normalizes()
    test_hash_different_inputs_diverge()
    print("\n=== all hash tests PASS ===")
