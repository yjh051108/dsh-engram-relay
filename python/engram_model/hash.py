"""
Engram Hash Addressing (DeepSeek Engram 论文移植)

对 token 序列做 2/3-gram 多项式哈希（multi-head，多素数取模），
O(1) 确定性寻址到巨大记忆表的槽位。确定性寻址 = 相同模式永远命中
相同槽位——「比普通向量索引更强」的根源（精确匹配，非相似度近似）。

移植自 deepseek-ai/Engram engram_demo_v1.py 的 NgramHashMapping，
用 numpy 实现以匹配论文的确定性随机数生成。
"""

from __future__ import annotations

import numpy as np

from sympy import isprime


class CompressedTokenizer:
    """归一化 tokenizer：NFKC + 大小写折叠 + 空白归一（论文 §CompressedTokenizer）。"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.lookup_table, self.num_new_token = self._build_lookup_table()

    def _build_lookup_table(self):
        old2new = {}
        key2new = {}
        new_tokens = []
        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)
            if "\ufffd" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self._normalize_str(text)
                key = norm if norm else text
            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid
        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]
        return lookup, len(new_tokens)

    @staticmethod
    def _normalize_str(text: str) -> str:
        import unicodedata
        s = unicodedata.normalize("NFKC", text)
        s = unicodedata.normalize("NFD", s)
        # strip accents
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = s.lower()
        import re
        s = re.sub(r"[ \t\r\n]+", " ", s)
        s = s.strip()
        return s

    def __len__(self):
        return self.num_new_token

    def compress(self, input_ids):
        arr = np.asarray(input_ids, dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        valid = arr[pos_mask]
        out[pos_mask] = self.lookup_table[valid]
        return out


class NgramHashMapping:
    """多头 N-gram 哈希寻址（论文 NgramHashMapping 移植）。"""

    def __init__(
        self,
        tokenizer,
        engram_vocab_size=(129280 * 5, 129280 * 5),
        max_ngram_size=3,
        n_head_per_ngram=8,
        layer_ids=(1, 15),
        pad_id=2,
        seed=0,
    ):
        self.max_ngram_size = max_ngram_size
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        self.compressed_tokenizer = CompressedTokenizer(tokenizer)
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])

        max_long = np.iinfo(np.int64).max
        m_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, m_max // 2)
        PRIME_1 = 10007

        self.layer_multipliers = {}
        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            rng = np.random.default_rng(base_seed)
            r = rng.integers(low=0, high=half_bound, size=(self.max_ngram_size,), dtype=np.int64)
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers

        self.engram_vocab_size = engram_vocab_size
        self.vocab_size_across_layers = self._calculate_vocab_size_across_layers()

    def _calculate_vocab_size_across_layers(self):
        seen_primes = set()
        vocab = {}
        for layer_id in self.layer_ids:
            all_ngram_vocab = []
            for ngram in range(2, self.max_ngram_size + 1):
                heads = []
                vocab_size = self.engram_vocab_size[ngram - 2]
                start = vocab_size - 1
                for _ in range(self.n_head_per_ngram):
                    p = self._next_prime(start, seen_primes)
                    seen_primes.add(p)
                    heads.append(p)
                    start = p
                all_ngram_vocab.append(heads)
            vocab[layer_id] = all_ngram_vocab
        return vocab

    @staticmethod
    def _next_prime(start, seen):
        c = start + 1
        while True:
            if isprime(c) and c not in seen:
                return c
            c += 1

    def _get_ngram_hashes(self, input_ids, layer_id):
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape
        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k):
            if k == 0:
                return x
            shifted = np.pad(x, ((0, 0), (k, 0)), mode="constant", constant_values=self.pad_id)[:, :T]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]
        all_hashes = []
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = tokens[0] * multipliers[0]
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            head_vocab = self.vocab_size_across_layers[layer_id][n_gram_index]
            for j in range(self.n_head_per_ngram):
                mod = int(head_vocab[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))
        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids):
        input_ids = self.compressed_tokenizer.compress(input_ids)
        hashes = {}
        for layer_id in self.layer_ids:
            hashes[layer_id] = self._get_ngram_hashes(input_ids, layer_id)
        return hashes


def build_hasher(tokenizer, **kwargs):
    """便捷工厂。"""
    return NgramHashMapping(tokenizer, **kwargs)
