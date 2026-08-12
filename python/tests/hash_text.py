"""JS 版 NgramHashAddressing 的 Python 移植（仿真用，与生产一致）。

含 2026-08-13 的中英混合词修复（逐字/逐段拆分）。
"""
import re

MULTIPLIER_PRIMES = [4099, 4127, 4133, 4139]


def _mulberry32(seed):
    a = seed & 0xffffffff
    while True:
        a = (a + 0x6d2b79f5) & 0xffffffff
        t = a ^ ((a << 15) & 0xffffffff)
        t = (t + ((t ^ (t >> 7)) * 61) & 0xffffffff) & 0xffffffff
        yield ((t ^ (t >> 14)) & 0xffffffff) / 4294967296


def _is_prime(n):
    if n < 2:
        return False
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in [2, 3, 5, 7, 11, 13, 17]:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(1, s):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _fnv1a(s):
    h = 0x811c9dc5
    for ch in s:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xffffffff
    return h


class TextHashAddressing:
    def __init__(self, max_ngram=3, heads=4, slots=4096, seed=0):
        self.max_ngram = max_ngram
        self.heads = heads
        self.slots = slots
        rng = _mulberry32(seed)
        self.multipliers = []
        for n in range(2, max_ngram + 1):
            self.multipliers.append([int(next(rng) * 0x3fffffff) * 2 + 1 for _ in range(n)])
        self.primes = []
        seen = set()
        start = slots - 1
        for _n in range(2, max_ngram + 1):
            row = []
            for _h in range(heads):
                p = start + 1
                while not (_is_prime(p) and p not in seen):
                    p += 1
                seen.add(p)
                row.append(p)
                start = p
            self.primes.append(row)

    def normalize(self, text):
        t = text.lower().strip()
        t = re.sub(r'[ \t\r\n]+', ' ', t)
        tokens = []
        for w in t.split(' '):
            cleaned = re.sub(r'[^\w\u4e00-\u9fff-]+', '', w)
            if not cleaned:
                continue
            # 逐字（汉字）/逐段（字母数字）拆分——中英混合词修复
            for p in re.findall(r'[\u4e00-\u9fff]|[\w-]+', cleaned):
                tokens.append(p)
        return tokens

    def hash(self, text):
        tokens = self.normalize(text)
        slot_set = set()
        for n in range(2, self.max_ngram + 1):
            mults = self.multipliers[n - 2]
            primes = self.primes[n - 2]
            for i in range(n - 1, len(tokens)):
                mix = 0
                for k in range(n):
                    mix = (mix + _fnv1a(tokens[i - n + 1 + k]) * mults[k]) % 2147483647
                for h in range(self.heads):
                    slot_set.add(f'n{n}h{h}:{mix % primes[h]}')
        return slot_set


class TextHashIndex:
    """槽位索引 + lookup（仿真用）。"""

    def __init__(self, hasher=None):
        self.hasher = hasher or TextHashAddressing()
        self.slot_index = {}

    def add(self, node_id, text):
        for slot in self.hasher.hash(text):
            self.slot_index.setdefault(slot, []).append(node_id)

    def lookup(self, text, limit=256):
        seen = set()
        hits = []
        for slot in self.hasher.hash(text):
            for nid in self.slot_index.get(slot, []):
                if nid not in seen:
                    seen.add(nid)
                    hits.append(nid)
        return hits[:limit]
