"""
哈希 vs Embedding 检索质量对比（同一图谱、同一查询集）。

对照：
  A. 哈希寻址（当前实现）——精确率基线 54%
  B. Embedding 相似度（bge-small-zh-v1.5）——语义匹配
  C. 混合：哈希粗筛 top-N + embedding 精排（生产管线形态）

指标：命中率 / 精确率（命中的入口是否属于目标主题）
运行：PYTHONIOENCODING=utf-8 python python/embed_compare.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sentence_transformers import SentenceTransformer

# 与 tests/quality-graph.mjs 同一图谱数据
THEMES = [
    {'name': '部署', 'nodes': ['部署方案决策', '部署实施', '灰度验证', '线上发布']},
    {'name': '缓存', 'nodes': ['缓存选型', '缓存接入', '缓存压测', '缓存上线']},
    {'name': '数据库', 'nodes': ['数据库选型', '数据库迁移', '数据校验', '数据上线']},
    {'name': '监控', 'nodes': ['监控方案', '监控接入', '告警演练', '监控上线']},
]
NODE_TEXT = {
    '部署方案决策': '容器化部署方案，端口映射 8080',
    '部署实施': 'Docker Compose 改造完成',
    '灰度验证': '灰度流量 24 小时无异常',
    '线上发布': '全量切换，回滚预案就绪',
    '缓存选型': 'Redis 集群方案',
    '缓存接入': '热点路径接入完成',
    '缓存压测': '压测 QPS 达标',
    '缓存上线': '缓存层全量生效',
    '数据库选型': 'PostgreSQL 主从方案',
    '数据库迁移': '存量数据迁移完成',
    '数据校验': '对账校验通过',
    '数据上线': '数据库切换完成',
    '监控方案': 'Prometheus + Grafana',
    '监控接入': '指标采集接入',
    '告警演练': '告警链路演练通过',
    '监控上线': '监控面板全量展示',
}
ALL_TITLES = [n for t in THEMES for n in t['nodes']]

EMBED_MODEL_PATH = os.environ.get(
    'ENGRAM_EMBED_MODEL', ''
)


# 伪随机（与 quality-graph.mjs 的 mulberry32(7) 同序）
class mulberry32:
    def __init__(self, seed):
        self.a = seed & 0xFFFFFFFF

    def __call__(self):
        self.a = (self.a + 0x6D2B79F5) & 0xFFFFFFFF
        t = self.a
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t = ((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296


def build_queries(n=80, seed=7):
    rng = mulberry32(seed)
    queries = []
    for _ in range(n):
        title = ALL_TITLES[int(rng() * len(ALL_TITLES))]
        theme = next(t for t in THEMES if title in t['nodes'])
        queries.append((f"{theme['name']} {title}", theme['nodes']))
    return queries


def evaluate(match_fn, queries, k=3):
    hits = 0
    prec_sum = 0.0
    for query, theme_nodes in queries:
        top = match_fn(query, k)
        if len(top) > 0:
            hits += 1
            themed = sum(1 for t in top if t in theme_nodes)
            prec_sum += themed / len(top)
    return hits / len(queries), prec_sum / len(queries)


# ---- 哈希寻址（与 src/engram/hash.ts 同一算法：per-position 多头 N-gram 多项式哈希）----
# 忠实移植：mulberry32 乘子 + 素数模数 + FNV-1a token 哈希 + 纯 CJK 按字切分。
def _mulberry32(seed):
    a = seed & 0xFFFFFFFF
    def rng():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t = ((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return rng


def _is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def _next_prime(start, seen):
    c = start + 1
    while True:
        if _is_prime(c) and c not in seen:
            return c
        c += 1


def _hash_str(s):
    h = 0x811C9DC5
    for ch in s:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h & 0xFFFFFFFF


class NgramHash:
    """与 src/engram/hash.ts 的 NgramHashAddressing 同构。"""

    def __init__(self, max_ngram_size=3, heads_per_ngram=4, slots_per_ngram=4096, seed=0):
        self.max_ngram_size = max_ngram_size
        self.heads_per_ngram = heads_per_ngram
        rng = _mulberry32(seed)
        self.multipliers = []
        self.primes = []
        seen = set()
        for n in range(2, max_ngram_size + 1):
            mults = [(int(rng() * 0x3FFFFFFF) * 2 + 1) for _ in range(n)]
            self.multipliers.append(mults)
            heads = []
            start = slots_per_ngram - 1
            for _ in range(heads_per_ngram):
                p = _next_prime(start, seen)
                seen.add(p)
                heads.append(p)
                start = p
            self.primes.append(heads)

    def normalize(self, text):
        import re
        t = text.lower()
        t = re.sub(r'[ \t\r\n]+', ' ', t).strip()
        if not t:
            return []
        tokens = []
        for w in t.split(' '):
            if not w:
                continue
            cleaned = re.sub(r'[^\w\u4e00-\u9fff-]+', '', w)
            if not cleaned:
                continue
            cjk = re.findall(r'[\u4e00-\u9fff]', cleaned)
            if cjk and len(cjk) == len(cleaned):
                tokens.extend(cjk)
            else:
                tokens.append(cleaned)
        return [x for x in tokens if x]

    def hash_tokens(self, tokens):
        MOD = 2147483647
        slots = set()
        for n_idx, n in enumerate(range(2, self.max_ngram_size + 1)):
            mults = self.multipliers[n_idx]
            primes = self.primes[n_idx]
            for i in range(n - 1, len(tokens)):
                mix = 0
                for k in range(n):
                    mix = (mix + _hash_str(tokens[i - n + 1 + k]) * mults[k]) % MOD
                for h, p in enumerate(primes):
                    slots.add(f"n{n}h{h}:{mix % p}")
        return slots

    def hash(self, text):
        return self.hash_tokens(self.normalize(text))


class HashMatcher:
    def __init__(self, titles):
        self.hasher = NgramHash()
        self.slots = {}
        for t in titles:
            for k in self.hasher.hash(t):
                self.slots.setdefault(k, []).append(t)

    def match(self, query, k):
        scores = {}
        for key in self.hasher.hash(query):
            for t in self.slots.get(key, []):
                scores[t] = scores.get(t, 0) + 1
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [t for t, _ in ranked]


def main():
    print("=== 哈希 vs Embedding 检索质量对比 ===")
    print(f"Embedding 模型: {EMBED_MODEL_PATH}")
    print("加载模型...")
    model = SentenceTransformer(EMBED_MODEL_PATH)
    print(f"模型加载完成，向量维度 {model.get_sentence_embedding_dimension()}")

    queries = build_queries()

    # 预编码：节点向量（title + summary 拼接）
    node_emb = np.asarray(model.encode(
        [f"{t}：{NODE_TEXT[t]}" for t in ALL_TITLES],
        normalize_embeddings=True,
    ))

    # A. 纯哈希（生产当前基线）
    hash_matcher = HashMatcher(ALL_TITLES)
    hash_match = lambda q, k: hash_matcher.match(q, k)

    # B. 纯 Embedding：查询向量 vs 节点向量 余弦相似度 top-k
    def emb_match(query, k):
        qv = model.encode([query], normalize_embeddings=True)
        sims = node_emb @ np.asarray(qv).T
        top_idx = np.argsort(sims[:, 0])[::-1][:k]
        return [ALL_TITLES[i] for i in top_idx]

    # C. 混合（生产管线形态）：哈希粗筛 top-16 → embedding 精排 top-k
    def hybrid_match(query, k, coarse=16):
        cands = hash_matcher.match(query, coarse)
        if not cands:
            return []
        idx = [ALL_TITLES.index(c) for c in cands]
        qv = model.encode([query], normalize_embeddings=True)
        sims = node_emb[idx] @ np.asarray(qv).T
        order = np.argsort(sims[:, 0])[::-1][:k]
        return [cands[i] for i in order]

    print("--- 结果 ---")
    hit_a, prec_a = evaluate(hash_match, queries)
    hit_b, prec_b = evaluate(emb_match, queries)
    hit_c, prec_c = evaluate(hybrid_match, queries)
    print(f"A. 哈希寻址:     命中 {hit_a:.0%} | 精确率 {prec_a:.0%}")
    print(f"B. Embedding:    命中 {hit_b:.0%} | 精确率 {prec_b:.0%}")
    print(f"C. 混合(粗16+精排): 命中 {hit_c:.0%} | 精确率 {prec_c:.0%}")
    print(f"提升: B 精确率 {(prec_b - prec_a) * 100:+.0f} 个百分点, "
          f"C 精确率 {(prec_c - prec_a) * 100:+.0f} 个百分点")

    # 样例：一条哈希误命中的查询，embedding 如何表现
    print("\n--- 样例（哈希易误命中：共享词） ---")
    shown = 0
    for q, theme_nodes in queries:
        top_a = hash_match(q, 3)
        top_c = hybrid_match(q, 3)
        hit_a_ = sum(1 for t in top_a if t in theme_nodes)
        hit_c_ = sum(1 for t in top_c if t in theme_nodes)
        if hit_a_ < 3 and shown < 3:
            shown += 1
            print(f"  Q: {q}")
            print(f"    目标: {theme_nodes}")
            print(f"    hash top3: {top_a}（主题命中 {hit_a_}/3）")
            print(f"    hyb  top3: {top_c}（主题命中 {hit_c_}/3）")


if __name__ == "__main__":
    main()
