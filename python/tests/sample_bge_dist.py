"""真实 bge 相似度分布采样——校准唤醒仿真模型。

对真实记忆样本与查询对，统计：
  - 相关对（查询与记忆同主题）：余弦相似度分布（均值/方差/分位数）
  - 无关对（查询与记忆不同主题）：余弦相似度分布
以此校准仿真里的 fake embedder（0.42 阈值的选取依据）。

用法：python python/tests/sample_bge_dist.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

# 采样用真实记忆条目（从 engrams.jsonl 读标题+摘要）
STORE = os.path.expanduser('~/.dsh/engram-relay/engrams.jsonl')
EMBED_MODEL = os.environ.get('ENGRAM_EMBED_MODEL', '')

QUERIES = [
    '缓存命中率优化方案',          # 主题:缓存
    '插件热重载和注入',            # 主题:注入
    '浏览器操控链路',              # 主题:浏览器
    '路由残留自愈',                # 主题:路由
    '记忆图谱设计',                # 主题:记忆
    '红烧肉的家常做法',            # 无关
    '今天天气怎么样',              # 无关
    '明天开会时间',                # 无关
]

TOPIC_KEYWORDS = {
    '缓存': ['缓存', '命中', 'cache', 'token'],
    '注入': ['注入', '插件', 'inject', '热重载'],
    '浏览器': ['浏览器', 'browser', '面板', 'panel'],
    '路由': ['路由', 'route', '自愈', 'duplicate'],
    '记忆': ['记忆', 'engram', '图谱', '唤醒'],
}


def topic_of(text):
    for topic, kws in TOPIC_KEYWORDS.items():
        for kw in kws:
            if kw in text:
                return topic
    return None


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb or 1)


def main():
    # 读真实记忆
    memories = []
    if os.path.exists(STORE):
        for line in open(STORE, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            try:
                node = json.loads(line)
            except Exception:
                continue
            if node.get('status') == 'pending':
                continue
            memories.append(node)

    print(f'真实记忆样本: {len(memories)} 条')
    if len(memories) > 60:
        memories = memories[:60]

    # 加载 bge（与生产 server 一致：sentence_transformers）
    if not EMBED_MODEL:
        raise SystemExit('请设置环境变量 ENGRAM_EMBED_MODEL 指向本地 bge-small-zh 目录')
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)

    def embed(texts):
        return model.encode(texts, normalize_embeddings=True)

    mem_texts = [f"{m.get('title', '')}：{m.get('summary', '')[:200]}" for m in memories]
    mem_vecs = embed(mem_texts)
    q_vecs = embed(QUERIES)

    rel, unrel = [], []
    pairs = []
    for qi, q in enumerate(QUERIES):
        qt = topic_of(q)
        for mi, m in enumerate(memories):
            mt = topic_of(f"{m.get('title', '')} {m.get('summary', '')}")
            s = cosine(q_vecs[qi], mem_vecs[mi])
            pairs.append((q, m.get('title', ''), qt, mt, s))
            if qt and mt == qt:
                rel.append(s)
            elif qt and mt and mt != qt:
                unrel.append(s)
            elif qt is None:
                unrel.append(s)

    def stats(xs):
        xs = sorted([float(x) for x in xs])
        n = len(xs)
        mean = sum(xs) / n
        var = sum((x - mean) ** 2 for x in xs) / n
        return {
            'n': n, 'mean': round(mean, 4), 'std': round(var ** 0.5, 4),
            'min': round(xs[0], 4), 'max': round(xs[-1], 4),
            'p25': round(xs[n // 4], 4), 'p50': round(xs[n // 2], 4), 'p75': round(xs[3 * n // 4], 4),
        }

    print('\n=== 相关对（同主题）分布 ===')
    print(json.dumps(stats(rel), ensure_ascii=False, indent=2))
    print('\n=== 无关对分布 ===')
    print(json.dumps(stats(unrel), ensure_ascii=False, indent=2))
    print('\n=== 阈值扫描（0.30/0.35/0.40/0.42/0.45/0.50） ===')
    for th in [0.30, 0.35, 0.40, 0.42, 0.45, 0.50]:
        rec = sum(1 for x in rel if x >= th) / len(rel)
        mis = sum(1 for x in unrel if x >= th) / len(unrel)
        print(f'阈值 {th}: 相关通过 {rec * 100:.0f}% / 无关误过 {mis * 100:.1f}%')


if __name__ == '__main__':
    main()
