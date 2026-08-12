"""真实历史数据唤醒仿真——真实记忆 × 真实会话查询 × 真实 bge 全链路。

链路（与生产一致）：
  哈希粗筛（字符 2-gram 交集近似）→ bge 余弦精排 → 阈值 0.42（+自适应多重校正）→ top-K → 分级渲染

ground truth（相关性标注）：查询与记忆的共享主题关键词（粗标注，诚实声明）。

用法：python python/tests/simulate_real.py
"""
import json
import math
import os

STORE = os.path.expanduser('~/.dsh/engram-relay/engrams.jsonl')
QUERIES_FILE = 'F:/dsh/.zcode/real-queries.txt'
EMBED_MODEL = 'F:/dsh/01-memory/engram-trial/bge-small-zh'

# 主题关键词（ground truth 标注用，粗粒度）
TOPIC_KW = {
    '热重载': ['热重载', '重载', 'pnpm', 'hmr', 'patch'],
    '注入': ['注入', 'inject', 'super-injector', '模组', '插件挂载', 'junction'],
    '缓存': ['缓存', '命中', 'cache', 'token', '注入预算'],
    '记忆': ['记忆', 'engram', '图谱', '唤醒', '蒸馏', 'evolve'],
    '浏览器': ['浏览器', 'browser', '面板', 'sider', 'sidebar'],
    '卸载': ['卸载', 'uninject', '卸载器'],
    '迁移': ['迁移', '打包', 'restore', '新电脑', 'zip'],
    '开源': ['开源', 'public', '转移', 'github', '仓库'],
    'solo': ['solo', 'code-server', 'vscode', '侧边栏', '工作区'],
}


def topic_of(text):
    hits = set()
    for topic, kws in TOPIC_KW.items():
        for kw in kws:
            if kw in text:
                hits.add(topic)
    return hits


def bigrams(text):
    return {text[i:i + 2] for i in range(len(text) - 1)} if len(text) >= 2 else set()


def main():
    memories = []
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
    queries = [q.strip() for q in open(QUERIES_FILE, encoding='utf-8') if q.strip()]

    print(f'真实记忆 {len(memories)} 条 × 真实查询 {len(queries)} 条')

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)

    mem_texts = [f"{m.get('title', '')}：{m.get('summary', '')[:200]}" for m in memories]
    mem_vecs = model.encode(mem_texts, normalize_embeddings=True)

    mem_topics = [topic_of(t) for t in mem_texts]

    results = []
    for qi, q in enumerate(queries):
        q_topic = topic_of(q)
        # 1. 哈希粗筛（2-gram 交集近似）
        qb = bigrams(q)
        cand_idx = [i for i, m in enumerate(memories) if qb & bigrams(m.get('title', '') + m.get('summary', ''))]
        if not cand_idx:
            results.append({'q': q, 'recalled': [], 'injected': 0, 'has_related': bool(q_topic), 'gt_hit': False})
            continue
        # 2. bge 精排（真实）
        cand_texts = [mem_texts[i] for i in cand_idx]
        cand_vecs = model.encode(cand_texts, normalize_embeddings=True)
        qv = model.encode([q], normalize_embeddings=True)[0]
        scores = [float(qv @ cv) for cv in cand_vecs]
        # 3. 阈值（0.42 + 自适应多重校正）
        n = len(cand_idx)
        threshold = 0.42 + 0.03 * math.log2(max(1, n / 16))
        top = sorted(
            [(i, s) for i, s in zip(cand_idx, scores) if s >= threshold],
            key=lambda x: -x[1],
        )[:3]
        recalled = [memories[i]['title'] for i, _ in top]
        # ground truth：查询主题与候选记忆主题相交 = 相关
        gt_related = [i for i, s in zip(cand_idx, scores) if q_topic & mem_topics[i] and s >= 0.45]
        gt_hit = any(memories[i]['title'] in recalled for i in gt_related)
        # 4. 分级注入 token 估算（与生产 renderInjection 一致）
        if recalled:
            injected = 40 + (len(recalled[0]) + 60) * 0.7 + sum(len(t) * 0.7 for t in recalled[1:3])
        else:
            injected = 0
        results.append({'q': q, 'recalled': recalled, 'injected': round(injected), 'has_related': bool(gt_related), 'gt_hit': gt_hit})

    related = [r for r in results if r['has_related']]
    unrelated = [r for r in results if not r['has_related']]
    recall = sum(1 for r in related if r['gt_hit']) / max(1, len(related))
    misrecall = sum(1 for r in unrelated if r['recalled']) / max(1, len(unrelated))
    avg_inj = sum(r['injected'] for r in results) / max(1, len(results))

    print('\n=== 真实数据全链路结果 ===')
    print(f'查询总数 {len(results)}（有相关记忆 {len(related)} / 无相关 {len(unrelated)}）')
    print(f'相关召回率: {recall * 100:.1f}%')
    print(f'无关误召率: {misrecall * 100:.1f}%')
    print(f'平均注入: {avg_inj:.0f} token/轮')
    print('\n样本（前 8 个有注入的查询）:')
    shown = 0
    for r in results:
        if r['recalled'] and shown < 8:
            print(f"  「{r['q'][:36]}」→ {r['recalled'][:2]}")
            shown += 1


if __name__ == '__main__':
    main()
