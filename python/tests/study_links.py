"""真实关联性消融实验——各关联维度（因果/时序/双链）对召回的真实贡献。

在真实记忆 × 真实查询 × 真实哈希 × 真实 bge 上，逐维度叠加：
  基线（语义 top-K）→ +因果传播 → +时序衰减 → +双链共现 → 全量
测量每步的召回增量。

关联边：真实记忆的因果边太少（历史记忆未声明），以主题链合成边替代
（机制验证用；边本身在真实系统由蒸馏自动因果补齐）。

用法：python python/tests/study_links.py
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hash_text import TextHashAddressing, TextHashIndex  # noqa: E402

STORE = os.path.expanduser('~/.dsh/engram-relay/engrams.jsonl')
QUERIES_FILE = os.environ.get('ENGRAM_STUDY_QUERIES', '')
EMBED_MODEL = os.environ.get('ENGRAM_EMBED_MODEL', '')

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
    '语音': ['语音', 'voice', '聊天'],
    '仿真': ['仿真', 'simulate', '压力', 'bge', '分布'],
}


def topic_of(text):
    hits = set()
    for topic, kws in TOPIC_KW.items():
        for kw in kws:
            if kw in text:
                hits.add(topic)
    return hits


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

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    mem_texts = [f"{m.get('title', '')}：{m.get('summary', '')[:200]}" for m in memories]
    mem_vecs = model.encode(mem_texts, normalize_embeddings=True)
    mem_topics = [topic_of(t) for t in mem_texts]

    idx = TextHashIndex()
    for i, t in enumerate(mem_texts):
        idx.add(i, t)

    # 合成因果链（主题内相邻记忆相连，模拟蒸馏自动因果建的边）
    causes = {}
    effects = {}
    for t in set(t for mt in mem_topics for t in mt):
        ids = [i for i, mt in enumerate(mem_topics) if t in mt]
        for a, b in zip(ids, ids[1:]):
            effects.setdefault(a, []).append(b)
            causes.setdefault(b, []).append(a)

    # 双链（同一主题的任意两条互链——共现近似）
    links = {}
    for t in set(t for mt in mem_topics for t in mt):
        ids = [i for i, mt in enumerate(mem_topics) if t in mt]
        for a in ids:
            for b in ids:
                if a != b:
                    links.setdefault(a, set()).add(b)

    # 时序（turn 排序，模拟节点创建顺序）
    turns = {i: m.get('turn', i) for i, m in enumerate(memories)}

    def run(use_causal, use_recency, use_links, threshold=0.42):
        related_n = 0
        hit = 0
        for q in queries:
            qt = topic_of(q)
            if not qt:
                continue
            cand = idx.lookup(q, 256)
            if not cand:
                continue
            qv = model.encode([q], normalize_embeddings=True)[0]
            base = {i: float(qv @ mem_vecs[i]) for i in cand}
            if not any(s >= threshold for s in base.values()):
                continue
            related_n += 1
            gt = [i for i in cand if qt & mem_topics[i]]
            # 激活：基线分数
            act = dict(base)
            # 因果传播：前因/后果的分数按 0.5 衰减注入
            if use_causal:
                for _round in range(2):
                    add = {}
                    for i, s in act.items():
                        for nb in causes.get(i, []) + effects.get(i, []):
                            add[nb] = max(add.get(nb, 0), s * 0.5)
                    for k, v in add.items():
                        act[k] = max(act.get(k, 0), v)
            # 时序衰减
            def recency(i):
                return 1 + (0.25 if use_recency and turns.get(i, 99999) > len(memories) - 30 else 0)
            # 双链共现加成
            def linkBoost(i):
                return 1 + (0.15 if use_links and any(base.get(j, 0) >= threshold for j in links.get(i, set())) else 0)
            # 排序 top-3
            top = sorted(cand, key=lambda i: act.get(i, 0) * recency(i) * linkBoost(i), reverse=True)[:3]
            if any(i in gt for i in top):
                hit += 1
        return hit, related_n

    h0, n0 = run(False, False, False)
    h1, _ = run(True, False, False)
    h2, _ = run(False, True, False)
    h3, _ = run(False, False, True)
    h4, n4 = run(True, True, True)

    print('=== 真实关联性消融（真实记忆 × 真实查询 × 真实 bge） ===')
    print(f'基线（纯语义）:       {h0}/{n0} = {h0 / max(1, n0) * 100:.1f}%')
    print(f'+ 因果传播:           {h1}/{n0} = {h1 / max(1, n0) * 100:.1f}%  (Δ{(h1 - h0) / max(1, n0) * 100:+.1f})')
    print(f'+ 时序衰减:           {h2}/{n0} = {h2 / max(1, n0) * 100:.1f}%  (Δ{(h2 - h0) / max(1, n0) * 100:+.1f})')
    print(f'+ 双链共现:           {h3}/{n0} = {h3 / max(1, n0) * 100:.1f}%  (Δ{(h3 - h0) / max(1, n0) * 100:+.1f})')
    print(f'全量（因果+时序+双链）: {h4}/{n4} = {h4 / max(1, n4) * 100:.1f}%  (Δ{(h4 - h0) / max(1, n0) * 100:+.1f})')


if __name__ == '__main__':
    main()
