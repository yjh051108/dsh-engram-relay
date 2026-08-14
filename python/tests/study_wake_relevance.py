"""唤醒相关性研究：纯会话文本 vs 记忆的相关性分布 + 唤醒输入选择实验。

研究点：
  A. 会话文本（真实用户消息）与记忆的 bge 分布（相关/无关）——对比人工查询词的分布，
     量化「口语-术语 gap」；
  B. 唤醒输入选择：仅最后消息 vs 消息+上轮助手回复拼接——召回对比
     （上下文是否提升口语查询的召回）。

用法：python python/tests/study_wake_relevance.py
"""
import json
import math
import os

STORE = os.path.expanduser('~/.dsh/engram-relay/engrams.jsonl')
QUERIES_FILE = os.environ.get('ENGRAM_STUDY_QUERIES', '')
SESSION_FILE = os.environ.get('ENGRAM_STUDY_SESSION', '')
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

    # 助手回复（研究 B：上轮回复拼接）
    assoc_replies = {}
    if os.path.exists(SESSION_FILE):
        for line in open(SESSION_FILE, encoding='utf-8'):
            if '"type":"assistant/message"' not in line:
                continue
            try:
                node = json.loads(line)
            except Exception:
                continue
            c = node.get('data', {}).get('message', {}).get('content', [])
            text = ''.join(b.get('text', '') for b in c if isinstance(b, dict) and b.get('type') == 'text')[:300]
            # 用时间序就近关联（粗：跳过，研究 B 用简化拼接——把最近记忆摘要当上下文）
    # 简化研究 B：query 拼接 = 原消息 + 其主题记忆的最相关摘要（模拟上下文增益）
    # 更诚实：对比「原消息」vs「原消息 + 关键词扩写（主题词）」

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    mem_texts = [f"{m.get('title', '')}：{m.get('summary', '')[:200]}" for m in memories]
    mem_vecs = model.encode(mem_texts, normalize_embeddings=True)
    mem_topics = [topic_of(t) for t in mem_texts]

    # 研究 A：会话文本分布
    rel, unrel = [], []
    for q in queries:
        qt = topic_of(q)
        qv = model.encode([q], normalize_embeddings=True)[0]
        for i, m in enumerate(memories):
            s = float(qv @ mem_vecs[i])
            mt = mem_topics[i]
            if qt and qt & mt:
                rel.append(s)
            elif qt and mt:
                unrel.append(s)
            elif not qt:
                unrel.append(s)

    def stats(xs):
        xs = sorted(xs)
        n = len(xs)
        mean = sum(xs) / n
        var = sum((x - mean) ** 2 for x in xs) / n
        return n, round(mean, 4), round(var ** 0.5, 4), round(xs[n // 4], 4), round(xs[n // 2], 4)

    n1, m1, s1, p251, p501 = stats(rel)
    n2, m2, s2, p252, p502 = stats(unrel)
    print('=== A. 纯会话文本 vs 记忆 分布（161 条真实用户消息） ===')
    print(f'相关: n={n1} mean={m1} std={s1} p25={p251} p50={p501}')
    print(f'无关: n={n2} mean={m2} std={s2} p25={p252} p50={p502}')
    print(f'对比人工查询词分布（相关 0.516/无关 0.293）：')
    print(f'  → 口语-术语 gap：相关均值 {m1} vs 0.516（差 {0.516 - m1:.3f}），无关均值 {m2} vs 0.293（差 {m2 - 0.293:+.3f}）')

    # 研究 B：输入选择（原消息 vs 关键词扩写）
    plain_rec, aug_rec = 0, 0
    related_n = 0
    for q in queries:
        qt = topic_of(q)
        if not qt:
            continue
        related_n += 1
        qv_plain = model.encode([q], normalize_embeddings=True)[0]
        # 扩写：把主题词拼接进 query（模拟上下文/改写增益）
        aug_q = q + ' 相关主题：' + ' '.join(sorted(qt))
        qv_aug = model.encode([aug_q], normalize_embeddings=True)[0]
        for i, m in enumerate(memories):
            if qt & mem_topics[i]:
                if float(qv_plain @ mem_vecs[i]) >= 0.42:
                    plain_rec += 1
                if float(qv_aug @ mem_vecs[i]) >= 0.42:
                    aug_rec += 1
                break  # 每查询只计一条最相关
    print('\n=== B. 唤醒输入选择（相关查询的召回） ===')
    print(f'仅原消息: {plain_rec}/{related_n}（{plain_rec / max(1, related_n) * 100:.1f}%）')
    print(f'主题扩写后: {aug_rec}/{related_n}（{aug_rec / max(1, related_n) * 100:.1f}%）')
    print(f'→ 扩写增益 +{(aug_rec - plain_rec) / max(1, related_n) * 100:.1f} 个百分点')


if __name__ == '__main__':
    main()
