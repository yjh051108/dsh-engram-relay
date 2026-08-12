import time
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('F:/dsh/01-memory/engram-trial/bge-small-zh')
# 预热
m.encode(['预热文本'], normalize_embeddings=True)
for n in [1, 16, 64, 128, 256]:
    texts = [f'记忆条目编号{i}的内容摘要' for i in range(n)]
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        m.encode(texts, normalize_embeddings=True)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    print(f'{n:4d} 条: 中位 {ts[2]:.0f} ms / 最小 {ts[0]:.0f} ms')
