# engram 端到端向量融合架构（一份缓存贯通粗筛→精排，目标 100M token 记忆库）

## 0. 设计原则

1. **一次嵌入，处处复用**：记忆在写入时 embed 一次 → 向量缓存（fp16）持久化——粗筛、精排、时序/因果加权、未来 reranker 全部消费同一份缓存，无重复计算、无重复分发。
2. **规模自适应**：<10 万条暴力内积；10 万-100 万 ANN（HNSW+PQ）；>100 万外部向量库。三层实现同一检索接口，切换不动上层。
3. **上下文恒定**：无论记忆总量多少，注入永远是 top-1~3 入口（渐进披露第一层）——记忆库规模与上下文窗口无关。
4. **纯 CPU、零显存**：全部模型走 onnxruntime-node CPU EP（bge-small-zh 24M 参数，CPU 内存 ~96MB fp32 / 24MB int8，单条 embed ~10ms）——无 GPU 依赖，普通笔记本即跑。这是选 bge-**small** 的根本理由。

## 1. 数据流（端到端三级检索）

```
写入：
  记忆 → bge embed（一次，~10ms）→ 512 维向量
       → 双量化落盘：int8 粗筛表 + fp16 精筛表（同一 embed 的两种量化）
       → vectors.i8.bin（1B/维）+ vectors.f16.bin（2B/维）追加
       → 内存索引（mmap 或分块加载）

检索（三级漏斗）：
  ① int8 全量内积粗筛 → top-50（快+省：内存减半，0.04 偏差不影响大致排序）
  ② fp16 精确余弦细筛 → 阈值 0.42/0.50 判定 + 因果传播 + 时序加权
  ③ bge-reranker 重排 top-10 → 最终 top-3（分数差大时跳过，省 50ms）
  注入 top-1~3 入口
```

## 2. 缓存格式（单一分发，双量化）

```
engram-relay/
  engrams.jsonl       # 节点元数据（现有）
  vectors.i8.bin      # int8 粗筛表 [N][512]（1B/维）
  vectors.f16.bin     # fp16 精筛表 [N][512]（2B/维）
  vectors.meta.json   # { count, dim: 512, version }
```

- **双量化来自同一次 embed**：写入时算一次 bge，同时产出 int8（量化到 [-127,127]）与 fp16 两份，无重复模型推理；
- 100M token（50 万条）规模：768MB 总缓存（256+512），单机可行；
- 增量写：新节点双表 append；删除/更新标记 tombstone（惰性压缩）。

## 3. 规模账

| 规模（记忆条数） | 向量内存（fp16） | 检索策略 | 单次查询 |
|---|---|---|---|
| ~5,000（现在） | 5MB | 暴力 | <1ms |
| 50 万（100M token） | 512MB | 暴力（SIMD/多线程） | 50-100ms ✅ |
| 100 万-1000 万 | 1-10GB | HNSW + PQ（faiss 同款） | <10ms |
| >1000 万（>2B token） | 磁盘/分布式 | 外部向量库（Milvus 等，接口预留） | 外部服务 |

**100M token 结论：单机暴力可行**（512MB 内存 + ~100ms 检索，LLM 推理 1-2s 面前可忽略）。

## 4. 检索接口抽象（三层共实现）

```ts
interface VectorIndex {
  add(id: string, vec: Float32Array): void          // 写入时调用
  search(q: Float32Array, k: number): { id: string; score: number }[]  // 粗筛
  remove(id: string): void
  persist(): void                                    // 落盘
}

class BruteForceIndex implements VectorIndex { /* 现在实现：全量内积 */ }
class HnswIndex implements VectorIndex { /* 未来：规模触发时替换 */ }
```

wake.ts 只依赖 `VectorIndex.search`——换实现零改动。

## 5. 与哈希的关系

- **哈希从主路径降级为兜底**：零模型（bge 不可用）时仍能按词汇粗筛 + 重要度排序；
- 向量索引主路径解决哈希的**词汇盲区**（口语-术语 gap 的根因：语义相关但无共享词）。

## 6. 第三级 reranker（设计内，实现后置）

- **主选：Qwen3-Reranker-0.6B**（cross-encoder，2025-2026 中文精度天花板，全面超过 bge-reranker 系；0.6B 是系列最小，CPU 推理 ~2.4GB 内存、~50ms/对）；
- 备选：bge-reranker-base（278M / ~1.1GB，内存紧张时，与 bge-small-zh 同源）；
- **重排语义（不是只排序，是"再筛选 + 重排 + 动态取数"）**：
  1. 输入：② 细筛的 top-10；
  2. 交叉注意力打分（每对 [query+入口] → 相关性 0~1）；
  3. **再筛选**：分数 < 0.3 截掉（reranker 是比双塔阈值 0.42 更准的第二道过滤，纠正边缘误过）；
  4. 按分数重排；
  5. **动态注入条数（分数团规则）**：以 top-1 分数为基准，取"分数 ≥ top1 × 0.9"的条目，上限 3——分数差大（第二名低于 top1×0.9）只注入 1 条（足够确定省噪声）；分数并列注入 2~3 条（多条真相关都给入口）。自动唤醒该动态团通常收窄到 1 条（极克制），手动 recall 可到 3 条；
- 成本：每对 ~50ms × 10 = 500ms；**分数差大时整体跳过**（② 细筛后 top-2 的余弦差距 >0.1 视为已足够确定，不跑 reranker）；
- 缓存不适用（逐对计算），与双量化向量缓存互补不冲突；
- 可配开关：`rerankerModel: 'qwen3-reranker-0.6b' | 'bge-reranker-base' | ''`（空 = 关闭第三级）。

## 7. 实施路线（按分发/收益排序）

1. **向量缓存 + BruteForceIndex**（本阶段：embedder 已 ONNX 化，写入时 embed + append，检索走全量内积替代哈希主路径）；
2. 规模自适应开关（条数 >10 万提示/切换 ANN——当前不做实现）；
3. reranker ONNX 导出与第三级精排；
4. （远期）HNSW 实现——只在真实规模信号出现时启动。

## 8. 诚实边界

- 当前几千条规模：TS 内嵌暴力已是最优（1ms、零索引成本）；
- 50 万条（100M token）：暴力仍可行，但要加**多线程/SIMD**（Node worker 或原生 onnxruntime 批量内积）；
- 1 亿条：必须外部向量库——接口已预留，但那是"插件接到 Milvus"的工作，不在插件内实现。
- **CPU 边界**：bge-small 的 CPU 推理是舒适的（10ms/条）；若未来换大 embedding 模型，CPU 会紧张——那时再谈 GPU 加速，当前设计不碰显存。
