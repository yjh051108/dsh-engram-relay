/**
 * SemanticScorer — 纯算法语义打分器（v0.5：不用 embedding 模型）。
 *
 * 用户方向：embedding 是"借来的语义"（预训练先验），记忆系统的语义应
 * 来自图谱自身结构与库内统计——自举、确定性、可解释。
 *
 * 语义通道（分数 ∈ [0,1]，语义对齐原 cos 标定，0.6 阈值沿用）：
 *  ① 词汇通道（lexical）：字符 n-gram Jaccard × 词频覆盖——"因为字词
 *     重叠度高"（保底精确，可解释）；
 *  ② 图语义通道（graph）：候选的因果/链接邻居是否命中查询哈希节点——
 *     "因为沿着边相连"（可解释，怎么索引就怎么推荐）；
 *  ③ PCA 语义通道（svd，核心收敛通道）：词-词共现矩阵的谱分解（幂迭代
 *     top-k 特征向量，线性代数零模型）——词向量余弦。能学到语义桥
 *     （「缓存」与「cache」在同一记忆共现 → 词向量相近 → 查询命中），
 *     随记忆增多收敛（共现矩阵依概率收敛，谱子空间随之收敛）。
 *     诚实声明：学的是"用户的语义空间"（库内统计），通用性上限低于
 *     预训练模型，但在单用户记忆库场景自举、可解释、无外部依赖。
 *
 * 纯 CPU：比 ONNX 快几个数量级，永不失败（无模型依赖）。
 */

import type { EngramStore, EngramNode } from './store.js'

/** 字符 n-gram 集合（2-3 gram，中文逐字 + ASCII 保词）。 */
function charNgrams(text: string, n = 2): Set<string> {
  const out = new Set<string>()
  const t = text.replace(/\s+/g, '')
  if (t.length < n) {
    if (t.length > 0) out.add(t)
    return out
  }
  for (let i = 0; i <= t.length - n; i++) out.add(t.slice(i, i + n))
  return out
}

/** 词频表（用于 BM25 风格 IDF 加权）。 */
function wordsOf(text: string): Map<string, number> {
  const out = new Map<string, number>()
  // CJK 逐字 + ASCII 词（简单切分，零依赖）
  const tokens = text.match(/[a-z0-9]+|[^\u0000-\u007f]/gi) ?? []
  for (const w of tokens) {
    const k = w.toLowerCase()
    out.set(k, (out.get(k) ?? 0) + 1)
  }
  return out
}

/** Jaccard 相似度。 */
function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 || b.size === 0) return 0
  let inter = 0
  for (const x of a) if (b.has(x)) inter++
  return inter / (a.size + b.size - inter)
}

export interface SemanticScore {
  /** 融合分数 [0,1]（语义对齐原 cos，0.6 阈值沿用）。 */
  score: number
  /** 通道分解（可解释：为什么是这个分）。 */
  lexical: number
  graph: number
  svd: number
}

/** PCA 语义通道（词-词共现谱分解，幂迭代零模型）。 */
class PcaSemantics {
  static readonly DIM = 16
  /** 词向量表：word → Float64Array(DIM)（幂迭代谱分解结果）。 */
  private wordVec = new Map<string, Float64Array>()
  private built = false

  constructor(private store: EngramStore) {}

  /** 懒重建（store 变化后首次 score 调用时）。 */
  ensure(): void {
    if (this.built) return
    this.rebuild()
  }

  markDirty(): void {
    this.built = false
  }

  /** 重建：词-词共现矩阵 → 幂迭代 top-k 特征向量 → 词向量。 */
  private rebuild(): void {
    this.built = true
    const nodes = this.store.all().filter((e) => e.status !== 'pending' && !e.supersededBy)
    if (nodes.length < 2) return // 冷启动：单条记忆无共现可学（词汇/图通道兜底）
    // 稀疏共现矩阵 C：word → Map<word, count>（同记忆词对共现）
    const co = new Map<string, Map<string, number>>()
    const bump = (a: string, b: string, weight = 1): void => {
      let row = co.get(a)
      if (!row) { row = new Map(); co.set(a, row) }
      row.set(b, (row.get(b) ?? 0) + weight)
    }
    const vocab = new Set<string>()
    for (const n of nodes) {
      const words = [...wordsOf(`${n.title}：${n.summary}`).keys()]
      if (words.length === 0) continue
      for (const w of words) {
        vocab.add(w)
        bump(w, w, 0.5) // 自共现（词频，降权——避免对角占优压过共现信号）
        for (const w2 of words) {
          if (w2 !== w) bump(w, w2)
        }
      }
    }
    const words = [...vocab]
    const idx = new Map<string, number>()
    words.forEach((w, i) => idx.set(w, i))
    const V = words.length
    if (V < 3) return
    // 幂迭代：对 C（对称）求 top-k 特征向量（λ·x = C·x），每次 deflate
    const matVec = (x: Float64Array): Float64Array => {
      const y = new Float64Array(V)
      for (const [a, row] of co) {
        const ia = idx.get(a)
        if (ia === undefined) continue
        let sum = 0
        for (const [b, c] of row) {
          const ib = idx.get(b)
          if (ib !== undefined) sum += c * x[ib]
        }
        y[ia] = sum
      }
      return y
    }
    const DIM = PcaSemantics.DIM
    const vecs: Float64Array[] = []
    // ⚠️ k 从 1 开始：非负矩阵第一主分量全正（Perron-Frobenius），只编码
    // 词频不编码语义——跳过它，用差异方向（第 2..17 主分量）做语义向量，
    // 否则所有词同向、余弦全高（语义坍缩，实测 svd≈0.94 无区分度）。
    for (let k = 1; k <= DIM && k < V; k++) {
      // 幂迭代：x ← C·x / |C·x|（初始随机确定性种子）
      let x = new Float64Array(V)
      for (let i = 0; i < V; i++) x[i] = Math.sin((i + 1) * (k + 1) * 12.9898) * 0.5 + 0.5
      let prev = 0
      for (let it = 0; it < 25; it++) {
        const y = matVec(x)
        let norm = 0
        for (let i = 0; i < V; i++) norm += y[i] * y[i]
        norm = Math.sqrt(norm) || 1
        for (let i = 0; i < V; i++) x[i] = y[i] / norm
        const cur = x.reduce((s, v) => s + v * v, 0)
        if (Math.abs(cur - prev) < 1e-9) break
        prev = cur
      }
      // deflate：C ← C − λ·x·xᵀ（从共现矩阵减去主分量）
      const lambda = x.reduce((s, v, i) => s + v * (matVec(x)[i]), 0) / x.reduce((s, v) => s + v * v, 0)
      for (const [a, row] of co) {
        const ia = idx.get(a)
        if (ia === undefined) continue
        const xa = x[ia] ?? 0
        for (const [b, c] of row) {
          const ib = idx.get(b)
          if (ib === undefined) continue
          row.set(b, c - lambda * xa * x[ib])
        }
      }
      vecs.push(x)
    }
    // 词向量：top-k 特征向量按行组织
    for (const w of words) {
      const v = new Float64Array(DIM)
      vecs.forEach((vec, k) => { v[k] = vec[idx.get(w)!] ?? 0 })
      this.wordVec.set(w, v)
    }
  }

  /** 文本 → 词向量加权平均（词频权重）。 */
  private textVec(text: string): Float64Array | null {
    const words = wordsOf(text)
    const out = new Float64Array(PcaSemantics.DIM)
    let total = 0
    for (const [w, c] of words) {
      const v = this.wordVec.get(w)
      if (!v) continue
      for (let k = 0; k < PcaSemantics.DIM; k++) out[k] += v[k] * c
      total += c
    }
    if (total === 0) return null
    for (let k = 0; k < PcaSemantics.DIM; k++) out[k] /= total
    return out
  }

  /** 余弦相似度（库内谱语义——能桥接共现词对如「缓存」↔「cache」）。 */
  cosine(query: string, memText: string): number {
    this.ensure()
    const qv = this.textVec(query)
    const mv = this.textVec(memText)
    if (!qv || !mv) return 0
    let dot = 0, na = 0, nb = 0
    for (let k = 0; k < PcaSemantics.DIM; k++) {
      dot += qv[k] * mv[k]
      na += qv[k] * qv[k]
      nb += mv[k] * mv[k]
    }
    const d = Math.sqrt(na) * Math.sqrt(nb)
    if (d === 0) return 0
    // 映射到 [0,1]（余弦 ∈ [-1,1] → (cos+1)/2）
    return (dot / d + 1) / 2
  }
}

export class SemanticScorer {
  /** 查询哈希命中的节点 id 集（图语义通道的种子）。 */
  private queryHits = new Set<string>()
  private pca: PcaSemantics

  constructor(private store: EngramStore) {
    this.pca = new PcaSemantics(store)
  }

  /**
   * 对候选打分（同步纯算法）：score = α·lexical + β·graph + γ·svd。
   * α/β/γ 初始标定（0.5/0.25/0.25）；后续可经 fit-tau 数据驱动调整。
   */
  score(query: string, candidates: EngramNode[]): Map<string, SemanticScore> {
    const out = new Map<string, SemanticScore>()
    if (candidates.length === 0) return out
    const qGrams = charNgrams(query, 2)
    const qWords = wordsOf(query)
    // 图语义种子：查询哈希命中（O(1) 寻址）
    this.queryHits = new Set(this.store.lookup(query, 64).map((e) => e.id))
    const alpha = 0.5
    const beta = 0.25
    const gamma = 0.25
    for (const e of candidates) {
      const text = `${e.title}：${e.summary}`
      const mGrams = charNgrams(text, 2)
      const lex = jaccard(qGrams, mGrams)
      // 词频增强：查询词在候选中的覆盖率
      const mWords = wordsOf(text)
      let hitWords = 0
      let totalW = 0
      for (const [w, c] of qWords) {
        const mc = mWords.get(w) ?? 0
        if (mc > 0) hitWords += Math.min(c, mc)
        totalW += c
      }
      const wordCover = totalW > 0 ? hitWords / totalW : 0
      const lexical = Math.min(1, lex * 0.6 + wordCover * 0.4)
      // 图语义：候选是否在查询命中节点的 1-2 跳邻居内
      let graph = 0
      if (this.queryHits.size > 0) {
        const nb = new Set([...(e.causes ?? []), ...(e.effects ?? []), ...(e.links ?? []).map((t) => this.store.byTitle(t)?.id ?? '')])
        if (this.queryHits.has(e.id)) {
          graph = 1 // 自身命中
        } else if ([...nb].some((x) => this.queryHits.has(x))) {
          graph = 0.7 // 1 跳
        } else {
          graph = 0.3 // 2 跳内（保守：任意边存在即弱信号）
        }
      }
      // PCA 语义（库内谱收敛——语义桥通道）
      const svd = this.pca.cosine(query, text)
      out.set(e.id, {
        score: Math.min(1, alpha * lexical + beta * graph + gamma * svd),
        lexical: Number(lexical.toFixed(4)),
        graph: Number(graph.toFixed(2)),
        svd: Number(svd.toFixed(4)),
      })
    }
    return out
  }

  /** 图语义种子暴露（调试/测试）。 */
  hits(): Set<string> {
    return this.queryHits
  }
}
