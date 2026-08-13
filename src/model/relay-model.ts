/**
 * RelayModel — 转接模型门面。
 *
 * 双轨：
 *  - 语义轨（v3 核心）：bge 专用嵌入模型（sentence-transformers）对
 *    hash 粗筛候选做余弦精排（混合检索：确定性寻址 + 语义重排）；
 *  - 蒸馏轨（遗留）：原 0.6B 魔改模型（Engram 条件记忆 × DSA 路由）
 *    的蒸馏/打分/原生回忆。模型目录未配置/缺失时全部优雅返回 null。
 *
 * 模型不可用（Python 缺失/模型未配置/服务崩溃）时自动降级：
 *  蒸馏 → 跳过；打分 → 重要度；记忆写入 → 无操作。插件始终可用。
 */

import type { Context as CordisContext } from 'cordis'
import type { EngramRelayConfig } from '../types.js'
import { PythonEngramClient, type DistillEntry } from './python-client.js'
import { embedWithOnnx } from './onnx-embedder.js'
import { SemanticScorer } from '../engram/semantic-scorer.js'
import type { EngramStore, EngramNode } from '../engram/store.js'
import type { CausalGraph } from '../engram/causal.js'

export class RelayModel {
  readonly python: PythonEngramClient
  /** v0.5 纯算法语义打分器（词汇 + 图通道，零模型——替代 embedding 精排）。 */
  readonly scorer: SemanticScorer
  private loadError: string | null = null

  constructor(private ctx: CordisContext, private config: EngramRelayConfig, store: EngramStore) {
    this.python = new PythonEngramClient(
      config.pythonPath,
      config.modelId,
      config.checkpoint ?? '',
      config.embedModel,
    )
    this.scorer = new SemanticScorer(store)
  }

  /** 预热：启动 Python 服务并加载模型（失败不抛出，记录后降级）。 */
  async warmup(): Promise<void> {
    if (this.warmupDone) return
    this.warmupDone = true
    try {
      const status = await this.python.load()
      if (status === null) this.loadError = 'python service unavailable'
    } catch (error) {
      this.loadError = String(error)
    }
  }

  private warmupDone = false

  /** 蒸馏：把对话转为记忆节点写入存储（模型不可用时跳过）。 */
  async distillTurn(store: EngramStore, graph: CausalGraph, conversation: string, sessionId: string, turn: number): Promise<EngramNode[]> {
    if (conversation === '') return []
    await this.warmup()
    const out = await this.python.distill(conversation)
    if (!out || !out.parsed) return []
    const p: DistillEntry = out.parsed
    const e = store.add({
      kind: p.kind ?? 'fact',
      title: p.label ?? '对话片段',
      summary: p.text ?? conversation.slice(0, 100),
      content: conversation,
      links: [],
      sessionId,
      turn,
      causes: p.causes ?? [],
      effects: [],
      importance: p.importance ?? 0.5,
    })
    for (const causeId of p.causes ?? []) graph.addEdge(causeId, e.id, 'causes', 1)
    return [e]
  }

  /**
   * 原始向量嵌入（向量缓存/检索用）：返回查询向量 + 候选向量（ONNX 优先，Python 回退）。
   */
  async embedRaw(query: string, texts: string[]): Promise<{ query_vec: number[]; vectors: number[][] } | null> {
    const q = query.slice(0, 500)
    const ts = await embedWithOnnx(texts, q, this.config.embedModel)
    if (ts) return ts
    await this.warmup()
    const out = await this.python.embed(texts, q)
    if (!out || !out.query_vec || !out.vectors) return null
    return { query_vec: out.query_vec, vectors: out.vectors }
  }

  /**
   * 语义打分（v0.5：纯算法 SemanticScorer——词汇 n-gram Jaccard + 词频
   * + 图语义传播 + PCA 共现谱分解，零 embedding 模型；确定性、可解释）。
   * 返回「候选 id → 融合分数 [0,1]」（0.6 阈值语义沿用；上层无需改动）。
   */
  async embed(query: string, candidates: EngramNode[]): Promise<Map<string, number> | null> {
    if (candidates.length === 0) return new Map()
    const scored = this.scorer.score(query, candidates)
    const out = new Map<string, number>()
    for (const [id, s] of scored) out.set(id, s.score)
    return out
  }

  /** 详细语义分（通道分解——查重/织网用 lexical 阈值，比融合分更稳）。 */
  semanticScores(query: string, candidates: EngramNode[]): Map<string, { score: number; lexical: number; graph: number; cooc: number }> {
    return this.scorer.score(query, candidates)
  }

  private cosineScores(candidates: EngramNode[], qv: number[], vectors: number[][]): Map<string, number> {
    const scores = new Map<string, number>()
    candidates.forEach((e, i) => {
      const v = vectors[i]
      if (!v || v.length !== qv.length) return
      let dot = 0
      let na = 0
      let nb = 0
      for (let k = 0; k < v.length; k += 1) {
        dot += v[k] * qv[k]
        na += v[k] * v[k]
        nb += qv[k] * qv[k]
      }
      scores.set(e.id, dot / (Math.sqrt(na) * Math.sqrt(nb) || 1))
    })
    return scores
  }

  /** 门控打分（遗留 0.6B 轨；模型不可用时返回空 Map（上层降级重要度）。 */
  async score(query: string, candidates: EngramNode[]): Promise<Map<string, number>> {
    await this.warmup()
    const out = await this.python.generate(
      `查询：「${query.slice(0, 200)}」\n记忆：「${candidates[0]?.title ?? ''}：${candidates[0]?.summary.slice(0, 100) ?? ''}」\n这条记忆与查询的相关度（只输出 0 到 1 的数字）：`,
      4,
      0,
    )
    if (!out) return new Map()
    const v = parseFloat(out.text.match(/\d+(\.\d+)?/)?.[0] ?? '')
    if (!Number.isFinite(v) || candidates.length === 0) return new Map()
    const map = new Map<string, number>()
    map.set(candidates[0].id, Math.min(1, Math.max(0, v)))
    return map
  }

  /**
   * 原生回忆：让训练好的记忆模型直接生成答案（forward 自动融合记忆表）。
   * 这是「回忆是模型行为」的对外接口——主模型转接层把回忆结果注入上下文。
   * 模型不可用/未训练时返回 null（调用方降级为纯 engram 文本注入）。
   */
  async recall(query: string, maxNewTokens = 32): Promise<string | null> {
    await this.warmup()  // 确保服务已加载（惰性）
    const out = await this.python.generate(query.slice(0, 200), maxNewTokens, 0)
    if (!out) return null
    const text = out.text.trim()
    return text === '' ? null : text
  }

  async describe(): Promise<Record<string, unknown>> {
    const status = await this.python.status().catch(() => null)
    return {
      modelId: this.config.modelId,
      embedModel: this.config.embedModel,
      pythonPath: this.config.pythonPath,
      loadError: this.loadError,
      service: status,
    }
  }

  stop(): void {
    this.python.stop()
  }
}
