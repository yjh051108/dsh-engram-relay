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
import type { EngramStore, EngramNode } from '../engram/store.js'
import type { CausalGraph } from '../engram/causal.js'

export class RelayModel {
  readonly python: PythonEngramClient
  private loadError: string | null = null

  constructor(private ctx: CordisContext, private config: EngramRelayConfig) {
    this.python = new PythonEngramClient(
      config.pythonPath,
      config.modelId,
      config.checkpoint ?? '',
      config.embedModel,
    )
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
   * 语义精排（混合检索核心）：对 hash 粗筛候选做 bge 余弦重排。
   * 返回「候选 id → 余弦相似度」；嵌入模型不可用时返回 null
   * （上层降级为重要度/遗留门控）。
   */
  async embed(query: string, candidates: EngramNode[]): Promise<Map<string, number> | null> {
    await this.warmup()
    if (candidates.length === 0) return new Map()
    const out = await this.python.embed(
      candidates.map((e) => `${e.title}：${e.summary.slice(0, 200)}`),
      query.slice(0, 500),
    )
    if (!out || !out.query_vec || !out.vectors || out.vectors.length !== candidates.length) return null
    const qv = out.query_vec
    const scores = new Map<string, number>()
    candidates.forEach((e, i) => {
      const v = out.vectors[i]
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
