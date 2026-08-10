/**
 * RelayModel — 转接模型门面。
 *
 * 双轨：
 *  - 本地轨（v2 核心）：spawn Python 魔改模型（Qwen3-0.6B + Engram
 *    模块 + DSA 路由），蒸馏/记忆写入走模型内部表示空间；
 *  - 云端轨：文本注入（systemPrompt 记忆段），供 API 主模型消费。
 *
 * 模型不可用（Python 缺失/下载失败/服务崩溃）时自动降级：
 *  蒸馏 → 跳过；打分 → 重要度；记忆写入 → 无操作。插件始终可用。
 */

import type { Context as CordisContext } from 'cordis'
import type { EngramRelayConfig } from '../types.js'
import { PythonEngramClient, type DistillEntry } from './python-client.js'
import type { EngramStore, Engram } from '../engram/store.js'
import type { CausalGraph } from '../engram/causal.js'

export class RelayModel {
  readonly python: PythonEngramClient
  private loadError: string | null = null

  constructor(private ctx: CordisContext, private config: EngramRelayConfig) {
    this.python = new PythonEngramClient(config.pythonPath, config.modelId, config.checkpoint ?? '')
  }

  /** 预热：启动 Python 服务并加载模型（失败不抛出，记录后降级）。 */
  async warmup(): Promise<void> {
    try {
      const status = await this.python.load()
      if (status === null) this.loadError = 'python service unavailable'
    } catch (error) {
      this.loadError = String(error)
    }
  }

  /** 蒸馏：把对话转为 engram 条目写入存储（模型不可用时跳过）。 */
  async distillTurn(store: EngramStore, graph: CausalGraph, conversation: string, sessionId: string, turn: number): Promise<Engram[]> {
    if (conversation === '') return []
    const out = await this.python.distill(conversation)
    if (!out || !out.parsed) return []
    const p: DistillEntry = out.parsed
    const e = store.add({
      kind: p.kind ?? 'fact',
      label: p.label,
      text: p.text,
      scope: null,
      sessionId,
      turn,
      causes: p.causes ?? [],
      effects: [],
      importance: p.importance ?? 0.5,
    })
    for (const causeId of p.causes ?? []) graph.addEdge(causeId, e.id, 'causes', 1)
    // 把记忆文本写入模型记忆表（本地轨融合）
    await this.python.writeMemory([{ text: p.text }]).catch(() => null)
    return [e]
  }

  /** 门控打分：模型不可用时返回空 Map（上层降级重要度）。 */
  async score(query: string, candidates: Engram[]): Promise<Map<string, number>> {
    const out = await this.python.generate(
      `查询：「${query.slice(0, 200)}」\n记忆：「${candidates[0]?.label ?? ''}：${candidates[0]?.text.slice(0, 100) ?? ''}」\n这条记忆与查询的相关度（只输出 0 到 1 的数字）：`,
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

  async describe(): Promise<Record<string, unknown>> {
    const status = await this.python.status().catch(() => null)
    return {
      modelId: this.config.modelId,
      pythonPath: this.config.pythonPath,
      loadError: this.loadError,
      service: status,
    }
  }

  stop(): void {
    this.python.stop()
  }
}
