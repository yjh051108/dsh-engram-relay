/**
 * CausalWakeEngine — 超稀疏精准主动唤醒。
 *
 * 唤醒管线（每回合自动执行，无需模型调用工具）：
 *  1. 打分：小模型对「当前请求 + 最近上下文」生成唤醒查询，并对候选
 *     engram 打分（语义 + 相关性）；
 *  2. 因果传播：从种子分数沿因果图双向传播（前因/后果），得到激活分数；
 *  3. 稀疏截断：按激活分数取 top-N（maxWakePerTurn），且总注入 token 受
 *     budget 约束（默认 600 token ≈ 100k 上下文的 <1%）；
 *  4. 渲染注入：格式化记忆段，供 systemPrompt.context 装配。
 *
 * 相比普通向量索引：向量索引只回答「语义上像什么」，本引擎额外回答
 * 「什么导致了它 / 它导致了什么」——因果链召回。
 */

import type { GenerateOptions } from '@deepseek-ai/dsh-llm'
import { CausalGraph } from './causal.js'
import { EngramStore, type Engram } from './store.js'
import type { EngramRelayConfig } from '../types.js'

export interface WakeHit {
  engrams: Engram[]
  reason: string
  injectedTokens: number
}

export class CausalWakeEngine {
  /** 最近一次唤醒结果（供 systemPrompt 渲染器读取）。 */
  private lastInjection: WakeHit = { engrams: [], reason: 'idle', injectedTokens: 0 }
  /** 回合计数（蒸馏节奏）。 */
  private turnCounter = 0

  constructor(
    private graph: CausalGraph,
    private store: EngramStore,
    private config: EngramRelayConfig,
    /** 打分回调：由 LocalRelayModel 提供；null = 纯因果图（模型未就绪时降级）。 */
    private scorer: ((query: string, candidates: Engram[]) => Promise<Map<string, number>>) | null = null,
  ) {}

  /** 每回合入口：收到一次模型请求时尝试唤醒。 */
  async maybeWake(sessionId: string, _options: GenerateOptions): Promise<WakeHit> {
    if (this.store.count() === 0) return { engrams: [], reason: 'empty-store', injectedTokens: 0 }

    // 从请求中提取查询文本（取最后一条 user 消息 + 系统提示要点）。
    const query = extractQuery(_options)
    if (query.trim() === '') return { engrams: [], reason: 'no-query', injectedTokens: 0 }

    const hit = await this.query(query, this.config.maxWakePerTurn)
    this.lastInjection = hit
    return hit
  }

  /** 核心查询：打分 → 因果传播 → 稀疏截断。 */
  async query(query: string, limit: number): Promise<WakeHit> {
    const candidates = this.store.all()

    // 1. 打分（模型就绪时用小模型，否则按 importance 降级）。
    let scores: Map<string, number>
    if (this.scorer) {
      scores = await this.scorer(query, candidates)
    } else {
      scores = new Map(candidates.map((e) => [e.id, e.importance]))
    }

    // 2. 因果传播。
    const activated = this.graph.propagate(scores)

    // 3. 稀疏截断：按激活分数排序取 top-N，且受 token 预算约束。
    const ranked = [...activated.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit)
    const picked: Engram[] = []
    let tokens = 0
    for (const [id] of ranked) {
      const e = this.store.get(id)
      if (!e) continue
      const cost = estimateTokens(e.label) + estimateTokens(e.text)
      if (tokens + cost > this.config.injectBudgetTokens && picked.length > 0) break
      picked.push(e)
      tokens += cost
      this.store.touch(id)
    }

    return {
      engrams: picked,
      reason: picked.length > 0 ? `causal-wake:${picked.length}` : 'below-threshold',
      injectedTokens: tokens,
    }
  }

  /** 渲染记忆注入段（超稀疏）。 */
  renderInjection(budgetTokens: number): string {
    const { engrams } = this.lastInjection
    if (engrams.length === 0) return ''
    const lines: string[] = []
    let tokens = 0
    for (const e of engrams) {
      if (tokens >= budgetTokens) break
      // 因果链标注：带上前因（若有）
      const causes = this.graph.causesOf(e.id)
      const causeNote = causes.length > 0
        ? ` (因: ${causes.map((c) => c.label).join('; ').slice(0, 80)})`
        : ''
      lines.push(`- [${e.kind}] ${e.label}${causeNote}: ${e.text.slice(0, 160)}`)
      tokens += estimateTokens(e.label) + estimateTokens(e.text)
    }
    return lines.length > 0
      ? `<engram-memory>（外置记忆，按因果唤醒）\n${lines.join('\n')}\n</engram-memory>`
      : ''
  }

  /** 供 status 工具读取。 */
  lastWake(): WakeHit {
    return this.lastInjection
  }

  bumpTurn(): void {
    this.turnCounter += 1
  }

  turnCount(): number {
    return this.turnCounter
  }
}

/** 从 GenerateOptions 提取查询文本（最后一条 user 消息的文本块）。 */
function extractQuery(options: GenerateOptions): string {
  const messages = (options as { messages?: Array<{ role?: string; content?: unknown }> }).messages
  if (!messages || messages.length === 0) return ''
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const m = messages[i]
    if (m.role !== 'user') continue
    const content = m.content
    if (typeof content === 'string') return content.slice(0, 2000)
    if (Array.isArray(content)) {
      const text = content
        .map((b) => (typeof b === 'object' && b !== null && 'text' in b ? String((b as { text: unknown }).text) : ''))
        .join(' ')
      if (text.trim() !== '') return text.slice(0, 2000)
    }
  }
  return ''
}

/** 粗略 token 估算：CJK 约 1 字 ≈ 1 token，ASCII ≈ 0.25 token/字符。 */
export function estimateTokens(text: string): number {
  let cjk = 0
  let ascii = 0
  for (const ch of text) {
    if (/[\u3000-\u9fff]/.test(ch)) cjk += 1
    else ascii += 1
  }
  return Math.ceil(cjk + ascii / 4)
}
