/**
 * EngramWakeEngine — 超稀疏精准主动唤醒。
 *
 * 唤醒管线（每回合自动执行，无需模型调用工具）：
 *  1. 哈希寻址：对当前请求文本做 N-gram 哈希（确定性，O(1)），
 *     命中外置 engram 表的槽位 → 候选记忆；
 *  2. 门控打分：<1B 模型对候选记忆与当前查询的相关性打分（门控），
 *     模型未就绪时降级为重要度排序（论文 gate 的转接层模拟）；
 *  3. 因果传播：从命中种子沿因果图双向扩散（前因/后果）——
 *     「什么导致了它 / 它导致了什么」，这是向量索引做不到的；
 *  4. 超稀疏截断：激活分数排序取 top-N（maxWakePerTurn），且总注入
 *     token 受预算约束（默认 600 token ≈ 100k 上下文的 <1%）。
 *
 * 相比普通向量索引：向量索引回答「语义上像什么」（近似），本引擎
 * 回答「确定命中了什么 + 因果上牵连什么」（精确 + 因果）。
 */

import type { GenerateOptions } from '@deepseek-ai/dsh-llm'
import { NgramHashAddressing } from './hash.js'
import { CausalGraph } from './causal.js'
import { EngramStore, type EngramNode } from './store.js'
import type { EngramRelayConfig } from '../types.js'

export interface WakeHit {
  engrams: EngramNode[]
  reason: string
  injectedTokens: number
}

export class EngramWakeEngine {
  /** 最近一次唤醒结果（供 systemPrompt 渲染器读取）。 */
  private lastInjection: WakeHit = { engrams: [], reason: 'idle', injectedTokens: 0 }

  constructor(
    private store: EngramStore,
    private graph: CausalGraph,
    private hasher: NgramHashAddressing,
    private config: EngramRelayConfig,
    /** 门控打分回调（由 LocalRelayModel 提供）；null = 纯哈希 + 重要度。 */
    private scorer: ((query: string, candidates: EngramNode[]) => Promise<Map<string, number>>) | null = null,
  ) {}

  /** 每回合入口：收到一次模型请求时尝试唤醒。 */
  async maybeWake(sessionId: string, _options: GenerateOptions): Promise<WakeHit> {
    if (this.store.count() === 0) return { engrams: [], reason: 'empty-store', injectedTokens: 0 }

    const query = extractQuery(_options)
    if (query.trim() === '') return { engrams: [], reason: 'no-query', injectedTokens: 0 }

    const hit = await this.query(query, this.config.maxWakePerTurn)
    this.lastInjection = hit
    return hit
  }

  /** 核心查询：哈希寻址 → 门控打分 → 因果传播 → 分层稀疏选择。 */
  async query(query: string, limit: number): Promise<WakeHit> {
    // 1. 确定性哈希寻址：当前查询命中哪些槽位（含跨会话记忆——
    //    全局/项目/规则记忆以固定种子文本写入，永远可命中）。
    const candidates = this.store.lookup(query, 32)
    if (candidates.length === 0) return { engrams: [], reason: 'no-hash-hit', injectedTokens: 0 }

    // 2. 门控打分（模型就绪时；否则重要度降级）。
    let scores: Map<string, number>
    if (this.scorer) {
      scores = await this.scorer(query, candidates)
    } else {
      scores = new Map(candidates.map((e) => [e.id, e.importance]))
    }

    // 3. 因果传播（前因/后果双向）。
    const activated = this.graph.propagate(scores)

    // 4. 分层稀疏选择（因果席位保证）：
    //    - 主席位：哈希命中的候选按激活分数排序；
    //    - 因果席位：传播激活的**因果邻居**占独立席位。注意：邻居可能
    //      也被哈希命中（n-gram 碰撞/共享），此时它若被主席位截断，
    //      仍应从因果席位进入——「带因果性」不被高分候选挤掉。
    const hitIds = new Set(candidates.map((e) => e.id))
    const causalSlots = Math.max(1, Math.ceil(limit / 2))

    const ranked: Array<[string, number]> = []
    // 主席位：哈希命中按分数排序（保留 limit - causalSlots 个）
    const hitRanked = [...activated.entries()]
      .filter(([id]) => hitIds.has(id))
      .sort((a, b) => b[1] - a[1])
    const mainQuota = Math.max(1, limit - causalSlots)
    const mainPicked = hitRanked.slice(0, mainQuota)
    ranked.push(...mainPicked)
    const mainIds = new Set(mainPicked.map(([id]) => id))

    // 因果席位：activated 中未进主席位的节点（含被截断的哈希命中邻居）
    // 按「因果传播增益」排序——激活分数高于自身重要性者优先
    const baseScores = scores
    const causalCandidates = [...activated.entries()]
      .filter(([id]) => !mainIds.has(id))
      .sort((a, b) => {
        const gainA = a[1] - (baseScores.get(a[0]) ?? 0)
        const gainB = b[1] - (baseScores.get(b[0]) ?? 0)
        return gainB - gainA || b[1] - a[1]
      })
    ranked.push(...causalCandidates.slice(0, causalSlots))
    // 主席位不足时用其余节点补齐
    let extra = ranked.length
    const rest = causalCandidates.slice(causalSlots)
    while (extra < limit && rest.length > 0) {
      ranked.push(rest[extra - ranked.length])
      extra += 1
    }
    ranked.length = Math.min(ranked.length, limit)

    const picked: EngramNode[] = []
    let tokens = 0
    for (const [id] of ranked) {
      const e = this.store.get(id)
      if (!e) continue
      const cost = estimateTokens(e.title) + estimateTokens(e.summary)
      if (tokens + cost > this.config.injectBudgetTokens && picked.length > 0) break
      picked.push(e)
      tokens += cost
      this.store.touch(id)
    }

    const hit: WakeHit = {
      engrams: picked,
      reason: picked.length > 0 ? `hash-wake:${picked.length}` : 'below-threshold',
      injectedTokens: tokens,
    }
    // query 是核心入口（maybeWake 与工具共用），结果供渲染器读取
    this.lastInjection = hit
    return hit
  }

  /** 渲染记忆注入段（渐进披露第一层：入口列表 + 簇概览，超稀疏）。 */
  renderInjection(budgetTokens: number): string {
    const { engrams } = this.lastInjection
    if (engrams.length === 0) return ''
    const lines: string[] = []
    let tokens = 0
    for (const e of engrams) {
      if (tokens >= budgetTokens) break
      // 入口层：title + summary（不含 content——按需展开）
      const causes = this.graph.causesOf(e.id)
      const effects = this.graph.effectsOf(e.id)
      const causeNote = causes.length > 0
        ? ` ↑因:${causes.map((c) => c.title).join(';').slice(0, 60)}`
        : ''
      const effectNote = effects.length > 0
        ? ` ↓果:${effects.map((c) => c.title).join(';').slice(0, 60)}`
        : ''
      lines.push(`- [[${e.title}]]${causeNote}${effectNote}: ${e.summary.slice(0, 120)}`)
      tokens += estimateTokens(e.title) + estimateTokens(e.summary)
    }
    // 自组织簇概览：让模型看到主题结构（不硬编码，连接密度自然成簇）
    const clusters = this.store.clusters()
    if (clusters.length > 1) {
      const overview = clusters
        .map((c) => `[[${c.label}]](+${c.members.length})`)
        .join(' · ')
      if (tokens + estimateTokens(overview) <= budgetTokens) {
        lines.push(`  簇: ${overview}`)
      }
    }
    return lines.length > 0
      ? `<engram-memory>（大一统记忆图谱 · 入口，[[标题]] 可展开）\n${lines.join('\n')}\n</engram-memory>`
      : ''
  }

  /** 供 status 工具读取。 */
  lastWake(): WakeHit {
    return this.lastInjection
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
