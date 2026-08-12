/**
 * EngramWakeEngine — 超稀疏精准主动唤醒。
 *
 * 唤醒管线（每回合自动执行，无需模型调用工具）：
 *  1. 哈希粗筛：对当前请求文本做 N-gram 哈希（确定性，O(1)），
 *     命中外置 engram 表的槽位 → 候选记忆（精确寻址，不含近似性）；
 *  2. 语义精排：bge 嵌入模型对候选做余弦重排（修掉哈希的
 *     mode-level 跨主题误命中——实测 80 查询精确率 85% → 95%），
 *     嵌入不可用时降级为重要度/遗留门控打分；
 *  3. 因果传播：从命中种子沿因果图双向扩散（前因/后果）——
 *     「什么导致了它 / 它导致了什么」，这是向量索引做不到的；
 *  4. 超稀疏截断：激活分数排序取 top-N（maxWakePerTurn），且总注入
 *     token 受预算约束（默认 600 token ≈ 100k 上下文的 <1%）。
 *
 * 相比普通向量索引：向量索引回答「语义上像什么」（近似），本引擎
 * 回答「确定命中了什么 + 语义上相关什么 + 因果上牵连什么」（精确 +
 * 语义 + 因果）。
 */

import type { GenerateOptions } from '@deepseek-ai/dsh-llm'
import { NgramHashAddressing } from './hash.js'
import { CausalGraph } from './causal.js'
import { EngramStore, isVisible, type EngramLayer, type EngramNode } from './store.js'
import type { EngramRelayConfig } from '../types.js'

export interface WakeHit {
  engrams: EngramNode[]
  reason: string
  injectedTokens: number
}

/**
 * 查看者视角（分层准入依据）：
 *  - global：所有会话可唤醒；
 *  - project：仅 node.projectId === viewer.cwd 的会话；
 *  - session：仅 node.sessionId === viewer.sessionId 的本会话。
 * 无 cwd/sessionId 的视角（subagent 等）只看 global 层。
 */
export interface WakeViewer {
  sessionId?: string
  cwd?: string
}

/** 打分回调：embedder（语义精排）优先，scorer（遗留门控）兜底。 */
export interface WakeScorers {
  embedder?: (query: string, candidates: EngramNode[]) => Promise<Map<string, number> | null>
  scorer?: (query: string, candidates: EngramNode[]) => Promise<Map<string, number>>
}

export class EngramWakeEngine {
  /** 最近一次唤醒结果（供 systemPrompt 渲染器读取）。 */
  private lastInjection: WakeHit = { engrams: [], reason: 'idle', injectedTokens: 0 }

  constructor(
    private store: EngramStore,
    private graph: CausalGraph,
    private hasher: NgramHashAddressing,
    private config: EngramRelayConfig,
    /** 打分器（bge 语义精排 + 遗留门控）；缺省 = 纯哈希 + 重要度。 */
    private scorers: WakeScorers | null = null,
  ) {}

  /** 每回合入口：收到一次模型请求时尝试唤醒。 */
  async maybeWake(sessionId: string, _options: GenerateOptions, viewer: WakeViewer = {}): Promise<WakeHit> {
    if (this.store.count() === 0) return { engrams: [], reason: 'empty-store', injectedTokens: 0 }

    const query = extractQuery(_options)
    if (query.trim() === '') return { engrams: [], reason: 'no-query', injectedTokens: 0 }

    const hit = await this.query(query, this.config.maxWakePerTurn, { sessionId, ...viewer })
    this.lastInjection = hit
    return hit
  }

  /** 核心查询：哈希粗筛 → 分层准入 → 语义精排（bge）/门控兜底 → 因果传播 → 分层稀疏选择。 */
  async query(query: string, limit: number, viewer: WakeViewer = {}): Promise<WakeHit> {
    // 1. 确定性哈希粗筛（多取候选：分层准入会过滤掉一部分，保证命中不因
    //    层过滤而丢失——global 常驻候选始终可见）。
    let candidates = this.store.lookup(query, 256)
    // 分层准入：global 所有会话 / project 同 cwd / session 本会话。
    // 这是「跨会话记忆」的可见性边界——看不到的记忆不会被唤醒注入。
    candidates = candidates.filter((e) => isVisible(e, viewer))
    if (candidates.length === 0) return { engrams: [], reason: 'no-hash-hit', injectedTokens: 0 }

    // 2. 打分：bge 语义精排（唯一语义判断）→ 重要度仅作排序兜底。
    //    ⚠️ 相关性门槛（宁缺毋滥）：不相关的记忆一律不注入——
    //      哈希命中只是粗筛，语义余弦低于阈值的候选直接剔除；
    //      embedder 不可用时无法判断语义相关性，本轮不注入
    //      （重要度垫底会带来弱相关污染 + 每轮注入的缓存损耗，宁可空手）。
    const semanticMin = this.config.semanticMinScore ?? 0.42
    // 多重比较校正（温和版）：候选越多误过概率越高，但真实系统哈希第一道防线
    // 已把无关候选压到个位数——仅对候选异常膨胀时温和收紧。
    const threshold = semanticMin + 0.03 * Math.log2(Math.max(1, candidates.length / 16))
    let raw: Map<string, number> | null | undefined
    if (this.scorers?.embedder) {
      raw = await this.scorers.embedder(query, candidates).catch(() => null)
    }
    if (!raw || raw.size === 0) {
      return { engrams: [], reason: 'no-embedder', injectedTokens: 0 }
    }
    const relevant = candidates.filter((e) => (raw!.get(e.id) ?? 0) >= threshold)
    if (relevant.length === 0) {
      return { engrams: [], reason: 'below-threshold', injectedTokens: 0 }
    }
    candidates = relevant
    const scores = new Map<string, number>(
      candidates.map((e) => [e.id, raw!.get(e.id) ?? e.importance]),
    )

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
      reason: picked.length > 0 ? `hybrid-wake:${picked.length}` : 'below-threshold',
      injectedTokens: tokens,
    }
    // query 是核心入口（maybeWake 与工具共用），结果供渲染器读取
    this.lastInjection = hit
    return hit
  }

  /** 渲染记忆注入段（动态预算：按相关度分级——高分完整入口、中分标题+摘要、低分仅标题）。 */
  renderInjection(budgetTokens: number): string {
    const { engrams } = this.lastInjection
    if (engrams.length === 0) return ''
    const HEADER = '（记忆：recall检索/open展开/store写入/link因果）'
    const lines: string[] = []
    let tokens = estimateTokens(HEADER)
    engrams.forEach((e, idx) => {
      if (tokens >= budgetTokens) return
      // 分级渲染（engrams 已按激活分数降序）：
      //  - 第 1 条（最高分）：完整入口（标题+层+因果+摘要）
      //  - 第 2-3 条：标题+摘要（省因果注）
      //  - 其余：仅 [[标题]]（最大化覆盖，预算内尽量多挂入口）
      let line: string
      if (idx === 0) {
        const causes = this.graph.causesOf(e.id)
        const effects = this.graph.effectsOf(e.id)
        const causeNote = causes.length > 0 ? ` ↑因:${causes.map((c) => c.title).join(';').slice(0, 60)}` : ''
        const effectNote = effects.length > 0 ? ` ↓果:${effects.map((c) => c.title).join(';').slice(0, 60)}` : ''
        line = `- [[${e.title}]][${e.layer}]${causeNote}${effectNote}: ${e.summary.slice(0, 120)}`
      } else if (idx <= 2) {
        line = `- [[${e.title}]][${e.layer}]: ${e.summary.slice(0, 80)}`
      } else {
        line = `- [[${e.title}]]`
      }
      const cost = estimateTokens(line)
      if (tokens + cost > budgetTokens) return
      lines.push(line)
      tokens += cost
    })
    return lines.length > 0
      ? `<engram-memory>${HEADER}\n${lines.join('\n')}\n</engram-memory>`
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

/** 粗略 token 估算：CJK 约 1 字 ≈ 0.7 token（DeepSeek 中文实测 ~1.4 字/token），ASCII ≈ 0.25 token/字符。 */
export function estimateTokens(text: string): number {
  let cjk = 0
  let ascii = 0
  for (const ch of text) {
    if (/[\u3000-\u9fff]/.test(ch)) cjk += 1
    else ascii += 1
  }
  return Math.ceil(cjk * 0.7 + ascii / 4)
}
