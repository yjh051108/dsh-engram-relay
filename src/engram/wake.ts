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
 *  - project：仅 node.projectId === viewer.cwd 的会话。
 * 无 cwd 的视角（subagent 等）只看 global 层。
 * （v0.3：session 层删除；sessionId 字段仅保留来源记录，不再参与准入。）
 */
export interface WakeViewer {
  sessionId?: string
  cwd?: string
  /** 当前回合号（时序衰减用：近期记忆加权） */
  turn?: number
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
    /** 候选预筛钩子（向量索引粗筛用）：返回候选 id 列表；null = 回退哈希 lookup。 */
    private prefilter: ((query: string) => Promise<string[] | null>) | null = null,
    /** 类脑激活缓存（B=ln(Σt^-d)）：排序融合（阶段 3）；缺省 = 无激活加权。 */
    private activation: import('./activation.js').ActivationCache | null = null,
  ) {}

  /** 每回合入口（自动唤醒，极克制）：哈希预筛 → 查询质量门 → 自动阈值 0.5 → top-1。 */
  async maybeWake(sessionId: string, _options: GenerateOptions, viewer: WakeViewer = {}): Promise<WakeHit> {
    if (this.store.count() === 0) return { engrams: [], reason: 'empty-store', injectedTokens: 0 }

    const query = extractQuery(_options)
    if (query.trim() === '') return { engrams: [], reason: 'no-query', injectedTokens: 0 }

    // ① 向量/哈希预筛（零成本）：词汇/语义与任何记忆无重叠 → 直接跳过（零注入）。
    //    这是最大的噪声过滤器——闲聊/过程轮基本在此命中退出。
    if (this.prefilter) {
      const ids = await this.prefilter(query).catch(() => null)
      if (ids && ids.length === 0) {
        return { engrams: [], reason: 'no-hash-hit', injectedTokens: 0 }
      }
    } else if (this.store.lookup(query, 1).length === 0) {
      return { engrams: [], reason: 'no-hash-hit', injectedTokens: 0 }
    }
    // ② 查询质量门：太短（口语无锚，如"这个怎么弄"）没有语义锚 → 跳过。
    const anchorText = query.trim()
    if (anchorText.length < 8) {
      return { engrams: [], reason: 'short-query', injectedTokens: 0 }
    }
    // ③④ 自动模式：阈值 +0.08（≈0.50，明显相关才注入）+ top-1（唯一入口，干扰最小）。
    const hit = await this.query(query, 1, { sessionId, ...viewer }, { auto: true })
    this.lastInjection = hit
    return hit
  }

  /** 核心查询：向量/哈希粗筛 → 分层准入 → 语义精排（bge）→ 因果传播 → 分层稀疏选择。 */
  async query(query: string, limit: number, viewer: WakeViewer = {}, opts: { auto?: boolean } = {}): Promise<WakeHit> {
    // 1. 粗筛：向量索引（prefilter 钩子，语义无盲区）→ 回退哈希 lookup。
    //    多取候选：分层准入会过滤掉一部分，保证命中不因层过滤而丢失。
    let candidates: EngramNode[]
    if (this.prefilter) {
      const ids = await this.prefilter(query).catch(() => null)
      candidates = ids
        ? ids.map((id) => this.store.get(id)).filter((e): e is EngramNode => !!e)
        : this.store.lookup(query, 256)
    } else {
      candidates = this.store.lookup(query, 256)
    }
    // 分层准入：global 所有会话 / project 同 cwd / session 本会话。
    // 这是「跨会话记忆」的可见性边界——看不到的记忆不会被唤醒注入。
    candidates = candidates.filter((e) => isVisible(e, viewer))
    if (candidates.length === 0) return { engrams: [], reason: 'no-hash-hit', injectedTokens: 0 }

    // 2. 打分降级链：embedder（bge 语义精排）→ scorer（遗留门控）→ 重要度垫底。
    //    ⚠️ 契约（AGENTS.md + hybrid/wake 测试）：哈希命中永不因打分器缺失被
    //    丢弃——打分只决定排序与截断顺序，不做硬性剔除（「宁缺毋滥」由自动
    //    唤醒的 top-1 注入执行，而非在候选层丢命中）。
    let raw: Map<string, number> | null | undefined
    if (this.scorers?.embedder) {
      raw = await this.scorers.embedder(query, candidates).catch(() => null)
    }
    if (!raw || raw.size === 0) {
      // 降级第二级：遗留 scorer（embedder 抛错/不可用时的门控打分）
      if (this.scorers?.scorer) {
        raw = await this.scorers.scorer(query, candidates).catch(() => null)
      }
    }
    const scores = new Map<string, number>()
    if (raw && raw.size > 0) {
      // 打分器部分缺分 → 重要度垫底（打分器不丢哈希命中）
      for (const e of candidates) scores.set(e.id, raw.get(e.id) ?? e.importance)
    } else {
      // 纯重要度兜底 + 同文本精确命中加权：与查询完全同文的条目（确定性
      // 寻址的精确保底）优先于同槽位的弱相关条目（测试契约：同文本必中）。
      const q = query.trim()
      for (const e of candidates) {
        let score = e.importance
        if (q && (e.summary.includes(q) || e.title.includes(q))) score += 0.5
        scores.set(e.id, score)
      }
    }
    // 哈希候选全保留（混合保底：语义分只决定排序，不硬剔哈希命中）。
    const relevant = candidates

    // 3. 因果传播（前因/后果双向）——基于全候选分数（含弱分邻居种子）。
    const activated = this.graph.propagate(scores)

    // 4. 分层稀疏选择（因果席位保证）：
    //    - 主席位：全部哈希候选按激活分数排序（混合保底，不做语义硬剔）；
    //    - 因果席位：传播激活的因果邻居占独立席位。
    const relevantIds = new Set(relevant.map((e) => e.id))
    const hitIds = new Set(candidates.map((e) => e.id))
    const causalSlots = Math.max(1, Math.ceil(limit / 2))

    // 时序衰减权重：近期记忆加分（w·e^(-Δturn/20) 指数衰减；w = recencyWeight
    // 配置可调——仿真标定显示方向需数据驱动（默认 0.25 保守，0 关闭，负=旧优先）。
    // 无 turn 信息时退化为 1（不调制）。
    const curTurn = viewer.turn
    const nodeById = new Map(candidates.map((e) => [e.id, e]))
    const recency = (id: string): number => {
      const e = nodeById.get(id)
      if (!e || typeof curTurn !== 'number' || typeof e.turn !== 'number') return 1
      const w = this.config.recencyWeight ?? 0.25
      if (w === 0) return 1
      const d = Math.max(0, curTurn - e.turn)
      return 1 + w * Math.exp(-d / 20)
    }

    // 类脑激活加权（阶段 3）：排序分数 = 语义激活 × sigmoid(基础激活 - 基准)，
    // 强化历史（命中/展开/链接）驱动——使用即巩固、闲置即遗忘。
    // 无激活缓存时退化为纯语义排序。
    const actBias = (id: string): number => {
      if (!this.activation) return 1
      const b = this.activation.get(id)
      // sigmoid 温和提升：B 高（近期多强化）加权，B 低（久未强化）不压死
      return 1 + 0.6 / (1 + Math.exp(-(b + 1.5))) // B≈0 时 ~1.09，B 大趋 1.6
    }

    const ranked: Array<[string, number]> = []
    // 主席位：全部哈希候选按「语义激活 × 激活加权 × 时序权重」排序
    const hitRanked = [...activated.entries()]
      .filter(([id]) => hitIds.has(id) && relevantIds.has(id))
      .sort((a, b) => b[1] * actBias(b[0]) * recency(b[0]) - a[1] * actBias(a[0]) * recency(a[0]))
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

    // 稀疏注入：limit 内全部按序注入（哈希命中不因分数差距被丢弃——契约
    // 「混合保底」；稀疏性由 limit + token 预算 + 自动唤醒 top-1 保证）。
    const picked: EngramNode[] = []
    let tokens = 0
    for (const [id, _score] of ranked) {
      if (picked.length >= limit) break
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
    // 实战样本积累：记录查询/候选分数/注入选择（异步落盘，不阻塞主流程；
    // 供融合权重离线拟合——仿真阶段 ① 的 τ 标定数据源）
    if (this.config.wakeSampleLog) {
      const actOf = (id: string): number | null => {
        if (!this.activation) return null
        try { return Number(this.activation.get(id).toFixed(4)) } catch { return null }
      }
      void this.appendSample({
        time: Date.now(),
        auto: opts.auto === true,
        limit,
        query: query.slice(0, 500),
        cwd: viewer.cwd ?? null,
        candidates: candidates.slice(0, 20).map((e) => ({
          id: e.id, title: e.title, layer: e.layer,
          cos: raw?.get(e.id) !== undefined ? Number((raw.get(e.id) as number).toFixed(4)) : null,
          act: actOf(e.id),
          importance: e.importance,
        })),
        picked: picked.map((e) => e.id),
      }).catch(() => { /* 采样失败不影响唤醒 */ })
    }
    // query 是核心入口（maybeWake 与工具共用），结果供渲染器读取
    this.lastInjection = hit
    return hit
  }

  /** 唤醒采样落盘：storeDir/wake-samples.jsonl（轮转：>8MB 时归档为 .1）。 */
  private async appendSample(record: Record<string, unknown>): Promise<void> {
    const { appendFile, stat, rename } = await import('node:fs/promises')
    const { join } = await import('node:path')
    const file = join(this.store.dir, 'wake-samples.jsonl')
    try {
      const st = await stat(file).catch(() => null)
      if (st && st.size > 8 * 1024 * 1024) {
        await rename(file, `${file}.1`).catch(() => {})
      }
    } catch { /* 归档失败忽略 */ }
    await appendFile(file, `${JSON.stringify(record)}\n`, 'utf8')
  }

  /** 渲染记忆注入段（动态预算：按相关度分级——高分完整入口、中分标题+摘要、低分仅标题）。 */
  renderInjection(budgetTokens: number): string {
    const { engrams } = this.lastInjection
    if (engrams.length === 0) return ''
    // 考究的使用引导：让模型知道「入口可直接用、细节可展开、更多可检索」——
    // 渐进披露的入口层语义（摘要够用直接用，不够才展开，避免无谓的 open 调用）。
    const HEADER = '（跨会话记忆：recall 检索 / open 展开 / store 写入 / link 织网。以下为相关记忆入口——摘要足够可直接用，需细节对 [[标题]] 用 engram_open 展开，需更多用 engram_recall）'
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

/** 从 GenerateOptions 提取查询文本（最后一条 user 消息 + 前一条 assistant 回复摘要拼接）。 */
function extractQuery(options: GenerateOptions): string {
  const messages = (options as { messages?: Array<{ role?: string; content?: unknown }> }).messages
  if (!messages || messages.length === 0) return ''
  const textOf = (content: unknown): string => {
    if (typeof content === 'string') return content
    if (Array.isArray(content)) {
      return content
        .map((b) => (typeof b === 'object' && b !== null && 'text' in b ? String((b as { text: unknown }).text) : ''))
        .join(' ')
    }
    return ''
  }
  let userText = ''
  let lastAssistant = ''
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const m = messages[i]
    const t = textOf(m.content).trim()
    if (t === '') continue
    if (m.role === 'user' && userText === '') {
      userText = t
      // 继续向上找最近一条 assistant 回复（上下文增益：口语-术语 gap 缩小，实测 +6.7% 召回）
    } else if (m.role === 'assistant' && userText !== '' && lastAssistant === '') {
      lastAssistant = t
      break
    }
  }
  if (userText === '') return ''
  const base = userText.slice(0, 1200)
  if (lastAssistant === '') return base
  // 拼接上轮回复（截断保证查询不长——bge 对长查询也敏感）
  return `${lastAssistant.slice(0, 400)}\n${base}`
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
