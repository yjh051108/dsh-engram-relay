/**
 * EngramRelay — 转接核心：大 engram 小 KV 的落地实现。
 *
 * 链路（全部挂在公开 seam 上，零核心改动）：
 *
 * 1. 请求前唤醒（读取）：`llm/stream` 旁路观察当前请求 → N-gram 哈希
 *    寻址外置 engram 表 → 门控打分 → 因果传播 → 超稀疏注入
 *    （systemPrompt 记忆段，预算默认 600 token）。
 *
 * 2. 回合后蒸馏（写入）：`agent/turn-stopping`（回合关闭边界）→ 从
 *    `agent.session.deriveMessages()` 提取最近回合文本 → <1B 模型蒸馏
 *    为 engram 条目写入外置表。**实时留底**：在官方 compact 折叠之前
 *    细节已进记忆表，折叠后仍可唤醒找回。
 *
 * 3. 与官方 compact 共存：不阻止、不替代官方折叠（它负责腾 KV，是
 *    成熟的有损总结式压缩）。engram 的职责在官方折叠之前完成——细节
 *    保真（可检索、带因果），官方负责空间（surface 替换）。
 */

import type { Context as CordisContext } from 'cordis'
import type LlmService from '@deepseek-ai/dsh-llm'
import { ReasoningEffortId } from '@deepseek-ai/dsh-llm'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import type SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import type ToolRegistry from '@deepseek-ai/dsh-tools'
import type CompactionEngine from '@deepseek-ai/dsh-compaction'

import { EngramStore } from './engram/store.js'
import { BruteForceIndex } from './engram/vector-index.js'
import { CausalGraph } from './engram/causal.js'
import { NgramHashAddressing } from './engram/hash.js'
import { ActivationCache } from './engram/activation.js'
import { EngramWakeEngine, type WakeViewer } from './engram/wake.js'
import { RelayModel } from './model/relay-model.js'
import { installGraphApi } from './graph-api.js'
import { ENGRAM_LAYERS, type EngramKind, type EngramLayer, type EngramNode } from './engram/store.js'
import type { EngramRelayConfig } from './types.js'
import { appendFileSync } from 'node:fs'
import { join } from 'node:path'

export interface EngramRelayDeps {
  llm: LlmService
  systemPrompt: SystemPrompt
  tools: ToolRegistry
  compaction?: CompactionEngine
}

/** 唤醒结果：本次请求注入的记忆痕迹（哈希命中 + 因果激活，超稀疏）。 */
export interface WakeResult {
  engrams: import('./engram/store.js').EngramNode[]
  reason: string
  injectedTokens: number
}

export class EngramRelay {
  readonly store: EngramStore
  readonly graph: CausalGraph
  readonly hasher: NgramHashAddressing
  readonly wake: EngramWakeEngine
  readonly model: RelayModel
  /** 类脑激活缓存（B=ln(Σt^-d)，强化事件驱动；wake 阶段 3 接入排序）。 */
  readonly activation: import('./engram/activation.js').ActivationCache
  /** 向量索引（int8 粗筛 + fp32 精筛双表；prefilter 候选来源）。 */
  readonly vectorIndex: import('./engram/vector-index.js').BruteForceIndex

  /** τ 融合权重（v0.4：写入织网与检索召回共用同一加权——可逆可解释）。 */
  get fusionTau(): { sem: number; time: number; cause: number } {
    return {
      sem: this.config.tauSem ?? 1,
      time: this.config.tauTime ?? 0,
      cause: this.config.tauCause ?? 0,
    }
  }

  private disposers: Array<() => void> = []

  constructor(private ctx: CordisContext, private config: EngramRelayConfig) {
    this.store = new EngramStore(config.storeDir ?? '')
    this.hasher = new NgramHashAddressing({ seed: 0 })
    this.graph = new CausalGraph(this.store)
    this.model = new RelayModel(ctx, config, this.store)
    this.activation = new ActivationCache()
    this.activation.rebuild(this.store.all())
    this.vectorIndex = new BruteForceIndex(this.store.dir)
    this.wake = new EngramWakeEngine(this.store, this.graph, this.hasher, config, {
      embedder: (query, candidates) => this.model.embed(query, candidates),
      scorer: (query, candidates) => this.model.score(query, candidates),
    }, async () => null, this.activation)
    // v0.5：纯算法语义（SemanticScorer 词汇+图通道）已替换 embedding 精排；
    // 向量 prefilter 停用（哈希粗筛兜底）——BM25 倒排索引替代待 P5 实施。
  }

  /**
   * 向量预筛（prefilter 钩子）：查询向量 → int8 全量内积 top-50 → 候选 id。
   * 含懒补 ensure：新记忆未入向量表时差量 embed 补入；embedder 不可用返回 null（哈希兜底）。
   */
  private async vectorPrefilter(query: string): Promise<string[] | null> {
    try {
      // 懒补：store 节点数 > 向量表行数 → 差量补算（写入后首次检索前补）
      const all = this.store.all().filter((e) => e.status !== 'pending')
      if (all.length > this.vectorIndex.size) {
        const missing = all.filter((e) => !this.vectorIndex.has(e.id))
        if (missing.length > 0) {
          const raw = await this.model.embedRaw(query, missing.map((e) => `${e.title}：${e.summary.slice(0, 200)}`))
          if (raw) {
            missing.forEach((e, i) => {
              if (raw.vectors[i]) this.vectorIndex.add(e.id, Float32Array.from(raw.vectors[i]))
            })
            this.vectorIndex.persist()
          }
        }
      }
      if (this.vectorIndex.size === 0) return null
      const raw = await this.model.embedRaw(query, [query])
      if (!raw) return null
      const hits = this.vectorIndex.search(Float32Array.from(raw.query_vec), 50)
      return hits.map((h) => h.id)
    } catch {
      return null
    }
  }

  /** 挂载所有 seam。 */
  install(): () => void {
    // 0. 预热 Python 魔改模型服务（失败降级，不阻塞）。
    void this.model.warmup()

    // 1. 请求前唤醒（llm/stream 旁路观察，不包装流）。
    this.ctx.on('llm/stream', (options, next) => {
      if (this.config.enabled) {
        // 捕获最近一次模型调用的路由（LLM 蒸馏复用同一 provider/model）
        this.lastLlmRoute = { provider: options.provider, model: options.model }
        const sessionId = (options as { sessionId?: string }).sessionId
        if (sessionId) {
          // 分层准入需要查看者视角：sessionId + 当前工作目录（cwd 经
          // turn-stopping 持续追踪）
          void this.wake.maybeWake(sessionId, options, { cwd: this.currentCwd ?? undefined, turn: this.lastTurnAt }).catch((error) => {
            this.ctx.logger?.warn?.('[engram-relay] wake failed: %s', String(error))
          })
          // 训练模型的原生回忆（异步，结果缓存供记忆段渲染）
          const query = extractQueryText(options)
          if (query) {
            void this.maybeRecall(query).catch((error) => {
              this.ctx.logger?.warn?.('[engram-relay] recall failed: %s', String(error))
            })
          }
          // 记忆注入（缓存友好）：追加到消息流末尾而非 system——system 内
          // 任何动态内容都会使其后 tools+历史缓存全失效；尾部注入变化只在
          // 历史末端，system+tools+前缀历史保持命中。
          try {
            const injection = this.renderMemorySection()
            if (injection) {
              const messages = (options as { messages?: unknown[] }).messages
              if (Array.isArray(messages)) {
                ;(messages as unknown[]).push({ role: 'system', content: injection } as never)
              }
            }
          } catch (error) {
            this.ctx.logger?.warn?.('[engram-relay] injection failed: %s', String(error))
          }
        }
      }
      return next()
    })

    // 2. 记忆能力说明（固定文本，零动态：静态到头——order 靠前，工具 schema
    //    变更时静态段仍缓存命中；动态召回内容走消息尾注入（下方），永不进
    //    system 头部）。
    //    ⚠️ 必须挂 ctx.effect：裸注册在 fiber 重建（热重载/失败回滚）时不注销，
    //    残留导致下次 apply duplicate "engram:relay already registered"。
    this.ctx.effect(() => this.ctx.systemPrompt.context({
      name: 'engram:relay',
      order: -85,
      text: '本环境有 engram 记忆图谱（分层/因果/链接）。engram_recall 等 engram_* 工具直接可用；每轮请求会自动把相关记忆追加到消息末尾。',
    }), 'engram-relay: memory-context')

    // 3. 回合后蒸馏（agent/turn-stopping：回合关闭边界，serial 不 veto）。
    //    从 agent.session.deriveMessages() 提取最近回合文本 → <1B 模型
    //    蒸馏为 engram（实时留底——在官方 compact 折叠之前，细节已进
    //    外置记忆表，折叠后仍可唤醒找回）。
    this.ctx.on('agent/turn-stopping', ({ agent, turn }) => {
      if (!this.config.enabled) return
      this.lastTurnAt = turn
      this.currentSessionId = agent.session.id
      // 持续追踪当前工作目录（分层准入：project 层按 cwd 过滤）
      const cwd = agent.session.header?.cwd
      if (typeof cwd === 'string' && cwd !== '') this.currentCwd = cwd
      const messages = extractRecentTurn(agent.session.deriveMessages(), turn)
      this.lastConversationText = messages
      // 工作快照（远景场景 6"继续昨天的工作"）：聚合最近写入的进行中状态
      if (typeof cwd === 'string' && cwd !== '') {
        try {
          this.store.upsertSnapshot(cwd, turn, this.currentSessionId)
        } catch (error) {
          this.ctx.logger?.warn?.('[engram-relay] snapshot failed: %s', String(error))
        }
      }
      // 硬上限（v0.5）：每回合惰性触发归档淘汰（不阻塞）
      try {
        const archived = this.store.enforceLimit(this.config.maxNodes ?? 10000)
        if (archived > 0) {
          this.ctx.logger?.info?.('[engram-relay] archive: %d nodes (max=%d, count=%d)', archived, this.config.maxNodes, this.store.count())
        }
      } catch { /* 淘汰失败不阻塞 */ }
      void this.maybeDistill().catch((error) => {
        this.ctx.logger?.warn?.('[engram-relay] distill failed: %s', String(error))
      })
    })

    // 4. 图谱 Web API（web-only）：记忆图谱 Tab 的数据面（分层准入）。
    this.ctx.inject(['webServer'], (webCtx) => {
      const disposeGraphApi = installGraphApi(webCtx as never, this)
      this.disposers.push(disposeGraphApi)
    })

    return () => {
      this.disposers.forEach((d) => d())
      this.disposers = []
    }
  }

  private renderMemorySection(): string {
    // engram 文本注入（哈希唤醒）
    const base = this.wake.renderInjection(this.config.injectBudgetTokens)
    // 训练模型的「原生回忆」结果（异步缓存，唤醒时填充）
    if (this.lastRecallText && base !== '') {
      return `${base}\n<engram-recall>（记忆模型原生回忆）${this.lastRecallText.slice(0, 160)}</engram-recall>`
    }
    return base
  }

  /** 异步触发训练模型的原生回忆（由 llm/stream 旁路调用，缓存结果）。 */
  async maybeRecall(query: string): Promise<void> {
    if (!this.config.enabled) return
    try {
      const recalled = await this.model.recall(query)
      if (recalled) this.lastRecallText = recalled
    } catch {
      // 回忆失败静默（降级为纯 engram 注入）
    }
  }

  private lastRecallText: string | null = null

  /** 回合后蒸馏：LLM 把最近回合内容提取为 engram（⏳待确认，用户确认后生效）。 */
  private async maybeDistill(): Promise<void> {
    this.debugLog(`distill called: everyTurns=${this.config.distillEveryTurns} convLen=${(this.lastConversationText ?? '').length} route=${this.lastLlmRoute ? `${this.lastLlmRoute.provider}/${this.lastLlmRoute.model}` : 'none'}`)
    if (this.config.distillEveryTurns === 0) return
    const conversation = this.lastConversationText ?? ''
    if (!conversation.trim()) {
      this.debugLog('distill skip: conversation empty')
      return
    }
    const route = this.lastLlmRoute
    if (!route) {
      this.ctx.logger?.warn?.('[engram-relay] distill skipped: no llm route captured yet')
      this.debugLog('distill skip: no llm route')
      return
    }
    let text = ''
    try {
      // 已有记忆入口（供因果推断引用——入口级，控制体积）
      const entryList = this.store.all()
        .map((n) => `[[${n.title}]] ${n.summary.slice(0, 40)}`)
        .join('\n')
      const userText = `已有记忆入口：\n${entryList.slice(0, 1800)}\n\n最近对话回合：\n${conversation.slice(0, 3800)}`
      const stream = this.ctx.llm.stream({
        provider: route.provider,
        model: route.model,
        system: DISTILL_SYSTEM_PROMPT,
        messages: [createUserMessage({
          source: { kind: 'user' },
          content: [{ type: 'text', text: userText }],
        })],
        temperature: 0.3,
        // 蒸馏是结构化抽取任务，不需要思考链：显式关闭 reasoning，
        // 否则 reasoningEffort=max 的思考会吃光 800 token 预算导致输出为空
        // （实测 distill-debug.log 大量 `outLen=0`）。
        reasoningEffort: ReasoningEffortId('off'),
        maxTokens: 1500,
      })
      for await (const chunk of stream) {
        if (chunk.type === 'text-delta') text += chunk.text
      }
      this.debugLog(`distill llm ok: outLen=${text.length}`)
    } catch (error) {
      this.ctx.logger?.warn?.('[engram-relay] distill llm failed: %s', String(error))
      this.debugLog(`distill llm FAILED: ${String(error)}`)
      return
    }
    const parsed = parseDistillJson(text)
    if (!parsed || parsed.length === 0) {
      this.ctx.logger?.warn?.('[engram-relay] distill output unparsable/empty: %s', text.slice(0, 160))
      this.debugLog(`distill output unparsable/empty: ${text.slice(0, 160)}`)
      return
    }
    this.debugLog(`distill parsed: ${parsed.length} items`)
    // 自动沉淀 → ⏳待确认（用户确认制合流：不擅自写入生效记忆）
    let proposed = 0
    const sessionId = this.currentSessionId ?? ''
    const cwd = this.currentCwd ?? null
    for (const item of parsed) {
      const layer = item.layer as EngramLayer
      if (!ENGRAM_LAYERS.includes(layer)) continue
      if (layer === 'project' && !cwd) continue
      if (!item.title || !item.summary) continue
      // 自动因果：causesOf 引用的已有记忆标题 → 建因果边（前因 → 新节点）。
      const causeIds: string[] = []
      for (const causeTitle of (Array.isArray(item.causesOf) ? item.causesOf : [])) {
        const cause = this.store.all().find((n) => n.title === causeTitle && n.id !== undefined)
        if (cause) causeIds.push(cause.id)
      }
      this.store.add({
        kind: (item.kind && KINDS.has(item.kind)) ? item.kind as EngramKind : 'note',
        layer,
        projectId: layer === 'project' ? cwd : null,
        title: item.title,
        summary: item.summary,
        content: item.content ?? '',
        links: [],
        sessionId,
        turn: this.lastTurnAt,
        causes: causeIds,
        effects: [],
        importance: 0.6,
        // 无确认模式（默认）：蒸馏直接 confirmed 立即生效；distillRequireConfirm
        // true 时写 pending，等用户 engram_confirm。
        ...(this.config.distillRequireConfirm ? { status: 'pending' as const } : {}),
      })
      // 建边 + 前因节点 effects 双写
      const added = this.store.all().find((n) => n.title === item.title)
      if (added) {
        for (const causeId of causeIds) {
          this.graph.addEdge(causeId, added.id, 'causes', 1)
          this.store.update(causeId, { effects: [...(this.store.get(causeId)?.effects ?? []), added.id] })
        }
      }
      proposed++
    }
    this.debugLog(`distill proposed: ${proposed}`)
    this.ctx.logger?.info?.('[engram-relay] distill: %d 条回合记忆已沉淀为 ⏳待确认节点', proposed)
  }

  /** 蒸馏排查日志（写入图谱目录 distill-debug.log）。 */
  private debugLog(msg: string): void {
    try {
      appendFileSync(join(this.store.dir, 'distill-debug.log'), `${new Date().toISOString()} ${msg}\n`)
    } catch {
      // 日志失败静默
    }
  }

  private lastConversationText = ''
  /** 最近一次模型调用的路由（llm/stream 拦截时捕获；LLM 蒸馏复用）。 */
  private lastLlmRoute: { provider: string; model: string } | null = null
  /** 当前会话 id（工具写入时归属；会话结束清理用）。 */
  currentSessionId: string | null = null
  /** 当前回合号（工具写入时归属）。 */
  lastTurnAt = 0
  /** 当前工作目录（分层准入：project 层按 cwd 过滤；turn-stopping 持续追踪）。 */
  currentCwd: string | null = null

  /**
   * 供工具使用的唤醒查询入口。
   * @param viewer - 查看者视角（分层准入：{ sessionId, cwd }）。
   * @param layer - 可选层过滤（逗号分隔如 'global,project'；缺省不过滤，
   *   由 viewer 准入决定可见层）。
   */
  async recall(query: string, limit?: number, viewer: WakeViewer = {}, layer?: string): Promise<WakeResult> {
    const hit = await this.wake.query(query, limit ?? this.config.maxWakePerTurn, viewer)
    let engrams = hit.engrams
    if (layer !== undefined && layer.trim() !== '') {
      const layers = layer.split(',').map((s) => s.trim()).filter(Boolean) as EngramLayer[]
      if (layers.length > 0) engrams = engrams.filter((e) => layers.includes(e.layer))
    }
    return {
      engrams,
      reason: hit.reason,
      injectedTokens: hit.injectedTokens,
    }
  }

  async status(): Promise<Record<string, unknown>> {
    return {
      enabled: this.config.enabled,
      storeDir: this.store.dir,
      engramCount: this.store.count(),
      pendingCount: this.store.pending().length,
      layerCounts: this.store.layerCounts(),
      stateCounts: this.store.stateCounts(),
      archivedCount: this.store.archivedCount(),
      maxNodes: this.config.maxNodes ?? 10000,
      slotCount: this.store.slotCount(),
      graphEdges: this.graph.edgeCount(),
      model: await this.model.describe(),
      budgetTokens: this.config.injectBudgetTokens,
      currentCwd: this.currentCwd,
      compactCoexist: this.ctx.get('compaction') !== undefined,
    }
  }
}

/**
 * 从会话消息投影中提取「最近一个回合」的文本（供蒸馏）。
 * deriveMessages 返回完整历史（frozen Message[]）；取最后一条 user
 * 消息起至末尾的文本块拼接。长度上限由调用方（distillTurn）截断。
 */
function extractRecentTurn(messages: Array<{ role?: string; content?: unknown }>, _turn: number): string {
  // 找最后一条 user 消息的起点
  let start = 0
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i]?.role === 'user') {
      start = i
      break
    }
  }
  const parts: string[] = []
  for (let i = start; i < messages.length; i += 1) {
    const m = messages[i]
    if (!m) continue
    const role = m.role ?? 'unknown'
    const content = m.content
    if (typeof content === 'string') {
      parts.push(`[${role}] ${content}`)
    } else if (Array.isArray(content)) {
      const text = content
        .map((b) => (typeof b === 'object' && b !== null && 'text' in b ? String((b as { text: unknown }).text) : ''))
        .filter((t) => t !== '')
        .join(' ')
      if (text !== '') parts.push(`[${role}] ${text}`)
    }
  }
  return parts.join('\n').slice(0, 4000)
}

/** 从 GenerateOptions 提取最后一条 user 消息文本（供原生回忆查询）。 */
function extractQueryText(options: unknown): string {
  const messages = (options as { messages?: Array<{ role?: string; content?: unknown }> }).messages
  if (!messages || messages.length === 0) return ''
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const m = messages[i]
    if (m?.role !== 'user') continue
    const content = m.content
    if (typeof content === 'string') return content.slice(0, 300)
    if (Array.isArray(content)) {
      const text = content
        .map((b) => (typeof b === 'object' && b !== null && 'text' in b ? String((b as { text: unknown }).text) : ''))
        .join(' ')
      if (text.trim() !== '') return text.slice(0, 300)
    }
  }
  return ''
}

/** 蒸馏允许的 kind 集合。 */
const KINDS = new Set(['fact', 'decision', 'event', 'note'])

/** LLM 自动蒸馏的系统提示（回合 → 记忆条目，JSON 数组输出）。 */
const DISTILL_SYSTEM_PROMPT = `你是 engram 记忆提取器。从用户提供的「最近对话回合」中提取值得长期记住的信息（事实/决策/事件/约定/踩坑），最多 3 条。只提取可复用、有长期价值的；寒暄、过程性、一次性内容一律不提取（返回空数组 []）。
每条输出 JSON 对象：
- kind: fact(事实/约定) / decision(决策/方案) / event(事件/进展) / note(笔记/其它)
- layer: project(默认——归属当前项目) / global(通用知识——换个项目还有用：技术模式/平台坑/偏好)
- title: 简短入口标题（10 字内，如「部署端口决策」）
- summary: 一句话摘要（30 字内）
- content: 完整细节（关键参数、上下文，200 字内）
- causesOf: 导致这条记忆的**已有记忆标题**数组（从下方「已有记忆入口」列表精确引用，最多 2 个；没有因果关系的填 []）——例如本轮"修复了 X"是由之前的「Y 故障定位」导致的，则 causesOf: ["Y 故障定位"]
- 额外：若本回合的内容与「已有记忆入口」中的多条记忆构成**反复出现的模式/规律**（如"每次改 X 都会引发 Y"），可再输出一条规律记忆（kind=fact，title 以「规律」开头，content 描述模式与证据）——没有明显规律则不输出。
只输出 JSON 数组，不要任何其他文字。`

/** 宽松解析蒸馏输出：剥代码围栏 → 取首个 JSON 数组。 */
function parseDistillJson(text: string): Array<{
  kind?: string
  layer?: string
  title?: string
  summary?: string
  content?: string
  causesOf?: string[]
}> | null {
  let t = text.trim()
  const fence = t.match(/```(?:json)?\s*([\s\S]*?)\s*```/)
  if (fence) t = fence[1]
  const start = t.indexOf('[')
  const end = t.lastIndexOf(']')
  if (start < 0 || end <= start) return null
  try {
    const arr = JSON.parse(t.slice(start, end + 1))
    return Array.isArray(arr)
      ? arr as Array<{ kind?: string; layer?: string; title?: string; summary?: string; content?: string; causesOf?: string[] }>
      : null
  } catch {
    return null
  }
}
