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
import type CompactService from '@deepseek-ai/dsh-compact'

import { EngramStore } from './engram/store.js'
import { CausalGraph } from './engram/causal.js'
import { NgramHashAddressing } from './engram/hash.js'
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
  compact?: CompactService
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

  private disposers: Array<() => void> = []

  constructor(private ctx: CordisContext, private config: EngramRelayConfig) {
    this.store = new EngramStore(config.storeDir ?? '')
    this.hasher = new NgramHashAddressing({ seed: 0 })
    this.graph = new CausalGraph(this.store)
    this.model = new RelayModel(ctx, config)
    this.wake = new EngramWakeEngine(this.store, this.graph, this.hasher, config, {
      embedder: (query, candidates) => this.model.embed(query, candidates),
      scorer: (query, candidates) => this.model.score(query, candidates),
    })
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
          void this.wake.maybeWake(sessionId, options, { cwd: this.currentCwd ?? undefined }).catch((error) => {
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

    // 2. 记忆能力说明（固定文本，零动态：system 稳定 → 前缀缓存保持命中）。
    //    动态召回内容走消息尾注入（上方），需要时也可用 engram_recall 工具。
    this.ctx.systemPrompt.context({
      name: 'engram:relay',
      order: 9997,
      text: '本环境有 engram 跨会话记忆图谱（global/project/session 分层、因果链接、渐进披露）。需要时用 engram_recall 检索、engram_store 写入、engram_open 展开；每轮请求会自动把相关记忆追加到消息末尾。',
    })

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
      void this.maybeDistill().catch((error) => {
        this.ctx.logger?.warn?.('[engram-relay] distill failed: %s', String(error))
      })
    })

    // 4. 会话结束（分层生命周期）：只清该会话的 session 层临时记忆；
    //    global/project 跨会话层持久保留——跨会话沉淀的核心转变。
    this.ctx.on('agent/disposed', ({ agent }) => {
      if (!this.config.enabled) return
      const cleared = this.store.clearSession(agent.session.id)
      if (cleared > 0) {
        this.ctx.logger?.info?.('[engram-relay] session %s ended, cleared %d session-layer engrams (global/project kept)', agent.session.id, cleared)
      }
    })

    // 5. 图谱 Web API（web-only）：记忆图谱 Tab 的数据面（分层准入）。
    this.ctx.inject(['httpServer'], (webCtx) => {
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
      const stream = this.ctx.llm.stream({
        provider: route.provider,
        model: route.model,
        system: DISTILL_SYSTEM_PROMPT,
        messages: [createUserMessage({
          source: { kind: 'user' },
          content: [{ type: 'text', text: conversation.slice(0, 6000) }],
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
      if (layer === 'session' && !sessionId) continue
      if (!item.title || !item.summary) continue
      this.store.add({
        kind: (item.kind && KINDS.has(item.kind)) ? item.kind as EngramKind : 'note',
        layer,
        projectId: layer === 'project' ? cwd : null,
        title: item.title,
        summary: item.summary,
        content: item.content ?? '',
        links: [],
        sessionId: layer === 'session' ? sessionId : null,
        turn: this.lastTurnAt,
        causes: [],
        effects: [],
        importance: 0.6,
        // 无确认模式（默认）：蒸馏直接 confirmed 立即生效；distillRequireConfirm
        // true 时写 pending，等用户 engram_confirm。
        ...(this.config.distillRequireConfirm ? { status: 'pending' as const } : {}),
      })
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
      slotCount: this.store.slotCount(),
      graphEdges: this.graph.edgeCount(),
      model: await this.model.describe(),
      budgetTokens: this.config.injectBudgetTokens,
      currentCwd: this.currentCwd,
      compactCoexist: this.ctx.get('compact') !== undefined,
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
- layer: global(跨项目通用，如环境/工具/偏好) / project(仅当前项目相关，如架构/踩坑/约定) / session(仅本次会话，如临时进度)
- title: 简短入口标题（10 字内，如「部署端口决策」）
- summary: 一句话摘要（30 字内）
- content: 完整细节（关键参数、上下文，200 字内）
只输出 JSON 数组，不要任何其他文字。`

/** 宽松解析蒸馏输出：剥代码围栏 → 取首个 JSON 数组。 */
function parseDistillJson(text: string): Array<{
  kind?: string
  layer?: string
  title?: string
  summary?: string
  content?: string
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
      ? arr as Array<{ kind?: string; layer?: string; title?: string; summary?: string; content?: string }>
      : null
  } catch {
    return null
  }
}
