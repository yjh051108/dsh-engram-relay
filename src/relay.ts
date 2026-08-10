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
import type SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import type ToolRegistry from '@deepseek-ai/dsh-tools'
import type CompactService from '@deepseek-ai/dsh-compact'

import { EngramStore } from './engram/store.js'
import { CausalGraph } from './engram/causal.js'
import { NgramHashAddressing } from './engram/hash.js'
import { EngramWakeEngine } from './engram/wake.js'
import { RelayModel } from './model/relay-model.js'
import type { EngramRelayConfig } from './types.js'

export interface EngramRelayDeps {
  llm: LlmService
  systemPrompt: SystemPrompt
  tools: ToolRegistry
  compact?: CompactService
}

/** 唤醒结果：本次请求注入的记忆痕迹（哈希命中 + 因果激活，超稀疏）。 */
export interface WakeResult {
  engrams: import('./engram/store.js').Engram[]
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
  private lastTurnAt = 0

  constructor(private ctx: CordisContext, private config: EngramRelayConfig) {
    this.store = new EngramStore(config.storeDir ?? '')
    this.hasher = new NgramHashAddressing({ seed: 0 })
    this.graph = new CausalGraph(this.store)
    this.model = new RelayModel(ctx, config)
    this.wake = new EngramWakeEngine(this.store, this.graph, this.hasher, config, (query, candidates) =>
      this.model.score(query, candidates),
    )
  }

  /** 挂载所有 seam。 */
  install(): () => void {
    // 0. 预热 Python 魔改模型服务（失败降级，不阻塞）。
    void this.model.warmup()

    // 1. 请求前唤醒（llm/stream 旁路观察，不包装流）。
    this.ctx.on('llm/stream', (options, next) => {
      if (this.config.enabled) {
        const sessionId = (options as { sessionId?: string }).sessionId
        if (sessionId) {
          void this.wake.maybeWake(sessionId, options).catch((error) => {
            this.ctx.logger?.warn?.('[engram-relay] wake failed: %s', String(error))
          })
        }
      }
      return next()
    })

    // 2. 记忆段注入（systemPrompt 装配时渲染最新唤醒结果，超稀疏）。
    this.ctx.systemPrompt.context({
      name: 'engram:relay',
      order: 800,
      text: () => this.renderMemorySection(),
    })

    // 3. 回合后蒸馏（agent/turn-stopping：回合关闭边界，serial 不 veto）。
    //    从 agent.session.deriveMessages() 提取最近回合文本 → <1B 模型
    //    蒸馏为 engram（实时留底——在官方 compact 折叠之前，细节已进
    //    外置记忆表，折叠后仍可唤醒找回）。
    this.ctx.on('agent/turn-stopping', ({ agent, turn }) => {
      if (!this.config.enabled) return
      this.lastTurnAt = turn
      this.currentSessionId = agent.session.id
      const messages = extractRecentTurn(agent.session.deriveMessages(), turn)
      this.lastConversationText = messages
      void this.maybeDistill().catch((error) => {
        this.ctx.logger?.warn?.('[engram-relay] distill failed: %s', String(error))
      })
    })

    // 4. 与官方 compact 共存：不阻止、不替代官方折叠（它负责腾 KV）。
    //    本插件的职责在官方折叠**之前**完成——每回合蒸馏已把细节留底；
    //    官方 compact 折叠后，细节经 engram 哈希/因果唤醒找回。

    return () => {
      this.disposers.forEach((d) => d())
      this.disposers = []
    }
  }

  private renderMemorySection(): string {
    return this.wake.renderInjection(this.config.injectBudgetTokens)
  }

  /** 回合后蒸馏：<1B 模型把最近回合内容蒸馏为 engram。 */
  private async maybeDistill(): Promise<void> {
    if (this.config.distillEveryTurns === 0) return
    const conversation = this.lastConversationText ?? ''
    await this.model.distillTurn(this.store, this.graph, conversation, this.currentSessionId ?? '', this.lastTurnAt)
  }

  private lastConversationText = ''
  private currentSessionId: string | null = null

  /** 供工具使用的唤醒查询入口。 */
  async recall(query: string, limit?: number): Promise<WakeResult> {
    const hit = await this.wake.query(query, limit ?? this.config.maxWakePerTurn)
    return {
      engrams: hit.engrams,
      reason: hit.reason,
      injectedTokens: hit.injectedTokens,
    }
  }

  async status(): Promise<Record<string, unknown>> {
    return {
      enabled: this.config.enabled,
      storeDir: this.store.dir,
      engramCount: this.store.count(),
      slotCount: this.store.slotCount(),
      graphEdges: this.graph.edgeCount(),
      model: await this.model.describe(),
      budgetTokens: this.config.injectBudgetTokens,
      compactCoexist: (this.ctx as unknown as { compact?: unknown }).compact !== undefined,
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
