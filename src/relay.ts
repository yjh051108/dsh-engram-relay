/**
 * EngramRelay — 转接核心：大 engram 小 KV 的落地实现。
 *
 * 三条链路（全部挂在公开 seam 上，零核心改动）：
 *
 * 1. 请求前唤醒（读取）：`llm/stream` 旁路观察当前请求 → N-gram 哈希
 *    寻址外置 engram 表 → 门控打分 → 因果传播 → 超稀疏注入
 *    （systemPrompt 记忆段，预算默认 600 token）。
 *
 * 2. 回合后蒸馏（写入）：`agent/turn-stopping`（回合关闭边界）→ <1B
 *    模型把本回合对话蒸馏为 engram 条目（全量，含跨会话全局/项目/
 *    规则记忆的提炼），写入外置表。hash 键由蒸馏文本决定——相同主题
 *    永远命中相同槽位。
 *
 * 3. 历史折叠（小 KV）：`agent/pre-step` waterfall → token 压力超阈值
 *    → <1B 模型把早期历史蒸馏成 engram 摘要 → 调 `ctx.compact`
 *    compactNow 折叠 → 折叠内容随时按哈希唤醒找回。等效：上下文窗口
 *    保持小，记忆容量无限（100k → 1M+）。
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
    this.ctx.on('agent/turn-stopping', ({ turn }) => {
      if (!this.config.enabled) return
      this.lastTurnAt = turn
      void this.maybeDistill().catch((error) => {
        this.ctx.logger?.warn?.('[engram-relay] distill failed: %s', String(error))
      })
    })

    // 4. 历史折叠（agent/pre-step：waterfall，可替换进入 step 的消息）。
    //    在官方 compact 压力触发点之外，由本插件的 engram 折叠策略驱动：
    //    先蒸馏早期历史为 engram（保底可召回），再走官方 compact 折叠。
    this.ctx.on('agent/pre-step', async ({ agent }, next) => {
      if (!this.config.enabled) return next()
      try {
        await this.maybeFold(agent)
      } catch (error) {
        this.ctx.logger?.warn?.('[engram-relay] fold failed: %s', String(error))
      }
      return next()
    })

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
    // TODO(relay): 从会话投影读取最近回合文本（后续接 session 读取）。
    const conversation = this.lastConversationText ?? ''
    await this.model.distillTurn(this.store, this.graph, conversation, this.currentSessionId ?? '', this.lastTurnAt)
  }

  /** 历史折叠：token 压力超阈值时蒸馏早期历史 → 官方 compact 折叠。 */
  private async maybeFold(_agent: { session: { id: string } }): Promise<void> {
    // TODO(relay): 读取会话 token 压力 → RelayModel.foldHistory → compactNow。
    // v2 折叠核心在 Python 侧（记忆表写入），Node 侧待会话投影接入后实现。
  }

  /** 供外部注入最近会话文本（relay 集成 session 投影后使用）。 */
  setConversationContext(text: string, sessionId: string, turn: number): void {
    this.lastConversationText = text
    this.currentSessionId = sessionId
    this.lastTurnAt = turn
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
      foldEnabled: (this.ctx as unknown as { compact?: unknown }).compact !== undefined,
    }
  }
}
