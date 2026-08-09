/**
 * EngramRelay — 转接核心。
 *
 * 职责：
 *  1. 请求前唤醒：拦截 `llm/stream`，用小模型对当前请求打分，沿因果图
 *     传播激活，把超稀疏唤醒结果注入 systemPrompt 记忆段（不污染消息流）；
 *  2. 回合后蒸馏：监听会话事件，每 N 回合调用小模型把新内容蒸馏为 engram
 *     并写入外置存储（带因果边）；
 *  3. 状态面：向工具注册面提供 recall / store / status 能力。
 */

import type { Context as CordisContext } from 'cordis'
import type LlmService from '@deepseek-ai/dsh-llm'
import type SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import type ToolRegistry from '@deepseek-ai/dsh-tools'

import { EngramStore } from './engram/store.js'
import { CausalGraph } from './engram/causal.js'
import { CausalWakeEngine } from './engram/wake.js'
import { LocalRelayModel } from './model/local.js'
import type { EngramRelayConfig } from './types.js'

export interface EngramRelayDeps {
  llm: LlmService
  systemPrompt: SystemPrompt
  tools: ToolRegistry
}

/**
 * 唤醒结果：本次请求注入的记忆痕迹（已按因果激活排序、超稀疏截断）。
 */
export interface WakeResult {
  engrams: import('./engram/store.js').Engram[]
  reason: string
  injectedTokens: number
}

export class EngramRelay {
  readonly store: EngramStore
  readonly graph: CausalGraph
  readonly wake: CausalWakeEngine
  readonly model: LocalRelayModel

  private disposers: Array<() => void> = []

  constructor(private ctx: CordisContext, private config: EngramRelayConfig) {
    this.store = new EngramStore(config.storeDir ?? '')
    this.graph = new CausalGraph(this.store)
    this.model = new LocalRelayModel(ctx, config)
    // 打分回调：模型就绪时用小模型打分（语义种子），否则纯因果图降级。
    this.wake = new CausalWakeEngine(this.graph, this.store, config, (query, candidates) =>
      this.model.score(query, candidates),
    )
  }

  /** 挂载所有 seam（llm/stream 拦截、systemPrompt 注入、会话监听）。 */
  install(): () => void {
    // 1. 请求前唤醒：llm/stream 是 waterfall（必须返回流），这里只做
    //    旁路观测——捕获当前请求上下文触发异步唤醒，不包装流本身。
    //    唤醒结果由 systemPrompt.context 渲染器在装配时读取。
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

    // 2. 记忆段注入：systemPrompt 装配时渲染最新唤醒结果（超稀疏）。
    this.ctx.systemPrompt.context({
      name: 'engram:relay',
      order: 800,
      text: () => this.renderMemorySection(),
    })

    // 3. 回合后蒸馏：回合关闭边界（serial）触发，每 N 回合蒸馏一次。
    //    listener 返回 void 不 veto，安全。
    this.ctx.on('agent/turn-stopping', () => {
      if (!this.config.enabled) return
      void this.maybeDistill().catch((error) => {
        this.ctx.logger?.warn?.('[engram-relay] distill failed: %s', String(error))
      })
    })

    return () => {
      this.disposers.forEach((d) => d())
      this.disposers = []
    }
  }

  private renderMemorySection(): string {
    return this.wake.renderInjection(this.config.injectBudgetTokens)
  }

  private async maybeDistill(): Promise<void> {
    await this.model.distillTurn(this.store, this.graph)
  }

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
      graphEdges: this.graph.edgeCount(),
      model: await this.model.describe(),
      budgetTokens: this.config.injectBudgetTokens,
    }
  }
}
