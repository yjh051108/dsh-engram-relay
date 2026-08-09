/**
 * dsh-engram-relay — 外置 engram 转接模型插件。
 *
 * 内置 <1B 本地模型（transformers.js/ONNX），作为 DSH 主模型与外部记忆
 * （engram 存储）之间的转接层：
 *
 *  - 写入：回合结束后，小模型把会话沉淀为结构化 engram（事实/决策/事件），
 *    带因果边（导致/依赖/引用）持久化到外置存储；
 *  - 唤醒：每次主模型请求前，小模型判断需要唤醒哪些记忆，沿因果图传播
 *    激活分数，只注入极少数超稀疏痕迹（预算默认 600 token）；
 *  - 转接：经 `llm/stream` waterfall 拦截模型调用（请求前注入、回合后
 *    蒸馏），零核心改动。
 *
 * @module dsh-engram-relay
 */

import type { Context as CordisContext } from 'cordis'
import type LlmService from '@deepseek-ai/dsh-llm'
import type SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import type ToolRegistry from '@deepseek-ai/dsh-tools'
import z from 'schemastery'

import { EngramRelay } from './relay.js'
import { installEngramTools } from './tools.js'

type Context = CordisContext & {
  llm: LlmService
  systemPrompt: SystemPrompt
  tools: ToolRegistry
}

export const name = 'dsh-engram-relay'
export const inject = ['llm', 'systemPrompt', 'tools']

export interface Config {
  modelId: string
  dtype: string
  storeDir: string | null
  injectBudgetTokens: number
  maxWakePerTurn: number
  distillEveryTurns: number
  enabled: boolean
}

export const Config: z<Config> = z.object({
  modelId: z.string().default('onnx-community/Qwen2.5-0.5B-Instruct')
    .description('内置转接模型 id（<1B，transformers.js/ONNX 格式）'),
  dtype: z.string().default('q8')
    .description('ONNX 量化档（q8/q4/fp32）'),
  storeDir: z.string().default('')
    .description('engram 持久化目录（空 = ~/.dsh/engram-relay/）'),
  injectBudgetTokens: z.number().min(0).max(8192).default(600)
    .description('单次唤醒注入的 token 预算（超稀疏：相对 100k 上下文 <1%）'),
  maxWakePerTurn: z.number().min(0).max(32).default(3)
    .description('每回合最多唤醒的 engram 条数'),
  distillEveryTurns: z.number().min(0).max(100).default(1)
    .description('每 N 回合蒸馏一次（0 = 关闭自动蒸馏）'),
  enabled: z.boolean().default(true)
    .description('总开关'),
})

export function apply(ctx: Context, config: Config): void {
  const relay = new EngramRelay(ctx, {
    modelId: config.modelId,
    dtype: config.dtype,
    storeDir: config.storeDir ?? '',
    injectBudgetTokens: config.injectBudgetTokens,
    maxWakePerTurn: config.maxWakePerTurn,
    distillEveryTurns: config.distillEveryTurns,
    enabled: config.enabled,
  })

  // 转接核心：llm/stream waterfall 拦截 + systemPrompt 记忆注入
  ctx.effect(() => relay.install(), 'dsh-engram-relay: relay')

  // 模型面工具
  ctx.effect(() => installEngramTools(ctx, relay), 'dsh-engram-relay: tools')
}
