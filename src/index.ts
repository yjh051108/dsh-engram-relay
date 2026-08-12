/**
 * dsh-engram-relay — 外置 engram 转接模型插件。
 *
 * 大一统记忆图谱 + 超稀疏精准主动唤醒（**跨会话分层记忆**）：
 *
 *  - 分层：预设 3 层（global=全局持久 / project=项目持久·按工作目录 /
 *    session=会话临时·结束清理），归属由模型 engram_store 时**自主决策**；
 *    层是节点属性（大一统图谱不分家）。
 *  - 唤醒：每次主模型请求前，N-gram 哈希确定性寻址（O(1)，精确命中）
 *    粗筛候选 → **分层准入**（global 所有会话 / project 同 cwd / session
 *    本会话）→ bge 专用嵌入模型语义精排（修跨主题误命中）→ 因果图
 *    双向传播（前因/后果）→ 只注入极少数超稀疏痕迹（预算默认 600
 *    token），渐进披露（入口 = [[标题]] + 摘要，按需展开全文）；
 *  - 写入：模型经 engram_store 工具落节点（分层/标题/摘要/正文/链接/
 *    因果），session 层在会话结束清理，global/project 跨会话持久；
 *  - 维护：engram_search（盘点）/ link（织图谱）/ update / remove /
 *    promote（session→project/global 转长期）——类 LSP 的能力声明 +
 *    按需请求-响应；
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
  pythonPath: string
  pythonTimeoutMs: number
  checkpoint: string
  embedModel: string
}

export const Config: z<Config> = z.object({
  modelId: z.string().default('')
    .description('遗留：0.6B 蒸馏模型目录（已弃用；空 = 不加载，蒸馏/回忆降级）'),
  dtype: z.string().default('bfloat16')
    .description('模型精度（bfloat16/float16/float32）'),
  storeDir: z.string().default('')
    .description('engram 持久化目录（空 = ~/.dsh/engram-relay/）'),
  injectBudgetTokens: z.number().min(0).max(8192).default(600)
    .description('单次唤醒注入的 token 预算（超稀疏：相对 100k 上下文 <1%）'),
  maxWakePerTurn: z.number().min(0).max(32).default(3)
    .description('每回合最多唤醒的 engram 条数'),
  distillEveryTurns: z.number().min(0).max(100).default(1)
    .description('每 N 回合蒸馏一次（0 = 关闭自动蒸馏；0.6B 已移除，默认实际不生效）'),
  enabled: z.boolean().default(true)
    .description('总开关'),
  pythonPath: z.string().default('python')
    .description('Python 解释器路径（spawn 转接服务用）'),
  pythonTimeoutMs: z.number().min(1000).max(600000).default(120000)
    .description('Python 服务预热超时'),
  checkpoint: z.string().default('')
    .description('遗留：训练好的原生 engram checkpoint 路径（0.6B 已移除）'),
  embedModel: z.string().default('')
    .description('bge 嵌入模型目录（本地路径；空 = 服务端 ENGRAM_EMBED_MODEL 环境变量，再空则禁用语义精排）'),
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
    pythonPath: config.pythonPath,
    pythonTimeoutMs: config.pythonTimeoutMs,
    checkpoint: config.checkpoint ?? '',
    embedModel: config.embedModel,
  })

  // 转接核心：llm/stream waterfall 拦截 + systemPrompt 记忆注入
  ctx.effect(() => relay.install(), 'dsh-engram-relay: relay')

  // 模型面工具
  ctx.effect(() => installEngramTools(ctx, relay), 'dsh-engram-relay: tools')
}
