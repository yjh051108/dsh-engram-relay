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
 *    因果），记忆跨会话持久（global/project 两层，session 层已删除）；
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
  distillRequireConfirm: boolean
  semanticMinScore: number
  recencyWeight: number
  wakeSampleLog: boolean
  tauSem: number
  tauTime: number
  tauCause: number
  maxNodes: number
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
    .description('bge 嵌入模型目录（本地路径；空 = 优先包内 model/bge-small-zh（仓库自带 int8），再空则服务端 ENGRAM_EMBED_MODEL 环境变量）'),
  distillRequireConfirm: z.boolean().default(false)
    .description('蒸馏产物是否需确认才生效：true=写 ⏳pending（确认后才参与检索），false=无确认模式，蒸馏直接 confirmed 立即生效（Obsidian 式开箱即用）'),
  semanticMinScore: z.number().min(0).max(1).default(0.42)
    .description('唤醒语义阈值：bge 余弦相似度下限（低于此值不注入；无关记忆零注入）'),
  recencyWeight: z.number().min(-2).max(2).default(0.25)
    .description('时序 recency 加权强度（1+w·e^(-Δturn/20)；仿真标定显示方向需数据驱动：0=关闭，负=旧记忆优先，默认 0.25 保守）'),
  wakeSampleLog: z.boolean().default(true)
    .description('唤醒采样日志（storeDir/wake-samples.jsonl）：记录查询/候选分数/注入选择，供融合权重离线拟合'),
  tauSem: z.number().min(0).max(10).default(1)
    .description('τ 融合：语义通道权重（z-score 语义分）'),
  tauTime: z.number().min(-5).max(5).default(0)
    .description('τ 融合：时序通道权重（z-score 激活；默认 0=纯语义，fit-tau 拟合后更新）'),
  tauCause: z.number().min(0).max(5).default(0)
    .description('τ 融合：因果通道权重（因果 1 跳可达 0/1）'),
  maxNodes: z.number().min(0).max(1000000).default(10000)
    .description('主库硬上限：超过触发归档淘汰（superseded→dormant→低激活，归档可恢复）；0=无限'),
})

/** 包内模型解析：空配置 → 仓库自带 model/bge-small-zh（int8 免下载）；解析失败 → 空串（走 Python 服务 ENGRAM_EMBED_MODEL / 重要度兜底）。 */
function resolveEmbedModel(configured: string): string {
  if (configured.trim() !== '') return configured
  try {
    const bundled = new URL('../model/bge-small-zh/', import.meta.url).pathname
    // Windows 路径修复（file:// URL 的 /C:/ 前缀 + 非 ASCII 目录百分号编码解码）
    const bundledPath = process.platform === 'win32'
      ? decodeURIComponent(bundled).replace(/^\/([A-Za-z]:)/, '$1').replace(/\//g, '\\')
      : decodeURIComponent(bundled)
    if (bundledPath) return bundledPath
  } catch { /* 解析失败回退 */ }
  return ''
}

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
    // 包内模型优先：空配置时用仓库自带 model/bge-small-zh（int8，免下载）；
    // 显式配置（本地 fp32 目录）时沿用配置。
    embedModel: resolveEmbedModel(config.embedModel),
    distillRequireConfirm: config.distillRequireConfirm,
    semanticMinScore: config.semanticMinScore,
    recencyWeight: config.recencyWeight,
    wakeSampleLog: config.wakeSampleLog,
    tauSem: config.tauSem,
    tauTime: config.tauTime,
    tauCause: config.tauCause,
    maxNodes: config.maxNodes,
  })

  // 转接核心：llm/stream waterfall 拦截 + systemPrompt 记忆注入
  ctx.effect(() => relay.install(), 'dsh-engram-relay: relay')

  // 模型面工具
  ctx.effect(() => installEngramTools(ctx, relay), 'dsh-engram-relay: tools')
}
