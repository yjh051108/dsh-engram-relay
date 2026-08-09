/**
 * LocalRelayModel — 内置 <1B 转接模型（transformers.js / ONNX 本地推理）。
 *
 * 模型职责（转接层的「智能」）：
 *  - distill：把回合内容蒸馏为结构化 engram（类型/摘要/因果边/重要度）；
 *  - score：对候选 engram 与唤醒查询打分（语义相关性，作为因果传播的种子）；
 *  - describe：模型元数据（id / 状态）。
 *
 * 实现：transformers.js pipeline（text-generation + feature-extraction），
 * 模型权重首次使用时自动下载并缓存（~/.dsh/engram-relay/models/），
 * 之后完全离线运行。onnxruntime-node 版本经 overrides 固定为兼容档。
 */

import type { Context as CordisContext } from 'cordis'
import type { EngramRelayConfig } from '../types.js'
import type { EngramStore } from '../engram/store.js'
import type { CausalGraph } from '../engram/causal.js'
import type { Engram } from '../engram/store.js'

interface TransformersJs {
  pipeline: (
    task: string,
    model: string,
    options?: Record<string, unknown>,
  ) => Promise<unknown>
  env: Record<string, unknown>
}

export class LocalRelayModel {
  private pipelinePromise: Promise<unknown> | null = null
  private embedPromise: Promise<unknown> | null = null
  private loadError: string | null = null

  constructor(private ctx: CordisContext, private config: EngramRelayConfig) {}

  /** 惰性加载 transformers.js（首次使用时）。 */
  private async tf(): Promise<TransformersJs> {
    try {
      const mod = await import('@huggingface/transformers')
      return mod as unknown as TransformersJs
    } catch (error) {
      this.loadError = String(error)
      throw error
    }
  }

  private async generator(): Promise<unknown> {
    if (!this.pipelinePromise) {
      this.pipelinePromise = this.tf().then((tf) =>
        tf.pipeline('text-generation', this.config.modelId, { dtype: this.config.dtype }),
      )
    }
    return this.pipelinePromise
  }

  private async embedder(): Promise<unknown> {
    if (!this.embedPromise) {
      this.embedPromise = this.tf().then((tf) =>
        tf.pipeline('feature-extraction', this.config.modelId, { dtype: this.config.dtype }),
      )
    }
    return this.embedPromise
  }

  /** 蒸馏一个回合：生成结构化 engram 写入存储。 */
  async distillTurn(store: EngramStore, graph: CausalGraph): Promise<Engram[]> {
    const generator = await this.generator().catch(() => null)
    if (!generator) {
      // 模型不可用：降级为不蒸馏（插件仍可用，只是没有新记忆写入）
      this.ctx.logger?.warn?.('[engram-relay] model unavailable, skip distill')
      return []
    }
    // TODO(relay): 从会话读取最近回合文本，构造蒸馏 prompt，解析 JSON 结果。
    // 蒸馏 prompt 模板与 JSON 解析器在 model/prompts.ts 中实现。
    return []
  }

  /** 打分：query 与候选 engram 的相关性分数（因果传播种子）。 */
  async score(query: string, candidates: Engram[]): Promise<Map<string, number>> {
    const embedder = await this.embedder().catch(() => null)
    if (!embedder || candidates.length === 0) {
      return new Map()
    }
    // TODO(relay): 用 feature-extraction 对 query 与各 engram 文本取嵌入，
    // 余弦相似度作为种子分数（与因果传播叠加）。
    return new Map()
  }

  async describe(): Promise<Record<string, unknown>> {
    return {
      modelId: this.config.modelId,
      dtype: this.config.dtype,
      loadError: this.loadError,
      loaded: this.pipelinePromise !== null,
    }
  }
}
