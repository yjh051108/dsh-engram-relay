/**
 * TS 内嵌 bge ONNX embedder（免 Python 语义精排）。
 *
 * 降级链：TS ONNX（本模块，优先）→ Python 服务（回退）→ null（重要度兜底）。
 * 模型目录：config.embedModel（含 model.onnx + tokenizer.json + config.json）。
 * bge 的嵌入 = CLS 向量 + L2 归一化（pooling: 'cls' + normalize）。
 */

export interface OnnxEmbedResult {
  query_vec: number[]
  vectors: number[][]
}

let pipelinePromise: Promise<unknown> | null = null

export async function embedWithOnnx(texts: string[], query: string, modelDir: string): Promise<OnnxEmbedResult | null> {
  if (!modelDir || modelDir.trim() === '') return null
  try {
    const { pipeline, env } = await import('@huggingface/transformers')
    env.allowRemoteModels = false
    env.useBrowserCache = false
    if (!pipelinePromise) {
      pipelinePromise = pipeline('feature-extraction', modelDir, { dtype: 'fp32' })
    }
    const extractor: any = await pipelinePromise
    const all = await extractor([query, ...texts], { pooling: 'cls', normalize: true })
    const toVec = (t: unknown): number[] => {
      const data = (t as { data: ArrayLike<number> }).data
      return data ? Array.from(data) : []
    }
    const queryVec = toVec(all[0])
    if (queryVec.length === 0) return null
    const vectors = all.slice(1).map(toVec)
    if (vectors.some((v: number[]) => v.length !== queryVec.length)) return null
    return { query_vec: queryVec, vectors }
  } catch {
    return null
  }
}
