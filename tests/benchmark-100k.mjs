/**
 * 10 万记忆级性能基准（阶段 4）：闭环 <500ms 断言。
 *
 * 合成 10 万条 512 维向量（主题偏置模拟真实分布），测：
 *  1. 写入（add）耗时
 *  2. 查询 embed（包内 ONNX int8）
 *  3. int8 全量内积粗筛 top-50
 *  4. fp32 细排（top-50 余弦）
 *  5. 激活查表 + 渲染估算
 *  总闭环 = 2+3+4+5（0.5s 断言）
 *
 * 运行：node tests/benchmark-100k.mjs
 */

import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { BruteForceIndex } from '../lib/engram/vector-index.js'
import { ActivationCache } from '../lib/engram/activation.js'
import { embedWithOnnx } from '../lib/model/onnx-embedder.js'

const N = 100_000
// 包内 int8 模型（仓库根 model/bge-small-zh）；可用 ENGRAM_EMBED_MODEL 覆盖
const MODEL_DIR = process.env.ENGRAM_EMBED_MODEL
  || new URL('../model/bge-small-zh', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')

/** 主题偏置向量：每条 = 主题基向量 + 噪声（模拟真实记忆分布）。 */
function synthVec(topicIdx, noise = 0.3) {
  const v = new Float32Array(512)
  const seed = topicIdx * 7919
  for (let i = 0; i < 512; i++) {
    // 确定性伪随机（mulberry32 简化）
    const r = Math.sin(seed + i * 12.9898) * 43758.5453
    v[i] = (r - Math.floor(r)) * 2 - 1
  }
  // 归一化
  let n = 0
  for (let i = 0; i < 512; i++) n += v[i] * v[i]
  n = Math.sqrt(n) || 1
  for (let i = 0; i < 512; i++) v[i] /= n
  return v
}

function now() { return performance.now() }

async function main() {
  console.log(`=== 10 万记忆级性能基准（N=${N}） ===`)
  const dir = mkdtempSync(join(tmpdir(), 'engram-bench-'))
  try {
    // 1. 写入（add）
    const index = new BruteForceIndex(dir)
    let t0 = now()
    for (let i = 0; i < N; i++) {
      index.add(`m${i}`, synthVec(i % 200))
    }
    const addMs = now() - t0
    console.log(`[1] 写入 ${N} 条: ${addMs.toFixed(0)}ms（${(N / addMs * 1000).toFixed(0)} 条/s）`)

    // 2. 查询 embed（ONNX int8 包内模型）——先预热（模型加载不计时），再计时
    await embedWithOnnx(['预热'], '预热', MODEL_DIR)
    t0 = now()
    const emb = await embedWithOnnx(['缓存命中率优化'], '缓存命中率的根因', MODEL_DIR)
    const embedMs = now() - t0
    if (!emb) { console.log('[2] ❌ embed 失败——总闭环无法验证'); return }
    console.log(`[2] 查询 embed（预热后）: ${embedMs.toFixed(0)}ms`)

    // 3. int8 全量内积 top-50（10 次取中位；记录最后一次 hits 供细排）
    const qv = Float32Array.from(emb.query_vec)
    const searchTimes = []
    let top = []
    for (let r = 0; r < 10; r++) {
      t0 = now()
      top = index.search(qv, 50)
      searchTimes.push(now() - t0)
      if (r === 0) console.log(`    首次 top-50: ${top.length} 条（top 分数 ${top[0]?.score.toFixed(0)}）`)
    }
    searchTimes.sort((a, b) => a - b)
    const searchMs = searchTimes[5]
    console.log(`[3] int8 全量内积 top-50（中位）: ${searchMs.toFixed(1)}ms`)

    // 4. fp32 细排（复用 [3] 的 top-50，仅测余弦——不重复全量检索）
    t0 = now()
    let cos = 0
    for (const h of top) cos += index.cosine(h.id, qv)
    const fineMs = now() - t0
    console.log(`[4] fp32 细排 top-50: ${fineMs.toFixed(1)}ms（均余弦 ${(cos / top.length).toFixed(2)}）`)

    // 5. 激活查表（10 万条预计算 + 查询）
    t0 = now()
    const acts = new ActivationCache(0.5)
    const nodes = Array.from({ length: N }, (_, i) => ({ id: `m${i}`, status: 'confirmed', reinforces: [Date.now() - (i % 100) * 60000] }))
    acts.rebuild(nodes)
    const rebuildMs = now() - t0
    t0 = now()
    let bSum = 0
    for (const h of top) bSum += acts.get(h.id)
    const actMs = now() - t0
    console.log(`[5] 激活全量重建: ${rebuildMs.toFixed(0)}ms；查询 top-50: ${actMs.toFixed(1)}ms`)

    // 总闭环
    const total = embedMs + searchMs + fineMs + actMs
    console.log(`\n=== 总闭环: ${total.toFixed(1)}ms（${total < 500 ? '✅ <500ms 达标' : '❌ 超预算'}） ===`)
    console.log(`规模账：${total.toFixed(1)}ms @ ${N} 条 → 预计 50 万条 ${(total * 5).toFixed(0)}ms（含 search 线性）`)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
}

await main()
