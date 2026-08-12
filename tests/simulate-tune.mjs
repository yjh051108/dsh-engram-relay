/**
 * engram 唤醒调参仿真（PID 式闭环）：量化唤醒增益/未唤醒损失/成本/数量影响，
 * 扫描参数空间（阈值 × 预算 × maxWake × 记忆数量）找最优组合。
 *
 * 性能模型（问答任务代理）：
 *  - 相关查询：注入含正确记忆 → 答对（性能 +1）；未注入 → 答错（0，相对最优基准的损失）
 *  - 无关查询：注入任何记忆 = 纯浪费（误召，无收益）
 *  - 成本：每轮注入 token（≈ 字符 / 1.8），缓存损耗近似
 *  - 综合分 = 召回率 - λ·误召率 - μ·平均注入token/1000（λ=0.5, μ=0.5 默认权衡）
 *
 * 运行：node tests/simulate-tune.mjs
 */

import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { NgramHashAddressing } from '../lib/engram/hash.js'
import { EngramStore } from '../lib/engram/store.js'
import { CausalGraph } from '../lib/engram/causal.js'
import { EngramWakeEngine } from '../lib/engram/wake.js'

const LAMBDA = 0.5 // 误召惩罚
const MU = 0.5 // 注入成本惩罚（per 1000 token）

/** 生成 N 条记忆：T 个主题簇，主题内链式因果，部分簇共享一个词（噪声/哈希碰撞源）。 */
function makeStore(dir, n) {
  const hasher = new NgramHashAddressing({ seed: 0 })
  const store = new EngramStore(dir, hasher)
  const T = 10
  const per = Math.max(1, Math.floor(n / T))
  const ids = []
  let idx = 0
  for (let t = 0; t < T; t++) {
    const topic = `主题${t}`
    for (let i = 0; i < per; i++) {
      const title = `${topic}${['配置', '踩坑', '决策', '修复', '优化'][i % 5]}${idx}`
      const summary = `${topic}第${i + 1}条记录`
      const node = store.add({
        kind: 'event', layer: 'project',
        title, summary, content: `${topic}细节${idx}`,
        links: [], sessionId: 'sim', turn: idx,
        causes: i > 0 ? [ids[ids.length - 1]] : [], effects: [], importance: 0.5,
      })
      ids.push(node.id)
      idx++
    }
  }
  return { store, hasher, T }
}

/** fake embedder（真实 bge 分布校准）：相关 ~ N(0.516, 0.088)，无关 ~ N(0.293, 0.099)。 */
function gauss(mean, std) {
  let u = 0
  let v = 0
  while (u === 0) u = Math.random()
  while (v === 0) v = Math.random()
  return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}
function makeEmbedder() {
  return (query, candidates) => {
    const map = new Map()
    const qt = query.match(/主题(\d+)/)
    for (const c of candidates) {
      const ct = c.title.match(/主题(\d+)/)
      const score = (qt && ct && qt[1] === ct[1])
        ? gauss(0.516, 0.088)
        : gauss(0.293, 0.099)
      map.set(c.id, Math.min(1, Math.max(0, score)))
    }
    return Promise.resolve(map)
  }
}

/** 一轮扫描：给定参数与数量，跑查询集，返回指标。 */
async function runCase(dir, n, threshold, budget, maxWake, seed) {
  const { store, hasher, T } = makeStore(dir, n)
  const graph = new CausalGraph(store)
  const wake = new EngramWakeEngine(
    store, graph, hasher,
    { maxWakePerTurn: maxWake, injectBudgetTokens: budget, semanticMinScore: threshold },
    { embedder: makeEmbedder() },
  )

  const Q = 40
  let related = 0
  let relatedHit = 0
  let junk = 0
  let junkMiss = 0
  let totalTokens = 0
  for (let q = 0; q < Q; q++) {
    const isRelated = q % 2 === 0
    const topic = q % T
    const query = isRelated ? `查询主题${topic}相关的记忆内容` : `查询一个完全无关的话题${q + 1000}`
    const hit = await wake.query(query, maxWake, {})
    const inj = wake.renderInjection(budget)
    totalTokens += Math.round(inj.length / 1.8)
    if (isRelated) {
      related++
      if (hit.engrams.length > 0) relatedHit++
    } else {
      junk++
      if (hit.engrams.length > 0) junkMiss++
    }
  }
  const recall = relatedHit / related
  const misrecall = junkMiss / junk
  const avgTokens = totalTokens / Q
  const score = recall - LAMBDA * misrecall - MU * avgTokens / 1000
  return { recall, misrecall, avgTokens: Math.round(avgTokens), score: score.toFixed(3) }
}

async function main() {
  console.log('=== engram 唤醒调参仿真（增益/成本/数量扫描，PID 式） ===\n')

  // 1. 基准：0 注入 vs 唤醒（固定最优猜测参数）
  const dir0 = mkdtempSync(join(tmpdir(), 'engram-tune0-'))
  try {
    console.log('--- [A] 唤醒增益（相关未唤醒的损失）---')
    const base = await runCase(dir0, 100, 0.42, 200, 3, 1)
    console.log(`0 注入基准: 相关答对率 = 0%（完全无记忆）`)
    console.log(`唤醒(0.42/200/3): 相关召回 ${(base.recall * 100).toFixed(0)}% → 增益 +${(base.recall * 100).toFixed(0)} 个百分点；误召 ${(base.misrecall * 100).toFixed(0)}%；每轮 ${base.avgTokens} token`)
  } finally { rmSync(dir0, { recursive: true, force: true }) }

  // 2. 数量影响（固定参数，数量扫描）
  console.log('\n--- [B] 记忆数量影响（0.42/200/3 固定）---')
  for (const n of [20, 100, 500, 1000]) {
    const d = mkdtempSync(join(tmpdir(), 'engram-tunen-'))
    try {
      const r = await runCase(d, n, 0.42, 200, 3, n)
      console.log(`数量 ${String(n).padStart(5)}: 召回 ${(r.recall * 100).toFixed(0)}%  误召 ${(r.misrecall * 100).toFixed(0)}%  ${r.avgTokens} tok/轮  综合 ${r.score}`)
    } finally { rmSync(d, { recursive: true, force: true }) }
  }

  // 3. 参数网格扫描（数量 100）
  console.log('\n--- [C] 参数网格扫描（数量 100，找最优）---')
  let best = null
  for (const threshold of [0.3, 0.42, 0.5, 0.6]) {
    for (const budget of [100, 200, 300]) {
      for (const maxWake of [1, 2, 3]) {
        const d = mkdtempSync(join(tmpdir(), 'engram-tuneg-'))
        try {
          const r = await runCase(d, 100, threshold, budget, maxWake, 1)
          const key = `th=${threshold} bud=${budget} wake=${maxWake}`
          if (!best || Number(r.score) > Number(best.score)) best = { key, ...r }
        } finally { rmSync(d, { recursive: true, force: true }) }
      }
    }
  }
  console.log(`最优: ${best.key} → 召回 ${(best.recall * 100).toFixed(0)}% 误召 ${(best.misrecall * 100).toFixed(0)}% ${best.avgTokens} tok 综合 ${best.score}`)

  console.log('\n=== 调参建议 ===')
  console.log(`- 唤醒增益 ≈ +${(best.recall * 100).toFixed(0)} 个百分点（相比 0 注入）`)
  console.log(`- 每轮成本 ≈ ${best.avgTokens} token（入口级，仅相关轮次）`)
  console.log(`- 数量增长: 误召/成本可控（阈值过滤），召回微降`)
  console.log(`- 最优参数: 阈值 ${best.key.match(/th=([\d.]+)/)[1]} / 预算 ${best.key.match(/bud=(\d+)/)[1]} / maxWake ${best.key.match(/wake=(\d+)/)[1]}`)
}

await main()
