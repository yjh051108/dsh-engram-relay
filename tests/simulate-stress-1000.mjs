/**
 * 1000 轮压力测试（逻辑维度）：模拟长期运行——
 * 每轮唤醒（相关/无关）+ 蒸馏记忆增长（每 2 轮 +1 条，带因果），
 * 观察 1000 轮后：记忆规模、召回退化、误召、注入成本、候选膨胀。
 *
 * 运行：node tests/simulate-stress-1000.mjs
 */

import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { NgramHashAddressing } from '../lib/engram/hash.js'
import { EngramStore } from '../lib/engram/store.js'
import { CausalGraph } from '../lib/engram/causal.js'
import { EngramWakeEngine } from '../lib/engram/wake.js'

const ROUNDS = 1000
const TOPICS = 20
const RELATED_PROB = 0.55

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
    const qt = query.match(/领域(\d+)/)
    for (const c of candidates) {
      const ct = c.title.match(/领域(\d+)/)
      const score = (qt && ct && qt[1] === ct[1])
        ? gauss(0.516, 0.088)
        : gauss(0.293, 0.099)
      map.set(c.id, Math.min(1, Math.max(0, score)))
    }
    return Promise.resolve(map)
  }
}

async function main() {
  console.log('=== 1000 轮压力测试（唤醒 + 蒸馏记忆增长） ===\n')
  const dir = mkdtempSync(join(tmpdir(), 'engram-stress-'))
  try {
    const hasher = new NgramHashAddressing({ seed: 0 })
    const store = new EngramStore(dir, hasher)
    // 初始 100 条（20 领域 × 5）
    const ids = []
    let seq = 0
    for (let t = 0; t < TOPICS; t++) {
      for (let i = 0; i < 5; i++) {
        const e = store.add({
          kind: 'event', layer: 'project',
          title: `领域${t}经验${i}${seq}`, summary: `领域${t}记录${seq}`,
          content: `细节${seq}`, links: [], sessionId: 'sim', turn: seq,
          causes: i > 0 ? [ids[ids.length - 1]] : [], effects: [], importance: 0.5,
        })
        ids.push(e.id)
        seq++
      }
    }
    const graph = new CausalGraph(store)
    const wake = new EngramWakeEngine(
      store, graph, hasher,
      { maxWakePerTurn: 3, injectBudgetTokens: 200, semanticMinScore: 0.42 },
      { embedder: makeEmbedder() },
    )

    let related = 0
    let hit = 0
    let junkInject = 0
    let injTokens = 0
    let candSum = 0
    let candN = 0

    for (let round = 1; round <= ROUNDS; round++) {
      const isRelated = Math.random() < RELATED_PROB
      const topic = round % TOPICS
      const q = isRelated
        ? `处理领域${topic}的问题，查历史经验`
        : `闲聊${round + 9999}与领域无关`

      const t0 = Date.now()
      const hitRes = await wake.query(q, 3, {})
      const ms = Date.now() - t0
      const inj = wake.renderInjection(200)
      injTokens += Math.round(inj.length / 1.4)

      // 统计候选数（从 store.lookup 侧量）
      const cand = store.lookup(q, 256).length
      candSum += cand
      candN++

      if (isRelated) {
        related++
        if (hitRes.engrams.length > 0) hit++
      } else if (hitRes.engrams.length > 0) {
        junkInject++
      }

      // 蒸馏：每 2 轮 +1 条记忆（同领域链式因果）
      if (round % 2 === 0 && store.count() < 5000) {
        const t = round % TOPICS
        const prev = store.all().find((n) => n.title.startsWith(`领域${t}`) && n.turn <= seq)
        store.add({
          kind: 'event', layer: 'project',
          title: `领域${t}经验${seq % 5}${seq}`, summary: `领域${t}新记录${seq}`,
          content: `细节${seq}`, links: [], sessionId: 'sim', turn: seq,
          causes: prev ? [prev.id] : [], effects: [], importance: 0.5,
        })
        seq++
      }

      if (round % 100 === 0) {
        const recall = (hit / Math.max(1, related) * 100).toFixed(0)
        const mis = (junkInject / Math.max(1, round - related) * 100).toFixed(0)
        console.log(`轮 ${String(round).padStart(4)}: 记忆 ${store.count()} 条 | 召回 ${recall}% | 误召 ${mis}% | 注入 ${Math.round(injTokens / round)} tok/轮 | 平均候选 ${Math.round(candSum / candN)}`)
      }
    }

    const recall = (hit / Math.max(1, related) * 100).toFixed(1)
    const mis = (junkInject / Math.max(1, ROUNDS - related) * 100).toFixed(1)
    console.log('\n=== 1000 轮结果 ===')
    console.log(`最终记忆: ${store.count()} 条（初始 100 + 蒸馏 +400）`)
    console.log(`相关召回率: ${recall}%`)
    console.log(`无关误召率: ${mis}%`)
    console.log(`平均注入: ${Math.round(injTokens / ROUNDS)} tok/轮`)
    console.log(`平均哈希候选: ${Math.round(candSum / candN)} 条`)
    console.log(`因果边: ${graph.edgeCount()} 条`)
    console.log(`\n结论: 记忆增长至 ${store.count()} 条后系统 ${Number(recall) > 60 ? '健康（召回可维持）' : '退化（需调参）'}`)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
}

await main()
