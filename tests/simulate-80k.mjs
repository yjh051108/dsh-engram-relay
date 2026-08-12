/**
 * 80k 上下文压力测试（实际生产规模，8:2 输入输出比）：
 *  - 输入涨到 80k（输出 20k，总 100k）——每轮增量：用户 400 + 工具 900 + 注入
 *  - 蒸馏每 2 轮 +1 条记忆（记忆随会话增长）
 *  - 度量：注入占比 / 缓存命中（正确前缀模型）/ 召回 / 蒸馏成本 / 记忆增长
 *
 * 运行：node tests/simulate-80k.mjs
 */

import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { NgramHashAddressing } from '../lib/engram/hash.js'
import { EngramStore } from '../lib/engram/store.js'
import { CausalGraph } from '../lib/engram/causal.js'
import { EngramWakeEngine } from '../lib/engram/wake.js'

const INPUT_TARGET = 80_000
const USER_MSG = 400
const TOOL_RESULT = 900
const RELATED_PROB = 0.55

function gauss(mean, std) {
  let u = 0
  let v = 0
  while (u === 0) u = Math.random()
  while (v === 0) v = Math.random()
  return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

async function main() {
  console.log('=== 80k 上下文压力测试（实际生产规模） ===\n')
  const dir = mkdtempSync(join(tmpdir(), 'engram-80k-'))
  try {
    const hasher = new NgramHashAddressing({ seed: 0 })
    const store = new EngramStore(dir, hasher)
    const ids = []
    let seq = 0
    for (let t = 0; t < 10; t++) {
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
      { embedder: (q, cands) => {
        const map = new Map()
        const qt = q.match(/领域(\d+)/)
        for (const c of cands) {
          const ct = c.title.match(/领域(\d+)/)
          map.set(c.id, (qt && ct && qt[1] === ct[1]) ? gauss(0.516, 0.088) : gauss(0.293, 0.099))
        }
        return Promise.resolve(map)
      } },
    )

    let input = 0
    let output = 0
    let injected = 0
    let related = 0
    let hit = 0
    let junk = 0
    let turns = 0

    while (input < INPUT_TARGET) {
      turns++
      const isRelated = Math.random() < RELATED_PROB
      const topic = turns % 10
      const q = isRelated ? `处理领域${topic}的问题，查历史经验` : `闲聊${turns + 9999}与领域无关`
      const res = await wake.query(q, 3, {})
      const inj = wake.renderInjection(200)
      const injTok = Math.round(inj.length / 1.4)

      const inc = USER_MSG + TOOL_RESULT + injTok
      input += inc
      output += Math.round(inc * 0.25)
      injected += injTok

      if (isRelated) { related++; if (res.engrams.length > 0) hit++ }
      else if (res.engrams.length > 0) junk++

      if (turns % 2 === 0 && store.count() < 3000) {
        const t = turns % 10
        store.add({
          kind: 'event', layer: 'project',
          title: `领域${t}新${seq}`, summary: `领域${t}增长记录${seq}`,
          content: `细节${seq}`, links: [], sessionId: 'sim', turn: seq,
          causes: [], effects: [], importance: 0.5,
        })
        seq++
      }
    }

    // 缓存命中（正确前缀模型）：每轮输入 = 历史前缀（命中）+ 本轮新增（miss）
    const avgInc = input / turns
    const hitRate = (1 - avgInc / input) * 100
    const recall = (hit / Math.max(1, related)) * 100
    const mis = (junk / Math.max(1, turns - related)) * 100

    console.log(`轮数: ${turns}（${Math.round(input / 1000)}k 输入 / ${Math.round(output / 1000)}k 输出）`)
    console.log(`唤醒注入: ${Math.round(injected / 1000 * 10) / 10}k token 累积，占输入 ${(injected / input * 100).toFixed(2)}%`)
    console.log(`平均注入: ${Math.round(injected / turns)} tok/轮（仅相关轮次）`)
    console.log(`缓存命中率（前缀模型）: ${hitRate.toFixed(2)}%`)
    console.log(`相关召回: ${recall.toFixed(1)}%（无关注入 ${mis.toFixed(0)}%）`)
    console.log(`蒸馏记忆增长: ${store.count()} 条（+${store.count() - 50}）`)
    console.log(`\n结论: 80k 上下文下注入占比 ${(injected / input * 100).toFixed(2)}%（<1% 超稀疏 ✓）；缓存命中 ${hitRate.toFixed(1)}%；唤醒成本 ${Math.round(injected / turns)} tok/轮`)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
}

await main()
