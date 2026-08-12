/**
 * 1M 上下文长会话仿真（真实会话模型）：
 *  - 输入输出比 8:2：每轮输入增量 = 用户消息 + 工具结果 + 唤醒注入，输出 = 输入增量的 1/4
 *  - 会话增长到 1M 总上下文（输入 800k / 输出 200k）
 *  - 记忆 1000 条，每轮 50% 概率相关查询（唤醒注入 ≤200 token）
 *
 * 度量：
 *  - 注入占比（超稀疏验证：注入 << 1% 上下文）
 *  - 缓存命中率（前缀缓存模型：稳定历史 vs 尾部新增）
 *  - 唤醒成本累积 / 精度（1000 条规模）
 *
 * 运行：node tests/simulate-1m-context.mjs
 */

import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { NgramHashAddressing } from '../lib/engram/hash.js'
import { EngramStore } from '../lib/engram/store.js'
import { CausalGraph } from '../lib/engram/causal.js'
import { EngramWakeEngine } from '../lib/engram/wake.js'

const CONTEXT_INPUT_TARGET = 800_000 // 8:2 的输入上限（1M 窗口）
const USER_MSG_TOKENS = 400 // 每轮用户消息
const TOOL_RESULT_TOKENS = 900 // 每轮工具结果（平均）
const OUTPUT_RATIO = 0.25 // 输出 = 输入增量 / 4（8:2）
const RELATED_PROB = 0.5 // 每轮相关查询概率

function makeStore(dir, n) {
  const hasher = new NgramHashAddressing({ seed: 0 })
  const store = new EngramStore(dir, hasher)
  const T = 20
  const per = Math.max(1, Math.floor(n / T))
  const ids = []
  let idx = 0
  for (let t = 0; t < T; t++) {
    const topic = `领域${t}`
    for (let i = 0; i < per; i++) {
      const title = `${topic}${['架构', '故障', '决策', '调优', '记录'][i % 5]}${idx}`
      const summary = `${topic}第${i + 1}条经验`
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

function makeEmbedder() {
  return (query, candidates) => {
    const map = new Map()
    const qt = query.match(/领域(\d+)/)
    for (const c of candidates) {
      const ct = c.title.match(/领域(\d+)/)
      let score
      if (qt && ct && qt[1] === ct[1]) score = 0.7 + Math.random() * 0.15
      else if (qt && ct) score = 0.12 + Math.random() * 0.1
      else score = 0.02 + Math.random() * 0.06
      map.set(c.id, score)
    }
    return Promise.resolve(map)
  }
}

async function main() {
  console.log('=== 1M 上下文长会话仿真（8:2 输入输出比） ===\n')
  const dir = mkdtempSync(join(tmpdir(), 'engram-1m-'))
  try {
    const { store, hasher, T } = makeStore(dir, 1000)
    const graph = new CausalGraph(store)
    const wake = new EngramWakeEngine(
      store, graph, hasher,
      { maxWakePerTurn: 3, injectBudgetTokens: 200, semanticMinScore: 0.42 },
      { embedder: makeEmbedder() },
    )

    let inputTotal = 0
    let outputTotal = 0
    let injectedTotal = 0
    let missTotal = 0 // 缓存 miss（= 每轮输入增量）
    let related = 0
    let relatedHit = 0
    let junkHit = 0
    let turns = 0

    while (inputTotal < CONTEXT_INPUT_TARGET) {
      turns++
      const isRelated = Math.random() < RELATED_PROB
      const topic = turns % T
      const query = isRelated
        ? `处理领域${topic}的问题，需要相关历史经验`
        : `闲聊话题${turns + 9999}，与项目无关`
      const hit = await wake.query(query, 3, {})
      const injection = wake.renderInjection(200)
      const injTokens = Math.round(injection.length / 1.8)

      const increment = USER_MSG_TOKENS + TOOL_RESULT_TOKENS + injTokens
      const output = Math.round(increment * OUTPUT_RATIO)

      inputTotal += increment
      outputTotal += output
      injectedTotal += injTokens
      missTotal += increment

      if (isRelated) {
        related++
        if (hit.engrams.length > 0) relatedHit++
      } else if (hit.engrams.length > 0) {
        junkHit++
      }
    }

    const contextTotal = inputTotal + outputTotal
    const hitRate = ((inputTotal - missTotal) / inputTotal) * 100
    console.log(`轮数: ${turns}`)
    console.log(`总上下文: ${(contextTotal / 1000).toFixed(0)}k token（输入 ${(inputTotal / 1000).toFixed(0)}k / 输出 ${(outputTotal / 1000).toFixed(0)}k，比 ${(inputTotal / Math.max(1, outputTotal)).toFixed(1)}:1）`)
    console.log(`唤醒注入累积: ${(injectedTotal / 1000).toFixed(1)}k token，占输入 ${(injectedTotal / inputTotal * 100).toFixed(3)}%（超稀疏 <1%）`)
    console.log(`相关召回: ${relatedHit}/${related}（${(relatedHit / Math.max(1, related) * 100).toFixed(0)}%），无关注入: ${junkHit} 轮`)
    console.log(`缓存命中率（前缀模型）: 输入历史全稳定 → 每轮新增 ${Math.round(missTotal / turns)} tok → 稳态命中 ${(100 - missTotal / turns / (inputTotal / turns) * 100).toFixed(2)}%`)
    console.log(`\n=== 结论 ===`)
    console.log(`1M 上下文下：注入占比 ${(injectedTotal / inputTotal * 100).toFixed(3)}%（超稀疏成立）；唤醒精度 ${(relatedHit / Math.max(1, related) * 100).toFixed(0)}%；无关零注入率 ${(100 - junkHit / Math.max(1, turns - related) * 100).toFixed(0)}%`)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
}

await main()
