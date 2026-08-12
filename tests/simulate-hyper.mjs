/**
 * 超高倍速综合仿真：验证 engram 新逻辑全链路（无真实 LLM/bge，毫秒级跑完）。
 *
 * 覆盖：
 *  1. 语义阈值 0.42：同主题（相似 0.75）通过、无关（0.05）被挡；
 *  2. embedder 不可用 → 宁缺毋滥（不注入）；
 *  3. 因果传播：查询命中链首 → 因果邻居被带回；
 *  4. 注入预算 200：renderInjection 输出封顶（入口级）；
 *  5. 蒸馏自动因果：causes 写入 → CausalGraph 重建边 → 边数增长。
 *
 * 运行：node tests/simulate-hyper.mjs
 */

import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { NgramHashAddressing } from '../lib/engram/hash.js'
import { EngramStore } from '../lib/engram/store.js'
import { CausalGraph } from '../lib/engram/causal.js'
import { EngramWakeEngine } from '../lib/engram/wake.js'

/** 主题相似度矩阵（模拟 bge 余弦）：同主题 0.75、无关 0.05。 */
const THEMES = ['缓存', '浏览器', '路由', '记忆', '构建']
const THEME_SCORE = { same: 0.75, cross: 0.15, none: 0.05 }

function themeOf(text) {
  for (const t of THEMES) if (text.includes(t)) return t
  return null
}

/** fake embedder：按主题词匹配给余弦分数。 */
function fakeEmbedder(query, candidates) {
  const qt = themeOf(query)
  const map = new Map()
  for (const c of candidates) {
    const ct = themeOf(c.title + c.summary)
    if (qt && ct === qt) map.set(c.id, THEME_SCORE.same)
    else if (qt && ct) map.set(c.id, THEME_SCORE.cross)
    else map.set(c.id, THEME_SCORE.none)
  }
  return Promise.resolve(map)
}

async function main() {
  console.log('=== engram 超高倍速综合仿真（阈值 0.42 + 因果 + 入口注入） ===')
  const dir = mkdtempSync(join(tmpdir(), 'engram-hyper-'))
  try {
    const hasher = new NgramHashAddressing({ seed: 0 })
    const store = new EngramStore(dir, hasher)

    // 1. 5 主题 × 4 条 = 20 条，主题内因果链
    const ids = []
    for (const t of THEMES) {
      for (let i = 1; i <= 4; i++) {
        const node = store.add({
          kind: i === 1 ? 'decision' : 'event',
          layer: 'project',
          title: `${t}${'一二三四'[i - 1]}决策`,
          summary: `${t}主题第${i}步记录`,
          content: `${t}第${i}步细节`,
          links: [],
          sessionId: 'sim',
          turn: i,
          causes: i > 1 ? [ids[ids.length - 1]] : [],
          effects: [],
          importance: 0.6,
        })
        ids.push(node.id)
      }
    }
    const graph = new CausalGraph(store)
    const baselineEdges = graph.edgeCount()
    console.log(`[1] 节点 ${store.count()}，因果边（重建后）${baselineEdges}（应 15）`)

    const CONFIG = { maxWakePerTurn: 3, injectBudgetTokens: 200 }

    // 2. 带 embedder：阈值 0.42
    const wake = new EngramWakeEngine(store, graph, hasher, CONFIG, { embedder: fakeEmbedder })

    let relatedHits = 0
    let causalNeighborHits = 0
    const relatedQueries = THEMES.map((t) => `查询${t}主题第一步的决策方案`)
    for (const q of relatedQueries) {
      const hit = await wake.query(q, 3, {})
      const titles = hit.engrams.map((e) => e.title)
      const theme = themeOf(q)
      if (titles.some((x) => x.includes(theme))) relatedHits++
      if (titles.length > 1) causalNeighborHits++
    }
    console.log(`[2] 相关查询 ${relatedQueries.length}：主题命中 ${relatedHits}（应 ${relatedQueries.length}），因果带回邻居 ${causalNeighborHits}`)

    // 3. 无关查询：应全空
    const junkQueries = ['红烧肉的做法', '今天天气怎么样', '晚上吃什么']
    let junkInjected = 0
    for (const q of junkQueries) {
      const hit = await wake.query(q, 3, {})
      if (hit.engrams.length > 0) junkInjected++
    }
    console.log(`[3] 无关查询 ${junkQueries.length}：误注入 ${junkInjected}（应 0）`)

    // 4. 注入渲染（预算 200，入口级）
    const sampleHit = await wake.query(relatedQueries[0], 3, {})
    const injection = wake.renderInjection(200)
    console.log(`[4] 注入渲染：${injection.length} 字符 ≈ ${Math.round(injection.length / 1.8)} token（预算 200）`)

    // 5. 无 embedder：宁缺毋滥
    const bare = new EngramWakeEngine(store, graph, hasher, CONFIG)
    const bareHit = await bare.query(relatedQueries[0], 3, {})
    console.log(`[5] 无 embedder：reason=${bareHit.reason}（应 no-embedder），注入 ${bareHit.injectedTokens}（应 0）`)

    // 6. 蒸馏自动因果模拟
    const cause = store.all().find((n) => n.title.includes('记忆四'))
    const fresh = store.add({
      kind: 'decision', layer: 'project',
      title: '记忆五蒸馏补强', summary: '蒸馏自动因果新节点',
      content: '由记忆四决策导致的新进展', links: [], sessionId: 'sim', turn: 5,
      causes: [cause.id], effects: [], importance: 0.6,
    })
    graph.addEdge(cause.id, fresh.id, 'causes', 1)
    store.update(cause.id, { effects: [...(store.get(cause.id)?.effects ?? []), fresh.id] })
    const rebuilt = new CausalGraph(store)
    console.log(`[6] 蒸馏自动因果：边 ${baselineEdges} → ${rebuilt.edgeCount()}（应 +1）`)

    console.log('\n=== 结论 ===')
    console.log(`阈值 0.42: 相关召回 ${relatedHits}/${relatedQueries.length}，无关误召 ${junkInjected}`)
    console.log(`因果传播: ${causalNeighborHits}/${relatedQueries.length} 查询带回邻居`)
    console.log(`无 embedder: ${bareHit.reason}`)
    console.log(`入口注入: ${injection.length} 字符`)
    console.log(`蒸馏因果: 自动建边 ${baselineEdges}→${rebuilt.edgeCount()}`)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
}

await main()
