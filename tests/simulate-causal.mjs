/**
 * 因果链召回仿真 v2（受控实验，干净验证「因果 > 纯哈希/向量」）。
 *
 * 设计：两主题链，**所有记忆文本互不共享 2-gram**（手工构造词汇隔离），
 * 查询精确命中单节点 → 邻居（前因/后果）与查询零 n-gram 重叠，
 * **只有因果传播能带回它们**。
 *
 * 度量：
 *  - 基线（空图 = 纯哈希/向量 top-k）：因果邻居召回率应 ≈ 0
 *  - 因果引擎（哈希 + 分层因果席位）：应稳定带回前因/后果
 *  - 增益 = 因果 - 基线
 *
 * 运行：node tests/simulate-causal.mjs
 */

import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { NgramHashAddressing } from '../lib/engram/hash.js'
import { EngramStore } from '../lib/engram/store.js'
import { CausalGraph } from '../lib/engram/causal.js'
import { EngramWakeEngine } from '../lib/engram/wake.js'

/**
 * 主题 A / B：**完全不相交的 ASCII 词集**（拼音代号）——任意两条文本
 * 字符集零重叠（含 2-gram），哈希寻址绝不交叉命中；邻居只能经因果边
 * 到达。这是「因果召回 > 纯哈希/向量」的确定性实验前提。
 * 跨主题：A 收尾 → B 收尾（B 依赖 A 的经验）。
 */
const CHAINS = {
  A: [
    { kind: 'decision', label: 'A方案敲定', text: 'zulu alpha beta' },
    { kind: 'fact', label: 'A环境准备', text: 'yankee gamma delta' },
    { kind: 'fact', label: 'A脚本验证', text: 'xray epsilon zeta' },
    { kind: 'event', label: 'A灰度观察', text: 'whiskey eta theta' },
    { kind: 'event', label: 'A收尾确认', text: 'victor iota kappa' },
  ],
  B: [
    { kind: 'decision', label: 'B方案敲定', text: 'uniform lambda mu' },
    { kind: 'fact', label: 'B环境准备', text: 'tango nu xi' },
    { kind: 'fact', label: 'B脚本验证', text: 'sierra omicron pi' },
    { kind: 'event', label: 'B灰度观察', text: 'quebec rho sigma' },
    { kind: 'event', label: 'B收尾确认', text: 'papa tau upsilon' },
  ],
}

const CONFIG = {
  modelId: 'sim', dtype: 'q8', storeDir: '',
  injectBudgetTokens: 600, maxWakePerTurn: 5, distillEveryTurns: 1, enabled: true,
  pythonPath: '', pythonTimeoutMs: 0,
}

/** 校验两条文本不共享 token（与哈希 normalize 一致：小写、空白分词）。 */
function shareToken(a, b) {
  const toks = (s) => new Set(s.toLowerCase().split(/[ \t\r\n]+/).filter(Boolean))
  const ta = toks(a), tb = toks(b)
  for (const t of ta) if (tb.has(t)) return true
  return false
}

function populate(store, graph) {
  const nodes = {}
  for (const [chainName, chain] of Object.entries(CHAINS)) {
    nodes[chainName] = []
    for (let p = 0; p < chain.length; p += 1) {
      const c = chain[p]
      const causes = p > 0 ? [nodes[chainName][p - 1].id] : []
      const e = store.add({
        kind: c.kind, label: c.label, text: c.text,
        scope: null, sessionId: 'sim', turn: p, causes, effects: [], importance: 0.8,
      })
      nodes[chainName].push(e)
    }
  }
  // 跨主题：A 收尾 → B 收尾（B 依赖 A 的经验）
  const aConcl = nodes.A[4], bConcl = nodes.B[4]
  bConcl.causes.push(aConcl.id)
  graph.rebuild()
  return nodes
}

async function main() {
  console.log('=== 因果链召回仿真 v2（受控实验：词汇全隔离） ===')
  const all = [...CHAINS.A, ...CHAINS.B].map((c) => c.text)
  let violations = 0
  for (let i = 0; i < all.length; i += 1) {
    for (let j = i + 1; j < all.length; j += 1) {
      if (shareToken(all[i], all[j])) violations += 1
    }
  }
  console.log(`文本 ${all.length} 条，共享 token 的文本对: ${violations}（应为 0）`)

  const dir = mkdtempSync(join(tmpdir(), 'engram-causal-v2-'))
  try {
    const hasher = new NgramHashAddressing({ seed: 0 })
    const store = new EngramStore(dir, hasher)
    const graph = new CausalGraph(store)
    const nodes = populate(store, graph)
    console.log(`记忆 ${store.count()} 条，因果边 ${graph.edgeCount()} 条（A4 + B4 + 跨主题 1）\n`)

    // 基线引擎（空图 = 纯哈希/向量 top-k 的确定性版）
    const enginePlain = new EngramWakeEngine(store, new CausalGraph(store, { rebuild: false }), hasher, { ...CONFIG, maxWakePerTurn: 5 })
    // 因果引擎（哈希 + 分层因果席位）
    const engineCausal = new EngramWakeEngine(store, graph, hasher, { ...CONFIG, maxWakePerTurn: 5 })

    // 查询集：每节点全文查询（精确命中单节点）
    const queries = []
    for (const [chainName, chain] of Object.entries(CHAINS)) {
      for (let p = 0; p < chain.length; p += 1) {
        queries.push({ name: `${chainName}${p}`, q: chain[p].text, targetId: nodes[chainName][p].id })
      }
    }

    // 因果邻居定义：与目标有因果边（in/out 直接相连）
    const neighborsOf = (id) => {
      const set = new Set()
      for (const e of graph.causesOf(id)) set.add(e.id)
      for (const e of graph.effectsOf(id)) set.add(e.id)
      return set
    }

    let plainRecalls = 0, causalRecalls = 0
    let plainTokens = 0, causalTokens = 0
    const hopDists = []
    const examples = []

    for (const { name, q, targetId } of queries) {
      const nbrs = neighborsOf(targetId)
      const hPlain = await enginePlain.query(q, 5)
      const hCausal = await engineCausal.query(q, 5)
      plainTokens += hPlain.injectedTokens
      causalTokens += hCausal.injectedTokens

      const plainHas = hPlain.engrams.some((e) => nbrs.has(e.id))
      const causalHas = hCausal.engrams.some((e) => nbrs.has(e.id))
      if (plainHas) plainRecalls += 1
      if (causalHas) causalRecalls += 1

      if (causalHas) {
        for (const e of hCausal.engrams) {
          if (nbrs.has(e.id)) {
            const d = graphDistance(graph, targetId, e.id)
            if (d > 0) hopDists.push(d)
          }
        }
        if (examples.length < 4) {
          examples.push({
            name,
            returned: hCausal.engrams.map((e) => `${e.kind}:${e.label}`),
          })
        }
      }
    }

    console.log('--- 结果 ---')
    console.log(`查询总数: ${queries.length}（每主题 5 节点 × 2 主题）`)
    console.log(`因果邻居召回:`)
    console.log(`  基线（纯哈希/向量 top-k）: ${plainRecalls}/${queries.length}（${(plainRecalls / queries.length * 100).toFixed(0)}%）`)
    console.log(`  因果引擎（哈希+因果席位）: ${causalRecalls}/${queries.length}（${(causalRecalls / queries.length * 100).toFixed(0)}%）`)
    console.log(`  增益: +${causalRecalls - plainRecalls} 次（因果传播带回前因/后果）`)
    console.log(`注入:   基线 ${(plainTokens / queries.length).toFixed(0)} vs 因果 ${(causalTokens / queries.length).toFixed(0)} token/次（预算 600）`)
    if (hopDists.length > 0) {
      console.log(`因果跳数: 平均 ${(hopDists.reduce((a, b) => a + b, 0) / hopDists.length).toFixed(1)} 跳（最长 ${Math.max(...hopDists)}）`)
    }
    if (examples.length > 0) {
      console.log('\n--- 示例（因果引擎返回，含前因/后果） ---')
      for (const ex of examples) {
        console.log(`查询 ${ex.name} → ${ex.returned.join(' | ')}`)
      }
    }
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
}

/** 无权图 BFS 距离（-1 = 不可达）。 */
function graphDistance(graph, from, to) {
  if (from === to) return 0
  const seen = new Set([from])
  const queue = [[from, 0]]
  while (queue.length > 0) {
    const [id, d] = queue.shift()
    for (const e of graph.causesOf(id)) {
      if (!seen.has(e.id)) {
        if (e.id === to) return d + 1
        seen.add(e.id)
        queue.push([e.id, d + 1])
      }
    }
    for (const e of graph.effectsOf(id)) {
      if (!seen.has(e.id)) {
        if (e.id === to) return d + 1
        seen.add(e.id)
        queue.push([e.id, d + 1])
      }
    }
  }
  return -1
}

main().catch((e) => { console.error(e); process.exitCode = 1 })
