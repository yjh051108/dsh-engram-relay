/**
 * CausalGraph + CausalWakeEngine 单元测试：
 * 核心卖点 —— 因果图传播唤醒（比纯向量索引更强的召回）。
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { EngramStore } from '../lib/engram/store.js'
import { CausalGraph } from '../lib/engram/causal.js'
import { NgramHashAddressing } from '../lib/engram/hash.js'
import { EngramWakeEngine } from '../lib/engram/wake.js'

const CONFIG = {
  modelId: 'test',
  dtype: 'q8',
  storeDir: '',
  injectBudgetTokens: 600,
  maxWakePerTurn: 3,
  distillEveryTurns: 1,
  enabled: true,
}

function makeEnv() {
  const dir = mkdtempSync(join(tmpdir(), 'engram-causal-'))
  const hasher = new NgramHashAddressing({ seed: 0 })
  const store = new EngramStore(dir, hasher)
  const graph = new CausalGraph(store)
  return { dir, store, graph, hasher }
}

test('graph: propagate activates causes and effects', () => {
  const { dir, store, graph, hasher } = makeEnv()
  try {
    // 因果链：A（种子）→ B → C
    const a = store.add({ kind: 'decision', title: 'A 采用 engram', summary: 'A', sessionId: null, turn: 1, causes: [], effects: [], importance: 0.5, scope: null })
    const b = store.add({ kind: 'fact', title: 'B 实现细节', summary: 'B', sessionId: null, turn: 2, causes: [a.id], effects: [], importance: 0.5, scope: null })
    const c = store.add({ kind: 'event', title: 'C 上线', summary: 'C', sessionId: null, turn: 3, causes: [b.id], effects: [], importance: 0.5, scope: null })
    graph.rebuild()

    // 从 A 出发传播：B、C 都应被激活，且分数随跳数衰减
    const scores = graph.propagate(new Map([[a.id, 1.0]]))
    assert.ok((scores.get(b.id) ?? 0) > 0, 'B 应被因果传播激活')
    assert.ok((scores.get(c.id) ?? 0) > 0, 'C 应被因果传播激活（经 B）')
    assert.ok((scores.get(b.id) ?? 0) > (scores.get(c.id) ?? 0), '分数随跳数衰减')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('graph: propagate also walks reverse direction (effects -> causes)', () => {
  const { dir, store, graph, hasher } = makeEnv()
  try {
    const a = store.add({ kind: 'decision', title: 'A 根因', summary: 'A', sessionId: null, turn: 1, causes: [], effects: [], importance: 0.5, scope: null })
    const b = store.add({ kind: 'event', title: 'B 后果', summary: 'B', sessionId: null, turn: 2, causes: [a.id], effects: [], importance: 0.5, scope: null })
    graph.rebuild()

    // 从后果 B 出发：应能回溯到前因 A —— 向量索引做不到的因果召回
    const scores = graph.propagate(new Map([[b.id, 1.0]]))
    assert.ok((scores.get(a.id) ?? 0) > 0, 'A（前因）应被反向因果传播激活')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('wake: hash hit then sparse truncation', async () => {
  const { dir, store, graph, hasher } = makeEnv()
  try {
    // 同主题写入 10 条（共享槽位），其中一条与查询文本完全一致
    for (let i = 0; i < 10; i += 1) {
      store.add({ kind: 'fact', title: `事实 ${i}`, summary: i === 3 ? '项目部署端口是 8080' : `这是第 ${i} 条很长的记忆内容用来测试稀疏截断逻辑是否正确`, content: '', links: [], sessionId: null, turn: i, causes: [], effects: [], importance: 0.9 })
    }
    graph.rebuild()

    const engine = new EngramWakeEngine(store, graph, hasher, { ...CONFIG, maxWakePerTurn: 3 })
    // 无模型打分时降级为 importance 排序（种子 = importance）
    const hit = await engine.query('项目部署端口是 8080', 3)
    assert.ok(hit.engrams.length <= 3, '唤醒条数受 maxWakePerTurn 限制')
    assert.ok(hit.injectedTokens <= 600, '注入 token 受预算限制')
    assert.ok(hit.engrams.some((e) => e.title === '事实 3'), '与查询同文本的条目必须命中')
    assert.ok(hit.reason.startsWith('hash-wake') || hit.reason === 'below-threshold')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('wake: hash seed + causal propagation recalls dependents', async () => {
  const { dir, store, graph, hasher } = makeEnv()
  try {
    // A 与查询同文本 → 哈希命中 A；B 与 A 词汇零重叠、因果依赖 A → 因果传播召回 B
    // （importance 需 ≥ threshold/decay = 0.2 才能传播过阈值）
    const a = store.add({ kind: 'decision', title: 'A 种子', summary: '项目部署端口是 8080 且使用 PostgreSQL', sessionId: null, turn: 1, causes: [], effects: [], importance: 0.8, scope: null })
    const b = store.add({ kind: 'fact', title: 'B 依赖 A', summary: '防火墙白名单规则已按既定方案更新完成', sessionId: null, turn: 2, causes: [a.id], effects: [], importance: 0.8, scope: null })
    graph.rebuild()

    // 无模型打分（重要度降级）：A/B 同为 0.1
    const engine = new EngramWakeEngine(store, graph, hasher, CONFIG)
    const hit = await engine.query('项目部署端口是 8080 且使用 PostgreSQL', 3)
    const ids = hit.engrams.map((e) => e.id)
    assert.ok(ids.includes(a.id), '哈希种子命中 A')
    assert.ok(ids.includes(b.id), '因果后继 B 被召回（因果传播/碰撞任一路径）')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('wake: empty store short-circuits', async () => {
  const { dir, store, graph, hasher } = makeEnv()
  try {
    const engine = new EngramWakeEngine(store, graph, hasher, CONFIG)
    const hit = await engine.query('anything', 3)
    assert.equal(hit.engrams.length, 0)
    assert.equal(hit.reason, 'no-hash-hit')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
