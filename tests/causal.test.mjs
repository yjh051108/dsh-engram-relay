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
import { CausalWakeEngine } from '../lib/engram/wake.js'

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
  const store = new EngramStore(dir)
  const graph = new CausalGraph(store)
  return { dir, store, graph }
}

test('graph: propagate activates causes and effects', () => {
  const { dir, store, graph } = makeEnv()
  try {
    // 因果链：A（种子）→ B → C
    const a = store.add({ kind: 'decision', label: 'A 采用 engram', text: 'A', sessionId: null, turn: 1, causes: [], effects: [], importance: 0.5 })
    const b = store.add({ kind: 'fact', label: 'B 实现细节', text: 'B', sessionId: null, turn: 2, causes: [a.id], effects: [], importance: 0.5 })
    const c = store.add({ kind: 'event', label: 'C 上线', text: 'C', sessionId: null, turn: 3, causes: [b.id], effects: [], importance: 0.5 })
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
  const { dir, store, graph } = makeEnv()
  try {
    const a = store.add({ kind: 'decision', label: 'A 根因', text: 'A', sessionId: null, turn: 1, causes: [], effects: [], importance: 0.5 })
    const b = store.add({ kind: 'event', label: 'B 后果', text: 'B', sessionId: null, turn: 2, causes: [a.id], effects: [], importance: 0.5 })
    graph.rebuild()

    // 从后果 B 出发：应能回溯到前因 A —— 向量索引做不到的因果召回
    const scores = graph.propagate(new Map([[b.id, 1.0]]))
    assert.ok((scores.get(a.id) ?? 0) > 0, 'A（前因）应被反向因果传播激活')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('wake: sparse truncation respects budget and limit', async () => {
  const { dir, store, graph } = makeEnv()
  try {
    for (let i = 0; i < 10; i += 1) {
      store.add({ kind: 'fact', label: `事实 ${i}`, text: `这是第 ${i} 条很长的记忆内容，用来测试稀疏截断逻辑是否正确。`, sessionId: null, turn: i, causes: [], effects: [], importance: 0.9 })
    }
    graph.rebuild()

    const engine = new CausalWakeEngine(graph, store, { ...CONFIG, maxWakePerTurn: 3 })
    // 无模型打分时降级为 importance 排序
    const hit = await engine.query('测试', 3)
    assert.ok(hit.engrams.length <= 3, '唤醒条数受 maxWakePerTurn 限制')
    assert.ok(hit.injectedTokens <= 600, '注入 token 受预算限制')
    assert.ok(hit.reason.startsWith('causal-wake') || hit.reason === 'below-threshold')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('wake: custom scorer seeds propagation', async () => {
  const { dir, store, graph } = makeEnv()
  try {
    const a = store.add({ kind: 'decision', label: 'A 种子', text: 'A', sessionId: null, turn: 1, causes: [], effects: [], importance: 0.1 })
    const b = store.add({ kind: 'fact', label: 'B 依赖 A', text: 'B', sessionId: null, turn: 2, causes: [a.id], effects: [], importance: 0.1 })
    graph.rebuild()

    // 模拟模型打分：只有 A 命中
    const engine = new CausalWakeEngine(graph, store, CONFIG, async (_q, candidates) => {
      return new Map(candidates.filter((c) => c.id === a.id).map((c) => [c.id, 1.0]))
    })
    const hit = await engine.query('什么导致了 A', 3)
    const ids = hit.engrams.map((e) => e.id)
    assert.ok(ids.includes(a.id), '种子命中')
    assert.ok(ids.includes(b.id), '因果后继被激活并召回')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('wake: empty store short-circuits', async () => {
  const { dir, store, graph } = makeEnv()
  try {
    const engine = new CausalWakeEngine(graph, store, CONFIG)
    const hit = await engine.query('anything', 3)
    assert.equal(hit.engrams.length, 0)
    assert.equal(hit.reason, 'below-threshold')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
