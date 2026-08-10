/**
 * 混合检索测试：hash 粗筛 + bge 语义精排（stub embedder）+ 因果传播。
 *
 * 验证 wake 管线在 v3（0.6B 移除，bge 语义精排上位）下的行为：
 *  1. hash 粗筛候选永远保留（精确寻址保底）；
 *  2. embedding 分数决定主席位顺序（语义重排）；
 *  3. embedder 缺失/抛错时依次降级：遗留 scorer → 重要度。
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { NgramHashAddressing } from '../lib/engram/hash.js'
import { EngramStore } from '../lib/engram/store.js'
import { CausalGraph } from '../lib/engram/causal.js'
import { EngramWakeEngine } from '../lib/engram/wake.js'

const CONFIG = {
  modelId: '', dtype: 'q8', storeDir: '',
  injectBudgetTokens: 600, maxWakePerTurn: 3, distillEveryTurns: 1, enabled: true,
  pythonPath: '', pythonTimeoutMs: 0, checkpoint: '', embedModel: '',
}

function makeEnv() {
  const dir = mkdtempSync(join(tmpdir(), 'engram-hybrid-'))
  const hasher = new NgramHashAddressing({ seed: 0 })
  const store = new EngramStore(dir, hasher)
  const graph = new CausalGraph(store, { rebuild: false })
  return { dir, store, graph, hasher }
}

function seedPair(store, graph) {
  // 「缓存上线」与「数据上线」共享「上线」n-gram → 一定同时进 hash 候选
  store.add({ kind: 'fact', title: '缓存上线', summary: '缓存层全量生效', content: '', links: [], sessionId: 's1', turn: 1, causes: [], effects: [], importance: 0.3 })
  store.add({ kind: 'fact', title: '数据上线', summary: '数据库切换完成', content: '', links: [], sessionId: 's1', turn: 2, causes: [], effects: [], importance: 0.3 })
  graph.rebuild()
}

test('hybrid: hash 粗筛候选保留，embedding 精排决定主席位', async () => {
  const { dir, store, graph, hasher } = makeEnv()
  try {
    seedPair(store, graph)
    const wake = new EngramWakeEngine(store, graph, hasher, CONFIG, {
      embedder: async (_query, candidates) =>
        new Map(candidates.map((e) => [e.id, e.title === '缓存上线' ? 0.9 : 0.4])),
    })
    const hit = await wake.query('缓存上线后性能如何', 3)
    const titles = hit.engrams.map((e) => e.title)
    assert.ok(titles.includes('缓存上线'), '语义命中者保留')
    assert.ok(titles.includes('数据上线'), 'hash 粗筛候选不被精排丢弃（混合保底）')
    assert.equal(titles[0], '缓存上线', 'embedding 高分者占主席位')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('hybrid: embedder 缺失时降级重要度，哈希命中不丢', async () => {
  const { dir, store, graph, hasher } = makeEnv()
  try {
    seedPair(store, graph)
    const wake = new EngramWakeEngine(store, graph, hasher, CONFIG, {
      embedder: async () => null,
    })
    const hit = await wake.query('缓存上线', 3)
    assert.ok(hit.engrams.length >= 2, 'embedder 缺失时仍按重要度唤醒全部 hash 命中')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('hybrid: embedder 抛错回退遗留 scorer', async () => {
  const { dir, store, graph, hasher } = makeEnv()
  try {
    seedPair(store, graph)
    const wake = new EngramWakeEngine(store, graph, hasher, CONFIG, {
      embedder: async () => { throw new Error('embed service down') },
      scorer: async (_q, candidates) =>
        new Map(candidates.map((e) => [e.id, e.title === '缓存上线' ? 0.8 : 0.2])),
    })
    const hit = await wake.query('缓存上线', 3)
    const titles = hit.engrams.map((e) => e.title)
    assert.ok(titles.includes('缓存上线'), 'scorer 兜底仍可唤醒')
    assert.equal(titles[0], '缓存上线', 'scorer 分数决定主席位')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('hybrid: 部分候选缺分时用重要度垫底（打分器不丢哈希命中）', async () => {
  const { dir, store, graph, hasher } = makeEnv()
  try {
    const a = store.add({ kind: 'fact', title: '缓存上线', summary: '缓存层全量生效', content: '', links: [], sessionId: 's1', turn: 1, causes: [], effects: [], importance: 0.9 })
    const b = store.add({ kind: 'fact', title: '数据上线', summary: '数据库切换完成', content: '', links: [], sessionId: 's1', turn: 2, causes: [], effects: [], importance: 0.1 })
    graph.rebuild()
    // embedder 只给「数据上线」打分（模拟服务端漏返回）
    const wake = new EngramWakeEngine(store, graph, hasher, CONFIG, {
      embedder: async (_q, candidates) =>
        new Map(candidates.filter((e) => e.id === b.id).map((e) => [e.id, 0.95])),
    })
    const hit = await wake.query('缓存上线', 3)
    const titles = hit.engrams.map((e) => e.title)
    assert.ok(titles.includes(a.title), '未打分候选以重要度垫底仍可唤醒')
    assert.equal(titles[0], '数据上线', '高语义分者优先')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
