/**
 * 巩固状态机测试（v0.3 双维度：可见性×巩固度）。
 *
 * 覆盖：写入=episodic；hits≥3→semantic；30 天无强化→dormant；
 * 持久化保留；旧数据默认 episodic；渲染按 state 分级。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { EngramStore, dormantOf } from '../lib/engram/store.js'
import { CausalGraph } from '../lib/engram/causal.js'
import { NgramHashAddressing } from '../lib/engram/hash.js'
import { EngramWakeEngine } from '../lib/engram/wake.js'

function tempDir() {
  return mkdtempSync(join(tmpdir(), 'engram-state-'))
}

function makeStore(dir) {
  return new EngramStore(dir, new NgramHashAddressing({ seed: 0 }))
}

function addNode(store, over = {}) {
  return store.add({
    kind: 'fact', layer: 'project', projectId: '/w', title: '记忆' + Math.random().toString(36).slice(2, 6),
    summary: '测试记忆文本 端口 8080 缓存 Redis', content: '', links: [], sessionId: null,
    turn: 1, causes: [], effects: [], importance: 0.6, ...over,
  })
}

test('state: 写入 = episodic，hits≥3 迁移 semantic', () => {
  const dir = tempDir()
  try {
    const store = makeStore(dir)
    const e = addNode(store)
    assert.equal(e.state, 'episodic', '新写入是 episodic')
    store.touch(e.id)
    store.touch(e.id)
    assert.equal(store.get(e.id).state, 'episodic', 'hits<3 仍 episodic')
    store.touch(e.id)
    assert.equal(store.get(e.id).state, 'semantic', 'hits≥3 → semantic')
    // 持久化保留
    const reloaded = makeStore(dir)
    assert.equal(reloaded.get(e.id).state, 'semantic')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('state: 30 天无强化 = dormant（派生状态，命中即复苏）', () => {
  const dir = tempDir()
  try {
    const store = makeStore(dir)
    // 陈年记忆：31 天前创建、无后续强化、hits=0
    const old = store.add({ kind: 'fact', layer: 'project', projectId: '/w', title: '陈年记忆', summary: '很久以前', content: '', links: [], sessionId: null, turn: 1, causes: [], effects: [], importance: 0.5, createdAt: Date.now() - 31 * 86400000 })
    const node = store.get(old.id)
    node.reinforces = [Date.now() - 31 * 86400000]
    // 派生判定：沉默 >30 天 → dormant
    assert.equal(dormantOf(store.get(old.id)), true, '31 天无强化 → dormant')
    assert.deepEqual(store.stateCounts(), { episodic: 0, semantic: 0, dormant: 1 })
    // 命中即复苏：touch 推入当前强化 → 不再 dormant
    store.touch(old.id)
    assert.equal(dormantOf(store.get(old.id)), false, '命中后复苏')
    // hits≥3 固化 semantic
    store.touch(old.id); store.touch(old.id)
    assert.equal(store.get(old.id).state, 'semantic')
    assert.deepEqual(store.stateCounts(), { episodic: 0, semantic: 1, dormant: 0 })
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('state: 旧数据加载默认 episodic', () => {
  const dir = tempDir()
  try {
    writeFileSync(join(dir, 'engrams.jsonl'),
      '{"id":"old-1","kind":"fact","layer":"project","projectId":"/w","title":"旧记忆","summary":"s","content":"","links":[],"causes":[],"effects":[],"sessionId":null,"turn":1,"importance":0.5,"hits":0,"createdAt":1,"reinforces":[1],"slots":[],"status":"confirmed"}\n',
      'utf8')
    const store = makeStore(dir)
    assert.equal(store.all()[0].state, 'episodic', '无 state 字段归一化为 episodic')
    // 注：reinforces=[1]（1970 年）→ 派生 dormant；近期强化的旧数据才是 episodic
    writeFileSync(join(dir, 'engrams.jsonl'),
      `{"id":"old-2","kind":"fact","layer":"project","projectId":"/w","title":"旧记忆2","summary":"s","content":"","links":[],"causes":[],"effects":[],"sessionId":null,"turn":1,"importance":0.5,"hits":0,"createdAt":${Date.now()},"reinforces":[${Date.now()}],"slots":[],"status":"confirmed"}\n`,
      'utf8')
    const store2 = makeStore(dir)
    assert.equal(store2.all()[0].state, 'episodic')
    assert.deepEqual(store2.stateCounts(), { episodic: 1, semantic: 0, dormant: 0 })
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('state: 渲染按巩固状态分级（semantic 完整 / episodic 摘要 / dormant 仅标题）', async () => {
  const dir = tempDir()
  try {
    const store = makeStore(dir)
    const sem = store.add({ kind: 'fact', layer: 'global', projectId: null, title: '语义记忆', summary: '去情景化真理：端口 8080 部署', content: '', links: [], sessionId: null, turn: 1, causes: [], effects: [], importance: 0.9 })
    const epi = store.add({ kind: 'event', layer: 'global', projectId: null, title: '事件记忆', summary: '今天修了缓存 bug', content: '', links: [], sessionId: null, turn: 1, causes: [], effects: [], importance: 0.8 })
    const dor = store.add({ kind: 'note', layer: 'global', projectId: null, title: '沉睡记忆', summary: '很久以前的事', content: '', links: [], sessionId: null, turn: 1, causes: [], effects: [], importance: 0.7 })
    // 手工设定状态：semantic 固化 + dormant 派生（31 天前强化）
    store.get(sem.id).state = 'semantic'
    store.get(dor.id).reinforces = [Date.now() - 31 * 86400000]
    store.get(dor.id).hits = 0
    const graph = new CausalGraph(store)
    const wake = new EngramWakeEngine(store, graph, new NgramHashAddressing({ seed: 0 }), {
      injectBudgetTokens: 600, maxWakePerTurn: 3, distillEveryTurns: 0, enabled: true,
      modelId: '', dtype: 'bfloat16', storeDir: '', pythonPath: 'python', pythonTimeoutMs: 10000,
      checkpoint: '', embedModel: '', recencyWeight: 0, wakeSampleLog: false,
    }, null)
    // 直接构造 lastInjection（渲染读 lastInjection；不经 query——query 的
    // touch 会刷新强化时间戳，破坏 dormant 派生判定）
    wake.lastInjection = { engrams: [store.get(sem.id), store.get(epi.id), store.get(dor.id)], reason: 'test', injectedTokens: 0 }
    const rendered = wake.renderInjection(600)
    assert.ok(rendered.includes('[[语义记忆]][global]'), 'semantic 完整入口（标题+层）')
    assert.ok(!rendered.includes('[[语义记忆]] ↑'), 'semantic 无因果注（无边）')
    assert.ok(rendered.includes('[[事件记忆]][global]: 今天修了缓存 bug'), 'episodic 标题+摘要')
    assert.ok(!rendered.includes('[[沉睡记忆]][global]:'), 'dormant 不渲染摘要')
    assert.ok(rendered.includes('[[沉睡记忆]]'), 'dormant 仅标题入口')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
