/**
 * 版本链测试（P2 治理缺口①真理维护）。
 *
 * 覆盖：supersede 双向指针 / lookup 过滤废止 / byTitle 可追溯 /
 * 持久化保留 / 自动修订路径（tools 层 findDuplicate 高置信查重）。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { EngramStore, isSuperseded } from '../lib/engram/store.js'
import { CausalGraph } from '../lib/engram/causal.js'
import { NgramHashAddressing } from '../lib/engram/hash.js'
import { EngramWakeEngine } from '../lib/engram/wake.js'
import { installEngramTools } from '../lib/tools.js'

function tempDir() {
  return mkdtempSync(join(tmpdir(), 'engram-version-'))
}

function makeStore(dir) {
  return new EngramStore(dir, new NgramHashAddressing({ seed: 0 }))
}

function addNode(store, title, summary, over = {}) {
  return store.add({
    kind: 'fact', layer: 'project', projectId: '/w', title, summary,
    content: '', links: [], sessionId: null, turn: 1, causes: [], effects: [],
    importance: 0.6, ...over,
  })
}

const WAKE_CONFIG = {
  injectBudgetTokens: 600, maxWakePerTurn: 3, distillEveryTurns: 0, enabled: true,
  modelId: '', dtype: 'bfloat16', storeDir: '', pythonPath: 'python', pythonTimeoutMs: 10000,
  checkpoint: '', embedModel: '', recencyWeight: 0, wakeSampleLog: false,
}

test('version: supersede 双向指针 + 持久化保留', () => {
  const dir = tempDir()
  try {
    const store = makeStore(dir)
    const v1 = addNode(store, '缓存策略', '旧版：缓存命中率下降用清缓存')
    const v2 = addNode(store, '缓存策略', '新版：缓存命中率下降查版本链')
    assert.equal(store.supersede(v1.id, v2.id), true)
    assert.equal(store.get(v1.id).supersededBy, v2.id, '旧版指向新版')
    assert.deepEqual(store.get(v2.id).supersedes, [v1.id], '新版记录取代列表')
    assert.equal(isSuperseded(store.get(v1.id)), true)
    assert.equal(isSuperseded(store.get(v2.id)), false)
    // 持久化
    const reloaded = makeStore(dir)
    assert.equal(reloaded.get(v1.id).supersededBy, v2.id)
    assert.deepEqual(reloaded.get(v2.id).supersedes, [v1.id])
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('version: 检索过滤废止——lookup/wake 只命中当前版，byTitle 可追溯', async () => {
  const dir = tempDir()
  try {
    const store = makeStore(dir)
    const v1 = addNode(store, '部署端口', '旧版：端口 8080 部署配置')
    const v2 = addNode(store, '部署端口', '新版：端口 9090 部署配置')
    store.supersede(v1.id, v2.id)
    // lookup 不含废止
    const hits = store.lookup('部署端口 8080 部署配置', 16)
    assert.ok(!hits.some((e) => e.id === v1.id), 'lookup 不含废止节点')
    assert.ok(hits.some((e) => e.id === v2.id), 'lookup 含当前版')
    // wake 注入不含废止
    const graph = new CausalGraph(store)
    const wake = new EngramWakeEngine(store, graph, new NgramHashAddressing({ seed: 0 }), WAKE_CONFIG, null)
    const hit = await wake.query('部署端口 8080', 5, { cwd: '/w' })
    assert.ok(!hit.engrams.some((e) => e.id === v1.id), '注入不含废止')
    // byTitle 可追溯（同名消歧取最近 = v2；byTitles 含 v1）
    assert.ok(store.byTitles('部署端口').some((e) => e.id === v1.id), 'byTitles 可追溯旧版')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('version: 工具自动修订——同主题高置信写入 = 修订而非新增', async () => {
  const dir = tempDir()
  try {
    const store = makeStore(dir)
    const graph = new CausalGraph(store)
    const tools = []
    const ctx = { tools: { register: (d) => { tools.push(d); return () => {} } } }
    const relay = {
      store, graph,
      currentSessionId: 's1', lastTurnAt: 1,
      fusionTau: { sem: 1, time: 0, cause: 0 },
      activation: { get: () => 0 },
      // embed stub：精确返回"同主题"高分（模拟真实语义高置信）
      model: {
        embed: async (text, cands) => new Map(cands.map((e, i) => [e.id, i === 0 ? 0.72 : 0.3])),
        semanticScores: (text, cands) => new Map(cands.map((e, i) => [e.id, { score: i === 0 ? 0.72 : 0.3, lexical: i === 0 ? 0.8 : 0.2, graph: 0, svd: 0 }])),
      },
      recall: async () => ({ engrams: [], reason: 'stub', injectedTokens: 0 }),
      status: async () => ({ engramCount: store.count() }),
    }
    installEngramTools(ctx, relay)
    const storeTool = tools.find((d) => d.name === 'engram_store')
    const exec = { agent: { session: { id: 's1', header: { cwd: '/w' } } } }

    // 第一次写入：新增
    const r1 = await storeTool.execute({ layer: 'project', kind: 'fact', title: '缓存修复', summary: '缓存命中率下降的排查步骤', causes: [] }, exec)
    assert.match(r1, /已写入记忆节点/)
    const count1 = store.count()

    // 第二次同主题写入：自动修订（embed stub 0.72 ≥ 0.6）
    const r2 = await storeTool.execute({ layer: 'project', kind: 'fact', title: '缓存修复', summary: '缓存命中率下降的排查步骤（修订版）', causes: [] }, exec)
    assert.match(r2, /已修订记忆/)
    assert.equal(store.count(), count1 + 1, '修订 = 新增当前版 + 旧版废止（不删）')
    const old = store.byTitles('缓存修复').find((e) => isSuperseded(e))
    const neu = store.byTitles('缓存修复').find((e) => !isSuperseded(e))
    assert.ok(old && neu, '存在废止版 + 当前版')
    assert.equal(store.get(old.id).supersededBy, neu.id, '版本链指针正确')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('version: 织网窗口——lexical 高但融合分 <0.6：织网触发但不修订', async () => {
  const dir = tempDir()
  try {
    const store = makeStore(dir)
    const graph = new CausalGraph(store)
    const tools = []
    const ctx = { tools: { register: (d) => { tools.push(d); return () => {} } } }
    // stub：候选 0 的 lexical 高（0.8）但融合分 0.55（织网窗口）
    const relay = {
      store, graph,
      currentSessionId: 's1', lastTurnAt: 1,
      fusionTau: { sem: 1, time: 0, cause: 0 },
      activation: { get: () => 0 },
      model: {
        embed: async () => new Map(),
        semanticScores: () => new Map([['cand-0', { score: 0.55, lexical: 0.8, graph: 0, svd: 0 }]]),
      },
      recall: async () => ({ engrams: [], reason: 'stub', injectedTokens: 0 }),
      status: async () => ({ engramCount: store.count() }),
    }
    installEngramTools(ctx, relay)
    const storeTool = tools.find((d) => d.name === 'engram_store')
    const exec = { agent: { session: { id: 's1', header: { cwd: '/w' } } } }

    // 先造一个候选（cand-0 是 lookup 能返回的节点）
    const seed = store.add({ kind: 'fact', layer: 'project', projectId: '/w', title: '主题甲', summary: '主题甲的内容', content: '', links: [], sessionId: null, turn: 1, causes: [], effects: [], importance: 0.5 })
    // 修正 stub：dynamic 返回 seed 的分数（候选 id 运行时才知道）
    relay.model.semanticScores = () => new Map([[seed.id, { score: 0.55, lexical: 0.8, graph: 0, svd: 0 }]])
    const r = await storeTool.execute({ layer: 'project', kind: 'fact', title: '主题甲补充', summary: '主题甲的内容补充说明', causes: [] }, exec)
    // 不修订（0.55 < 0.6）
    assert.match(r, /已写入记忆节点/)
    assert.ok(!/已修订/.test(r), '融合分 0.55 不触发修订')
    // 织网（lexical 0.8 ≥ 0.5）→ 双向链接建立
    assert.match(r, /自动织网/, '织网触发提示')
    const neu = store.byTitle('主题甲补充')
    assert.ok(neu.links.includes('主题甲'), '新节点链接到候选')
    const seedReloaded = store.get(seed.id)
    assert.ok(seedReloaded.links.includes('主题甲补充'), '候选反向链接')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
