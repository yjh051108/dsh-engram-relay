/**
 * 跨会话分层记忆（v0.3 两层版）测试。
 *
 * 覆盖：
 *  - store 分层：add 两层 / query 过滤 / layerCounts / promote / clearProject / update
 *  - wake 分层准入：global 所有会话可见 / project 仅同 cwd
 *  - 工具契约：12 工具注册；engram_store 的 layer 决策校验（project 无 cwd
 *    拒绝、非法 layer 拒绝）；engram_promote 转层；engram_link 显式连接
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
import { installEngramTools } from '../lib/tools.js'

function tempDir() {
  return mkdtempSync(join(tmpdir(), 'dsh-engram-layers-'))
}

function makeStore(dir) {
  return new EngramStore(dir, new NgramHashAddressing({ seed: 0 }))
}

/** 一条记忆（固定种子文本，任何会话可哈希命中）。 */
function addNode(store, { layer, title, summary, cwd = '/proj-a', projectId }) {
  return store.add({
    kind: 'fact',
    layer,
    projectId: layer === 'project' ? (projectId ?? cwd) : null,
    title,
    summary,
    content: '',
    links: [],
    sessionId: 'sess-a',
    turn: 1,
    causes: [],
    effects: [],
    importance: 0.8,
  })
}

const WAKE_CONFIG = {
  injectBudgetTokens: 600,
  maxWakePerTurn: 3,
  distillEveryTurns: 0,
  enabled: true,
  modelId: '',
  dtype: 'bfloat16',
  storeDir: '',
  pythonPath: 'python',
  pythonTimeoutMs: 10000,
  checkpoint: '',
  embedModel: '',
  recencyWeight: 0,
  wakeSampleLog: false,
}

// ---------------------------------------------------------------------------
// store：分层 CRUD
// ---------------------------------------------------------------------------

test('store: add 两层 + query/layerCounts 过滤', () => {
  const dir = tempDir()
  const store = makeStore(dir)
  const g = addNode(store, { layer: 'global', title: '全局偏好', summary: '喜欢简洁文档' })
  const p = addNode(store, { layer: 'project', title: '项目约定', summary: '本项目用 pnpm', cwd: '/proj-a' })
  const p2 = addNode(store, { layer: 'project', title: '别项目约定', summary: 'B 项目用 npm', cwd: '/proj-b' })

  assert.equal(g.layer, 'global')
  assert.equal(p.projectId, '/proj-a')
  assert.deepEqual(store.layerCounts(), { global: 1, project: 2 })

  // query 过滤
  assert.equal(store.query({ layer: 'global' }).length, 1)
  assert.equal(store.query({ layer: 'project', projectId: '/proj-a' }).length, 1)
  assert.equal(store.query({ layer: 'project', projectId: '/proj-b' })[0].title, '别项目约定')
  rmSync(dir, { recursive: true, force: true })
})

test('store: promote project→global 后跨项目可见 + 只升不降', async () => {
  const dir = tempDir()
  const store = makeStore(dir)
  const p = addNode(store, { layer: 'project', title: '端口决策', summary: '决定用 8080', cwd: '/proj-a' })
  const promoted = store.promote(p.id, 'global')
  assert.equal(promoted.layer, 'global')
  assert.equal(promoted.projectId, null, 'global 层 projectId 置空')
  assert.equal(promoted.sessionId, 'sess-a', 'sessionId 保留作溯源')
  // global 节点跨 cwd 可见
  const other = new EngramWakeEngine(store, new CausalGraph(store), new NgramHashAddressing({ seed: 0 }), WAKE_CONFIG, null)
  const hit = await other.query('端口决策 8080', 5, { cwd: '/proj-z' })
  assert.ok(hit.engrams.map((e) => e.title).includes('端口决策'), '提升到 global 后跨项目可见')
  rmSync(dir, { recursive: true, force: true })
})

test('store: update 修正字段 + clearProject 清空项目层', () => {
  const dir = tempDir()
  const store = makeStore(dir)
  const p = addNode(store, { layer: 'project', title: '项目约定', summary: '旧摘要', cwd: '/proj-a' })
  const updated = store.update(p.id, { summary: '新摘要', importance: 0.9 })
  assert.equal(updated.summary, '新摘要')
  assert.equal(updated.importance, 0.9)
  const cleared = store.clearProject('/proj-a')
  assert.equal(cleared, 1)
  assert.equal(store.count(), 0)
  rmSync(dir, { recursive: true, force: true })
})

// ---------------------------------------------------------------------------
// wake：分层准入（跨会话可见性边界）
// ---------------------------------------------------------------------------

test('wake: 全可见（v0.4 项目不隔离——任意 cwd 命中全部记忆）', async () => {
  const dir = tempDir()
  const store = makeStore(dir)
  addNode(store, { layer: 'global', title: '全局事实', summary: '团队用 pnpm 且部署端口 8080' })
  addNode(store, { layer: 'project', title: '项目A决策', summary: 'A 项目部署端口 8080', cwd: '/proj-a' })
  const graph = new CausalGraph(store)
  const wake = new EngramWakeEngine(store, graph, new NgramHashAddressing({ seed: 0 }), WAKE_CONFIG, null)

  // 会话 A（cwd=/proj-a）：全可见
  const a = await wake.query('部署端口 8080', 5, { cwd: '/proj-a' })
  const aTitles = a.engrams.map((e) => e.title)
  assert.ok(aTitles.includes('全局事实'), 'global 可见')
  assert.ok(aTitles.includes('项目A决策'), '同项目可见')

  // 会话 C（cwd=/proj-c）：同样全可见（融会贯通）
  const c = await wake.query('部署端口 8080', 5, { cwd: '/proj-c' })
  const cTitles = c.engrams.map((e) => e.title)
  assert.ok(cTitles.includes('全局事实'))
  assert.ok(cTitles.includes('项目A决策'), 'v0.4：跨项目全可见（项目即标签）')
  rmSync(dir, { recursive: true, force: true })
})

test('wake: 跨会话命中（global 常驻，另一天同项目仍能命中 project）', async () => {
  const dir = tempDir()
  const store = makeStore(dir)
  addNode(store, { layer: 'global', title: '通用约定', summary: '所有项目部署端口都用 8080 且缓存用 Redis' })
  addNode(store, { layer: 'project', title: 'A项目踩坑', summary: 'A 项目防火墙规则导致端口 8080 不通', cwd: '/proj-a' })
  // 会话 B（另一天、同项目）仍能命中跨会话记忆
  const graph = new CausalGraph(store)
  const wake = new EngramWakeEngine(store, graph, new NgramHashAddressing({ seed: 0 }), WAKE_CONFIG, null)
  const hit = await wake.query('端口 8080', 5, { cwd: '/proj-a' })
  const titles = hit.engrams.map((e) => e.title)
  assert.ok(titles.includes('通用约定'), 'global 跨会话')
  assert.ok(titles.includes('A项目踩坑'), 'project 跨会话同 cwd')
  rmSync(dir, { recursive: true, force: true })
})

// ---------------------------------------------------------------------------
// 工具契约
// ---------------------------------------------------------------------------

/** fake ctx：tools.register 收集工具定义。 */
function fakeToolCtx() {
  const tools = []
  return {
    tools: {
      register(def) {
        tools.push(def)
        return () => { tools.splice(tools.indexOf(def), 1) }
      },
    },
    collected: tools,
  }
}

/** 半真实 relay（真实 store/graph，recall/status stub）——避免触发 Python。 */
function fakeRelay(dir) {
  const store = makeStore(dir)
  const graph = new CausalGraph(store)
  return {
    store,
    graph,
    currentSessionId: null,
    lastTurnAt: 0,
    recall: async () => ({ engrams: [], reason: 'stub', injectedTokens: 0 }),
    model: { embed: async () => null }, // 无模型：查重退化为标题精确匹配
    status: async () => ({ engramCount: store.count(), layerCounts: store.layerCounts() }),
  }
}

test('tools: 注册完整工具集（12 个，含用户确认制 propose/confirm/reject）', () => {
  const ctx = fakeToolCtx()
  const relay = fakeRelay(tempDir())
  const dispose = installEngramTools(ctx, relay)
  const names = ctx.collected.map((d) => d.name).sort()
  assert.deepEqual(names, [
    'engram_confirm',
    'engram_link',
    'engram_open',
    'engram_promote',
    'engram_propose',
    'engram_recall',
    'engram_reject',
    'engram_remove',
    'engram_search',
    'engram_status',
    'engram_store',
    'engram_update',
  ])
  dispose()
})

test('tools: engram_store 的 layer 决策校验（非法层拒绝 / project 无 cwd 拒绝）', async () => {
  const dir = tempDir()
  const ctx = fakeToolCtx()
  const relay = fakeRelay(dir)
  installEngramTools(ctx, relay)
  const storeTool = ctx.collected.find((d) => d.name === 'engram_store')

  // 非法 layer
  const bad = await storeTool.execute({ layer: 'invalid', kind: 'fact', title: 'T', summary: 'S' }, { agent: { session: { id: 's1', header: { cwd: '/w' } } } })
  assert.match(bad, /layer 必须是/)

  // project 层无 cwd
  const noCwd = await storeTool.execute({ layer: 'project', kind: 'fact', title: 'T', summary: 'S' }, { agent: { session: { id: 's1', header: {} } } })
  assert.match(noCwd, /需要当前工作目录/)

  // 合法写入：project 层带 cwd → 落 project + projectId=cwd
  const ok = await storeTool.execute({ layer: 'project', kind: 'fact', title: '端口约定', summary: '用 8080', causes: [] }, { agent: { session: { id: 's1', header: { cwd: '/proj-x' } } } })
  assert.match(ok, /已写入记忆节点/)
  const saved = relay.store.byTitle('端口约定')
  assert.equal(saved.layer, 'project')
  assert.equal(saved.projectId, '/proj-x')
  rmSync(dir, { recursive: true, force: true })
})

test('tools: engram_promote 只升不降（project→global）', async () => {
  const dir = tempDir()
  const ctx = fakeToolCtx()
  const relay = fakeRelay(dir)
  installEngramTools(ctx, relay)
  const promoteTool = ctx.collected.find((d) => d.name === 'engram_promote')
  const node = relay.store.add({
    kind: 'fact', layer: 'project', projectId: '/w', title: '项目结论', summary: '调试结论',
    content: '', links: [], sessionId: 's1', turn: 1, causes: [], effects: [], importance: 0.8,
  })
  // 非法目标层（session 已不存在）
  const bad = await promoteTool.execute({ ref: node.id, layer: 'session' }, { agent: { session: { id: 's1', header: { cwd: '/w' } } } })
  assert.match(bad, /目标层/)
  // 正常提升 project→global
  const up = await promoteTool.execute({ ref: node.id, layer: 'global' }, { agent: { session: { id: 's1', header: { cwd: '/w' } } } })
  assert.match(up, /已提升/)
  assert.equal(relay.store.get(node.id).layer, 'global')
  // global 不能再升
  const again = await promoteTool.execute({ ref: node.id, layer: 'global' }, { agent: { session: { id: 's1', header: { cwd: '/w' } } } })
  assert.match(again, /最高/)
  rmSync(dir, { recursive: true, force: true })
})

test('tools: engram_link 显式因果连接（store 因果数组同步）', async () => {
  const dir = tempDir()
  const ctx = fakeToolCtx()
  const relay = fakeRelay(dir)
  installEngramTools(ctx, relay)
  const linkTool = ctx.collected.find((d) => d.name === 'engram_link')
  const a = relay.store.add({
    kind: 'decision', layer: 'project', projectId: '/w', title: '根因', summary: '磁盘满了',
    content: '', links: [], sessionId: null, turn: 1, causes: [], effects: [], importance: 0.8,
  })
  const b = relay.store.add({
    kind: 'event', layer: 'project', projectId: '/w', title: '故障', summary: '服务挂了',
    content: '', links: [], sessionId: null, turn: 2, causes: [], effects: [], importance: 0.8,
  })
  const out = await linkTool.execute(
    { from: a.id, to: b.id, kind: 'causes', bidirectional: true },
    { agent: { session: { id: 's1', header: { cwd: '/w' } } } },
  )
  assert.match(out, /已连接/)
  // store 因果数组同步（graph rebuild 不丢边）
  assert.ok(relay.store.get(a.id).effects.includes(b.id), 'from.effects 加 to')
  assert.ok(relay.store.get(b.id).causes.includes(a.id), 'to.causes 加 from')
  // 双向链接
  assert.ok(relay.store.get(b.id).links.includes(a.title), '双向链接 b→a')
  // graph 重建后边仍在
  relay.graph.rebuild()
  assert.equal(relay.graph.effectsOf(a.id)[0]?.id, b.id, 'graph rebuild 后因果边保留')
  rmSync(dir, { recursive: true, force: true })
})
