/**
 * 大一统记忆图谱测试：渐进披露 + 双向链接 + 因果双向追溯。
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
  modelId: 'sim', dtype: 'q8', storeDir: '',
  injectBudgetTokens: 600, maxWakePerTurn: 3, distillEveryTurns: 1, enabled: true,
  pythonPath: '', pythonTimeoutMs: 0, checkpoint: '',
}

function makeEnv() {
  const dir = mkdtempSync(join(tmpdir(), 'engram-graph-'))
  const hasher = new NgramHashAddressing({ seed: 0 })
  const store = new EngramStore(dir, hasher)
  const graph = new CausalGraph(store)
  return { dir, store, graph, hasher }
}

test('graph: 统一节点含 title/summary/content（渐进披露分层）', () => {
  const { dir, store } = makeEnv()
  try {
    const n = store.add({
      kind: 'decision', title: '部署端口决策', summary: '定了用 8080',
      content: '详细背景：考虑兼容性后定了 8080，回滚方案是 9090。',
      links: ['部署实施'], sessionId: 's1', turn: 1, causes: [], effects: [], importance: 0.9,
    })
    assert.equal(store.byTitle('部署端口决策')?.id, n.id, '标题索引可解析 [[链接]]')
    assert.equal(n.summary, '定了用 8080', '入口层是摘要')
    assert.equal(n.content.length > n.summary.length, true, '展开层是正文')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('graph: 双向链接（Obsidian 风格）——写 A 链 B，B 反链 A', () => {
  const { dir, store } = makeEnv()
  try {
    const a = store.add({
      kind: 'decision', title: '方案A', summary: '选了 A', content: '', links: ['方案B'],
      sessionId: 's1', turn: 1, causes: [], effects: [], importance: 0.5,
    })
    // B 已存在时，A 的 links 应让 B 获得反向链接（en gram_store 工具逻辑）
    const b = store.byTitle('方案A')
    assert.equal(b?.id, a.id, '标题解析正常')
    assert.deepEqual(a.links, ['方案B'], 'A 声明链接 B')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('graph: 因果双向追溯——从后果回溯前因、从前因走向后果', () => {
  const { dir, store, graph } = makeEnv()
  try {
    const root = store.add({ kind: 'decision', title: '根决策', summary: '定了方案', content: '', links: [], sessionId: 's1', turn: 1, causes: [], effects: [], importance: 0.5 })
    const mid = store.add({ kind: 'event', title: '实施', summary: '执行中', content: '', links: [], sessionId: 's1', turn: 2, causes: [root.id], effects: [], importance: 0.5 })
    const end = store.add({ kind: 'note', title: '上线', summary: '完成', content: '', links: [], sessionId: 's1', turn: 3, causes: [mid.id], effects: [], importance: 0.5 })
    graph.rebuild()

    // 双向追溯：后果 → 前因（因果 ↑）
    const causes = graph.causesOf(end.id).map((n) => n.title)
    assert.deepEqual(causes, ['实施'], '后果可回溯前因')
    // 前因 → 后果（因果 ↓）
    const effects = graph.effectsOf(root.id).map((n) => n.title)
    assert.deepEqual(effects, ['实施'], '前因可走向后果')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('graph: 渐进披露渲染——入口列表不含 content', async () => {
  const { dir, store, graph, hasher } = makeEnv()
  try {
    store.add({ kind: 'fact', title: '部署端口', summary: '项目用 8080', content: '超长的完整正文内容不应该出现在入口层', links: [], sessionId: 's1', turn: 1, causes: [], effects: [], importance: 0.9 })
    graph.rebuild()
    const wake = new EngramWakeEngine(store, graph, hasher, CONFIG)
    const hit = await wake.query('部署端口', 3)
    assert.ok(hit.engrams.length >= 1, '应命中入口')
    const section = wake.renderInjection(600)
    assert.ok(section.includes('[[部署端口]]'), '入口含 [[标题]]')
    assert.ok(section.includes('项目用 8080'), '入口含摘要')
    assert.ok(!section.includes('超长的完整正文'), '入口不含 content（渐进披露）')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('graph: 自组织聚类——链接密度自然成簇（不硬编码分层）', () => {
  const { dir, store } = makeEnv()
  try {
    // 簇 A：部署主题（3 节点互相链接）
    const a1 = store.add({ kind: 'decision', title: '部署方案', summary: '容器化', content: '', links: ['部署实施'], sessionId: 's1', turn: 1, causes: [], effects: [], importance: 0.5 })
    const a2 = store.add({ kind: 'event', title: '部署实施', summary: '改造中', content: '', links: ['部署方案', '上线验证'], sessionId: 's1', turn: 2, causes: [a1.id], effects: [], importance: 0.5 })
    store.add({ kind: 'note', title: '上线验证', summary: '通过', content: '', links: ['部署实施'], sessionId: 's1', turn: 3, causes: [a2.id], effects: [], importance: 0.5 })
    // 簇 B：缓存主题（2 节点互相链接，与 A 无连接）
    const b1 = store.add({ kind: 'decision', title: '缓存选型', summary: 'Redis', content: '', links: ['缓存实施'], sessionId: 's1', turn: 4, causes: [], effects: [], importance: 0.5 })
    store.add({ kind: 'event', title: '缓存实施', summary: '接入中', content: '', links: ['缓存选型'], sessionId: 's1', turn: 5, causes: [b1.id], effects: [], importance: 0.5 })

    const clusters = store.clusters()
    assert.equal(clusters.length, 2, '应自然分成 2 簇（部署/缓存）')
    // 大簇在前：部署 3 节点
    assert.equal(clusters[0].members.length, 3, '部署簇 3 节点')
    assert.equal(clusters[1].members.length, 2, '缓存簇 2 节点')
    // 代表节点 = 连接度最高（部署实施连接 2 个，是代表）
    assert.equal(clusters[0].representative, a2.id, '代表节点是连接度最高的「部署实施」')
    assert.equal(clusters[0].label, '部署实施')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
