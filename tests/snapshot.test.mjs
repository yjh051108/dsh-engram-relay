/**
 * 工作快照测试（远景场景 6"继续昨天的工作"）。
 *
 * 覆盖：快照创建（聚合最近写入）/ 幂等（内容未变不写盘）/ 内容变化更新+强化 /
 * 检索命中（"继续昨天的工作"类查询召回快照）。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { EngramStore } from '../lib/engram/store.js'
import { NgramHashAddressing } from '../lib/engram/hash.js'
import { EngramWakeEngine } from '../lib/engram/wake.js'
import { CausalGraph } from '../lib/engram/causal.js'

function tempDir() {
  return mkdtempSync(join(tmpdir(), 'engram-snapshot-'))
}

function makeStore(dir) {
  return new EngramStore(dir, new NgramHashAddressing({ seed: 0 }))
}

const WAKE_CONFIG = {
  injectBudgetTokens: 600, maxWakePerTurn: 3, distillEveryTurns: 0, enabled: true,
  modelId: '', dtype: 'bfloat16', storeDir: '', pythonPath: 'python', pythonTimeoutMs: 10000,
  checkpoint: '', embedModel: '', recencyWeight: 0, wakeSampleLog: false,
}

test('snapshot: 创建（聚合最近写入）→ 幂等 → 内容变化更新 + 强化', () => {
  const dir = tempDir()
  try {
    const store = makeStore(dir)
    store.add({ kind: 'decision', layer: 'project', projectId: '/proj', title: '缓存改造', summary: '决定引入 Redis 缓存层', content: '', links: [], sessionId: null, turn: 1, causes: [], effects: [], importance: 0.8 })
    store.add({ kind: 'event', layer: 'project', projectId: '/proj', title: '注入器调试', summary: '正在修热重载 fiber 问题', content: '', links: [], sessionId: null, turn: 2, causes: [], effects: [], importance: 0.8 })

    // 创建
    const snap = store.upsertSnapshot('/proj', 3, 's1')
    assert.ok(snap !== null)
    assert.equal(snap.kind, 'snapshot')
    assert.equal(snap.layer, 'project')
    assert.equal(snap.projectId, '/proj')
    assert.match(snap.title, /^工作快照·/)
    assert.ok(snap.content.includes('[[缓存改造]]'), '聚合最近写入')
    assert.ok(snap.content.includes('[[注入器调试]]'))
    assert.ok(snap.links.includes('缓存改造'), '快照链接到最近记忆')
    const snapId = snap.id
    const reinforces0 = snap.reinforces.length

    // 幂等：内容未变 → 不写盘（reinforces 不增）
    const again = store.upsertSnapshot('/proj', 4, 's1')
    assert.equal(again.id, snapId)
    assert.equal(store.get(snapId).reinforces.length, reinforces0, '内容未变不强化不写盘')

    // 内容变化 → 更新 + 强化
    store.add({ kind: 'fact', layer: 'project', projectId: '/proj', title: '缓存命中率', summary: '命中率恢复到 95%', content: '', links: [], sessionId: null, turn: 5, causes: [], effects: [], importance: 0.8 })
    const updated = store.upsertSnapshot('/proj', 6, 's1')
    assert.equal(updated.id, snapId, '原地更新同节点')
    assert.ok(store.get(snapId).content.includes('[[缓存命中率]]'), '聚合新写入')
    assert.ok(store.get(snapId).reinforces.length > reinforces0, '更新即强化（快照不沉睡）')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('snapshot: "继续昨天的工作"类查询召回快照', async () => {
  const dir = tempDir()
  try {
    const store = makeStore(dir)
    store.add({ kind: 'decision', layer: 'project', projectId: '/proj', title: '缓存改造', summary: '决定引入 Redis 缓存层', content: '', links: [], sessionId: null, turn: 1, causes: [], effects: [], importance: 0.8 })
    const snap = store.upsertSnapshot('/proj', 3, 's1')
    assert.ok(snap !== null)
    const graph = new CausalGraph(store)
    const wake = new EngramWakeEngine(store, graph, new NgramHashAddressing({ seed: 0 }), WAKE_CONFIG, null)
    // "继续昨天的工作"类查询：哈希命中快照标题/内容中的项目与主题词
    const hit = await wake.query('继续昨天的缓存改造工作', 3, { cwd: '/proj' })
    assert.ok(hit.engrams.length > 0, '快照可被唤醒')
    // 快照不参与因果传播被丢弃（kind=snapshot 正常参与候选）
    assert.ok(hit.engrams.some((e) => e.kind === 'snapshot') || hit.engrams.some((e) => e.title.includes('工作快照')), '快照在注入结果中')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
