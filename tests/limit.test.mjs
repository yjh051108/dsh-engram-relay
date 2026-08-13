/**
 * 硬上限 + 归档淘汰测试（v0.5）。
 *
 * 覆盖：归档写入 archived.jsonl 且主库移除 / 淘汰顺序（superseded →
 * dormant → 低激活）/ 硬上限触发 / 归档可读回恢复。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync, existsSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { EngramStore, isSuperseded, dormantOf } from '../lib/engram/store.js'
import { NgramHashAddressing } from '../lib/engram/hash.js'

function tempDir() {
  return mkdtempSync(join(tmpdir(), 'engram-limit-'))
}

function makeStore(dir) {
  return new EngramStore(dir, new NgramHashAddressing({ seed: 0 }))
}

function addNode(store, over = {}) {
  return store.add({
    kind: 'fact', layer: 'project', projectId: '/w', title: '记忆' + Math.random().toString(36).slice(2, 8),
    summary: '测试文本 端口 8080', content: '', links: [], sessionId: null,
    turn: 1, causes: [], effects: [], importance: 0.5, ...over,
  })
}

test('limit: 归档——主库移除 + archived.jsonl 落盘 + 可恢复', () => {
  const dir = tempDir()
  try {
    const store = makeStore(dir)
    const e = addNode(store)
    assert.equal(store.archiveNode(e.id), true)
    assert.equal(store.get(e.id), undefined, '主库移除')
    assert.equal(store.lookup('测试文本 端口 8080', 16).length, 0, '退出检索')
    const archiveFile = join(dir, 'archived.jsonl')
    assert.ok(existsSync(archiveFile), '归档文件存在')
    const lines = readFileSync(archiveFile, 'utf8').split('\n').filter((l) => l.trim() !== '')
    assert.equal(lines.length, 1)
    const record = JSON.parse(lines[0])
    assert.equal(record.id, e.id)
    assert.ok(record.archivedAt > 0, '带归档时间戳')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('limit: 淘汰顺序——superseded 优先，其次 dormant，最后低激活', () => {
  const dir = tempDir()
  try {
    const store = makeStore(dir)
    // 低激活（保留）：近期强化
    const keep = addNode(store, { reinforces: [Date.now()], createdAt: Date.now() })
    // dormant（应淘汰）：31 天前强化
    const dor = addNode(store, { reinforces: [Date.now() - 31 * 86400000], createdAt: Date.now() - 31 * 86400000 })
    // superseded（最先淘汰）
    const sup = addNode(store)
    const sup2 = addNode(store, { title: '同主题新' })
    store.supersede(sup.id, sup2.id)
    assert.equal(dormantOf(store.get(dor.id)), true, 'dormant 派生成立')

    // 上限 = 2（当前 4 条 → 淘汰 2 条：superseded + dormant）
    const archived = store.enforceLimit(2)
    assert.equal(archived, 2)
    assert.equal(store.count(), 2)
    assert.equal(store.get(sup.id), undefined, 'superseded 被淘汰')
    assert.equal(store.get(dor.id), undefined, 'dormant 被淘汰')
    assert.ok(store.get(keep.id), '近期记忆保留')
    assert.ok(store.get(sup2.id), '当前版保留')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('limit: 未超限不动作 + 归档计数', () => {
  const dir = tempDir()
  try {
    const store = makeStore(dir)
    const a = addNode(store)
    const b = addNode(store)
    assert.equal(store.enforceLimit(10), 0, '未超限不淘汰')
    assert.equal(store.count(), 2)
    store.archiveNode(a.id)
    assert.equal(store.archivedCount(), 1)
    // 归档节点可读回（恢复路径：解析 archived.jsonl）
    const lines = readFileSync(join(dir, 'archived.jsonl'), 'utf8').split('\n').filter((l) => l.trim() !== '')
    const restored = JSON.parse(lines[0])
    assert.equal(restored.title, a.title)
    void b
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
