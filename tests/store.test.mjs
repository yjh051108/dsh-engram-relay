/**
 * EngramStore 单元测试：JSONL 持久化 / 因果边 / 唤醒计数。
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync, writeFileSync, rmSync, existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { EngramStore } from '../lib/engram/store.js'

function tempDir() {
  return mkdtempSync(join(tmpdir(), 'engram-store-'))
}

test('store: add/persist/reload roundtrip', () => {
  const dir = tempDir()
  try {
    const store = new EngramStore(dir)
    const e = store.add({
      kind: 'decision',
      title: 'engram 转接层决策',
      summary: '决定采用因果图传播的稀疏唤醒，不引入向量数据库。',
      sessionId: 's1',
      turn: 3,
      causes: [],
      effects: [],
      importance: 0.9,
    })
    assert.equal(store.count(), 1)
    assert.ok(store.get(e.id))

    // 重新加载（模拟重启）
    const reloaded = new EngramStore(dir)
    assert.equal(reloaded.count(), 1)
    const got = reloaded.get(e.id)
    assert.equal(got?.title, 'engram 转接层决策')
    assert.equal(got?.kind, 'decision')

    // 持久化文件是 JSONL
    const raw = readFileSync(join(dir, 'engrams.jsonl'), 'utf8')
    assert.ok(raw.includes('decision'))
    assert.ok(raw.trim().endsWith('}'))
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('store: touch updates hits and persists', () => {
  const dir = tempDir()
  try {
    const store = new EngramStore(dir)
    const e = store.add({
      kind: 'fact',
      title: '部署端口',
      summary: '项目部署端口是 8080。',
      sessionId: null,
      turn: 0,
      causes: [],
      effects: [],
      importance: 0.5,
    })
    store.touch(e.id)
    const reloaded = new EngramStore(dir)
    assert.equal(reloaded.get(e.id)?.hits, 1)
    assert.ok(reloaded.get(e.id)?.lastHitAt)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('store: remove deletes and persists', () => {
  const dir = tempDir()
  try {
    const store = new EngramStore(dir)
    const e = store.add({
      kind: 'preference',
      title: '回复偏好',
      summary: '用户偏好中文回复。',
      sessionId: null,
      turn: 0,
      causes: [],
      effects: [],
      importance: 0.8,
    })
    assert.equal(store.remove(e.id), true)
    assert.equal(store.count(), 0)
    assert.equal(existsSync(join(dir, 'engrams.jsonl')), true)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('store: 旧数据加载归一化——缺 layer/slots/links 的持久化不产生 undefined/NaN', () => {
  const dir = tempDir()
  try {
    // 手工写入 v0.2.0 分层前格式的旧节点（缺 layer/projectId/slots/links/causes/effects）
    writeFileSync(
      join(dir, 'engrams.jsonl'),
      '{"id":"e-old-1","kind":"fact","title":"旧节点","summary":"分层前写入","content":"","sessionId":"s1","turn":0,"importance":0.5,"hits":0,"lastHitAt":null,"createdAt":1}\n',
      'utf8',
    )
    const store = new EngramStore(dir)
    assert.equal(store.count(), 1, '旧节点应加载成功（不被 slots 缺失丢弃）')
    const n = store.get('e-old-1')
    assert.equal(n?.layer, 'project', '缺 layer 归一化为 project（v0.3 默认层）')
    assert.equal(n?.projectId, null)
    assert.deepEqual(n?.links, [])
    assert.deepEqual(n?.causes, [])
    assert.deepEqual(n?.effects, [])
    assert.deepEqual(n?.slots, [])
    // layerCounts 不再出现 undefined 键（曾 JSON 序列化为 "undefined":null）
    assert.deepEqual(store.layerCounts(), { global: 0, project: 1 })
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
