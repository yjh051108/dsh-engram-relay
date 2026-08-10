/**
 * EngramStore 单元测试：JSONL 持久化 / 因果边 / 唤醒计数。
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync, rmSync, existsSync } from 'node:fs'
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
