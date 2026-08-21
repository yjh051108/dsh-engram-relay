/**
 * 会话隔离测试：单会话上下文增强——会话结束即弃，不跨会话残留。
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { NgramHashAddressing } from '../lib/engram/hash.js'
import { EngramStore } from '../lib/engram/store.js'

function makeEnv() {
  const dir = mkdtempSync(join(tmpdir(), 'engram-session-'))
  const store = new EngramStore(dir, new NgramHashAddressing({ seed: 0 }))
  return { dir, store }
}

test('session: engrams are attributed to sessionId', () => {
  const { dir, store } = makeEnv()
  try {
    const a = store.add({ kind: 'fact', label: '会话A', text: '会话A的部署端口是 8080', sessionId: 'sess-a', turn: 1, causes: [], effects: [], importance: 0.8 })
    const b = store.add({ kind: 'decision', label: '会话B', text: '会话B决定采用 PostgreSQL', sessionId: 'sess-b', turn: 1, causes: [], effects: [], importance: 0.8 })
    assert.equal(store.get(a.id)?.sessionId, 'sess-a')
    assert.equal(store.get(b.id)?.sessionId, 'sess-b')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('session: clearSession removes only that session (会话结束即弃)', () => {
  const { dir, store } = makeEnv()
  try {
    store.add({ kind: 'fact', label: '会话A', text: '会话A的部署端口是 8080', sessionId: 'sess-a', turn: 1, causes: [], effects: [], importance: 0.8 })
    store.add({ kind: 'fact', label: '会话A2', text: '会话A的缓存策略是 Redis', sessionId: 'sess-a', turn: 2, causes: [], effects: [], importance: 0.8 })
    store.add({ kind: 'decision', label: '会话B', text: '会话B决定采用 PostgreSQL', sessionId: 'sess-b', turn: 1, causes: [], effects: [], importance: 0.8 })

    const cleared = store.clearSession('sess-a')
    assert.equal(cleared, 2, '应清空会话A的 2 条')
    assert.equal(store.count(), 1, '会话B的记忆应保留')
    // 会话A的记忆不可再查询到
    const hits = store.lookup('会话A的部署端口是 8080')
    assert.equal(hits.length, 0, '已清空会话的记忆不应被唤醒')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('session: clearSession persists across reload', () => {
  const { dir, store } = makeEnv()
  try {
    store.add({ kind: 'fact', label: '会话A', text: '会话A的部署端口是 8080', sessionId: 'sess-a', turn: 1, causes: [], effects: [], importance: 0.8 })
    store.clearSession('sess-a')
    const reloaded = new EngramStore(dir, new NgramHashAddressing({ seed: 0 }))
    assert.equal(reloaded.count(), 0, '清空后重载不应残留')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
