/**
 * v0.3 分层迁移测试：session 层删除——旧 session 数据归一化为 project，
 * 会话结束不再清理（跨会话沉淀），sessionId 仅保留来源记录。
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { NgramHashAddressing } from '../lib/engram/hash.js'
import { EngramStore } from '../lib/engram/store.js'

function makeEnv() {
  const dir = mkdtempSync(join(tmpdir(), 'engram-session-'))
  const store = new EngramStore(dir, new NgramHashAddressing({ seed: 0 }))
  return { dir, store }
}

test('session: engrams are attributed to sessionId (来源记录保留)', () => {
  const { dir, store } = makeEnv()
  try {
    const a = store.add({ kind: 'fact', label: '会话A', text: '会话A的部署端口是 8080', sessionId: 'sess-a', turn: 1, causes: [], effects: [], importance: 0.8 })
    const b = store.add({ kind: 'decision', label: '会话B', text: '会话B决定采用 PostgreSQL', sessionId: 'sess-b', turn: 1, causes: [], effects: [], importance: 0.8 })
    assert.equal(store.get(a.id)?.sessionId, 'sess-a')
    assert.equal(store.get(b.id)?.sessionId, 'sess-b')
    // 默认层 = project（session 层已删）
    assert.equal(store.get(a.id)?.layer, 'project')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('v0.3: 旧 session 层数据加载归一化为 project（不丢记忆，不再清理）', () => {
  const dir = mkdtempSync(join(tmpdir(), 'engram-session-'))
  try {
    // 手工构造旧格式数据：layer='session'
    writeFileSync(join(dir, 'engrams.jsonl'), [
      JSON.stringify({ id: 'old-1', kind: 'fact', layer: 'session', projectId: null, title: '旧会话记忆', summary: '迁移测试', content: '', links: [], causes: [], effects: [], sessionId: 'sess-a', turn: 1, importance: 0.5, hits: 0, createdAt: 1, reinforces: [1], slots: [], status: 'confirmed' }),
    ].join('\n'), 'utf8')
    const store = new EngramStore(dir, new NgramHashAddressing({ seed: 0 }))
    const node = store.all()[0]
    assert.equal(node?.layer, 'project', '旧 session 节点应迁移为 project')
    assert.equal(store.count(), 1, '迁移不丢记忆')
    // 会话结束不再有任何清理 API（clearSession 已删除）——跨会话沉淀
    assert.equal(typeof store.clearSession, 'undefined')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
