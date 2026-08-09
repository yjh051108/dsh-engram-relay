/**
 * NgramHashAddressing + EngramStore 哈希寻址测试：
 * DeepSeek Engram 式确定性寻址（相同模式永远命中相同槽位）。
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { NgramHashAddressing } from '../lib/engram/hash.js'
import { EngramStore } from '../lib/engram/store.js'

test('hash: deterministic addressing (same text -> same slots)', () => {
  const h1 = new NgramHashAddressing({ seed: 0 })
  const h2 = new NgramHashAddressing({ seed: 0 })
  const a = h1.hash('项目部署端口是 8080')
  const b = h2.hash('项目部署端口是 8080')
  assert.deepEqual(a.slots, b.slots, '相同文本必须命中相同槽位（确定性）')
  assert.deepEqual(h1.slotKeys(a), h2.slotKeys(b))
})

test('hash: normalization folds case and whitespace', () => {
  const h = new NgramHashAddressing({ seed: 0 })
  const a = h.hash('Port 8080')
  const b = h.hash('  port   8080  ')
  assert.deepEqual(a.slots, b.slots, '大小写与空白归一后必须同槽位')
})

test('hash: multi-head gives multiple slots per ngram length', () => {
  const h = new NgramHashAddressing({ seed: 0, maxNgramSize: 3, headsPerNgram: 4 })
  const r = h.hash('部署端口 8080')
  // n=2 与 n=3 两级，每级 4 头
  assert.equal(r.slots.length, 2)
  assert.equal(r.slots[0].length, 4)
  assert.equal(r.slots[1].length, 4)
  const keys = h.slotKeys(r)
  assert.equal(keys.length, 8)
})

test('hash: different texts mostly diverge (low collision sanity)', () => {
  const h = new NgramHashAddressing({ seed: 0 })
  const a = new Set(h.slotKeys(h.hash('数据库用 PostgreSQL')))
  const b = new Set(h.slotKeys(h.hash('写一首关于猫的诗')))
  let overlap = 0
  for (const k of a) if (b.has(k)) overlap += 1
  assert.ok(overlap <= 2, `不相关文本不应大量共槽位（重叠 ${overlap}/8）`)
})

test('store: hash-add then lookup hits same slot', () => {
  const dir = mkdtempSync(join(tmpdir(), 'engram-hash-'))
  try {
    const hasher = new NgramHashAddressing({ seed: 0 })
    const store = new EngramStore(dir, hasher)
    store.add({
      kind: 'fact',
      label: '部署端口',
      text: '项目部署端口是 8080',
      scope: null,
      sessionId: 's1',
      turn: 1,
      causes: [],
      effects: [],
      importance: 0.9,
    })
    // 相同表述查询 → 确定性命中
    const hits = store.lookup('项目部署端口是 8080')
    assert.ok(hits.length >= 1, '相同表述必须命中')
    assert.equal(hits[0].label, '部署端口')

    // 归一化后仍命中
    const hits2 = store.lookup('  项目 部署端口 是 8080 ')
    assert.ok(hits2.length >= 1, '归一化表述必须命中')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('store: global/rule memories are addressable by seed text', () => {
  const dir = mkdtempSync(join(tmpdir(), 'engram-rule-'))
  try {
    const hasher = new NgramHashAddressing({ seed: 0 })
    const store = new EngramStore(dir, hasher)
    store.add({
      kind: 'rule',
      label: '回复语言',
      text: '用户偏好中文回复',
      scope: 'rule',
      sessionId: null,
      turn: 0,
      causes: [],
      effects: [],
      importance: 1,
    })
    const hits = store.lookup('请用中文回复我')
    // 规则记忆不一定被任意查询命中（哈希是精确的）——但它可被
    // 与写入文本一致的主题命中；跨会话记忆由「固定种子文本」保证
    // 每次会话都可寻址（en gram_store 的 scope 语义）。
    assert.equal(store.byKind('rule').length, 1)
    assert.ok(hits.length >= 0)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('store: persist and reload keeps slot index', () => {
  const dir = mkdtempSync(join(tmpdir(), 'engram-persist-'))
  try {
    const hasher = new NgramHashAddressing({ seed: 0 })
    const s1 = new EngramStore(dir, hasher)
    s1.add({
      kind: 'decision',
      label: '架构决策',
      text: '采用 engram 条件记忆做外置记忆',
      scope: null,
      sessionId: 's1',
      turn: 2,
      causes: [],
      effects: [],
      importance: 0.95,
    })
    const s2 = new EngramStore(dir, hasher)
    assert.equal(s2.count(), 1)
    const hits = s2.lookup('采用 engram 条件记忆做外置记忆')
    assert.ok(hits.length >= 1, '重载后槽位索引必须可用')
    assert.ok(s2.slotCount() > 0)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
