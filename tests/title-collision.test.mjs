/**
 * 标题多值索引测试（P0 地基：同名节点不再互相顶掉）。
 *
 * 覆盖：同名 add/解析消歧/remove/update 改标题/持久化重载保留多值/
 * byTitles 盘点。Node ≥22.6 直接 import TS（type stripping）。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { EngramStore } from '../lib/engram/store.js'

function tempDir() {
  return mkdtempSync(join(tmpdir(), 'engram-title-'))
}

test('title: 同名节点共存——byTitle 最近优先，byTitles 全量', () => {
  const dir = tempDir()
  try {
    const store = new EngramStore(dir)
    const a = store.add({ kind: 'fact', title: '同名记忆', summary: '第一条', importance: 0.5 })
    const b = store.add({ kind: 'fact', title: '同名记忆', summary: '第二条', importance: 0.8 })
    assert.notEqual(a.id, b.id)

    // byTitle 消歧：最近写入（b）优先——不丢、可解析
    assert.equal(store.byTitle('同名记忆')?.id, b.id)
    assert.equal(store.byTitle('同名记忆')?.summary, '第二条')
    // byTitles 全量盘点
    assert.deepEqual(store.byTitles('同名记忆').map((e) => e.id).sort(), [a.id, b.id].sort())
    // 不存在的标题
    assert.equal(store.byTitle('不存在'), undefined)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('title: remove 只摘自身，不影响同名其他节点', () => {
  const dir = tempDir()
  try {
    const store = new EngramStore(dir)
    const a = store.add({ kind: 'fact', title: '同名X', summary: 'a' })
    const b = store.add({ kind: 'fact', title: '同名X', summary: 'b' })
    assert.equal(store.remove(a.id), true)
    // b 仍可解析
    assert.equal(store.byTitle('同名X')?.id, b.id)
    assert.equal(store.byTitles('同名X').length, 1)
    // 全部移除后索引清空
    assert.equal(store.remove(b.id), true)
    assert.equal(store.byTitle('同名X'), undefined)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('title: update 改标题——旧标题摘除、新标题登记', () => {
  const dir = tempDir()
  try {
    const store = new EngramStore(dir)
    const a = store.add({ kind: 'fact', title: '旧标题', summary: 'a' })
    store.add({ kind: 'fact', title: '旧标题', summary: 'b' }) // 同名另一条
    store.update(a.id, { title: '新标题' })
    // 旧标题仍能解析到另一条
    assert.equal(store.byTitle('旧标题')?.summary, 'b')
    // 新标题解析到 a
    assert.equal(store.byTitle('新标题')?.id, a.id)
    assert.equal(store.byTitles('旧标题').length, 1)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('title: 持久化重载后多值保留（不再互相覆盖丢失）', () => {
  const dir = tempDir()
  try {
    const store = new EngramStore(dir)
    const a = store.add({ kind: 'fact', title: '持久化同名', summary: '一' })
    const b = store.add({ kind: 'fact', title: '持久化同名', summary: '二' })
    const reloaded = new EngramStore(dir)
    assert.equal(reloaded.byTitles('持久化同名').length, 2)
    assert.equal(reloaded.byTitle('持久化同名')?.id, b.id)
    // 链接解析（links 双向引用同名标题 → 最近节点，不报错不丢）
    assert.equal(reloaded.byTitle('持久化同名')?.summary, '二')
    assert.ok(reloaded.get(a.id))
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
