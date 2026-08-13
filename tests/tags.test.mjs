/**
 * 自由标签测试（v0.4：一节点多标签，自由分类，命名空间约定）。
 *
 * 覆盖：旧数据迁移默认标签 / 多标签保存 / 缺省自动生成 / update 覆盖。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { EngramStore, defaultTags } from '../lib/engram/store.js'
import { NgramHashAddressing } from '../lib/engram/hash.js'

function tempDir() {
  return mkdtempSync(join(tmpdir(), 'engram-tags-'))
}

function makeStore(dir) {
  return new EngramStore(dir, new NgramHashAddressing({ seed: 0 }))
}

test('tags: 默认标签生成——project→项目:目录名 / global→全局', () => {
  assert.deepEqual(defaultTags('project', 'D:/x/y/proj-a'), ['项目:proj-a'])
  assert.deepEqual(defaultTags('project', 'C:\\dev\\my-proj'), ['项目:my-proj'])
  assert.deepEqual(defaultTags('global', null), ['全局'])
  assert.deepEqual(defaultTags('project', null), ['全局'])
})

test('tags: 多标签保存 + 缺省自动生成 + update 覆盖', () => {
  const dir = tempDir()
  try {
    const store = makeStore(dir)
    // 显式多标签（跨命名空间：项目 + 教训）
    const e = store.add({
      kind: 'fact', layer: 'project', projectId: '/w/engram', title: '标签测试',
      summary: '多标签自由分类', content: '', links: [], sessionId: null, turn: 1,
      causes: [], effects: [], importance: 0.6,
      tags: ['项目:engram', '教训:代码'],
    })
    assert.deepEqual(store.get(e.id).tags, ['项目:engram', '教训:代码'], '多标签保存且去重')

    // 缺省：project 层自动生成 项目:<目录名>
    const auto = store.add({
      kind: 'note', layer: 'project', projectId: '/w/other-proj', title: '自动标签',
      summary: '缺省生成', content: '', links: [], sessionId: null, turn: 1,
      causes: [], effects: [], importance: 0.6,
    })
    assert.deepEqual(store.get(auto.id).tags, ['项目:other-proj'])

    // update 覆盖 tags
    store.update(e.id, { tags: ['教训:思想', '全局'] })
    assert.deepEqual(store.get(e.id).tags, ['教训:思想', '全局'])

    // 持久化保留
    const reloaded = makeStore(dir)
    assert.deepEqual(reloaded.get(e.id).tags, ['教训:思想', '全局'])
    assert.deepEqual(reloaded.get(auto.id).tags, ['项目:other-proj'])
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('tags: 旧数据无 tags → 加载迁移默认标签', () => {
  const dir = tempDir()
  try {
    writeFileSync(join(dir, 'engrams.jsonl'),
      JSON.stringify({ id: 'old-1', kind: 'fact', layer: 'project', projectId: '/w/legacy', title: '旧记忆', summary: 's', content: '', links: [], causes: [], effects: [], sessionId: null, turn: 1, importance: 0.5, hits: 0, createdAt: 1, reinforces: [1], slots: [], status: 'confirmed' }) + '\n' +
      JSON.stringify({ id: 'old-2', kind: 'fact', layer: 'global', projectId: null, title: '旧全局', summary: 's2', content: '', links: [], causes: [], effects: [], sessionId: null, turn: 1, importance: 0.5, hits: 0, createdAt: 1, reinforces: [1], slots: [], status: 'confirmed' }) + '\n',
      'utf8')
    const store = makeStore(dir)
    assert.deepEqual(store.get('old-1').tags, ['项目:legacy'], '旧 project 迁移项目标签')
    assert.deepEqual(store.get('old-2').tags, ['全局'], '旧 global 迁移全局标签')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
