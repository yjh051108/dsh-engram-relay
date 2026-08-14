/**
 * 教训通道测试（v0.6）：tags 含「教训:」的节点在自动唤醒（auto）时用更低
 * 阈值（lessonMinScore）独立补位——同类操作时踩坑提醒必达。
 *
 * 覆盖：教训第 2 名补位（低分但 ≥ lessonMinScore）/ 低于 lessonMin 不补 /
 *       非教训不补 / lessonMinScore=0 关闭 / 渲染 ⚠️教训 标记。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { EngramStore } from '../lib/engram/store.js'
import { CausalGraph } from '../lib/engram/causal.js'
import { NgramHashAddressing } from '../lib/engram/hash.js'
import { EngramWakeEngine } from '../lib/engram/wake.js'

function tempDir() {
  return mkdtempSync(join(tmpdir(), 'engram-lesson-'))
}

function makeStore(dir) {
  return new EngramStore(dir, new NgramHashAddressing({ seed: 0 }))
}

const BASE_CONFIG = {
  injectBudgetTokens: 600, maxWakePerTurn: 3, distillEveryTurns: 0, enabled: true,
  modelId: '', dtype: 'bfloat16', storeDir: '', pythonPath: 'python', pythonTimeoutMs: 10000,
  checkpoint: '', embedModel: '', recencyWeight: 0, wakeSampleLog: false,
  lessonMinScore: 0.42, lessonBudgetTokens: 60,
}

/** 主题相关的种子节点：查询文本与它们哈希重叠才能进候选。 */
const COMMON = '缓存命中率下降的排查步骤：清缓存、查版本链、看注入预算'

/** fake embedder：按节点 title 给分（教训节点给 lessonScore，其余给 mainScore）。 */
function scorerWith(mainScore, lessonScore) {
  return {
    embedder: async (_q, cands) => new Map(cands.map((e) => {
      const isL = (e.tags ?? []).some((t) => typeof t === 'string' && t.startsWith('教训:'))
      return [e.id, isL ? lessonScore : mainScore]
    })),
  }
}

/** 建库：普通节点 + 可选教训节点（同主题，哈希可命中）。 */
function buildStore(dir, { withLesson = true, lessonTags = ['教训:代码'] } = {}) {
  const store = makeStore(dir)
  const main = store.add({ kind: 'fact', layer: 'project', projectId: '/w', title: '缓存排查流程', summary: COMMON, content: '', links: [], sessionId: null, turn: 1, causes: [], effects: [], importance: 0.7 })
  let lesson = null
  if (withLesson) {
    lesson = store.add({ kind: 'note', layer: 'project', projectId: '/w', title: '缓存坑：先查版本链', summary: '缓存不命中先查 superseded 版本链，别先清库（清库丢激活）' + COMMON.slice(0, 20), content: '', links: [], sessionId: null, turn: 2, causes: [], effects: [], importance: 0.6, tags: lessonTags })
  }
  return { store, main, lesson }
}

test('lesson: 教训第 2 名补位——低分（≥lessonMinScore）教训在 auto 下也注入', async () => {
  const dir = tempDir()
  try {
    const { store, main, lesson } = buildStore(dir)
    const graph = new CausalGraph(store)
    const wake = new EngramWakeEngine(store, graph, new NgramHashAddressing({ seed: 0 }), BASE_CONFIG, scorerWith(0.48, 0.45))
    const hit = await wake.query('缓存命中率下降 排查', 1, { cwd: '/w' }, { auto: true })
    const titles = hit.engrams.map((e) => e.title)
    assert.equal(hit.engrams.length, 2, `主 top-1 + 教训补位（实际: ${titles.join(', ')}）`)
    assert.ok(titles.includes('缓存排查流程'), '主 top-1 必选')
    assert.ok(titles.includes('缓存坑：先查版本链'), '教训节点补位注入')
    // 渲染：⚠️教训 标记 + 教训 tag
    const rendered = wake.renderInjection(600)
    assert.ok(rendered.includes('⚠️教训[[缓存坑：先查版本链]]'), '渲染带 ⚠️教训 标记')
    assert.ok(rendered.includes('(教训:代码)'), '渲染带教训分类 tag')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('lesson: 低于 lessonMinScore 不补位——教训也不是什么都提醒', async () => {
  const dir = tempDir()
  try {
    const { store, lesson } = buildStore(dir)
    const graph = new CausalGraph(store)
    const wake = new EngramWakeEngine(store, graph, new NgramHashAddressing({ seed: 0 }), BASE_CONFIG, scorerWith(0.48, 0.30))
    const hit = await wake.query('缓存命中率下降 排查', 1, { cwd: '/w' }, { auto: true })
    assert.equal(hit.engrams.length, 1, `低于 lessonMinScore 的教训不注入（实际: ${hit.engrams.map((e) => e.title).join(', ')}）`)
    assert.ok(!hit.engrams.some((e) => e.id === lesson.id))
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('lesson: 非教训节点低分不补位——通道只对「教训:」标签生效', async () => {
  const dir = tempDir()
  try {
    // 低分节点是普通记忆（无教训 tag）
    const { store, main, lesson } = buildStore(dir, { withLesson: false })
    const extra = store.add({ kind: 'fact', layer: 'project', projectId: '/w', title: '缓存小技巧', summary: '排查缓存的补充技巧' + COMMON.slice(0, 16), content: '', links: [], sessionId: null, turn: 3, causes: [], effects: [], importance: 0.5 })
    const graph = new CausalGraph(store)
    const wake = new EngramWakeEngine(store, graph, new NgramHashAddressing({ seed: 0 }), BASE_CONFIG, scorerWith(0.48, 0.45))
    const hit = await wake.query('缓存命中率下降 排查', 1, { cwd: '/w' }, { auto: true })
    assert.equal(hit.engrams.length, 1, `非教训低分节点不补位（实际: ${hit.engrams.map((e) => e.title).join(', ')}）`)
    assert.ok(!hit.engrams.some((e) => e.id === extra.id))
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('lesson: lessonMinScore=0 关闭通道', async () => {
  const dir = tempDir()
  try {
    const { store, lesson } = buildStore(dir)
    const graph = new CausalGraph(store)
    const wake = new EngramWakeEngine(store, graph, new NgramHashAddressing({ seed: 0 }), { ...BASE_CONFIG, lessonMinScore: 0 }, scorerWith(0.48, 0.45))
    const hit = await wake.query('缓存命中率下降 排查', 1, { cwd: '/w' }, { auto: true })
    assert.equal(hit.engrams.length, 1, '关闭通道后教训不补位')
    assert.ok(!hit.engrams.some((e) => e.id === lesson.id))
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('lesson: 手动 query（非 auto）不启用教训通道——显式检索按正常排序', async () => {
  const dir = tempDir()
  try {
    const { store, lesson } = buildStore(dir)
    const graph = new CausalGraph(store)
    const wake = new EngramWakeEngine(store, graph, new NgramHashAddressing({ seed: 0 }), BASE_CONFIG, scorerWith(0.48, 0.45))
    const hit = await wake.query('缓存命中率下降 排查', 1, { cwd: '/w' })
    assert.equal(hit.engrams.length, 1, '非 auto 只取 top-1（教训通道仅自动唤醒启用）')
    assert.ok(!hit.engrams.some((e) => e.id === lesson.id))
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
