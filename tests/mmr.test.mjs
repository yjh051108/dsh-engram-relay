/**
 * MMR 多样性测试（建模命题 1 缺口：top-K 同主题发散 → 簇间多样性）。
 *
 * 覆盖：同主题多条高分 + 异主题中分 → MMR 选异主题补位；
 *      第一名不受惩罚（与原 top-1 一致）。
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
  return mkdtempSync(join(tmpdir(), 'engram-mmr-'))
}

function makeStore(dir) {
  return new EngramStore(dir, new NgramHashAddressing({ seed: 0 }))
}

const WAKE_CONFIG = {
  injectBudgetTokens: 600, maxWakePerTurn: 3, distillEveryTurns: 0, enabled: true,
  modelId: '', dtype: 'bfloat16', storeDir: '', pythonPath: 'python', pythonTimeoutMs: 10000,
  checkpoint: '', embedModel: '', recencyWeight: 0, wakeSampleLog: false,
}

test('mmr: 同主题多条高分 → 只取最高分者，异主题中分补位', async () => {
  const dir = tempDir()
  try {
    const store = makeStore(dir)
    // 同主题 A1/A2（文本高度重叠 → 槽位 Jaccard 高）+ 异主题 C
    const a1 = store.add({ kind: 'fact', layer: 'project', projectId: '/w', title: '缓存方案A', summary: '缓存命中率下降的排查步骤：清缓存、查版本链、看注入预算', content: '', links: [], sessionId: null, turn: 1, causes: [], effects: [], importance: 0.9 })
    const a2 = store.add({ kind: 'fact', layer: 'project', projectId: '/w', title: '缓存方案B', summary: '缓存命中率下降的排查步骤：清缓存、查版本链、看注入预算（补充）', content: '', links: [], sessionId: null, turn: 2, causes: [], effects: [], importance: 0.88 })
    const c = store.add({ kind: 'fact', layer: 'project', projectId: '/w', title: '端口配置', summary: '部署端口 8080 映射 nginx 反代', content: '', links: [], sessionId: null, turn: 3, causes: [], effects: [], importance: 0.8 })
    const graph = new CausalGraph(store)
    const wake = new EngramWakeEngine(store, graph, new NgramHashAddressing({ seed: 0 }), WAKE_CONFIG, null)

    // 查询命中三者的主题（含缓存与端口词）
    const hit = await wake.query('缓存命中率下降 端口配置', 2, { cwd: '/w' })
    const titles = hit.engrams.map((e) => e.title)
    assert.equal(hit.engrams.length, 2)
    // 最高分 a1 必选
    assert.ok(titles.includes('缓存方案A'), 'top-1 必选')
    // MMR：a2 与 a1 槽位高相似被降权 → 选异主题 C（而不是 a2）
    assert.ok(titles.includes('端口配置'), `MMR 选异主题补位（实际: ${titles.join(', ')}）`)
    assert.ok(!titles.includes('缓存方案B'), '同主题第二条被多样性惩罚')
    void a2
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('mmr: 无相似候选时退化为纯分数排序（top-1 不变）', async () => {
  const dir = tempDir()
  try {
    const store = makeStore(dir)
    const x = store.add({ kind: 'fact', layer: 'project', projectId: '/w', title: '主题甲', summary: '甲 的内容 8080', content: '', links: [], sessionId: null, turn: 1, causes: [], effects: [], importance: 0.9 })
    const y = store.add({ kind: 'fact', layer: 'project', projectId: '/w', title: '主题乙', summary: '乙 的内容 9090', content: '', links: [], sessionId: null, turn: 2, causes: [], effects: [], importance: 0.7 })
    const z = store.add({ kind: 'fact', layer: 'project', projectId: '/w', title: '主题丙', summary: '丙 的内容 7070', content: '', links: [], sessionId: null, turn: 3, causes: [], effects: [], importance: 0.6 })
    const graph = new CausalGraph(store)
    const wake = new EngramWakeEngine(store, graph, new NgramHashAddressing({ seed: 0 }), WAKE_CONFIG, null)
    const hit = await wake.query('甲 乙 丙 内容', 3, { cwd: '/w' })
    const titles = hit.engrams.map((e) => e.title)
    // 三主题互不相似 → 按分数排 x, y, z
    assert.equal(titles[0], '主题甲')
    assert.ok(titles.includes('主题乙') && titles.includes('主题丙'))
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
