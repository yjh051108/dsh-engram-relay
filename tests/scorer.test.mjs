/**
 * SemanticScorer 测试（v0.5：纯算法语义——零 embedding 模型）。
 *
 * 覆盖：词汇通道（n-gram Jaccard + 词频）/ 图语义通道（边传播）/
 * 融合分数 / 阈值语义（0.6 对齐）。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { EngramStore } from '../lib/engram/store.js'
import { CausalGraph } from '../lib/engram/causal.js'
import { NgramHashAddressing } from '../lib/engram/hash.js'
import { SemanticScorer } from '../lib/engram/semantic-scorer.js'

function tempDir() {
  return mkdtempSync(join(tmpdir(), 'engram-scorer-'))
}

function makeStore(dir) {
  return new EngramStore(dir, new NgramHashAddressing({ seed: 0 }))
}

test('scorer: 词汇通道——n-gram 重叠高的候选分数更高', () => {
  const dir = tempDir()
  try {
    const store = makeStore(dir)
    const a = store.add({ kind: 'fact', layer: 'project', projectId: '/w', title: '缓存命中率', summary: '缓存命中率下降的排查步骤：清缓存、查版本链', content: '', links: [], sessionId: null, turn: 1, causes: [], effects: [], importance: 0.5 })
    const b = store.add({ kind: 'fact', layer: 'project', projectId: '/w', title: '部署端口', summary: '部署端口 8080 映射 nginx 反代', content: '', links: [], sessionId: null, turn: 2, causes: [], effects: [], importance: 0.5 })
    const scorer = new SemanticScorer(store)
    const scores = scorer.score('缓存命中率下降怎么办', [store.get(a.id), store.get(b.id)])
    assert.ok(scores.get(a.id).lexical > scores.get(b.id).lexical, '词汇重叠高的候选 lexical 更高')
    assert.ok(scores.get(a.id).score > scores.get(b.id).score, '融合分数一致')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('scorer: 图语义通道——因果邻居获得图分数加成', () => {
  const dir = tempDir()
  try {
    const store = makeStore(dir)
    const root = store.add({ kind: 'fact', layer: 'project', projectId: '/w', title: '磁盘满', summary: '磁盘满了导致服务挂', content: '', links: [], sessionId: null, turn: 1, causes: [], effects: [], importance: 0.5 })
    const child = store.add({ kind: 'fact', layer: 'project', projectId: '/w', title: '服务重启', summary: '服务挂后重启恢复', content: '', links: [], sessionId: null, turn: 2, causes: [root.id], effects: [], importance: 0.5 })
    const graph = new CausalGraph(store)
    graph.rebuild()
    const scorer = new SemanticScorer(store)
    // 查询命中 root（磁盘满）→ child 是 1 跳因果邻居 → 图分数 > 0
    const scores = scorer.score('磁盘满了', [store.get(child.id)])
    assert.ok(scores.get(child.id).graph > 0, '因果邻居获得图语义分数')
    assert.ok(scores.get(child.id).graph >= 0.7, '1 跳邻居 graph=0.7')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('scorer: 语义桥——PCA 通道桥接跨词共现（缓存↔cache，纯库内收敛）', () => {
  const dir = tempDir()
  try {
    const store = makeStore(dir)
    // 记忆 A 同时含「缓存」与「cache」（桥的锚点）；记忆 B 只含「缓存」
    const a = store.add({ kind: 'fact', layer: 'project', projectId: '/w', title: '缓存方案', summary: '缓存 cache 命中率下降的排查：清缓存、查版本链', content: '', links: [], sessionId: null, turn: 1, causes: [], effects: [], importance: 0.5 })
    const b = store.add({ kind: 'fact', layer: 'project', projectId: '/w', title: '缓存优化', summary: '缓存淘汰策略 LRU 与 TTL 配置', content: '', links: [], sessionId: null, turn: 2, causes: [], effects: [], importance: 0.5 })
    store.add({ kind: 'fact', layer: 'project', projectId: '/w', title: '无关主题', summary: '天气很好适合散步买菜做饭', content: '', links: [], sessionId: null, turn: 3, causes: [], effects: [], importance: 0.5 })
    const scorer = new SemanticScorer(store)
    // 查询用「cache」（与 B 无词汇重叠）——桥应让共现通道把 B 拉高
    const scores = scorer.score('cache 命中率', [store.get(a.id), store.get(b.id), store.all().find((e) => e.title === '无关主题')])
    const coocB = scores.get(b.id).cooc
    const coocA = scores.get(a.id).cooc
    assert.ok(coocA > 0.05, `锚点记忆 A 的共现分数非零 (${coocA})`)
    assert.ok(coocB > 0.05, `桥接：B 经「缓存↔cache」共现获得统计语义分 (${coocB})`)
    assert.ok(scores.get(b.id).score > 0, 'B 的融合分数非零（语义桥生效）')
    // 无关记忆的共现分应低于桥接记忆
    const coocU = scores.get(store.all().find((e) => e.title === '无关主题').id).cooc
    assert.ok(coocU < coocB, `无关记忆共现分应低 (${coocU} < ${coocB})`)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('scorer: 阈值语义——高相似候选 score ≥ 0.6（织网/查重门槛沿用）', () => {
  const dir = tempDir()
  try {
    const store = makeStore(dir)
    const a = store.add({ kind: 'fact', layer: 'project', projectId: '/w', title: '自动织网', summary: '自动织网用三维度加权选高置信邻居建双向链接', content: '', links: [], sessionId: null, turn: 1, causes: [], effects: [], importance: 0.5 })
    const b = store.add({ kind: 'fact', layer: 'project', projectId: '/w', title: '完全无关主题', summary: '天气很好适合出去散步买菜做饭', content: '', links: [], sessionId: null, turn: 2, causes: [], effects: [], importance: 0.5 })
    const scorer = new SemanticScorer(store)
    const scores = scorer.score('自动织网建链接', [store.get(a.id), store.get(b.id)])
    assert.ok(scores.get(a.id).score >= 0.6, `同主题候选过织网门槛 (${scores.get(a.id).score})`)
    assert.ok(scores.get(b.id).score < 0.6, '无关候选低于门槛')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
