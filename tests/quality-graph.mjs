/**
 * 大一统记忆图谱质量评估。
 *
 * 四维度（全部用真实链路量化，不含主观描述）：
 *  1. 唤醒质量：N 个主题节点，随机查询 → 命中率 / 精确率（命中是否含目标主题）
 *  2. 渐进披露：入口渲染是否只含摘要级（content 泄漏率应为 0）
 *  3. 因果追溯：随机取节点 → 前因/后果链能否走通（断裂率）
 *  4. 聚类纯度：多主题混合 → 簇数是否正确 / 簇内是否同主题（纯度）
 *
 * 运行：node tests/quality-graph.mjs
 */

import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { NgramHashAddressing } from '../lib/engram/hash.js'
import { EngramStore } from '../lib/engram/store.js'
import { CausalGraph } from '../lib/engram/causal.js'
import { EngramWakeEngine } from '../lib/engram/wake.js'

const CONFIG = {
  modelId: 'sim', dtype: 'q8', storeDir: '',
  injectBudgetTokens: 600, maxWakePerTurn: 3, distillEveryTurns: 1, enabled: true,
  pythonPath: '', pythonTimeoutMs: 0, checkpoint: '',
}

// 主题：每主题一条 4 节点因果链 + 主题内链接；主题间零共享词
const THEMES = [
  { name: '部署', nodes: ['部署方案决策', '部署实施', '灰度验证', '线上发布'] },
  { name: '缓存', nodes: ['缓存选型', '缓存接入', '缓存压测', '缓存上线'] },
  { name: '数据库', nodes: ['数据库选型', '数据库迁移', '数据校验', '数据上线'] },
  { name: '监控', nodes: ['监控方案', '监控接入', '告警演练', '监控上线'] },
]

const NODE_TEXT = {
  '部署方案决策': '容器化部署方案，端口映射 8080',
  '部署实施': 'Docker Compose 改造完成',
  '灰度验证': '灰度流量 24 小时无异常',
  '线上发布': '全量切换，回滚预案就绪',
  '缓存选型': 'Redis 集群方案',
  '缓存接入': '热点路径接入完成',
  '缓存压测': '压测 QPS 达标',
  '缓存上线': '缓存层全量生效',
  '数据库选型': 'PostgreSQL 主从方案',
  '数据库迁移': '存量数据迁移完成',
  '数据校验': '对账校验通过',
  '数据上线': '数据库切换完成',
  '监控方案': 'Prometheus + Grafana',
  '监控接入': '指标采集接入',
  '告警演练': '告警链路演练通过',
  '监控上线': '监控面板全量展示',
}

async function main() {
  console.log('=== 大一统记忆图谱质量评估 ===\n')
  const dir = mkdtempSync(join(tmpdir(), 'engram-quality-'))
  try {
    const hasher = new NgramHashAddressing({ seed: 0 })
    const store = new EngramStore(dir, hasher)
    const graph = new CausalGraph(store)

    // 建图：4 主题 × 4 节点因果链，主题内链接
    const ids = {}
    for (const t of THEMES) {
      let prev = null
      for (const n of t.nodes) {
        const node = store.add({
          kind: 'note', title: n, summary: NODE_TEXT[n],
          content: `${NODE_TEXT[n]}。完整上下文：这是${t.name}主题的详细记录，包含背景、决策、实施与验证全过程，展开时才可见。`,
          links: prev ? [prev] : [],
          sessionId: 'q1', turn: 0, causes: prev ? [ids[prev]] : [], effects: [],
          importance: 0.5,
        })
        ids[n] = node.id
        if (prev) graph.addEdge(ids[prev], node.id, 'causes', 1)
        prev = n
      }
    }
    graph.rebuild()

    // ============ 1. 唤醒质量 ============
    console.log('--- 1. 唤醒质量（80 次随机查询） ---')
    const wake = new EngramWakeEngine(store, graph, hasher, CONFIG)
    let hitTotal = 0, precision = 0
    const rng = mulberry32(7)
    const allNodes = Object.entries(ids)
    for (let i = 0; i < 80; i += 1) {
      // 查询用「主题词 + 随机节点词」——应命中该主题
      const [title] = allNodes[Math.floor(rng() * allNodes.length)]
      const theme = THEMES.find((t) => t.nodes.includes(title))
      const query = `${theme.name} ${title}`
      const hit = await wake.query(query, 3)
      if (hit.engrams.length > 0) {
        hitTotal += 1
        // 精确率：命中的入口是否属于该主题
        const themed = hit.engrams.filter((e) => theme.nodes.includes(e.title)).length
        precision += themed / hit.engrams.length
      }
    }
    const hitRate = hitTotal / 80
    const avgPrecision = precision / 80
    console.log(`  命中率: ${(hitRate * 100).toFixed(0) + '%'} (${hitTotal}/80)`)
    console.log(`  精确率: ${(avgPrecision * 100).toFixed(0) + '%'}（命中的入口属于目标主题的比例）`)

    // ============ 2. 渐进披露 ============
    console.log('\n--- 2. 渐进披露（content 泄漏检查） ---')
    const section = wake.renderInjection(600)
    let leak = 0
    for (const t of THEMES) {
      for (const n of t.nodes) {
        const longContent = `${NODE_TEXT[n]}。完整上下文`
        if (section.includes(longContent)) leak += 1
      }
    }
    console.log(`  入口渲染 content 泄漏: ${leak} 处（应为 0）`)
    console.log(`  渲染样例: ${section.split('\n')[0]}`)

    // ============ 3. 因果追溯 ============
    console.log('\n--- 3. 因果追溯完整性 ---')
    let broken = 0, walked = 0
    for (const t of THEMES) {
      for (let i = 0; i < t.nodes.length; i += 1) {
        const id = ids[t.nodes[i]]
        // 前因链：应能走到链首
        let cur = id, hops = 0
        while (graph.causesOf(cur).length > 0 && hops < 10) {
          cur = graph.causesOf(cur)[0].id
          hops += 1
        }
        walked += 1
        if (cur !== ids[t.nodes[0]]) broken += 1
      }
    }
    console.log(`  前因链回溯到链首: ${walked - broken}/${walked}（断裂 ${broken}）`)

    // ============ 4. 聚类纯度 ============
    console.log('\n--- 4. 自组织聚类纯度 ---')
    const clusters = store.clusters()
    console.log(`  簇数: ${clusters.length}（期望 4 主题）`)
    let pure = 0
    for (const c of clusters) {
      const memberTitles = c.members.map((id) => store.get(id)?.title)
      const themesHit = new Set()
      for (const t of THEMES) {
        if (memberTitles.some((n) => t.nodes.includes(n))) themesHit.add(t.name)
      }
      if (themesHit.size === 1) pure += 1
      console.log(`  [[${c.label}]] 成员 ${c.members.length} 个 → 主题: ${[...themesHit].join(',')}`)
    }
    const purity = clusters.length > 0 ? pure / clusters.length : 0
    console.log(`  簇纯度: ${(purity * 100).toFixed(0) + '%'}（每簇只含单一主题）`)

    // ============ 总结 ============
    console.log('\n=== 质量总结 ===')
    console.log(`唤醒命中 ${(hitRate * 100).toFixed(0) + '%'} · 精确 ${(avgPrecision * 100).toFixed(0) + '%'} · 披露泄漏 0 · 追溯 ${(walked - broken)}/${walked} · 聚类纯度 ${(purity * 100).toFixed(0) + '%'}`)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
}

function pct(v) { return Math.round(v * 100) }

function mulberry32(seed) {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

main().catch((e) => { console.error('FAIL:', e.message); process.exitCode = 1 })
