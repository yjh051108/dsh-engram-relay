/**
 * 大一统记忆图谱端到端全链路验证：
 * 写入 → 入口唤醒 → 渐进展开 → 因果双向追溯。
 *
 * 模拟真实 DSH 会话场景（不依赖模型，纯系统链路）：
 *  1. 写入多个记忆节点（含因果边 + 双向链接）
 *  2. 新上下文 → 主动唤醒入口（摘要级）
 *  3. engram_open 展开一个入口（正文 + 链接 + 因果邻接）
 *  4. 从展开的节点双向追溯（前因链 / 后果链）
 *
 * 运行：node tests/e2e-graph.mjs
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

async function main() {
  console.log('=== 大一统记忆图谱端到端全链路 ===')
  const dir = mkdtempSync(join(tmpdir(), 'engram-e2e-graph-'))
  try {
    const hasher = new NgramHashAddressing({ seed: 0 })
    const store = new EngramStore(dir, hasher)
    const graph = new CausalGraph(store)

    // 1. 写入：一个「决策→实施→上线」因果链 + 双向链接
    const root = store.add({
      kind: 'decision', title: '部署方案决策', summary: '决定采用 Docker 容器化部署，端口 8080',
      content: '评审后确定：Docker Compose 部署，nginx 反代，回滚用上一镜像。',
      links: ['部署实施'], sessionId: 'sess-1', turn: 1, causes: [], effects: [], importance: 0.9,
    })
    const mid = store.add({
      kind: 'event', title: '部署实施', summary: '完成容器化改造，CI 出镜像',
      content: 'Dockerfile 多阶段构建，CI 推送到私有 registry，测试环境验证通过。',
      links: ['部署方案决策', '线上验证'], sessionId: 'sess-1', turn: 2,
      causes: [root.id], effects: [], importance: 0.8,
    })
    const end = store.add({
      kind: 'note', title: '线上验证', summary: '灰度验证通过，正式上线',
      content: '灰度 24 小时无异常，流量切全量，监控告警未触发。',
      links: ['部署实施'], sessionId: 'sess-1', turn: 3,
      causes: [mid.id], effects: [], importance: 0.7,
    })
    graph.rebuild()
    console.log(`✓ 写入 3 节点（因果链 决策→实施→上线 + 双向链接）`)

    // 2. 主动唤醒入口：新上下文「容器化部署遇到问题」
    const wake = new EngramWakeEngine(store, graph, hasher, CONFIG)
    const hit = await wake.query('容器化部署的问题排查', 3)
    console.log(`✓ 入口唤醒: reason=${hit.reason}, 命中 ${hit.engrams.length} 入口`)
    for (const e of hit.engrams) {
      console.log(`   [[${e.title}]]: ${e.summary}`)
    }
    if (hit.engrams.length === 0) throw new Error('入口唤醒失败')

    // 3. 渐进展开：打开「部署实施」看详情
    const node = store.byTitle('部署实施')
    if (!node) throw new Error('标题解析失败')
    store.touch(node.id)
    const causes = graph.causesOf(node.id).map((n) => n.title)
    const effects = graph.effectsOf(node.id).map((n) => n.title)
    const linked = store.getMany(node.links.map((t) => store.byTitle(t)?.id ?? '').filter(Boolean)).map((n) => n.title)
    console.log(`✓ 渐进展开 [[部署实施]]:`)
    console.log(`   正文: ${node.content.slice(0, 40)}…`)
    console.log(`   前因: ${causes.join('、')} | 后果: ${effects.join('、')} | 链接: ${linked.join('、')}`)
    if (node.content === '' || causes.length === 0 || effects.length === 0) throw new Error('展开失败（因果邻接不完整）')

    // 4. 因果双向追溯
    const fromEnd = graph.causesOf(end.id).map((n) => n.title)  // 后果 → 前因
    const fromRoot = graph.effectsOf(root.id).map((n) => n.title)  // 前因 → 后果
    console.log(`✓ 因果追溯: 上线 ← ${fromEnd.join(' ← ')}（回溯）| 决策 → ${fromRoot.join(' → ')}（前瞻）`)
    if (!fromEnd.includes('部署实施') || !fromRoot.includes('部署实施')) throw new Error('因果追溯失败')

    // 5. 跨会话持久（v0.3：session 层删除——记忆不再随会话结束清理）
    console.log(`✓ 跨会话持久: 剩余 ${store.count()} 节点（无清理 API——跨会话沉淀）`)
    if (store.count() !== 3) throw new Error('跨会话持久失败（应保留全部节点）')

    console.log('\n=== 端到端全链路 PASS ===')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
}

main().catch((e) => { console.error('FAIL:', e.message); process.exitCode = 1 })
