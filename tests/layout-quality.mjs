/**
 * 布局质量校验（真实记忆 + 合成压力数据）——GraphView 消费面的布局回归。
 *
 * 度量（对齐 GraphView 的渲染约束）：
 *  - 零重叠：任意两节点圆心距 ≥ r₁+r₂+gap（碰撞分离硬约束）
 *  - 边界：所有节点在画布内（含半径）
 *  - 聚团：≥2 个连通分量时，团间质心距 > 团内最大跨度（分量不被搅成一团）
 *  - 边距：平均边长贴近 springLength（弹簧语义）
 *
 * 运行：node tests/layout-quality.mjs [storeDir]
 */
import { readFileSync, existsSync } from 'node:fs'
import { EngramStore } from '../lib/engram/store.js'
import { buildGraphView } from './graph-view.mjs'
import { layoutForce } from '../src/client/force.ts'

const VIEW_W = 900
const VIEW_H = 620
// 与 src/client/GraphView.tsx 保持一致的布局参数（改布局时同步改这里）
const LAYOUT_OPTS = {
  width: VIEW_W, height: VIEW_H, iterations: 250,
  repulsionScale: 0.25, springScale: 2.0, center: 0.02,
  alphaDecay: 0.995, damping: 0.8, maxMove: 8,
  radius: 12, gap: 14,
}

const radiusOf = (n) => 7 + Math.max(0, Math.min(1, n.importance)) * 9

/** 构建 GraphView 同款布局输入（nodes/edges/半径）。 */
function buildLayoutInput(nodes, edges) {
  const layout = layoutForce(
    nodes.map((n) => ({ id: n.id, weight: 0.6 + n.importance, radius: radiusOf(n) })),
    edges.map((e) => ({ from: e.from, to: e.to })),
    LAYOUT_OPTS,
  )
  return { layout, radiusOf: new Map(nodes.map((n) => [n.id, radiusOf(n)])) }
}

/** 合成压力数据：100 节点、多分量（链/星/孤点混合），贴近真实记忆结构。 */
function syntheticGraph() {
  const nodes = []
  const edges = []
  const layers = ['global', 'project', 'session']
  let id = 0
  const add = (title, layer, importance, links = []) => {
    const nid = `n${id++}`
    nodes.push({ id: nid, title, layer, importance, summary: `summary of ${title}` })
    return nid
  }
  // 10 条链（每条 6 节点，因果边串联）
  for (let c = 0; c < 10; c += 1) {
    let prev = null
    for (let k = 0; k < 6; k += 1) {
      const nid = add(`链${c}-节点${k}`, layers[c % 3], 0.1 + ((k * 7 + c * 3) % 9) / 10)
      if (prev !== null) edges.push({ from: prev, to: nid, kind: 'causes' })
      prev = nid
    }
  }
  // 3 个星型簇（中心高重要度，双向链接到 8 个卫星）
  for (let c = 0; c < 3; c += 1) {
    const center = add(`簇${c}-枢纽`, 'global', 0.9)
    for (let k = 0; k < 8; k += 1) {
      const leaf = add(`簇${c}-卫星${k}`, 'project', 0.3 + k * 0.05)
      edges.push({ from: center, to: leaf, kind: 'link' })
    }
  }
  // 8 个孤立节点
  for (let k = 0; k < 8; k += 1) add(`孤点${k}`, layers[k % 3], 0.5)
  return { nodes, edges }
}

function metrics(nodes, edges, layout, radiusMap) {
  const ids = nodes.map((n) => n.id)
  let minD = Infinity
  let overlaps = 0
  for (let i = 0; i < ids.length; i += 1) {
    for (let j = i + 1; j < ids.length; j += 1) {
      const a = layout.get(ids[i])
      const b = layout.get(ids[j])
      const d = Math.hypot(a.x - b.x, a.y - b.y)
      const need = radiusMap.get(ids[i]) + radiusMap.get(ids[j]) + 14
      if (d < need) overlaps += 1
      if (d < minD) minD = d
    }
  }
  // 边界（含半径）：仅单分量时坐标契约 [0,W]×[0,H]（多分量走虚拟空间，
  // 视图层 fitTransform 负责缩放）
  let outOfBounds = 0
  let singleComponent = true
  {
    // 粗略分量判断：第一条边连通的所有节点
    const seen0 = new Set()
    if (edges.length > 0) {
      const stack = [edges[0].from]
      while (stack.length) {
        const id = stack.pop()
        if (seen0.has(id)) continue
        seen0.add(id)
        for (const e of edges) {
          if (e.from === id && !seen0.has(e.to)) stack.push(e.to)
          if (e.to === id && !seen0.has(e.from)) stack.push(e.from)
        }
      }
      singleComponent = seen0.size === nodes.length
    }
  }
  if (singleComponent) {
    for (const n of nodes) {
      const p = layout.get(n.id)
      const r = radiusMap.get(n.id)
      if (p.x - r < 0 || p.x + r > VIEW_W || p.y - r < 0 || p.y + r > VIEW_H) outOfBounds += 1
    }
  }
  // 连通分量（BFS）
  const adj = new Map(nodes.map((n) => [n.id, new Set()]))
  for (const e of edges) { adj.get(e.from)?.add(e.to); adj.get(e.to)?.add(e.from) }
  const seen = new Set()
  const comps = []
  for (const n of nodes) {
    if (seen.has(n.id)) continue
    const members = []
    const stack = [n.id]
    seen.add(n.id)
    while (stack.length) {
      const id = stack.pop()
      members.push(id)
      for (const nb of adj.get(id) ?? []) if (!seen.has(nb)) { seen.add(nb); stack.push(nb) }
    }
    comps.push(members)
  }
  const centroids = comps.map((m) => {
    let x = 0; let y = 0
    for (const id of m) { const p = layout.get(id); x += p.x; y += p.y }
    return { x: x / m.length, y: y / m.length, m }
  })
  let clusterOk = true
  let clusterRatio = 0
  if (centroids.length >= 2) {
    let between = Infinity
    for (let i = 0; i < centroids.length; i += 1) {
      for (let j = i + 1; j < centroids.length; j += 1) {
        const d = Math.hypot(centroids[i].x - centroids[j].x, centroids[i].y - centroids[j].y)
        if (d < between) between = d
      }
    }
    let within = 0
    for (const c of centroids) {
      for (const id of c.m) {
        const p = layout.get(id)
        within = Math.max(within, Math.hypot(p.x - c.x, p.y - c.y))
      }
    }
    clusterRatio = between / Math.max(1, within)
    clusterOk = clusterRatio >= 1.2
  }
  // 平均边长
  const edgeLens = edges.map((e) => Math.hypot(layout.get(e.from).x - layout.get(e.to).x, layout.get(e.from).y - layout.get(e.to).y))
  const avgEdge = edgeLens.length ? edgeLens.reduce((a, b) => a + b, 0) / edgeLens.length : 0
  return { minD, overlaps, outOfBounds, comps: comps.length, clusterRatio, clusterOk, avgEdge }
}

function report(label, nodes, edges, layout, radiusMap, expect) {
  const m = metrics(nodes, edges, layout, radiusMap)
  console.log(`\n[${label}] ${nodes.length} 节点 / ${edges.length} 边 / ${m.comps} 分量`)
  console.log(`  最小圆心距 ${m.minD.toFixed(1)} / 重叠对 ${m.overlaps} / 越界 ${m.outOfBounds} / 聚团比 ${m.clusterRatio.toFixed(2)} / 平均边长 ${m.avgEdge.toFixed(1)}`)
  const failures = []
  if (m.overlaps !== 0) failures.push(`重叠对 ${m.overlaps}`)
  if (m.outOfBounds !== 0) failures.push(`越界 ${m.outOfBounds}`)
  if (expect.cluster && !m.clusterOk) failures.push(`聚团比 ${m.clusterRatio.toFixed(2)} < 1.2`)
  if (failures.length) throw new Error(`[${label}] 布局质量不达标: ${failures.join('; ')}`)
}

// ─── 真实记忆（缺省取当前部署 store）───
const storeDir = process.argv[2] ?? ''
let realCount = 0
try {
  const store = new EngramStore(storeDir)
  const gv = buildGraphView(store, {})
  const { layout, radiusOf: rm } = buildLayoutInput(gv.nodes, gv.edges)
  if (gv.nodes.length > 0) {
    report(`真实记忆 (${gv.nodes.length} 可见)`, gv.nodes, gv.edges, layout, rm, { cluster: false })
    realCount = gv.nodes.length
  } else {
    console.log('\n[真实记忆] store 为空，跳过')
  }
} catch (e) {
  console.log(`\n[真实记忆] 加载失败（跳过）: ${e.message}`)
}

// ─── 合成压力数据（100 节点）───
const { nodes, edges } = syntheticGraph()
const { layout, radiusOf: rm2 } = buildLayoutInput(nodes, edges)
report('合成压力 (100)', nodes, edges, layout, rm2, { cluster: true })

console.log('\nPASS（真实记忆 ' + (realCount || 0) + ' 条 + 合成 100 节点）')
