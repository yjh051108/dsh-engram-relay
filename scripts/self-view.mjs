/**
 * 自视脚本：用真实 store 数据 + 转译的 force.ts，复刻 GraphView 渲染链路，
 * 生成 SVG → sharp 转 PNG——让 agent 自己看到图谱效果（不再让用户当眼睛）。
 *
 * 用法：node scripts/self-view.mjs [画布宽] [画布高]
 */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import ts from 'typescript'
import { EngramStore } from '../lib/engram/store.js'
import { homedir } from 'node:os'

const __dirname = dirname(fileURLToPath(import.meta.url))
const W = Number(process.argv[2] ?? 1600)
const H = Number(process.argv[3] ?? 755) // 2.12:1 画布

// ── 1. 转译 force.ts（TS → JS，内存执行）──
const forceSrc = readFileSync(join(__dirname, '../src/client/force.ts'), 'utf8')
const js = ts.transpileModule(forceSrc, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 },
}).outputText
const mod = await import('data:text/javascript;base64,' + Buffer.from(js).toString('base64'))
const { layoutForce } = mod

// ── 2. 真实数据（与图谱 API 同过滤：废止/待确认/快照）──
const store = new EngramStore(join(homedir(), '.dsh/engram-relay'))
const nodes = store.all().filter((e) => !e.supersededBy && e.status !== 'pending' && e.kind !== 'snapshot')
const edges = []
const seen = new Set()
const addEdge = (a, b, kind) => {
  const k = `${a}|${b}|${kind}`
  if (seen.has(k)) return
  seen.add(k)
  edges.push({ from: a, to: b, kind })
}
const ids = new Set(nodes.map((n) => n.id))
for (const n of nodes) {
  for (const c of n.causes) if (ids.has(c)) addEdge(c, n.id, 'causes')
  for (const e of n.effects) if (ids.has(e)) addEdge(n.id, e, 'causes')
  for (const l of n.links) { const t = store.byTitle(l); if (t && ids.has(t.id) && t.id !== n.id) addEdge(n.id, t.id, 'link') }
}
console.log(`节点 ${nodes.length} | 边 ${edges.length}`)

// ── 3. 连通分量（簇）+ 项目分组 ──
const adj = new Map()
for (const n of nodes) adj.set(n.id, new Set())
for (const e of edges) { adj.get(e.from)?.add(e.to); adj.get(e.to)?.add(e.from) }
const clusterOf = new Map()
const clusterList = []
const visited = new Set()
for (const n of nodes) {
  if (visited.has(n.id)) continue
  const q = [n.id]; visited.add(n.id); const idsArr = []
  while (q.length) { const id = q.shift(); idsArr.push(id); for (const nb of adj.get(id) ?? []) if (!visited.has(nb)) { visited.add(nb); q.push(nb) } }
  const cid = `c${clusterList.length}`
  for (const id of idsArr) clusterOf.set(id, cid)
  clusterList.push(idsArr)
}
const projectGroups = new Map()
// ⚠️ 所有节点入组（null → __solo__）——通用节点不参与区域引力会散布
// 全图撑爆布局（与 GraphView 一致）
for (const n of nodes) projectGroups.set(n.id, n.projectId ?? '__solo__')

// ── 4. 布局（与 GraphView 同参数）──
const layout = layoutForce(
  nodes.map((n) => ({ id: n.id, weight: 0.6 + n.importance })),
  edges.map((e) => ({ from: e.from, to: e.to })),
  {
    width: W, height: H, iterations: 250,
    charge: -100, spring: 0.1, springLength: 110, collideRadius: 22, centerStrength: 0.08,
    clusters: clusterOf, clusterTarget: 110, clusterStrength: 0.04,
    projectGroups: projectGroups.size ? projectGroups : undefined,
    projectStrength: 0.8,
  },
)

// ── 5. 项目大圆 / 连通分量簇圆（≥5）/ 节点 ──
function projectColor(pid) {
  if (pid === null) return '#8a94a6'
  let h = 0
  for (const ch of pid) h = (h * 31 + ch.charCodeAt(0)) >>> 0
  return `hsl(${h % 360} 55% 55%)`
}
const byProject = new Map()
for (const n of nodes) {
  const arr = byProject.get(n.projectId) ?? []
  arr.push(n.id)
  byProject.set(n.projectId, arr)
}
const projectCircles = []
for (const [pid, idsArr] of byProject) {
  if (pid === null) continue // 通用不画项目圆（散布全图会罩住一切）
  if (idsArr.length < 3) continue
  const pts = idsArr.map((id) => layout.get(id)).filter(Boolean)
  if (pts.length < 3) continue
  const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length
  const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length
  const radius = Math.max(70, ...pts.map((p) => Math.hypot(p.x - cx, p.y - cy))) + 50
  const label = String(pid).split(/[\\/]/).pop()
  // ⚠️ 重叠检测（v0.6：只看圆心距会漏——半径大时圆重叠）
  projectCircles.forEach((o) => {
    const d = Math.hypot(o.cx - cx, o.cy - cy)
    if (d < o.radius + radius) {
      console.log(`  ⚠️ 重叠: [${o.label}]↔[${label}] 圆心距 ${d.toFixed(0)} < 半径和 ${(o.radius + radius).toFixed(0)}`)
    }
  })
  projectCircles.push({ cx, cy, radius, label, color: projectColor(pid) })
}
const clusterCircles = clusterList.filter((c) => c.length >= 5).map((idsArr) => {
  const pts = idsArr.map((id) => layout.get(id)).filter(Boolean)
  const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length
  const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length
  const radius = Math.max(60, ...pts.map((p) => Math.hypot(p.x - cx, p.y - cy))) + 40
  return { cx, cy, radius }
})
const degreeOf = new Map()
for (const e of edges) { degreeOf.set(e.from, (degreeOf.get(e.from) ?? 0) + 1); degreeOf.set(e.to, (degreeOf.get(e.to) ?? 0) + 1) }

// 包围盒统计（fit 后视口 = 包围盒，画布占满度）
let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
for (const n of nodes) {
  const p = layout.get(n.id)
  if (!p) continue
  minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x)
  minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y)
}
console.log(`布局范围 x[${minX.toFixed(0)}, ${maxX.toFixed(0)}] y[${minY.toFixed(0)}, ${maxY.toFixed(0)}]`)
console.log(`画布 ${W}x${H} | 节点包围盒 ${(maxX - minX).toFixed(0)}x${(maxY - minY).toFixed(0)}`)
console.log(`占画布宽 ${((maxX - minX) / W * 100).toFixed(0)}% | 高 ${((maxY - minY) / H * 100).toFixed(0)}%`)
console.log(`项目大圆 ${projectCircles.length} 个:`)
for (const c of projectCircles) console.log(`  [${c.label}] cx=${c.cx.toFixed(0)} cy=${c.cy.toFixed(0)} r=${c.radius.toFixed(0)} 色=${c.color}`)
console.log(`连通分量簇圆(≥5) ${clusterCircles.length} 个`)

// ── 6. SVG 渲染（复刻 GraphView）──
const pad = 80
const vx = minX - pad, vy = minY - pad, vw = Math.max(200, maxX - minX + 2 * pad), vh = Math.max(150, maxY - minY + 2 * pad)
const zc = 1 // fit 后 zoomScale = 900/vw < 1 → zc=1
let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${vx} ${vy} ${vw} ${vh}" width="${W}" height="${H}">`
svg += `<rect x="${vx - 2000}" y="${vy - 2000}" width="${vw + 4000}" height="${vh + 4000}" fill="#101318"/>`
// 点阵
for (let gx = Math.floor(vx / 40) * 40; gx < vx + vw + 40; gx += 40) {
  for (let gy = Math.floor(vy / 40) * 40; gy < vy + vh + 40; gy += 40) {
    svg += `<circle cx="${gx}" cy="${gy}" r="1.2" fill="rgba(255,255,255,0.05)"/>`
  }
}
// 项目大圆
for (const c of projectCircles) {
  svg += `<circle cx="${c.cx}" cy="${c.cy}" r="${c.radius}" fill="${c.color}" fill-opacity="0.07" stroke="${c.color}" stroke-opacity="0.4" stroke-width="1.5"/>`
  svg += `<text x="${c.cx}" y="${c.cy - c.radius + 24}" text-anchor="middle" font-size="13" font-weight="700" fill="#8a94a6" stroke="#101318" stroke-width="3" paint-order="stroke">${c.label}</text>`
}
// 簇圆
for (const c of clusterCircles) {
  svg += `<circle cx="${c.cx}" cy="${c.cy}" r="${c.radius}" fill="rgba(255,255,255,0.04)" stroke="rgba(255,255,255,0.15)" stroke-width="1"/>`
}
// 边
for (const e of edges) {
  const a = layout.get(e.from), b = layout.get(e.to)
  if (!a || !b) continue
  svg += `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="${e.kind === 'causes' ? projectColor(nodes.find((n) => n.id === e.from)?.projectId ?? null) : '#aab2c0'}" stroke-opacity="0.25" stroke-width="1"/>`
}
// 节点
for (const n of nodes) {
  const p = layout.get(n.id)
  if (!p) continue
  const deg = degreeOf.get(n.id) ?? 0
  const isSem = n.state === 'semantic'
  const isEvt = n.kind === 'event'
  const r = (7 + Math.min(9, deg * 1.2) + (isSem ? 3 : 0) + n.importance * 1.5) * (isEvt ? 0.7 : 1)
  const cid = clusterOf.get(n.id)
  const color = projectColor(n.projectId)
  svg += `<circle cx="${p.x}" cy="${p.y}" r="${r}" fill="${color}" fill-opacity="${isSem ? 0.95 : 0.8}" stroke="${isSem ? 'rgba(255,255,255,0.6)' : 'rgba(255,255,255,0.35)'}" stroke-width="${isSem ? 1.5 : 1}"/>`
  svg += `<text x="${p.x}" y="${p.y + r + 12}" text-anchor="middle" font-size="${isSem ? 11 : 10}" fill="${isSem ? '#e8eaee' : '#aab2c0'}" stroke="#101318" stroke-width="3" paint-order="stroke">${n.title.length > 14 ? n.title.slice(0, 13) + '…' : n.title}</text>`
}
svg += '</svg>'

const outDir = join(__dirname, '..', '..', 'engram-selfview')
mkdirSync(outDir, { recursive: true })
const svgPath = join(outDir, `graph-${W}x${H}.svg`)
const pngPath = join(outDir, `graph-${W}x${H}.png`)
writeFileSync(svgPath, svg, 'utf8')
const sharp = (await import('sharp')).default
await sharp(Buffer.from(svg)).resize(W, H).png().toFile(pngPath)
console.log(`已生成: ${pngPath}`)
