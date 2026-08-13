// 验证 d3-force 风格布局：真实 15 节点 / 14 边 → 输出坐标分布
import { layoutForce } from '../src/client/force.ts'

const nodes = [
  { id: 'emsqucruc-1', title: '注入器rc适配踩坑', importance: 1 },
  { id: 'emsqud2dx-2', title: '注入器rc适配', importance: 0.6 },
  { id: 'emsqud2ea-3', title: 'engram注入完成', importance: 0.6 },
  { id: 'emsqv66w2-3', title: 'engram模型主路径', importance: 0.6 },
  { id: 'emsqv5tsy-1', title: 'wake打分降级链', importance: 0.6 },
  { id: 'emsqv66wa-4', title: '空库不误判损坏', importance: 0.6 },
  { id: 'emsqvatb1-5', title: 'engram API迁移完成', importance: 1 },
  { id: 'emsqv5tt2-2', title: 'engram注入完成', importance: 0.6 },
  { id: 'emsqvakiz-1', title: 'engram热重载循环', importance: 0.6 },
  { id: 'emsqvakj4-2', title: 'API迁移收尾提交', importance: 0.6 },
  { id: 'emsqvatam-3', title: '注入插件UI修复', importance: 1 },
  { id: 'emsqvasdf-6', title: '注入器rc适配踩坑2', importance: 1 },
  { id: 'emsqvaaaa-7', title: '注入器UI注册缺陷', importance: 0.6 },
  { id: 'emsqvbbbb-8', title: 'client bundle构建路径', importance: 0.6 },
  { id: 'emsqvcccc-9', title: 'UI热重载待完善', importance: 0.6 },
]
const edges = [
  { from: 'emsqucruc-1', to: 'emsqud2dx-2', kind: 'causes' },
  { from: 'emsqucruc-1', to: 'emsqud2ea-3', kind: 'causes' },
  { from: 'emsqucruc-1', to: 'emsqvakiz-1', kind: 'link' },
  { from: 'emsqud2ea-3', to: 'emsqv66w2-3', kind: 'causes' },
  { from: 'emsqv5tsy-1', to: 'emsqv66wa-4', kind: 'causes' },
  { from: 'emsqv5tsy-1', to: 'emsqv5tt2-2', kind: 'link' },
  { from: 'emsqucruc-1', to: 'emsqv5tt2-2', kind: 'causes' },
  { from: 'emsqv66w2-3', to: 'emsqvatb1-5', kind: 'causes' },
  { from: 'emsqvakiz-1', to: 'emsqvatam-3', kind: 'causes' },
  { from: 'emsqvakiz-1', to: 'emsqvasdf-6', kind: 'causes' },
  { from: 'emsqvakiz-1', to: 'emsqvaaaa-7', kind: 'link' },
  { from: 'emsqvakj4-2', to: 'emsqud2dx-2', kind: 'causes' },
  { from: 'emsqvakj4-2', to: 'emsqud2ea-3', kind: 'causes' },
  { from: 'emsqvakj4-2', to: 'emsqvakiz-1', kind: 'link' },
]

const VIEW_W = 900
const VIEW_H = 620

// 参数：charge spring springLength collideRadius iterations centerStrength
const [cA, cB, cC, cD, cE, cF] = process.argv.slice(2)
const P = {
  charge: cA !== undefined ? Number(cA) : -300,
  spring: cB !== undefined ? Number(cB) : 0.1,
  springLength: cC !== undefined ? Number(cC) : 80,
  collideRadius: cD !== undefined ? Number(cD) : 24,
  iterations: cE !== undefined ? Number(cE) : 500,
  centerStrength: cF !== undefined ? Number(cF) : 0.08,
}

const layout = layoutForce(
  nodes.map((n) => ({ id: n.id, weight: 0.6 + n.importance })),
  edges.map((e) => ({ from: e.from, to: e.to })),
  {
    width: VIEW_W, height: VIEW_H, iterations: P.iterations,
    charge: P.charge,
    spring: P.spring,
    springLength: P.springLength,
    collideRadius: P.collideRadius,
    centerStrength: P.centerStrength,
  },
)

const pts = [...layout.entries()].map(([id, p]) => ({ id, ...p }))
const cx = VIEW_W / 2
const cy = VIEW_H / 2
const dists = pts.map((p) => Math.hypot(p.x - cx, p.y - cy))
const sorted = [...dists].sort((a, b) => a - b)
const avg = dists.reduce((s, d) => s + d, 0) / dists.length
let minPair = Infinity
for (let i = 0; i < pts.length; i++) {
  for (let j = i + 1; j < pts.length; j++) {
    const d = Math.hypot(pts[i].x - pts[j].x, pts[i].y - pts[j].y)
    if (d < minPair) minPair = d
  }
}
// 贴边检测：距画布边缘 < 60px 算贴边
const clamped = pts.filter((p) => p.x < 60 || p.x > VIEW_W - 60 || p.y < 60 || p.y > VIEW_H - 60).length
console.log(`[charge=${P.charge} spring=${P.spring} len=${P.springLength} col=${P.collideRadius} center=${P.centerStrength} iter=${P.iterations}]`)
console.log(`  到中心: min=${sorted[0].toFixed(0)} avg=${avg.toFixed(0)} max=${sorted.at(-1).toFixed(0)} | 最小节点间距=${minPair.toFixed(0)} | 贴边节点=${clamped}`)
