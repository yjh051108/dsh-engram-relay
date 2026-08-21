/**
 * 力导向布局（src/client/force.ts）测试。
 *
 * 覆盖：确定性（相同输入 → 相同输出）、单节点居中、空输入、边界约束
 * （坐标不越出画布）、节点在结果中一一对应、**碰撞分离（任意两节点不
 * 重叠）**、**连通分量感知初始化（团内聚拢、团间分离）**、弹簧长度单调。
 * Node ≥22.6 直接 import TS（type stripping）。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'

import { layoutForce } from '../src/client/force.ts'

const W = 900
const H = 620

test('force: 确定性——相同输入两次布局结果完全一致', () => {
  const nodes = [
    { id: 'a' }, { id: 'b' }, { id: 'c' }, { id: 'd' }, { id: 'e' },
  ]
  const edges = [
    { from: 'a', to: 'b' }, { from: 'b', to: 'c' }, { from: 'c', to: 'a' },
  ]
  const first = layoutForce(nodes, edges, { width: W, height: H })
  const second = layoutForce(nodes, edges, { width: W, height: H })
  for (const n of nodes) {
    assert.deepEqual(second.get(n.id), first.get(n.id), `node ${n.id} 布局确定性`)
  }
})

test('force: 单节点落在画布中心附近', () => {
  const out = layoutForce([{ id: 'solo' }], [], { width: W, height: H, iterations: 400 })
  const p = out.get('solo')
  assert.ok(p !== undefined)
  // 向心力收敛后应接近中心（容差 60px）
  assert.ok(Math.abs(p.x - W / 2) < 60, `x=${p.x} 接近中心`)
  assert.ok(Math.abs(p.y - H / 2) < 60, `y=${p.y} 接近中心`)
})

test('force: 空输入返回空 Map', () => {
  const out = layoutForce([], [], { width: W, height: H })
  assert.equal(out.size, 0)
})

test('force: 坐标不越出画布（硬边界）', () => {
  const nodes = Array.from({ length: 24 }, (_, i) => ({ id: `n${i}` }))
  const edges = []
  for (let i = 1; i < 24; i += 1) edges.push({ from: `n${i - 1}`, to: `n${i}` })
  const out = layoutForce(nodes, edges, { width: W, height: H, iterations: 400 })
  assert.equal(out.size, 24)
  for (const [id, p] of out) {
    assert.ok(p.x >= 0 && p.x <= W, `${id} x 在画布内 (${p.x})`)
    assert.ok(p.y >= 0 && p.y <= H, `${id} y 在画布内 (${p.y})`)
  }
})

test('force: 碰撞分离——任意两节点圆心距 ≥ r₁+r₂+gap（零重叠）', () => {
  // 30 个节点 + 若干边，紧凑参数（强斥力弱迭代）下仍必须零重叠
  const nodes = Array.from({ length: 30 }, (_, i) => ({ id: `n${i}`, radius: i % 3 === 0 ? 16 : 9 }))
  const edges = []
  for (let i = 1; i < 30; i += 1) edges.push({ from: `n${i - 1}`, to: `n${i}` })
  const out = layoutForce(nodes, edges, {
    width: W, height: H, iterations: 60, collisionIterations: 120,
    radius: 10, gap: 6, center: 0.001, repulsionScale: 0.5, springScale: 0.2, alphaDecay: 0.98,
  })
  assert.equal(out.size, 30)
  const ids = nodes.map((n) => n.id)
  for (let i = 0; i < ids.length; i += 1) {
    for (let j = i + 1; j < ids.length; j += 1) {
      const a = out.get(ids[i])
      const b = out.get(ids[j])
      const ra = nodes[i].radius
      const rb = nodes[j].radius
      const d = Math.hypot(a.x - b.x, a.y - b.y)
      assert.ok(d >= ra + rb + 6 - 1e-6, `n${i}(${a.x},${a.y}) 与 n${j}(${b.x},${b.y}) 重叠: d=${d.toFixed(1)} < ${ra + rb + 6}`)
    }
  }
})

test('force: 连通分量感知——团内聚拢、团间分离', () => {
  // 两个分量：A 链 (a0-a1-a2-a3) 与 B 链 (b0-b1-b2-b3)，分量质心相距应
  // 显著大于各自团内最大跨度（初始即分环 + 质心引力聚团）。
  const nodes = [
    { id: 'a0' }, { id: 'a1' }, { id: 'a2' }, { id: 'a3' },
    { id: 'b0' }, { id: 'b1' }, { id: 'b2' }, { id: 'b3' },
  ]
  const edges = [
    { from: 'a0', to: 'a1' }, { from: 'a1', to: 'a2' }, { from: 'a2', to: 'a3' },
    { from: 'b0', to: 'b1' }, { from: 'b1', to: 'b2' }, { from: 'b2', to: 'b3' },
  ]
  const out = layoutForce(nodes, edges, {
    width: W, height: H, iterations: 250, collisionIterations: 120,
    center: 0.02, repulsionScale: 0.25, springScale: 2.0, maxMove: 8,
  })
  const centroid = (ids) => {
    let x = 0; let y = 0
    for (const id of ids) { const p = out.get(id); x += p.x; y += p.y }
    return { x: x / ids.length, y: y / ids.length }
  }
  const ca = centroid(['a0', 'a1', 'a2', 'a3'])
  const cb = centroid(['b0', 'b1', 'b2', 'b3'])
  const between = Math.hypot(ca.x - cb.x, ca.y - cb.y)
  const spreadA = Math.max(...['a0', 'a1', 'a2', 'a3'].map((id) => Math.hypot(out.get(id).x - ca.x, out.get(id).y - ca.y)))
  const spreadB = Math.max(...['b0', 'b1', 'b2', 'b3'].map((id) => Math.hypot(out.get(id).x - cb.x, out.get(id).y - cb.y)))
  assert.ok(between > Math.max(spreadA, spreadB) * 1.5,
    `团间质心距 ${between.toFixed(1)} 应显著大于团内跨度 A=${spreadA.toFixed(1)} B=${spreadB.toFixed(1)}`)
})

test('force: 弹簧生效——springFactor 越大相邻节点间距越大', () => {
  // FR 弹簧把边距拉向 k·springFactor；k 由密度推导（双节点 → 钳到 kMax 110）。
  // 只断言单调性与显著差距，不依赖精确平衡位置。
  const nodes = [{ id: 'a' }, { id: 'b' }]
  const edges = [{ from: 'a', to: 'b' }]
  const dist = (out) => {
    const pa = out.get('a')
    const pb = out.get('b')
    return Math.hypot(pa.x - pb.x, pa.y - pb.y)
  }
  const short = layoutForce(nodes, edges, { width: W, height: H, springFactor: 0.6, iterations: 250 })
  const long = layoutForce(nodes, edges, { width: W, height: H, springFactor: 1.3, iterations: 250 })
  const shortD = dist(short)
  const longD = dist(long)
  // 1) 长弹簧 → 更大边距（单调性）
  assert.ok(shortD < longD, `短弹簧更紧凑 (${shortD} < ${longD})`)
  // 2) 差距显著（弹簧确实改变布局，而非噪声）
  assert.ok(longD - shortD > 30, `边距差距显著 (${longD} - ${shortD} > 30)`)
  // 3) 边距在合理范围（未被斥力完全压塌 / 未弹飞）
  assert.ok(shortD > 40 && shortD < 180, `短弹簧边距合理 (${shortD})`)
  assert.ok(longD > 80 && longD < 300, `长弹簧边距合理 (${longD})`)
})
