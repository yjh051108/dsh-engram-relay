/**
 * 力导向布局（src/client/force.ts）测试。
 *
 * 覆盖：确定性（相同输入 → 相同输出）、单节点居中、空输入、边界约束
 * （坐标不越出画布）、节点在结果中一一对应。
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

test('force: 弹簧生效——springLength 越大相邻节点间距越大', () => {
  // 弹簧的真实语义是"边距可调"：连边节点对在斥力/向心力/弹簧三方平衡下，
  // 平衡间距随 springLength 单调增大。双节点对称布局可直接量间距验证。
  // 注意：平衡点受斥力钳制与向心力压缩，不恰等于 springLength（L=60 实测
  // ~34，L=140 实测 ~98），因此只断言单调性与显著差距，不依赖精确平衡位置。
  const nodes = [{ id: 'a' }, { id: 'b' }]
  const edges = [{ from: 'a', to: 'b' }]
  const dist = (out) => {
    const pa = out.get('a')
    const pb = out.get('b')
    return Math.hypot(pa.x - pb.x, pa.y - pb.y)
  }
  const short = layoutForce(nodes, edges, { width: W, height: H, springLength: 60, iterations: 300 })
  const long = layoutForce(nodes, edges, { width: W, height: H, springLength: 140, iterations: 300 })
  const shortD = dist(short)
  const longD = dist(long)
  // 1) 长弹簧 → 更大边距（单调性）
  assert.ok(shortD < longD, `短弹簧更紧凑 (${shortD} < ${longD})`)
  // 2) 差距显著（弹簧确实改变布局，而非噪声）
  assert.ok(longD - shortD > 30, `边距差距显著 (${longD} - ${shortD} > 30)`)
  // 3) 边距在合理范围（未被斥力完全压塌 / 未被弹飞）
  assert.ok(shortD > 20 && shortD < 120, `短弹簧边距合理 (${shortD})`)
  assert.ok(longD > 40 && longD < 200, `长弹簧边距合理 (${longD})`)
})
