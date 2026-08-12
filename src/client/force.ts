/**
 * 确定性力导向布局（零依赖，纯函数）。
 *
 * 把图谱节点摊到二维平面：斥力（库仑，防重叠）+ 弹簧（胡克，沿边聚拢）
 * + 向心力（防飞散）。**确定性**：初始位置按索引均匀分布在圆环上（不用
 * 随机），固定迭代次数与参数 → 相同输入永远得到相同布局（用户偏好：不
 * 喜欢不可控的随机性；也便于测试断言）。
 */

export interface ForceNodeInput {
  id: string
  /** 斥力权重（大节点推得更远；缺省 1）。 */
  weight?: number
}

export interface ForceEdgeInput {
  from: string
  to: string
}

export interface ForcePoint {
  x: number
  y: number
}

export interface ForceLayoutOptions {
  width: number
  height: number
  /** 迭代轮数（越多越收敛；缺省 200）。 */
  iterations?: number
  /** 斥力强度（负值；越大越散）。 */
  charge?: number
  /** 弹簧强度（沿边聚拢）。 */
  spring?: number
  /** 弹簧自然长度（期望的边距）。 */
  springLength?: number
  /** 向心力（拉回中心，防飞散）。 */
  center?: number
  /** 速度阻尼（0-1；越小越快停）。 */
  damping?: number
  /** 单轮最大位移（防抖）。 */
  maxMove?: number
  /** 节点半径（斥力计算的最小间距；缺省 18）。 */
  radius?: number
}

/** 布局结果：nodeId → 中心点坐标。 */
export type ForceLayout = Map<string, ForcePoint>

const DEFAULT_RADIUS = 18

export function layoutForce(
  nodes: ForceNodeInput[],
  edges: ForceEdgeInput[],
  opts: ForceLayoutOptions,
): ForceLayout {
  const {
    width, height,
    iterations = 200,
    charge = -900,
    spring = 0.02,
    springLength = 100,
    center = 0.015,
    damping = 0.85,
    maxMove = 2.5,
    radius = DEFAULT_RADIUS,
  } = opts

  if (nodes.length === 0) return new Map()

  const cx = width / 2
  const cy = height / 2
  const ringRadius = Math.max(40, Math.min(width, height) / 2 - 60)
  const n = nodes.length

  // 确定性初始：均匀圆环（按输入顺序取角）。
  interface Body { x: number; y: number; vx: number; vy: number; weight: number }
  const bodies = new Map<string, Body>()
  nodes.forEach((node, i) => {
    const angle = (2 * Math.PI * i) / n
    bodies.set(node.id, {
      x: cx + ringRadius * Math.cos(angle),
      y: cy + ringRadius * Math.sin(angle),
      vx: 0,
      vy: 0,
      weight: Math.max(0.5, node.weight ?? 1),
    })
  })

  // 边索引（弹簧只沿实际连接）。
  const edgePairs = edges
    .map((e) => ({ a: bodies.get(e.from), b: bodies.get(e.to), aId: e.from, bId: e.to }))
    .filter((e): e is { a: Body; b: Body; aId: string; bId: string } => e.a !== undefined && e.b !== undefined && e.a !== e.b)

  for (let iter = 0; iter < iterations; iter += 1) {
    // ---- 斥力（两两，O(n²)；图谱规模小，可接受）----
    const list = [...bodies.entries()]
    for (let i = 0; i < list.length; i += 1) {
      const [, a] = list[i]!
      for (let j = i + 1; j < list.length; j += 1) {
        const [, b] = list[j]!
        let dx = b.x - a.x
        let dy = b.y - a.y
        let dist2 = dx * dx + dy * dy
        // 最小间距兜底（重叠节点避免除零/爆力）
        const minDist = radius + radius + 12
        if (dist2 < minDist * minDist) dist2 = minDist * minDist
        const dist = Math.sqrt(dist2)
        // 库仑力：F = charge * (w1*w2) / d²
        const force = (charge * a.weight * b.weight) / dist2
        const fx = (force * dx) / dist
        const fy = (force * dy) / dist
        a.vx -= fx
        a.vy -= fy
        b.vx += fx
        b.vy += fy
      }
    }

    // ---- 弹簧（沿边）----
    for (const { a, b } of edgePairs) {
      const dx = b.x - a.x
      const dy = b.y - a.y
      const dist = Math.max(1, Math.sqrt(dx * dx + dy * dy))
      const force = spring * (dist - springLength)
      const fx = (force * dx) / dist
      const fy = (force * dy) / dist
      a.vx += fx
      a.vy += fy
      b.vx -= fx
      b.vy -= fy
    }

    // ---- 向心力（拉回画布中心）----
    for (const body of bodies.values()) {
      body.vx += (cx - body.x) * center * body.weight
      body.vy += (cy - body.y) * center * body.weight
    }

    // ---- 积分 + 阻尼 + 限速 ----
    for (const body of bodies.values()) {
      body.vx *= damping
      body.vy *= damping
      const speed = Math.sqrt(body.vx * body.vx + body.vy * body.vy)
      if (speed > maxMove) {
        body.vx = (body.vx / speed) * maxMove
        body.vy = (body.vy / speed) * maxMove
      }
      body.x += body.vx
      body.y += body.vy
      // 硬边界（不越出画布）
      body.x = Math.max(radius, Math.min(width - radius, body.x))
      body.y = Math.max(radius, Math.min(height - radius, body.y))
    }
  }

  const out: ForceLayout = new Map()
  for (const [id, body] of bodies) out.set(id, { x: body.x, y: body.y })
  return out
}
