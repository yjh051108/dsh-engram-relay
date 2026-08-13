/**
 * 确定性力导向布局（d3-force 风格，零依赖，纯函数）。
 *
 * 机制参考 Obsidian Graph View（d3-force）重写，修复原实现的三处缺陷
 * （斥力符号反了导致引力坍缩 / 重叠节点斥力为零永久粘死 / 向心力把网络
 * 吸成团）：
 *  - manyBody：库仑斥力（负 strength = 排斥），1/d² 公式（d3 同款）；
 *  - link：弹簧沿边吸引（d3 forceLink 同款：strength 0-1 标定）；
 *  - collide：硬防重叠（按半径修正速度，节点永不重叠粘死）；
 *  - forceCenter：质心平移居中——**不加力**，不会把网络吸成团；
 *  - alpha 温度衰减：所有力 × alpha（1 → alphaMin），初期剧烈、后期收敛。
 *
 * 确定性：初始位置按索引均匀分布在圆环上（无随机，仅按索引确定性微扰
 * 打破完美对称），固定迭代次数与参数 → 相同输入永远得到相同布局
 * （用户偏好：不喜欢不可控的随机性；也便于测试断言）。
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
  /** 迭代轮数（越多越收敛；缺省 300）。 */
  iterations?: number
  /** manyBody 斥力强度（负值 = 排斥；缺省 -400，按 900×620 画布标定）。 */
  charge?: number
  /** 弹簧强度（0-1；越大边越硬，缺省 0.5）。 */
  spring?: number
  /** 弹簧自然长度（期望的边距，缺省 80）。 */
  springLength?: number
  /** 节点碰撞半径（防重叠的最小间距，缺省 24）。 */
  collideRadius?: number
  /** 软向心强度（d3 forceX/forceY；0-1，把网络温和拉回中心防无限扩散，
   *  缺省 0.05——远弱于斥力，只阻止撞边界，不压缩网络）。 */
  centerStrength?: number
  /** 速度衰减（0-1；每轮速度保留比例，缺省 0.55）。 */
  velocityDecay?: number
  /** alpha 温度衰减率（缺省 0.02 → 约 300 轮衰减到 0.001）。 */
  alphaDecay?: number
  /** 最大速度钳制（防抖；缺省 20，宽松即可）。 */
  maxMove?: number
}

/** 布局结果：nodeId → 中心点坐标。 */
export type ForceLayout = Map<string, ForcePoint>

const EPS = 1e-6

export function layoutForce(
  nodes: ForceNodeInput[],
  edges: ForceEdgeInput[],
  opts: ForceLayoutOptions,
): ForceLayout {
  const {
    width, height,
    iterations = 500,
    charge = -300,
    spring = 0.1,
    springLength = 80,
    collideRadius = 24,
    centerStrength = 0.08,
    velocityDecay = 0.55,
    alphaDecay = 0.02,
    maxMove = 40,
  } = opts

  if (nodes.length === 0) return new Map()

  const cx = width / 2
  const cy = height / 2
  const ringRadius = Math.max(40, Math.min(width, height) / 2 - 60)
  const n = nodes.length

  // 确定性初始：均匀圆环 + 按索引的确定性微扰（打破完美对称，防对称坍缩）。
  interface Body { x: number; y: number; vx: number; vy: number; weight: number }
  const bodies = new Map<string, Body>()
  nodes.forEach((node, i) => {
    const angle = (2 * Math.PI * i) / n
    const jig = ((i * 2654435761) % 1000) / 1000 - 0.5 // 确定性伪随机 [-0.5, 0.5)
    bodies.set(node.id, {
      x: cx + (ringRadius + jig * 4) * Math.cos(angle) + jig * 4,
      y: cy + (ringRadius + jig * 4) * Math.sin(angle) + jig * 4,
      vx: 0,
      vy: 0,
      weight: Math.max(0.5, node.weight ?? 1),
    })
  })

  // 边索引（弹簧只沿实际连接）。
  const edgePairs = edges
    .map((e) => ({ a: bodies.get(e.from), b: bodies.get(e.to) }))
    .filter((e): e is { a: Body; b: Body } => e.a !== undefined && e.b !== undefined && e.a !== e.b)

  // ---- alpha 温度：1 → alphaMin（alphaDecay 每轮衰减）----
  let alpha = 1
  const alphaMin = 0.001
  const alphaTarget = 0

  const list = [...bodies.entries()]

  for (let iter = 0; iter < iterations; iter += 1) {
    // 温度衰减
    alpha += (alphaTarget - alpha) * alphaDecay
    const a = alpha

    // ---- manyBody：库仑斥力（d3 公式：w = strength×α / d²，负 strength 排斥）----
    for (let i = 0; i < list.length; i += 1) {
      const [, bi] = list[i]!
      for (let j = i + 1; j < list.length; j += 1) {
        const [, bj] = list[j]!
        // d3 方向约定：x = 对方 − 自身（bj − bi），负 w → vx += x·w 把节点沿
        // 连线推开（⚠️ 曾写成 bi − bj，导致负 charge 变成引力，圆环上所有
        // 邻居都在内侧 → 全部被吸向圆心坍缩）
        const x = bj.x - bi.x
        const y = bj.y - bi.y
        let l = x * x + y * y
        if (l < EPS) l = EPS // 防除零（重叠节点仍有有限斥力，不会粘死）
        // strength 负值 → w 负 → vx += x·w 把节点沿连线推开
        const w = (charge * bi.weight * bj.weight * a) / l
        bi.vx += x * w
        bi.vy += y * w
        bj.vx -= x * w
        bj.vy -= y * w
      }
    }

    // ---- link：弹簧沿边（d3 forceLink：strength 0-1 标定）----
    for (const { a: s, b: t } of edgePairs) {
      const x = t.x - s.x
      const y = t.y - s.y
      let l = Math.sqrt(x * x + y * y)
      if (l < EPS) l = EPS
      l = ((l - springLength) / l) * a * spring
      s.vx += x * l
      s.vy += y * l
      t.vx -= x * l
      t.vy -= y * l
    }

    // ---- collide：硬防重叠（d3 forceCollide；不乘 alpha——硬约束必须在
    //      温度衰减后仍有效，否则后期节点被 link 拽到一起后冻结在聚团态）----
    for (let i = 0; i < list.length; i += 1) {
      const [, bi] = list[i]!
      for (let j = i + 1; j < list.length; j += 1) {
        const [, bj] = list[j]!
        const x = bj.x - bi.x
        const y = bj.y - bi.y
        let l = Math.sqrt(x * x + y * y)
        const r = collideRadius + collideRadius
        if (l < r) {
          if (l < EPS) { // 完全重叠：确定性方向微扰
            const ang = ((i * 2654435761 + j * 40503) % 628) / 100
            bj.x += Math.cos(ang) * 0.5
            bj.y += Math.sin(ang) * 0.5
            l = 0.5
          }
          l = (r - l) / l
          bi.vx -= x * l
          bi.vy -= y * l
          bj.vx += x * l
          bj.vy += y * l
        }
      }
    }

    // ---- forceX/forceY 软向心：弱弹簧把网络拉回中心（防无限扩散撞边界；
    //      强度远小于斥力，不会把网络压成团）----
    for (const body of bodies.values()) {
      body.vx += (cx - body.x) * centerStrength * a
      body.vy += (cy - body.y) * centerStrength * a
    }

    // ---- 积分 + 速度衰减 + 限速 + 边界 ----
    let sx = 0
    let sy = 0
    for (const body of bodies.values()) {
      sx += body.x
      sy += body.y
      body.vx *= velocityDecay
      body.vy *= velocityDecay
      const speed = Math.sqrt(body.vx * body.vx + body.vy * body.vy)
      if (speed > maxMove) {
        body.vx = (body.vx / speed) * maxMove
        body.vy = (body.vy / speed) * maxMove
      }
      body.x += body.vx
      body.y += body.vy
      // 无限画布（v0.3）：无硬边界墙——节点位置完全由力平衡决定
      // （软向心 forceX/forceY 拉回中心，不会飘走；视口由前端自由平移缩放）
    }

    // ---- forceCenter：质心平移居中（不加力，不会吸成团）----
    if (bodies.size > 0) {
      const mx = sx / bodies.size
      const my = sy / bodies.size
      for (const body of bodies.values()) {
        body.x += cx - mx
        body.y += cy - my
      }
    }
  }

  const out: ForceLayout = new Map()
  for (const [id, body] of bodies) out.set(id, { x: body.x, y: body.y })
  return out
}
