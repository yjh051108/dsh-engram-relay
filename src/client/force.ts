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
 *
 * v0.5 增量模拟器（createForceSimulator）：Obsidian 式**渐进生成**——
 * 节点按批次 addNode（初始从画布中心弹出，被斥力推开 = "动力学球一点
 * 一点生成、相互拥挤丰满"），每帧 step(n) 实时演化，alpha 续接不重置。
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
  /**
   * 同簇引力（v0.5 视觉：Obsidian 聚类感）——nodeId → clusterId。
   * 同簇节点对在距离 > clusterTarget 时被柔和拉近（弱弹簧风格，正比
   * (d−target)），簇自然聚团、簇间分离；距离内不受力（不压缩簇结构，
   * collide 已防重叠）。
   */
  clusters?: Map<string, string>
  /** 簇内目标间距（缺省 110——略大于弹簧长度，簇内结构舒展）。 */
  clusterTarget?: number
  /** 簇引力强度（0-1；缺省 0.04——弱，只聚团不压扁）。 */
  clusterStrength?: number
  /**
   * 项目引力（v0.6 用户诉求"每个项目一个大圆形"）——nodeId → projectId。
   * 同项目节点在距离 > projectTarget 时柔和拉近（比连通分量引力弱、
   * 距离远）——**项目级大簇**形成（内部连通分量子结构保持），不同项目
   * 分离。
   */
  projectGroups?: Map<string, string>
  /** 项目内目标间距（缺省 320——项目圆尺度）。 */
  projectTarget?: number
  /** 项目引力强度（缺省 0.025——弱，大尺度聚团不压扁子结构）。 */
  projectStrength?: number
}

/** 布局结果：nodeId → 中心点坐标。 */
export type ForceLayout = Map<string, ForcePoint>

const EPS = 1e-6

interface Body { x: number; y: number; vx: number; vy: number; weight: number }

export interface ForceSimulator {
  /** 迭代 n 轮（alpha 续接，不重置）。 */
  step(iterations: number): void
  /** 增量加入节点（初始位置 = 画布中心 + 确定性微扰——从中心弹出被
   *  斥力推开，Obsidian 渐进生成观感；alpha 回升让网络重新活动）。 */
  addNode(id: string, weight?: number): void
  /** 当前布局快照。 */
  layout(): ForceLayout
  /** 当前温度 alpha（动画收尾判断）。 */
  alpha(): number
}

/**
 * 创建力导向模拟器（可持续迭代 + 增量加节点）。
 * nodes/edges/clusters 为**全量**引用（边按 id 存，body 增量加入后自动生效）；
 * 初始 nodes 非空时按簇散布摆位（全量一次算场景），否则空开始全走 addNode。
 */
export function createForceSimulator(
  nodes: ForceNodeInput[],
  edges: ForceEdgeInput[],
  opts: ForceLayoutOptions,
): ForceSimulator {
  const {
    width, height,
    charge = -300,
    spring = 0.1,
    springLength = 80,
    collideRadius = 24,
    centerStrength = 0.08,
    velocityDecay = 0.55,
    alphaDecay = 0.02,
    maxMove = 40,
    clusters,
    clusterTarget = 110,
    clusterStrength = 0.04,
    projectGroups,
    projectTarget = 320,
    projectStrength = 0.025,
  } = opts

  const cx = width / 2
  const cy = height / 2
  const ringRadius = Math.max(40, Math.min(width, height) / 2 - 60)

  const bodies = new Map<string, Body>()
  // 边索引（存 id——body 增量加入后 step 时动态取，弹簧自动生效）
  const edgePairs: Array<{ from: string; to: string }> = edges
    .filter((e) => e.from !== e.to)
    .map((e) => ({ from: e.from, to: e.to }))

  // alpha 温度：1 → alphaMin（alphaDecay 每轮衰减）
  let alpha = 1
  const alphaMin = 0.001

  // 确定性伪随机 [-0.5, 0.5)
  const jigOf = (i: number, salt = 0): number => ((i * 2654435761 + salt * 40503) % 1000) / 1000 - 0.5

  // 初始摆位：全量节点按簇散布（obsidian-graph-spawn——簇天然分离）
  if (nodes.length > 0) {
    if (clusters !== undefined && clusters.size > 0) {
      const byCluster = new Map<string, string[]>()
      for (const node of nodes) {
        const c = clusters.get(node.id) ?? '__solo__'
        const arr = byCluster.get(c) ?? []
        arr.push(node.id)
        byCluster.set(c, arr)
      }
      const clusterIds = [...byCluster.keys()]
      const clusterCount = clusterIds.length
      clusterIds.forEach((cid, ci) => {
        const angle = (2 * Math.PI * ci) / clusterCount
        const ccx = cx + ringRadius * Math.cos(angle)
        const ccy = cy + ringRadius * Math.sin(angle)
        const members = byCluster.get(cid)!
        members.forEach((id, mi) => {
          const inner = clusterCount === 1 ? 0 : (2 * Math.PI * mi) / Math.max(1, members.length)
          const rr = Math.min(70, 26 + (mi % 5) * 8)
          const jig = jigOf(mi)
          const w = nodes.find((nd) => nd.id === id)?.weight ?? 1
          bodies.set(id, {
            x: ccx + (rr + jig * 3) * Math.cos(inner) + jig * 3,
            y: ccy + (rr + jig * 3) * Math.sin(inner) + jig * 3,
            vx: 0, vy: 0,
            weight: Math.max(0.5, w),
          })
        })
      })
    } else {
      const n = nodes.length
      nodes.forEach((node, i) => {
        const angle = (2 * Math.PI * i) / n
        const jig = jigOf(i)
        bodies.set(node.id, {
          x: cx + (ringRadius + jig * 4) * Math.cos(angle) + jig * 4,
          y: cy + (ringRadius + jig * 4) * Math.sin(angle) + jig * 4,
          vx: 0, vy: 0,
          weight: Math.max(0.5, node.weight ?? 1),
        })
      })
    }
  }

  /** 单轮迭代（所有力 + 积分）。 */
  const iterate = (a: number): void => {
    const list = [...bodies.entries()]
    // ---- manyBody：库仑斥力 ----
    for (let i = 0; i < list.length; i += 1) {
      const [, bi] = list[i]!
      for (let j = i + 1; j < list.length; j += 1) {
        const [, bj] = list[j]!
        const x = bj.x - bi.x
        const y = bj.y - bi.y
        let l = x * x + y * y
        if (l < EPS) l = EPS
        const w = (charge * bi.weight * bj.weight * a) / l
        bi.vx += x * w
        bi.vy += y * w
        bj.vx -= x * w
        bj.vy -= y * w
      }
    }
    // ---- link：弹簧沿边 ----
    for (const { from, to } of edgePairs) {
      const s = bodies.get(from)
      const t = bodies.get(to)
      if (s === undefined || t === undefined) continue
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
    // ---- 同簇引力 ----
    if (clusters !== undefined && clusters.size > 1 && list.length > 1) {
      for (let i = 0; i < list.length; i += 1) {
        const [ida, bi] = list[i]!
        const ca = clusters.get(ida)
        if (ca === undefined) continue
        for (let j = i + 1; j < list.length; j += 1) {
          const [idb, bj] = list[j]!
          if (clusters.get(idb) !== ca) continue
          const x = bj.x - bi.x
          const y = bj.y - bi.y
          const dist = Math.sqrt(x * x + y * y)
          const dx = dist - clusterTarget
          if (dx > 0) {
            const w = (dx / dist) * a * clusterStrength
            bi.vx += x * w
            bi.vy += y * w
            bj.vx -= x * w
            bj.vy -= y * w
          }
        }
      }
    }
    // ---- 项目引力（v0.6：同项目节点聚成项目级大簇——比连通分量引力
    // 弱、距离远：大尺度聚团，内部子结构不被压扁）----
    if (projectGroups !== undefined && projectGroups.size > 1 && list.length > 1) {
      for (let i = 0; i < list.length; i += 1) {
        const [ida, bi] = list[i]!
        const pa = projectGroups.get(ida)
        if (pa === undefined) continue
        for (let j = i + 1; j < list.length; j += 1) {
          const [idb, bj] = list[j]!
          if (projectGroups.get(idb) !== pa) continue
          const x = bj.x - bi.x
          const y = bj.y - bi.y
          const dist = Math.sqrt(x * x + y * y)
          const dx = dist - projectTarget
          if (dx > 0) {
            const w = (dx / dist) * a * projectStrength
            bi.vx += x * w
            bi.vy += y * w
            bj.vx -= x * w
            bj.vy -= y * w
          }
        }
      }
    }
    // ---- collide：硬防重叠 ----
    for (let i = 0; i < list.length; i += 1) {
      const [, bi] = list[i]!
      for (let j = i + 1; j < list.length; j += 1) {
        const [, bj] = list[j]!
        const x = bj.x - bi.x
        const y = bj.y - bi.y
        let l = Math.sqrt(x * x + y * y)
        const r = collideRadius + collideRadius
        if (l < r) {
          if (l < EPS) {
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
    // ---- 软向心 ----
    for (const body of bodies.values()) {
      body.vx += (cx - body.x) * centerStrength * a
      body.vy += (cy - body.y) * centerStrength * a
    }
    // ---- 积分 + 速度衰减 + 限速 ----
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
    }
    // ---- forceCenter：质心平移居中 ----
    if (bodies.size > 0) {
      const mx = sx / bodies.size
      const my = sy / bodies.size
      for (const body of bodies.values()) {
        body.x += cx - mx
        body.y += cy - my
      }
    }
  }

  return {
    step(iterations: number): void {
      for (let iter = 0; iter < iterations; iter += 1) {
        alpha += (0 - alpha) * alphaDecay
        iterate(alpha)
      }
    },
    addNode(id: string, weight = 1): void {
      if (bodies.has(id)) return
      // 从画布中心附近弹出（确定性微扰）——被斥力推开 = Obsidian 观感
      const jig = jigOf(bodies.size, 7)
      bodies.set(id, {
        x: cx + jig * 24,
        y: cy + jigOf(bodies.size, 13) * 24,
        vx: 0, vy: 0,
        weight: Math.max(0.5, weight),
      })
      // 新节点加入 → 网络重新活动（alpha 回升）
      alpha = Math.max(alpha, 0.6)
    },
    layout(): ForceLayout {
      const out: ForceLayout = new Map()
      for (const [id, body] of bodies) out.set(id, { x: body.x, y: body.y })
      return out
    },
    alpha(): number {
      return alpha
    },
  }
}

/** 一次性布局（兼容旧 API）：创建模拟器 + 全量迭代。 */
export function layoutForce(
  nodes: ForceNodeInput[],
  edges: ForceEdgeInput[],
  opts: ForceLayoutOptions,
): ForceLayout {
  if (nodes.length === 0) return new Map()
  const sim = createForceSimulator(nodes, edges, opts)
  sim.step(opts.iterations ?? 500)
  return sim.layout()
}
