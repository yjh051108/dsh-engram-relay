/**
 * GraphView — 记忆图谱可视化（DSH 会话页「图谱」Tab）。
 *
 * 数据面：host 的 /engram-relay/api/graph（分层准入：global + 本目录
 * project + 本会话 session）。渲染：确定性力导向布局（force.ts）+ SVG。
 *  - 节点 = 记忆（颜色按层：global 蓝 / project 绿 / session 橙）
 *  - 实线边 = 因果（causes），虚线边 = 双向链接（link）
 *  - 点击节点 → 拉取详情（渐进披露第二层：正文/前因/后果/关联）
 *  - 层过滤 + 刷新
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { layoutForce, type ForceLayout, type ForcePoint } from './force.ts'
import styles from './graph.module.css'

export interface GraphNodeData {
  id: string
  title: string
  summary: string
  kind: string
  layer: 'global' | 'project'
  projectId: string | null
  importance: number
  hits: number
  createdAt: number
  /** 巩固状态（v0.5 视觉分层：semantic=固化知识突出显示）。 */
  state: string
}

export interface GraphEdgeData {
  from: string
  to: string
  kind: 'causes' | 'link'
}

interface GraphData {
  nodes: GraphNodeData[]
  edges: GraphEdgeData[]
  total: number
}

interface NodeDetail extends GraphNodeData {
  content: string
  causes: GraphNodeData[]
  effects: GraphNodeData[]
  links: GraphNodeData[]
}

/** 层 → 颜色。 */
export const LAYER_COLORS: Record<string, string> = {
  global: '#4a7dff',
  project: '#34c98a',
}

const VIEW_W = 900
const VIEW_H = 620

interface GraphViewProps {
  t: (key: string, params?: Record<string, unknown>) => string
  /** 当前会话 id（host 端据此解析工作目录做分层准入）。 */
  sessionId?: string
}

export function GraphView({ t, sessionId }: GraphViewProps) {
  const [data, setData] = useState<GraphData | null>(null)
  const [detail, setDetail] = useState<NodeDetail | null>(null)
  const [filter, setFilter] = useState<string>('all')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  // 无限画布视口（viewBox 四元组）：世界坐标无限，视口自由缩放/平移。
  //  vx/vy = 视口左上角世界坐标，vw/vh = 视口尺寸（世界单位）
  const VIEW_DEFAULT = { vx: 0, vy: 0, vw: VIEW_W, vh: VIEW_H }
  const [view, setView] = useState(VIEW_DEFAULT)
  const viewRef = useRef(view)
  viewRef.current = view
  const svgRef = useRef<SVGSVGElement>(null)
  const dragRef = useRef<{ startX: number; startY: number; startView: typeof VIEW_DEFAULT; rect: DOMRect; moved: boolean } | null>(null)

  // v0.5 缩放补偿：节点/文字尺寸随视口缩放**反向补偿**——屏幕大小 =
  // 世界值 × (svgWidth/vw)，要屏幕恒定 → 世界值 = 基准 × (vw/VIEW_W) =
  // 基准 ÷ zoomScale（⚠️ 曾误乘 zoomScale 越放大越大——已修正）。
  // v0.6 折中（用户要求"缩得极小节点也得缩小"）：**放大恒定、缩小跟随**——
  // 补偿系数 zc = max(zoomScale, 1)：放大（zoomScale>1）屏幕恒定；
  // 缩小（zoomScale<1）世界尺寸不变 → 屏幕按比例变小，不挤成一坨。
  const zoomScale = VIEW_W / view.vw
  const zc = Math.max(zoomScale, 1)

  const loadGraph = (): void => {
    setLoading(true)
    setError('')
    const q = sessionId !== undefined && sessionId !== '' ? `?sessionId=${encodeURIComponent(sessionId)}` : ''
    void fetch(`/engram-relay/api/graph${q}`)
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then((d: GraphData) => setData(d))
      .catch((e: unknown) => setError(String((e as Error)?.message ?? e)))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    loadGraph()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId])

  // 层过滤（确定性力导向布局在簇计算之后——布局需要 clusterOf）。
  const { nodes, edges } = useMemo(() => {
    if (data === null) return { nodes: [], edges: [] }
    const visible = filter === 'all' ? data.nodes : data.nodes.filter((n) => n.layer === filter)
    const ids = new Set(visible.map((n) => n.id))
    const visibleEdges = data.edges.filter((e) => ids.has(e.from) && ids.has(e.to))
    return { nodes: visible, edges: visibleEdges }
  }, [data, filter])

  // 自发簇（v0.4 终态优先）：前端对边做连通分量——簇由链接/因果密度
  // 自然形成（不依赖项目标签）。项目标签只作簇名（硬编码簇 = 脚手架；
  // 簇跨项目 = 融会贯通/套娃在图上可见）。
  const clusters = useMemo(() => {
    if (nodes.length === 0) return { list: [], clusterOf: new Map<string, string>() }
    const adj = new Map<string, Set<string>>()
    for (const n of nodes) adj.set(n.id, new Set())
    for (const e of edges) {
      adj.get(e.from)?.add(e.to)
      adj.get(e.to)?.add(e.from)
    }
    const visited = new Set<string>()
    const list: Array<{ ids: string[]; projects: Set<string | null> }> = []
    const clusterOf = new Map<string, string>()
    for (const n of nodes) {
      if (visited.has(n.id)) continue
      const ids: string[] = []
      const projects = new Set<string | null>()
      const queue = [n.id]
      visited.add(n.id)
      while (queue.length > 0) {
        const id = queue.shift()!
        ids.push(id)
        projects.add(nodes.find((x) => x.id === id)?.projectId ?? null)
        for (const nb of adj.get(id) ?? []) {
          if (!visited.has(nb)) { visited.add(nb); queue.push(nb) }
        }
      }
      const cid = `c${list.length}`
      for (const id of ids) clusterOf.set(id, cid)
      list.push({ ids, projects })
    }
    return { list, clusterOf }
  }, [nodes, edges])
  const clusterList = clusters.list
  const clusterOf = clusters.clusterOf

  // ---- 实时演化布局（v0.6 用户要求：**实时渲染，不先算完再做动画**）----
  // 增量力导向模拟器：节点按 createdAt 时序分帧加入（从中心弹出被斥力
  // 推开），每帧 step(60) 实时演化渲染——加载即渲染、边加载边生长
  // （Obsidian 观感），**无"先算完布局"的等待**。alpha 回升温和避免
  // 旧节点乱晃；速度衰减 0.6 更平滑。
  const [layout, setLayout] = useState<ForceLayout>(new Map())
  const joinedAtRef = useRef(new Map<string, number>())
  const simRef = useRef<ForceSimulator | null>(null)
  useEffect(() => {
    if (nodes.length === 0) {
      setLayout(new Map())
      return
    }
    setTarget(null) // 切换过滤/重新加载 → 重置运镜目标（越界检测接管）
    const sim = createForceSimulator([], edges.map((e) => ({ from: e.from, to: e.to })), {
      width: VIEW_W, height: VIEW_H,
      charge: -100,
      spring: 0.1,
      springLength: 110,
      collideRadius: 30,
      centerStrength: 0.08,
      velocityDecay: 0.6,
      clusters: clusterOf.size > 0 ? clusterOf : undefined,
      clusterTarget: 110,
      clusterStrength: 0.04,
    })
    simRef.current = sim
    joinedAtRef.current = new Map()
    const sorted = [...nodes].sort((a, b) => a.createdAt - b.createdAt)
    let idx = 0
    let raf = 0
    const tick = (): void => {
      // 分帧加入（约 45 批——加载即渲染，无需等待）
      const batch = Math.max(3, Math.ceil(nodes.length / 45))
      for (let i = 0; i < batch && idx < sorted.length; i += 1, idx += 1) {
        const n = sorted[idx]!
        sim.addNode(n.id, 0.6 + n.importance)
        joinedAtRef.current.set(n.id, performance.now())
      }
      // 每帧 60 迭代——平滑演化（之前 24 太糙会乱晃）
      sim.step(60)
      setLayout(sim.layout())
      if (idx < sorted.length || sim.alpha() > 0.004) {
        raf = requestAnimationFrame(tick)
      }
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [nodes, edges, clusterOf])

  // 节点生长（v0.6 "繁殖壮大"感）：加入后 350ms 内从 0.12 小点弹性
  // 膨胀到 1.08 再回落 1（easeOutBack 微过冲）——"砰"地长出来
  const growScaleOf = (id: string): number => {
    const joined = joinedAtRef.current.get(id)
    if (joined === undefined) return 1
    const t = Math.min(1, (performance.now() - joined) / 350)
    const c1 = 1.70158
    const c3 = c1 + 1
    const u = t - 1
    return Math.max(0.12, 1 + c3 * u ** 3 + c1 * u ** 2)
  }
  const fadeOf = (id: string): number => {
    const joined = joinedAtRef.current.get(id)
    if (joined === undefined) return 1
    return Math.min(1, (performance.now() - joined) / 350)
  }

  // 节点度（链接数——取经 Obsidian Dynamic-Node-Size：hub 节点大小=骨架）
  const degreeOf = useMemo(() => {
    const deg = new Map<string, number>()
    for (const e of edges) {
      deg.set(e.from, (deg.get(e.from) ?? 0) + 1)
      deg.set(e.to, (deg.get(e.to) ?? 0) + 1)
    }
    return deg
  }, [edges])

  // 选中高亮（v0.5：**单击节点只高亮延展边，不弹详情框**——详情改双击
  // 打开；点空白取消高亮回到总图）。邻居边/邻居节点高亮，其余淡出。
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const highlight = useMemo(() => {
    const sel = selectedId
    if (sel === null) return null
    const nodeIds = new Set<string>([sel])
    const edgeKeys = new Set<string>()
    for (const e of edges) {
      if (e.from === sel || e.to === sel) {
        nodeIds.add(e.from)
        nodeIds.add(e.to)
        edgeKeys.add(`${e.from}|${e.to}|${e.kind}`)
      }
    }
    return { nodeIds, edgeKeys }
  }, [selectedId, edges])

  // ---- 闭环运镜系统（v0.6 用户建议：不开环瞬移——感知画面内外 +
  // 平滑摄像头移动）----
  // 目标视口计算（fitToCurrent：当前布局包围盒 ∪ 中心点——生成阶段
  // 节点从中心弹出，中心必须始终在画面内）
  const fitToCurrent = (): { vx: number; vy: number; vw: number; vh: number } | null => {
    const pts = nodes
      .map((n) => layout.get(n.id))
      .filter((p): p is ForcePoint => p !== undefined)
    if (pts.length === 0) return null
    pts.push({ x: VIEW_W / 2, y: VIEW_H / 2 }) // 闭环：中心始终在画面内
    const pad = 80
    const minX = pts.reduce((s, p) => Math.min(s, p.x), Infinity) - pad
    const maxX = pts.reduce((s, p) => Math.max(s, p.x), -Infinity) + pad
    const minY = pts.reduce((s, p) => Math.min(s, p.y), Infinity) - pad
    const maxY = pts.reduce((s, p) => Math.max(s, p.y), -Infinity) + pad
    return {
      vx: minX,
      vy: minY,
      vw: Math.max(200, maxX - minX),
      vh: Math.max(150, maxY - minY),
    }
  }
  const [target, setTarget] = useState<{ vx: number; vy: number; vw: number; vh: number } | null>(null)
  const targetRef = useRef(target)
  targetRef.current = target
  // fitToCurrent 的最新引用（拖拽 effect 的 onUp 闭包需要）
  const fitToCurrentRef = useRef(fitToCurrent)
  fitToCurrentRef.current = fitToCurrent
  // 用户手动缩放/拖动 → 停止自动运镜（避免打架）；0.6s 无操作后恢复跟随
  const lastUserOpRef = useRef(0)
  const isUserControlled = (): boolean => Date.now() - lastUserOpRef.current < 600
  // 闭环：目标视口变化（越界/回全图）→ PID 式指数平滑运镜（时间常数
  // 0.25s，等效 PD：平滑逼近无振荡，不瞬移）
  useEffect(() => {
    if (target === null) return
    let raf = 0
    let last = performance.now()
    const tick = (now: number): void => {
      const dt = Math.min(0.1, (now - last) / 1000)
      last = now
      // 用户正在操作 → 暂停跟随（等用户停手）
      if (isUserControlled()) {
        raf = requestAnimationFrame(tick)
        return
      }
      setView((v) => {
        const k = 1 - Math.exp(-dt / 0.25) // 指数平滑（PD 等效）
        const nv = {
          vx: v.vx + (target.vx - v.vx) * k,
          vy: v.vy + (target.vy - v.vy) * k,
          vw: v.vw + (target.vw - v.vw) * k,
          vh: v.vh + (target.vh - v.vh) * k,
        }
        const dv = Math.abs(nv.vx - target.vx) + Math.abs(nv.vy - target.vy)
          + Math.abs(nv.vw - target.vw) + Math.abs(nv.vh - target.vh)
        if (dv < 1) {
          // 已到位：直接设为目标并停（避免无限逼近）
          return target
        }
        return nv
      })
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [target])
  // 闭环越界检测（生成阶段绝对显示全）：实时演化期间每帧检查当前
  // 布局是否都在画面内（含余量）——越界则更新目标视口（视口平滑
  // 跟随生长中的网络展开）
  useEffect(() => {
    if (isUserControlled()) return
    const pts = nodes
      .map((n) => layout.get(n.id))
      .filter((p): p is ForcePoint => p !== undefined)
    if (pts.length === 0) return
    const pad = 40
    const minX = pts.reduce((s, p) => Math.min(s, p.x), Infinity) - pad
    const maxX = pts.reduce((s, p) => Math.max(s, p.x), -Infinity) + pad
    const minY = pts.reduce((s, p) => Math.min(s, p.y), Infinity) - pad
    const maxY = pts.reduce((s, p) => Math.max(s, p.y), -Infinity) + pad
    const cur = viewRef.current
    // 闭环判定：当前画面（含 padding 余量）是否完整包含所有节点
    const inside = cur.vx <= minX && cur.vx + cur.vw >= maxX && cur.vy <= minY && cur.vy + cur.vh >= maxY
    if (!inside) {
      setTarget({
        vx: minX,
        vy: minY,
        vw: Math.max(200, maxX - minX),
        vh: Math.max(150, maxY - minY),
      })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout, nodes])

  // 簇大圆：质心 + 半径（≥2 节点才画，避免视觉噪音）
  const clusterCircles = useMemo(() => clusterList
    .filter((c) => c.ids.length >= 2)
    .map((c) => {
      const pts = c.ids.map((id) => layout.get(id)).filter((p): p is ForcePoint => p !== undefined)
      if (pts.length === 0) return null
      const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length
      const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length
      const radius = Math.max(60, ...pts.map((p) => Math.hypot(p.x - cx, p.y - cy))) + 40
      const projNames = [...c.projects].map((p) => {
        if (p === null) return '通用'
        const parts = String(p).split(/[\\/]/)
        return parts[parts.length - 1] || '项目'
      })
      const label = c.projects.size > 1 ? `${projNames.join(' ⇄ ')}（融合）` : projNames[0]
      return { cx, cy, radius, label, multi: c.projects.size > 1 }
    })
    .filter((c): c is { cx: number; cy: number; radius: number; label: string; multi: boolean } => c !== null),
  [clusterList, layout])

  /** 项目着色（v0.4）：projectId 哈希取色；null（通用知识）灰色。 */
  const projectColor = (projectId: string | null): string => {
    if (projectId === null) return '#8a94a6'
    let h = 0
    for (const ch of projectId) h = (h * 31 + ch.charCodeAt(0)) >>> 0
    return `hsl(${h % 360} 55% 55%)`
  }

  /** 簇着色（v0.5 取经 Obsidian color groups：**同簇同色**——分类感知的
   *  核心。簇色相从簇 id 确定性哈希；孤立节点回退项目色）。 */
  const clusterColor = (clusterId: string): string => {
    let h = 0
    for (const ch of clusterId) h = (h * 131 + ch.charCodeAt(0)) >>> 0
    // 色相错开（黄金角）+ 高饱和亮色（暗色背景下醒目）
    return `hsl(${(h + 35) % 360} 65% 62%)`
  }

  // Ctrl+滚轮缩放（围绕鼠标位置，作用于视口）。必须用原生非 passive 监听：
  // React 的 onWheel 在根容器以 passive 注册，preventDefault 拦不住页面缩放。
  useEffect(() => {
    const svg = svgRef.current
    if (svg === null) return
    const onWheel = (e: WheelEvent): void => {
      if (!e.ctrlKey) return
      e.preventDefault()
      lastUserOpRef.current = Date.now() // 用户操作 → 暂停自动运镜
      const rect = svg.getBoundingClientRect()
      const mx = (e.clientX - rect.left) / rect.width // 鼠标归一化位置 [0,1]
      const my = (e.clientY - rect.top) / rect.height
      // viewBox 语义：vw 变小 = 看到更少 = 放大。故上滚（deltaY<0）缩小 vw
      // （放大），下滚（deltaY>0）增大 vw（缩小）——与 Obsidian 方向一致。
      const factor = e.deltaY < 0 ? 1 / 1.15 : 1.15
      setView((v) => {
        const vw2 = Math.min(100000, Math.max(10, v.vw * factor))
        const vh2 = v.vh * factor
        // 保持鼠标下的世界点在缩放前后不动：w = vx + mx·vw
        const wx = v.vx + mx * v.vw
        const wy = v.vy + my * v.vh
        return { vx: wx - mx * vw2, vy: wy - my * vh2, vw: vw2, vh: vh2 }
      })
    }
    svg.addEventListener('wheel', onWheel, { passive: false })
    return () => svg.removeEventListener('wheel', onWheel)
  }, [loading, nodes.length])

  // 拖拽平移视口（无限画布：空白处按下拖动 = 世界跟随鼠标）。
  // 节点/边上的按下不启动拖拽——保留点击高亮；空白按下同时取消高亮
  // （v0.5 用户要求：点空白回到总图，不卡在选中态）。
  useEffect(() => {
    const svg = svgRef.current
    if (svg === null) return
    const onDown = (e: MouseEvent): void => {
      if (e.button !== 0) return
      const target = e.target as Element
      if (target !== svg && target.tagName !== 'svg') return // 空白处才拖拽
      setSelectedId(null) // 点空白取消高亮（回到总图）
      lastUserOpRef.current = Date.now() // 用户操作 → 暂停自动运镜
      const rect = svg.getBoundingClientRect()
      dragRef.current = { startX: e.clientX, startY: e.clientY, startView: viewRef.current, rect, moved: false }
      e.preventDefault()
    }
    const onMove = (e: MouseEvent): void => {
      const d = dragRef.current
      if (d === null) return
      // 记录移动量（区分单击/拖拽）
      d.moved = d.moved || Math.hypot(e.clientX - d.startX, e.clientY - d.startY) > 3
      const dx = (e.clientX - d.startX) / d.rect.width // 屏幕像素 → 视口比例
      const dy = (e.clientY - d.startY) / d.rect.height
      setView((v) => ({
        ...v,
        vx: d.startView.vx - dx * d.startView.vw,
        vy: d.startView.vy - dy * d.startView.vh,
      }))
    }
    const onUp = (): void => {
      const d = dragRef.current
      dragRef.current = null
      // 单击空白（无位移）= 取消高亮 + **运镜回全图**（v0.6 用户要求）
      if (d !== null && !d.moved) {
        setSelectedId(null)
        const f = fitToCurrentRef.current()
        if (f !== null) setTarget(f)
      }
    }
    svg.addEventListener('mousedown', onDown)
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      svg.removeEventListener('mousedown', onDown)
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
  }, [loading, nodes.length])

  const openDetail = (node: GraphNodeData): void => {
    const q = sessionId !== undefined && sessionId !== '' ? `?sessionId=${encodeURIComponent(sessionId)}` : ''
    void fetch(`/engram-relay/api/node/${encodeURIComponent(node.title)}${q}`)
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then((d: NodeDetail) => setDetail(d))
      .catch((e: unknown) => setError(String((e as Error)?.message ?? e)))
  }

  return (
    <div className={styles.root}>
      <div className={styles.toolbar}>
        <div className={styles.filters} title={t('graph.filter.title')}>
          {['all', 'global', 'project'].map((f) => (
            <button
              key={f}
              className={`${styles.filterBtn} ${filter === f ? styles.filterActive : ''}`}
              onClick={() => setFilter(f)}
              title={f === 'all' ? t('graph.filter.title') : `${t(`graph.filter.${f}`)} — ${t('graph.filter.title')}`}
            >
              {t(`graph.filter.${f}`)}
            </button>
          ))}
        </div>
        <div className={styles.meta}>
          <span className={styles.count}>
            {data !== null ? t('graph.count', { nodes: nodes.length, edges: edges.length }) : ''}
          </span>
          <button className={styles.refresh} onClick={() => { const f = fitToCurrent(); if (f !== null) setTarget(f) }}>{t('graph.reset')}</button>
          <button className={styles.refresh} onClick={loadGraph}>{t('graph.refresh')}</button>
        </div>
      </div>

      {error !== '' && <div className={styles.error}>{error}</div>}

      {loading && <div className={styles.state}>{t('graph.loading')}</div>}

      {!loading && data !== null && nodes.length === 0 && (
        <div className={styles.state}>{t('graph.empty')}</div>
      )}

      {!loading && nodes.length > 0 && (
        <div className={styles.canvas}>
          {/* 无限画布：动态 viewBox 即视口（世界坐标无限，缩放/拖拽只动视口） */}
          <svg
            ref={svgRef}
            viewBox={`${view.vx} ${view.vy} ${view.vw} ${view.vh}`}
            className={styles.svg}
          >
            <defs>
              {/* 背景点阵（无限画布尺度感） */}
              <pattern id="dot-grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <circle cx="1.5" cy="1.5" r="1.2" fill="rgba(255,255,255,0.06)" />
              </pattern>
              {/* semantic 节点光环（固化知识发光） */}
              <radialGradient id="halo-grad">
                <stop offset="0%" stopColor="#4a7dff" stopOpacity="0.28" />
                <stop offset="100%" stopColor="#4a7dff" stopOpacity="0" />
              </radialGradient>
            </defs>
            {/* 背景点阵层 */}
            <rect
              x={view.vx - 2000}
              y={view.vy - 2000}
              width={view.vw + 4000}
              height={view.vh + 4000}
              fill="url(#dot-grid)"
            />
            {/* 自发簇大圆（连通分量；项目标签作簇名——簇跨项目即融合/套娃） */}
            {clusterCircles.map((c, i) => (
              <g key={`cluster-${i}`} className={styles.cluster}>
                <circle
                  cx={c.cx} cy={c.cy} r={c.radius}
                  fill={c.multi ? 'rgba(138,148,166,0.08)' : 'rgba(255,255,255,0.05)'}
                  stroke={c.multi ? 'rgba(138,148,166,0.5)' : 'rgba(255,255,255,0.16)'}
                  strokeWidth={1.5 / zc}
                  strokeDasharray={c.multi ? '6 4' : undefined}
                />
                <text
                  x={c.cx} y={c.cy - c.radius + 20 / zc}
                  textAnchor="middle"
                  className={styles.clusterLabel}
                  style={{
                    fontSize: 11 / zc,
                    paintOrder: 'stroke',
                    stroke: 'rgba(10,14,22,0.85)',
                    strokeWidth: 3 / zc,
                  }}
                >
                  {c.label}
                </text>
              </g>
            ))}
            {/* 边（v0.5：causes 带方向箭头 + 簇色；选中节点时其延展边
                高亮（亮色加粗），其余边淡出） */}
            {edges.map((e) => {
              const a = layout.get(e.from)
              const b = layout.get(e.to)
              if (a === undefined || b === undefined) return null
              const key = `${e.from}|${e.to}|${e.kind}`
              const isHighlighted = highlight !== null && highlight.edgeKeys.has(key)
              const dimmed = highlight !== null && !isHighlighted
              // 入场淡入：两端节点都出现后边才淡入（min 两端 fade）
              const edgeFade = Math.min(fadeOf(e.from), fadeOf(e.to))
              const baseOpacity = dimmed ? 0.04 : isHighlighted ? 0.95 : 0.35
              if (e.kind === 'causes') {
                const src = nodes.find((n) => n.id === e.from)
                const scid = src ? clusterOf.get(src.id) : undefined
                const color = scid !== undefined
                  ? clusterColor(scid)
                  : (src ? projectColor(src.projectId) : '#8a94a6')
                return (
                  <line
                    key={key}
                    x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                    stroke={isHighlighted ? '#ffd166' : color}
                    strokeOpacity={baseOpacity * edgeFade}
                    strokeWidth={(isHighlighted ? 2.5 : 1.5) / zc}
                    className={styles.edge}
                  />
                )
              }
              return (
                <line
                  key={key}
                  x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                  stroke={isHighlighted ? '#ffd166' : '#aab2c0'}
                  strokeOpacity={baseOpacity * edgeFade}
                  strokeWidth={(isHighlighted ? 2 : 1) / zc}
                  strokeDasharray="5 4"
                  className={styles.edge}
                />
              )
            })}
            {/* 节点（v0.5：同簇同色 + 大小按度 + semantic 光环 + 标签 halo；
                缩放补偿（尺寸/文字随 zoomScale 保持屏幕大小）；
                选中时其邻居高亮，其余淡出） */}
            {nodes.map((n) => {
              const p = layout.get(n.id)
              if (p === undefined) return null
              // 簇色优先（同簇同色）；孤立节点回退项目色（灰色）
              const cid = clusterOf.get(n.id)
              const color = cid !== undefined ? clusterColor(cid) : projectColor(n.projectId)
              const selected = selectedId === n.id
              const isSemantic = n.state === 'semantic'
              // v0.6 事件弱化：event（过程性记忆）缩小 + 降透明——知识
              // （fact/decision/note）突出，过时过程不抢眼
              const isEvent = n.kind === 'event'
              // 半径：度中心性为主（hub 大），semantic 加成，重要度微调，
              // event 弱化（×0.7）
              const deg = degreeOf.get(n.id) ?? 0
              const rBase = (7 + Math.min(9, deg * 1.2) + (isSemantic ? 3 : 0) + n.importance * 1.5) * (isEvent ? 0.7 : 1) + (selected ? 2 : 0)
              // ⚠️ 缩放补偿：屏幕大小 = 世界值 × (svgWidth/vw) → 要屏幕
              // 恒定，世界坐标半径 = 屏幕基准 ÷ zoomScale（放大后节点/文字
              // 保持屏幕大小，看的是更稀疏更清楚，不是更大）
              const r = rBase / zc
              const inHighlight = highlight !== null && highlight.nodeIds.has(n.id)
              const dimmed = highlight !== null && !inHighlight
              // 入场淡入（动画期间节点依次出现）+ 事件弱化（透明度低一档）
              const fade = fadeOf(n.id)
              const eventDim = isEvent ? 0.55 : 1
              const opacity = (dimmed ? 0.12 : 1) * fade * eventDim
              // 弹性生长缩放（v0.6："自我繁殖壮大"感——从 0.12 小点
              // 膨胀到 1.08 再回落 1，easeOutBack 微过冲）
              const grow = growScaleOf(n.id)
              return (
                <g
                  key={n.id}
                  className={`${styles.node} ${isSemantic ? styles.nodeSemantic : ''}`}
                  onClick={() => { setSelectedId(n.id); setDetail(null) }}
                  onDoubleClick={() => openDetail(n)}
                  role="button"
                  tabIndex={0}
                  opacity={opacity}
                  transform={`translate(${p.x}, ${p.y}) scale(${grow}) translate(${-p.x}, ${-p.y})`}
                  style={{ transition: 'opacity 0.2s' }}
                >
                  {isSemantic && (
                    <circle cx={p.x} cy={p.y} r={r + 9 / zc} fill="url(#halo-grad)" pointerEvents="none" />
                  )}
                  <circle
                    cx={p.x} cy={p.y}
                    r={r}
                    fill={color}
                    fillOpacity={selected ? 1 : isSemantic ? 0.95 : 0.82}
                    stroke={selected ? '#fff' : isSemantic ? 'rgba(255,255,255,0.6)' : 'rgba(255,255,255,0.35)'}
                    strokeWidth={(selected ? 2 : isSemantic ? 1.5 : 1) / zc}
                  />
                  <text
                    x={p.x} y={p.y + r + 12 / zc}
                    textAnchor="middle"
                    className={isSemantic ? styles.nodeLabelSemantic : styles.nodeLabel}
                    style={{
                      fontSize: (isSemantic ? 11 : 10) / zc,
                      paintOrder: 'stroke',
                      stroke: 'rgba(10,14,22,0.9)',
                      strokeWidth: 3 / zc,
                    }}
                  >
                    {n.title.length > 14 ? `${n.title.slice(0, 13)}…` : n.title}
                  </text>
                </g>
              )
            })}
          </svg>

          <div className={styles.legend}>
            <span><span className={styles.legendLineSolid} />{t('graph.legend.causes')}</span>
            <span><span className={styles.legendLineDash} />{t('graph.legend.link')}</span>
            <span>
              <span className={styles.legendDot} style={{ background: clusterColor('c0') }} />
              {t('graph.legend.cluster')}
            </span>
            <span>
              <span className={styles.legendDot} style={{ background: projectColor(null) }} />
              {t('graph.legend.solo')}
            </span>
            <span><span className={styles.legendCluster} />{t('graph.legend.clusterRing')}</span>
          </div>
        </div>
      )}

      {/* 详情侧栏（渐进披露第二层） */}
      {detail !== null && (
        <div className={styles.detail}>
          <div className={styles.detailHead}>
            <span className={styles.detailTitle}>
              [[{detail.title}]] ({detail.kind} · {t(`graph.layer.${detail.layer}`)})
            </span>
            <button className={styles.detailClose} onClick={() => setDetail(null)}>{t('graph.detail.close')}</button>
          </div>
          <p className={styles.detailSummary}>{detail.summary}</p>
          {detail.content !== '' && (
            <pre className={styles.detailContent}>{detail.content}</pre>
          )}
          <div className={styles.detailSection}>
            <span className={styles.detailSectionTitle}>{t('graph.detail.causes')}</span>
            {detail.causes.length > 0
              ? detail.causes.map((c) => (
                <button key={c.id} className={styles.detailNode} onClick={() => openDetail(c)}>[[{c.title}]]</button>
              ))
              : <span className={styles.detailNone}>{t('graph.detail.none')}</span>}
          </div>
          <div className={styles.detailSection}>
            <span className={styles.detailSectionTitle}>{t('graph.detail.effects')}</span>
            {detail.effects.length > 0
              ? detail.effects.map((c) => (
                <button key={c.id} className={styles.detailNode} onClick={() => openDetail(c)}>[[{c.title}]]</button>
              ))
              : <span className={styles.detailNone}>{t('graph.detail.none')}</span>}
          </div>
          <div className={styles.detailSection}>
            <span className={styles.detailSectionTitle}>{t('graph.detail.links')}</span>
            {detail.links.length > 0
              ? detail.links.map((c) => (
                <button key={c.id !== '' ? c.id : c.title} className={styles.detailNode} onClick={() => c.id !== '' && openDetail(c)}>
                  [[{c.title}]]
                </button>
              ))
              : <span className={styles.detailNone}>{t('graph.detail.none')}</span>}
          </div>
        </div>
      )}
    </div>
  )
}
