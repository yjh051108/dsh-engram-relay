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
  const dragRef = useRef<{ startX: number; startY: number; startView: typeof VIEW_DEFAULT; rect: DOMRect } | null>(null)

  // v0.5 缩放补偿：节点/文字尺寸随视口缩放**反向补偿**——屏幕大小 =
  // 世界值 × (svgWidth/vw)，要屏幕恒定 → 世界值 = 基准 × (vw/VIEW_W) =
  // 基准 ÷ zoomScale（⚠️ 曾误乘 zoomScale 越放大越大——已修正）。
  // 无限画布的要点：缩放只改变"看得多密"，不改变"元素多大"。
  const zoomScale = VIEW_W / view.vw

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

  // 最终布局（确定性力导向，全量一次收敛——布局质量与"静态图"一致；
  // v0.5 曾用增量模拟器实时演化，每帧迭代少 + alpha 回升导致旧节点被
  // 新节点推得乱晃 → 改为**预计算 + 入场插值**：动画只做"节点从中心
  // 飞向最终位置"，过程丝滑、终局不凌乱）。
  const finalLayout = useMemo(() => layoutForce(
    nodes.map((n) => ({ id: n.id, weight: 0.6 + n.importance })),
    edges.map((e) => ({ from: e.from, to: e.to })),
    {
      width: VIEW_W, height: VIEW_H, iterations: 500,
      // d3-force 风格参数（force.ts 注释）：负 charge = manyBody 斥力；
      // spring 0-1 弱标定；collide 硬防重叠；forceCenter 平移居中 +
      // forceX/forceY 软向心（弱弹簧拉回中心，网络铺开但不撞边界）。
      // collideRadius 24→34（容纳节点下方标签，防标签重叠）
      charge: -300,
      spring: 0.1,
      springLength: 90,
      collideRadius: 34,
      centerStrength: 0.08,
      clusters: clusterOf.size > 0 ? clusterOf : undefined,
      clusterTarget: 110,
      clusterStrength: 0.04,
    },
  ), [nodes, edges, clusterOf])

  // 入场动画进度 0→1（约 1.8 秒）：节点按 createdAt 时序依次从中心
  // 缓动飞向最终位置 + 淡入（Obsidian"动力学球一点点生成"的观感，
  // 但每个节点平滑到位——不凌乱）。
  const [animT, setAnimT] = useState(1)
  const animTRef = useRef(1)
  animTRef.current = animT
  useEffect(() => {
    if (nodes.length === 0) return
    setAnimT(0)
    let raf = 0
    let last = performance.now()
    const tick = (now: number): void => {
      const dt = (now - last) / 1000
      last = now
      const t = animTRef.current
      if (t < 1) {
        setAnimT(Math.min(1, t + dt * 0.55))
      } else {
        cancelAnimationFrame(raf)
        return
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [nodes, edges, clusterOf])

  // 节点时序序号（入场顺序 = createdAt 升序）
  const nodeOrder = useMemo(() => {
    const m = new Map<string, number>()
    nodes.slice()
      .sort((a, b) => a.createdAt - b.createdAt)
      .forEach((n, i) => m.set(n.id, i))
    return m
  }, [nodes])

  // 动画渲染用布局：位置 = 中心 → 最终位置（easeOutCubic 插值）
  const layout = useMemo(() => {
    const out: ForceLayout = new Map()
    const count = Math.max(1, nodes.length)
    const BATCH = 36
    for (const n of nodes) {
      const fp = finalLayout.get(n.id)
      if (fp === undefined) continue
      const idx = nodeOrder.get(n.id) ?? 0
      // 该节点在批次轴上的进度：animT×BATCH 推进时，节点依次 t 0→1
      const t = Math.max(0, Math.min(1, animT * BATCH - (idx / count) * BATCH))
      if (t >= 1) {
        out.set(n.id, { x: fp.x, y: fp.y })
      } else {
        const ease = 1 - (1 - t) ** 3 // easeOutCubic
        out.set(n.id, {
          x: VIEW_W / 2 + (fp.x - VIEW_W / 2) * ease,
          y: VIEW_H / 2 + (fp.y - VIEW_H / 2) * ease,
        })
      }
    }
    return out
  }, [finalLayout, animT, nodes, nodeOrder])

  // 节点淡入进度（动画期间）
  const fadeOf = (id: string): number => {
    if (animT >= 1) return 1
    const count = Math.max(1, nodes.length)
    const idx = nodeOrder.get(id) ?? 0
    return Math.max(0, Math.min(1, animT * 36 - (idx / count) * 36))
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
      const rect = svg.getBoundingClientRect()
      dragRef.current = { startX: e.clientX, startY: e.clientY, startView: viewRef.current, rect }
      e.preventDefault()
    }
    const onMove = (e: MouseEvent): void => {
      const d = dragRef.current
      if (d === null) return
      const dx = (e.clientX - d.startX) / d.rect.width // 屏幕像素 → 视口比例
      const dy = (e.clientY - d.startY) / d.rect.height
      setView((v) => ({
        ...v,
        vx: d.startView.vx - dx * d.startView.vw,
        vy: d.startView.vy - dy * d.startView.vh,
      }))
    }
    const onUp = (): void => { dragRef.current = null }
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
          <button className={styles.refresh} onClick={() => setView(VIEW_DEFAULT)}>{t('graph.reset')}</button>
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
              {/* 因果边方向箭头（v0.5 视觉优化：因果方向一目了然） */}
              <marker
                id="arrow-causes"
                viewBox="0 0 10 10"
                refX="9"
                refY="5"
                markerWidth={7 / zoomScale}
                markerHeight={7 / zoomScale}
                orient="auto-start-reverse"
              >
                <path d="M 0 1 L 9 5 L 0 9 z" fill="#7a8599" />
              </marker>
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
                  strokeWidth={1.5 / zoomScale}
                  strokeDasharray={c.multi ? '6 4' : undefined}
                />
                <text
                  x={c.cx} y={c.cy - c.radius + 20 / zoomScale}
                  textAnchor="middle"
                  className={styles.clusterLabel}
                  style={{
                    fontSize: 11 / zoomScale,
                    paintOrder: 'stroke',
                    stroke: 'rgba(10,14,22,0.85)',
                    strokeWidth: 3 / zoomScale,
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
                    strokeWidth={(isHighlighted ? 2.5 : 1.5) / zoomScale}
                    markerEnd="url(#arrow-causes)"
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
                  strokeWidth={(isHighlighted ? 2 : 1) / zoomScale}
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
              // 半径：度中心性为主（hub 大），semantic 加成，重要度微调
              const deg = degreeOf.get(n.id) ?? 0
              const rBase = 7 + Math.min(9, deg * 1.2) + (isSemantic ? 3 : 0) + n.importance * 1.5 + (selected ? 2 : 0)
              // ⚠️ 缩放补偿：屏幕大小 = 世界值 × (svgWidth/vw) → 要屏幕
              // 恒定，世界坐标半径 = 屏幕基准 ÷ zoomScale（放大后节点/文字
              // 保持屏幕大小，看的是更稀疏更清楚，不是更大）
              const r = rBase / zoomScale
              const inHighlight = highlight !== null && highlight.nodeIds.has(n.id)
              const dimmed = highlight !== null && !inHighlight
              // 入场淡入（动画期间节点依次出现）
              const fade = fadeOf(n.id)
              const opacity = (dimmed ? 0.12 : 1) * fade
              return (
                <g
                  key={n.id}
                  className={`${styles.node} ${isSemantic ? styles.nodeSemantic : ''}`}
                  onClick={() => setSelectedId(n.id)}
                  onDoubleClick={() => openDetail(n)}
                  role="button"
                  tabIndex={0}
                  opacity={opacity}
                  style={{ transition: 'opacity 0.2s' }}
                >
                  {isSemantic && (
                    <circle cx={p.x} cy={p.y} r={r + 9 / zoomScale} fill="url(#halo-grad)" pointerEvents="none" />
                  )}
                  <circle
                    cx={p.x} cy={p.y}
                    r={r}
                    fill={color}
                    fillOpacity={selected ? 1 : isSemantic ? 0.95 : 0.82}
                    stroke={selected ? '#fff' : isSemantic ? 'rgba(255,255,255,0.6)' : 'rgba(255,255,255,0.35)'}
                    strokeWidth={(selected ? 2 : isSemantic ? 1.5 : 1) / zoomScale}
                  />
                  <text
                    x={p.x} y={p.y + r + 12 / zoomScale}
                    textAnchor="middle"
                    className={isSemantic ? styles.nodeLabelSemantic : styles.nodeLabel}
                    style={{
                      fontSize: (isSemantic ? 11 : 10) / zoomScale,
                      paintOrder: 'stroke',
                      stroke: 'rgba(10,14,22,0.9)',
                      strokeWidth: 3 / zoomScale,
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
