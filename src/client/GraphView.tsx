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
import { layoutForce, type ForcePoint } from './force.ts'
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

  // 层过滤 + 布局（确定性力导向）。
  const { nodes, edges, layout } = useMemo(() => {
    if (data === null) return { nodes: [], edges: [], layout: new Map<string, ForcePoint>() }
    const visible = filter === 'all' ? data.nodes : data.nodes.filter((n) => n.layer === filter)
    const ids = new Set(visible.map((n) => n.id))
    const visibleEdges = data.edges.filter((e) => ids.has(e.from) && ids.has(e.to))
    const layout = layoutForce(
      visible.map((n) => ({ id: n.id, weight: 0.6 + n.importance })),
      visibleEdges.map((e) => ({ from: e.from, to: e.to })),
      {
        width: VIEW_W, height: VIEW_H, iterations: 500,
        // d3-force 风格参数（force.ts 注释）：负 charge = manyBody 斥力；
        // spring 0-1 弱标定；collide 硬防重叠；forceCenter 平移居中 +
        // forceX/forceY 软向心（弱弹簧拉回中心，网络铺开但不撞边界）。
        // v0.5 视觉优化：collideRadius 24→34（容纳节点下方标签，防标签
        // 重叠——力导向图最扎眼的视觉问题）；springLength 80→90（边舒展）。
        charge: -300,
        spring: 0.1,
        springLength: 90,
        collideRadius: 34,
        centerStrength: 0.08,
      },
    )
    return { nodes: visible, edges: visibleEdges, layout }
  }, [data, filter])

  // 自发簇（v0.4 终态优先）：前端对边做连通分量——簇由链接/因果密度
  // 自然形成（不依赖项目标签）。项目标签只作着色（硬编码簇 = 脚手架；
  // 簇跨项目 = 融会贯通/套娃在图上可见）。
  const clusters = useMemo(() => {
    if (nodes.length === 0) return []
    const adj = new Map<string, Set<string>>()
    for (const n of nodes) adj.set(n.id, new Set())
    for (const e of edges) {
      adj.get(e.from)?.add(e.to)
      adj.get(e.to)?.add(e.from)
    }
    const visited = new Set<string>()
    const out: Array<{ ids: string[]; projects: Set<string | null> }> = []
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
      out.push({ ids, projects })
    }
    return out
  }, [nodes, edges])

  // 簇大圆：质心 + 半径（≥2 节点才画，避免视觉噪音）
  const clusterCircles = useMemo(() => clusters
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
  [clusters, layout])

  /** 项目着色（v0.4）：projectId 哈希取色；null（通用知识）灰色。 */
  const projectColor = (projectId: string | null): string => {
    if (projectId === null) return '#8a94a6'
    let h = 0
    for (const ch of projectId) h = (h * 31 + ch.charCodeAt(0)) >>> 0
    return `hsl(${h % 360} 55% 55%)`
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
  // 节点/边上的按下不启动拖拽——保留点击开详情。
  useEffect(() => {
    const svg = svgRef.current
    if (svg === null) return
    const onDown = (e: MouseEvent): void => {
      if (e.button !== 0) return
      const target = e.target as Element
      if (target !== svg && target.tagName !== 'svg') return // 空白处才拖拽
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
        <div className={styles.filters}>
          {['all', 'global', 'project'].map((f) => (
            <button
              key={f}
              className={`${styles.filterBtn} ${filter === f ? styles.filterActive : ''}`}
              onClick={() => setFilter(f)}
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
                markerWidth="7"
                markerHeight="7"
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
                  strokeWidth={1.5}
                  strokeDasharray={c.multi ? '6 4' : undefined}
                />
                <text
                  x={c.cx} y={c.cy - c.radius + 20}
                  textAnchor="middle"
                  className={styles.clusterLabel}
                  style={{ paintOrder: 'stroke', stroke: 'rgba(10,14,22,0.85)', strokeWidth: 3 }}
                >
                  {c.label}
                </text>
              </g>
            ))}
            {/* 边（v0.5：causes 带方向箭头 + 按源节点色淡化——视觉与节点协调） */}
            {edges.map((e) => {
              const a = layout.get(e.from)
              const b = layout.get(e.to)
              if (a === undefined || b === undefined) return null
              if (e.kind === 'causes') {
                const src = nodes.find((n) => n.id === e.from)
                const color = src ? projectColor(src.projectId) : '#8a94a6'
                return (
                  <line
                    key={`${e.from}|${e.to}|${e.kind}`}
                    x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                    stroke={color}
                    strokeOpacity={0.45}
                    strokeWidth={1.5}
                    markerEnd="url(#arrow-causes)"
                    className={styles.edge}
                  />
                )
              }
              return (
                <line
                  key={`${e.from}|${e.to}|${e.kind}`}
                  x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                  stroke="#aab2c0"
                  strokeOpacity={0.3}
                  strokeWidth={1}
                  strokeDasharray="5 4"
                  className={styles.edge}
                />
              )
            })}
            {/* 节点（v0.5 视觉分层：semantic=固化知识 大+光环；episodic=新事件
                常规；大小随重要度；标签 halo 保证任何背景下可读） */}
            {nodes.map((n) => {
              const p = layout.get(n.id)
              if (p === undefined) return null
              // v0.4：按项目着色（哈希色板；通用知识灰色）——项目即标签
              const color = projectColor(n.projectId)
              const selected = detail !== null && detail.id === n.id
              const isSemantic = n.state === 'semantic'
              // 半径：semantic 14-16，episodic 9-13（随重要度）
              const r = (isSemantic ? 12 : 8) + n.importance * 4 + (selected ? 2 : 0)
              return (
                <g
                  key={n.id}
                  className={`${styles.node} ${isSemantic ? styles.nodeSemantic : ''}`}
                  onClick={() => openDetail(n)}
                  role="button"
                  tabIndex={0}
                >
                  {isSemantic && (
                    <circle cx={p.x} cy={p.y} r={r + 9} fill="url(#halo-grad)" pointerEvents="none" />
                  )}
                  <circle
                    cx={p.x} cy={p.y}
                    r={r}
                    fill={color}
                    fillOpacity={isSemantic ? 0.95 : 0.8}
                    stroke={selected ? '#fff' : isSemantic ? 'rgba(255,255,255,0.6)' : 'rgba(255,255,255,0.35)'}
                    strokeWidth={selected ? 2 : isSemantic ? 1.5 : 1}
                  />
                  <text
                    x={p.x} y={p.y + r + 12}
                    textAnchor="middle"
                    className={isSemantic ? styles.nodeLabelSemantic : styles.nodeLabel}
                    style={{ paintOrder: 'stroke', stroke: 'rgba(10,14,22,0.9)', strokeWidth: 3 }}
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
              <span className={styles.legendDot} style={{ background: projectColor('D:\\x') }} />
              {t('graph.layer.project')}
            </span>
            <span>
              <span className={styles.legendDot} style={{ background: projectColor(null) }} />
              {t('graph.layer.global')}
            </span>
            <span><span className={styles.legendCluster} />{t('graph.legend.cluster')}</span>
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
