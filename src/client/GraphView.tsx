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
import { useEffect, useMemo, useState } from 'react'
import { layoutForce, type ForcePoint } from './force.ts'
import styles from './graph.module.css'

export interface GraphNodeData {
  id: string
  title: string
  summary: string
  kind: string
  layer: 'global' | 'project' | 'session'
  projectId: string | null
  importance: number
  hits: number
  createdAt: number
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
  session: '#ff9f43',
}

/** 边类型 → 渲染样式。 */
export const EDGE_STYLE: Record<'causes' | 'link', { color: string; dash: string }> = {
  causes: { color: '#8a94a6', dash: '' },
  link: { color: '#aab2c0', dash: '5 4' },
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
        // d3-force 风格参数（force.ts 注释）：
        // 负 charge = manyBody 斥力；spring 0-1 弱标定（0.1 ≈ 15 节点网络
        // 边距 80-100px）；collide 硬防重叠；forceCenter 平移居中
        // （不加向心力，网络不会聚核）。
        charge: -800,
        spring: 0.1,
        springLength: 80,
        collideRadius: 24,
      },
    )
    return { nodes: visible, edges: visibleEdges, layout }
  }, [data, filter])

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
          {['all', 'global', 'project', 'session'].map((f) => (
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
          <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} className={styles.svg}>
            {/* 边 */}
            {edges.map((e) => {
              const a = layout.get(e.from)
              const b = layout.get(e.to)
              if (a === undefined || b === undefined) return null
              const style = EDGE_STYLE[e.kind] ?? EDGE_STYLE.link
              return (
                <line
                  key={`${e.from}|${e.to}|${e.kind}`}
                  x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                  stroke={style.color}
                  strokeWidth={e.kind === 'causes' ? 1.5 : 1}
                  strokeDasharray={style.dash}
                  className={styles.edge}
                />
              )
            })}
            {/* 节点 */}
            {nodes.map((n) => {
              const p = layout.get(n.id)
              if (p === undefined) return null
              const color = LAYER_COLORS[n.layer] ?? '#888'
              const selected = detail !== null && detail.id === n.id
              return (
                <g
                  key={n.id}
                  className={styles.node}
                  onClick={() => openDetail(n)}
                  role="button"
                  tabIndex={0}
                >
                  <circle
                    cx={p.x} cy={p.y}
                    r={selected ? 15 : 12}
                    fill={color}
                    fillOpacity={0.85}
                    stroke={selected ? '#fff' : 'rgba(255,255,255,0.35)'}
                    strokeWidth={selected ? 2 : 1}
                  />
                  <text
                    x={p.x} y={p.y + 30}
                    textAnchor="middle"
                    className={styles.nodeLabel}
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
            {(['global', 'project', 'session'] as const).map((layer) => (
              <span key={layer}>
                <span className={styles.legendDot} style={{ background: LAYER_COLORS[layer] }} />
                {t(`graph.layer.${layer}`)}
              </span>
            ))}
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
