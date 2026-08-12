/**
 * 图谱视图构建（GraphView 消费数据面）——仿真脚本共享辅助。
 *
 * 复刻 host 侧 graph-api.ts 的序列化语义（src/graph-api.ts），让仿真脚本
 * 能在 Node 侧直接验证「图谱 Tab 能消费的数据面」，补齐 sim:causal /
 * sim:1m 的图谱维度：
 *  - 节点视图：nodeView 最小集（GraphNodeData）
 *  - 边聚合：因果边（causes，前因/后果双向展开）+ 双向链接边（link），
 *    按 from|to|kind 去重
 *  - 分层准入：空 viewer（无 sessionId 且无 cwd）→ 只暴露 global 层
 *    （HTTP 隐私语义）；有 viewer → isVisible
 *
 * 运行：仅作为 simulate-*.mjs 的辅助模块被 import，不直接运行。
 */

import { isVisible } from '../lib/engram/store.js'

/** 节点视图（与 graph-api nodeView 同构）。 */
export function nodeView(n) {
  return {
    id: n.id,
    title: n.title,
    summary: n.summary,
    kind: n.kind,
    layer: n.layer,
    projectId: n.projectId,
    importance: n.importance,
    hits: n.hits,
    createdAt: n.createdAt,
  }
}

/** 构建 GraphView 可消费的图谱视图（与 graph-api GET /graph 同构）。 */
export function buildGraphView(store, viewer) {
  const visible = store.all().filter((n) =>
    viewer.sessionId === undefined && viewer.cwd === undefined
      ? n.layer === 'global'
      : isVisible(n, viewer))
  const ids = new Set(visible.map((n) => n.id))
  const edges = []
  const seen = new Set()
  const addEdge = (from, to, kind) => {
    const key = `${from}|${to}|${kind}`
    if (seen.has(key)) return
    seen.add(key)
    edges.push({ from, to, kind })
  }
  for (const n of visible) {
    // 因果边（前因/后果双向展开）
    for (const c of n.causes) if (ids.has(c)) addEdge(c, n.id, 'causes')
    for (const e of n.effects) if (ids.has(e)) addEdge(n.id, e, 'causes')
    // 双向链接边（Obsidian 风格 [[标题]]）
    for (const l of n.links) {
      const t = store.byTitle(l)
      if (t && ids.has(t.id) && t.id !== n.id) addEdge(n.id, t.id, 'link')
    }
  }
  return {
    nodes: visible.map(nodeView),
    edges,
    layerCounts: store.layerCounts(),
    total: visible.length,
  }
}

/**
 * 图谱数据完整性校验：
 *  - 无悬挂边（边 from/to 都在 nodes 中——GraphView 布局依赖此保证）
 *  - 层取值合法（LAYER_COLORS 查表依赖）
 *  - 边 kind 分类齐全（实线因果 / 虚线 link）
 *  - 边数符合预期
 *  - JSON 可序列化（GraphView 经 fetch().json() 消费）
 */
export function assertGraphView(gv, { label, expectedEdges }) {
  const problems = []
  const ids = new Set(gv.nodes.map((n) => n.id))
  const dangling = gv.edges.filter((e) => !ids.has(e.from) || !ids.has(e.to))
  if (dangling.length > 0) problems.push(`悬挂边 ${dangling.length} 条`)
  for (const n of gv.nodes) {
    if (!['global', 'project', 'session'].includes(n.layer)) problems.push(`非法层 ${n.layer}@${n.title}`)
    if (typeof n.title !== 'string' || typeof n.summary !== 'string') problems.push(`节点缺 title/summary@${n.id}`)
  }
  const causes = gv.edges.filter((e) => e.kind === 'causes').length
  const links = gv.edges.filter((e) => e.kind === 'link').length
  if (causes + links !== gv.edges.length) problems.push('边 kind 缺失')
  if (expectedEdges !== undefined && gv.edges.length !== expectedEdges) {
    problems.push(`边数 ${gv.edges.length} ≠ 预期 ${expectedEdges}`)
  }
  try {
    JSON.stringify(gv)
  } catch (e) {
    problems.push(`JSON 序列化失败: ${e.message}`)
  }
  if (problems.length > 0) {
    throw new Error(`图谱维度校验失败 [${label}]: ${problems.join('; ')}`)
  }
  return { causes, links }
}
