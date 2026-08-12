/**
 * EngramStore — 大一统记忆图谱（JSONL 持久化）。
 *
 * 模型（参考 Obsidian 双向链接 + skill 渐进式披露）：
 *  - **节点**：统一记忆（不预分轨/不硬编码分层）。每条记忆 =
 *      title（入口锚点）+ summary（一句话摘要，渐进披露第一层）
 *      + content（完整正文，按需展开）+ links（双向链接 [[title]]）
 *      + causes/effects（因果边，双向可追溯）
 *  - **索引**：N-gram 哈希寻址（NgramHashAddressing）→ 槽位 → 节点，
 *    确定性 O(1) 匹配当前上下文；
 *  - **自组织**：不手动分层——链接密度/主题关联自然形成结构，
 *    唤醒按关联度排序（类 Obsidian 图谱的局部密度）。
 *
 * 定位：**单次会话上下文增强**——本会话记忆写入、入口唤醒、渐进展开、
 * 因果双向追溯；会话结束即弃（clearSession），不做跨会话沉淀。
 */

import { mkdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs'
import { homedir } from 'node:os'
import { dirname, join, resolve } from 'node:path'

import { NgramHashAddressing, type HashResult } from './hash.js'

/** 记忆节点类型（统一，不预分轨；kind 仅作展示标签，非分层）。 */
export type EngramKind = 'fact' | 'decision' | 'event' | 'note'

/**
 * 记忆分层（预设骨架，归属由 AI 自主决策）——分层的本质 = 生命周期 × 可见范围：
 *  - global：全局持久，所有会话可见（长期事实/用户偏好）
 *  - project：项目持久，仅同工作目录（cwd）会话可见（项目约定/决策）
 *  - session：会话临时，仅本会话（会话结束清理）
 * 层是**节点属性**（大一统图谱，不分家），不是物理分库。
 */
export type EngramLayer = 'global' | 'project' | 'session'

/** 分层常量（工具 description 引用）。 */
export const ENGRAM_LAYERS: EngramLayer[] = ['global', 'project', 'session']

/**
 * 分层可见性判定（跨会话准入的单源逻辑，wake/tools/图谱 API 共用）。
 *  - global：所有会话可见；
 *  - project：仅 node.projectId === viewer.cwd 的会话；
 *  - session：仅 node.sessionId === viewer.sessionId 的本会话。
 * 空 viewer（无 sessionId 且无 cwd）向后兼容全可见（生产路径总传 viewer，
 * 缺省仅测试/直接调用）。
 */
export function isVisible(e: EngramNode, viewer: { sessionId?: string; cwd?: string }): boolean {
  if (viewer.sessionId === undefined && viewer.cwd === undefined) return true
  switch (e.layer) {
    case 'global':
      return true
    case 'project':
      return e.projectId !== null && e.projectId === viewer.cwd
    case 'session':
      return e.sessionId !== null && e.sessionId === viewer.sessionId
    default:
      return false
  }
}

/** 渐进披露层级。 */
export interface EngramNode {
  id: string
  kind: EngramKind
  /** 分层归属（AI 自主决策）：global=全局持久 / project=项目持久 / session=会话临时。 */
  layer: EngramLayer
  /** project 层标识（会话工作目录；global/session 层为 null）。 */
  projectId: string | null
  /** 入口锚点（唤醒列表展示；如 Obsidian 的页面标题）。 */
  title: string
  /** 一句话摘要（渐进披露第一层——入口列表只给这个）。 */
  summary: string
  /** 完整正文（渐进披露第二层——展开时给）。 */
  content: string
  /** 双向链接：关联节点的 title 集（Obsidian 风格 [[title]]）。 */
  links: string[]
  /** 因果边（前因）：导致本节点的节点 id 集。 */
  causes: string[]
  /** 因果边（后果）：本节点导致的节点 id 集。 */
  effects: string[]
  /** 来源会话 id（本会话内）。 */
  sessionId: string | null
  /** 来源回合序号。 */
  turn: number
  /** 创建时间（epoch ms）。 */
  createdAt: number
  /** 关联度 0-1（唤醒排序用；自组织：链接越多/被引用越多越高）。 */
  importance: number
  /** 被唤醒次数（LRU 衰减）。 */
  hits: number
  /** 最后唤醒时间。 */
  lastHitAt: number | null
  /** 该节点对应的哈希槽位（写入时固化，重哈希可重建）。 */
  slots: string[]
}

/** 渐进披露视图。 */
export interface EngramEntry {
  id: string
  title: string
  summary: string
  kind: EngramKind
  /** 因果邻接摘要（入口层展示：前因/后果标题）。 */
  causeTitles: string[]
  effectTitles: string[]
  /** 双向链接标题。 */
  linkTitles: string[]
}

let seq = 0

export function createEngramId(): string {
  seq += 1
  return `e${Date.now().toString(36)}-${seq.toString(36)}`
}

export class EngramStore {
  readonly dir: string
  private file: string
  private byId = new Map<string, EngramNode>()
  /** 槽位索引：slotKey -> Set<nodeId>（派生索引，写入/加载时构建）。 */
  private slotIndex = new Map<string, Set<string>>()
  /** 标题索引：title -> nodeId（双向链接解析用）。 */
  private titleIndex = new Map<string, string>()

  constructor(storeDir: string, private hasher: NgramHashAddressing = new NgramHashAddressing()) {
    this.dir = storeDir === '' ? join(homedir(), '.dsh', 'engram-relay') : resolve(storeDir)
    this.file = join(this.dir, 'engrams.jsonl')
    if (existsSync(this.file)) {
      this.load()
    } else {
      mkdirSync(this.dir, { recursive: true })
    }
  }

  private load(): void {
    for (const line of readFileSync(this.file, 'utf8').split('\n')) {
      if (line.trim() === '') continue
      try {
        const e = JSON.parse(line) as EngramNode
        this.byId.set(e.id, e)
        for (const s of e.slots) this.indexSlot(s, e.id)
        if (e.title) this.titleIndex.set(e.title, e.id)
      } catch {
        // 单条损坏跳过，不拖垮整个存储
      }
    }
  }

  private indexSlot(slot: string, id: string): void {
    let set = this.slotIndex.get(slot)
    if (!set) {
      set = new Set()
      this.slotIndex.set(slot, set)
    }
    set.add(id)
  }

  private persist(): void {
    mkdirSync(dirname(this.file), { recursive: true })
    const lines: string[] = []
    for (const e of this.byId.values()) lines.push(JSON.stringify(e))
    writeFileSync(this.file, lines.join('\n') + '\n', 'utf8')
  }

  /**
   * 写入/更新一个记忆节点：按 title+summary 哈希寻址，挂到命中槽位。
   * 渐进披露：title/summary 是入口层，content 是展开层。
   * layer 缺省 'session'（向后兼容：旧调用语义 = 会话级即弃）。
   */
  add(input: Omit<EngramNode, 'id' | 'createdAt' | 'hits' | 'lastHitAt' | 'slots' | 'layer' | 'projectId'>
    & { layer?: EngramLayer; projectId?: string | null }): EngramNode {
    const keyText = `${input.title} ${input.summary}`
    const result = this.hasher.hash(keyText)
    const slots = this.hasher.slotKeys(result)
    const node: EngramNode = {
      ...input,
      layer: input.layer ?? 'session',
      projectId: input.projectId ?? null,
      id: createEngramId(),
      createdAt: Date.now(),
      hits: 0,
      lastHitAt: null,
      slots,
    }
    this.byId.set(node.id, node)
    for (const s of slots) this.indexSlot(s, node.id)
    if (node.title) this.titleIndex.set(node.title, node.id)
    this.persist()
    return node
  }

  /** 按标题取节点（双向链接 [[title]] 解析）。 */
  byTitle(title: string): EngramNode | undefined {
    const id = this.titleIndex.get(title)
    return id ? this.byId.get(id) : undefined
  }

  /** 按文本哈希寻址，返回命中槽位的候选节点（去重，按关联度降序）。 */
  lookup(text: string, limit = 8): EngramNode[] {
    const result = this.hasher.hash(text)
    return this.lookupHash(result, limit)
  }

  /** 按已计算的哈希结果寻址（避免重复哈希）。 */
  lookupHash(result: HashResult, limit = 8): EngramNode[] {
    const keys = this.hasher.slotKeys(result)
    const seen = new Set<string>()
    const hits: EngramNode[] = []
    for (const k of keys) {
      const ids = this.slotIndex.get(k)
      if (!ids) continue
      for (const id of ids) {
        if (seen.has(id)) continue
        seen.add(id)
        const e = this.byId.get(id)
        if (e) hits.push(e)
      }
    }
    hits.sort((a, b) => b.importance - a.importance)
    return hits.slice(0, limit)
  }

  /** 渐进披露入口视图：摘要级 + 因果/链接邻接摘要。 */
  entry(node: EngramNode): EngramEntry {
    return {
      id: node.id,
      title: node.title,
      summary: node.summary,
      kind: node.kind,
      causeTitles: this.getMany(node.causes).map((n) => n.title),
      effectTitles: this.getMany(node.effects).map((n) => n.title),
      linkTitles: node.links.map((t) => this.byTitle(t)?.title ?? t),
    }
  }

  /** 批量入口视图。 */
  entries(nodes: EngramNode[]): EngramEntry[] {
    return nodes.map((n) => this.entry(n))
  }

  /**
   * 自组织聚类：按连接密度（links + causes/effects）自然成簇——不预定义
   * 主题、不硬编码分层。连通分量即簇；每簇选「代表节点」（连接度最高者）
   * 作为唤醒入口。类似 Obsidian 图谱的视觉密度：密集连接处自然成团。
   */
  clusters(): Array<{ label: string; members: string[]; representative: string }> {
    const all = this.all()
    if (all.length === 0) return []
    // 邻接：节点间有 links 或因果边即相连
    const adj = new Map<string, Set<string>>()
    for (const n of all) adj.set(n.id, new Set())
    for (const n of all) {
      // links（双向）
      for (const t of n.links) {
        const target = this.byTitle(t)
        if (target && target.id !== n.id) {
          adj.get(n.id)!.add(target.id)
          adj.get(target.id)!.add(n.id)
        }
      }
      // 因果边
      for (const c of n.causes) {
        if (this.byId.has(c)) {
          adj.get(n.id)!.add(c)
          adj.get(c)!.add(n.id)
        }
      }
      for (const e of n.effects) {
        if (this.byId.has(e)) {
          adj.get(n.id)!.add(e)
          adj.get(e)!.add(n.id)
        }
      }
    }
    // BFS 连通分量
    const visited = new Set<string>()
    const clusters: Array<{ label: string; members: string[]; representative: string }> = []
    for (const n of all) {
      if (visited.has(n.id)) continue
      const members: string[] = []
      const queue = [n.id]
      visited.add(n.id)
      while (queue.length > 0) {
        const id = queue.shift()!
        members.push(id)
        for (const nb of adj.get(id) ?? []) {
          if (!visited.has(nb)) {
            visited.add(nb)
            queue.push(nb)
          }
        }
      }
      // 代表节点：连接度最高（邻接数最多）；并列取 importance 高者
      const representative = members.reduce((best, id) => {
        const deg = adj.get(id)?.size ?? 0
        const bestDeg = adj.get(best)?.size ?? 0
        const node = this.byId.get(id)!
        const bestNode = this.byId.get(best)!
        return deg > bestDeg || (deg === bestDeg && node.importance > bestNode.importance) ? id : best
      })
      const repNode = this.byId.get(representative)!
      clusters.push({
        label: repNode.title,
        members,
        representative,
      })
    }
    // 簇按大小降序（大的主题簇在前）
    clusters.sort((a, b) => b.members.length - a.members.length)
    return clusters
  }

  get(id: string): EngramNode | undefined {
    return this.byId.get(id)
  }

  getMany(ids: string[]): EngramNode[] {
    const out: EngramNode[] = []
    for (const id of ids) {
      const e = this.byId.get(id)
      if (e) out.push(e)
    }
    return out
  }

  all(): EngramNode[] {
    return [...this.byId.values()]
  }

  count(): number {
    return this.byId.size
  }

  slotCount(): number {
    return this.slotIndex.size
  }

  /**
   * 分层统一查询（维护/检索入口）：按层/项目/会话/类型/时间过滤。
   * 缺省按 importance 降序；recent=true 按创建时间倒序。
   */
  query(filter: {
    layer?: EngramLayer
    projectId?: string | null
    sessionId?: string
    kind?: EngramKind
    since?: number
    until?: number
    limit?: number
    recent?: boolean
  } = {}): EngramNode[] {
    let list = this.all()
    if (filter.layer !== undefined) list = list.filter((e) => e.layer === filter.layer)
    if (filter.projectId !== undefined) list = list.filter((e) => e.projectId === filter.projectId)
    if (filter.sessionId !== undefined) list = list.filter((e) => e.sessionId === filter.sessionId)
    if (filter.kind !== undefined) list = list.filter((e) => e.kind === filter.kind)
    if (filter.since !== undefined) list = list.filter((e) => e.createdAt >= filter.since!)
    if (filter.until !== undefined) list = list.filter((e) => e.createdAt <= filter.until!)
    list = [...list]
    if (filter.recent) list.sort((a, b) => b.createdAt - a.createdAt)
    else list.sort((a, b) => b.importance - a.importance)
    if (filter.limit !== undefined && filter.limit > 0) list = list.slice(0, filter.limit)
    return list
  }

  /** 分层统计（status 工具用）。 */
  layerCounts(): Record<EngramLayer, number> {
    const counts: Record<EngramLayer, number> = { global: 0, project: 0, session: 0 }
    for (const e of this.byId.values()) counts[e.layer] += 1
    return counts
  }

  /**
   * 提升/转层：改 layer 与 projectId（保留 id/因果/链接——引用不失效）。
   * 会话结束前把 session 临时记忆提升为 project/global 跨会话持久。
   */
  promote(id: string, layer: EngramLayer, projectId: string | null = null): EngramNode | undefined {
    const e = this.byId.get(id)
    if (!e) return undefined
    e.layer = layer
    e.projectId = layer === 'project' ? projectId : null
    this.persist()
    return e
  }

  /** 修正节点字段（title 变更会同步标题索引；层变更用 promote）。 */
  update(id: string, patch: Partial<Pick<EngramNode, 'title' | 'summary' | 'content' | 'links' | 'causes' | 'effects' | 'importance'>>): EngramNode | undefined {
    const e = this.byId.get(id)
    if (!e) return undefined
    if (patch.title !== undefined && patch.title !== e.title) {
      this.titleIndex.delete(e.title)
      e.title = patch.title
      if (e.title) this.titleIndex.set(e.title, e.id)
    }
    if (patch.summary !== undefined) e.summary = patch.summary
    if (patch.content !== undefined) e.content = patch.content
    if (patch.links !== undefined) e.links = patch.links
    if (patch.causes !== undefined) e.causes = patch.causes
    if (patch.effects !== undefined) e.effects = patch.effects
    if (patch.importance !== undefined) e.importance = patch.importance
    this.persist()
    return e
  }

  /** 清空一个项目（project 层全部节点；项目移除/归档时）。 */
  clearProject(projectId: string): number {
    const doomed = this.all().filter((e) => e.layer === 'project' && e.projectId === projectId)
    for (const e of doomed) this.remove(e.id)
    return doomed.length
  }

  /** 登记一次唤醒（LRU 衰减）。 */
  touch(id: string): void {
    const e = this.byId.get(id)
    if (!e) return
    e.hits += 1
    e.lastHitAt = Date.now()
    this.persist()
  }

  remove(id: string): boolean {
    const e = this.byId.get(id)
    if (!e) return false
    this.byId.delete(id)
    if (e.title) this.titleIndex.delete(e.title)
    for (const s of e.slots) {
      const set = this.slotIndex.get(s)
      if (set) {
        set.delete(id)
        if (set.size === 0) this.slotIndex.delete(s)
      }
    }
    this.persist()
    return true
  }

  /**
   * 会话隔离（分层生命周期）：只清该会话的 **session 层** 临时记忆；
   * global/project 跨会话层保留——跨会话沉淀的核心转变。
   * 复用 remove() 统一清理索引（byId/titleIndex/slotIndex）。
   */
  clearSession(sessionId: string): number {
    const doomed = this.all().filter((e) => e.sessionId === sessionId && e.layer === 'session')
    for (const e of doomed) this.remove(e.id)
    return doomed.length
  }
}
