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

/** 渐进披露层级。 */
export interface EngramNode {
  id: string
  kind: EngramKind
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
   */
  add(input: Omit<EngramNode, 'id' | 'createdAt' | 'hits' | 'lastHitAt' | 'slots'>): EngramNode {
    const keyText = `${input.title} ${input.summary}`
    const result = this.hasher.hash(keyText)
    const slots = this.hasher.slotKeys(result)
    const node: EngramNode = {
      ...input,
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
   * 会话隔离：清空某会话的全部节点（单会话增强——会话结束即弃）。
   */
  clearSession(sessionId: string): number {
    const doomed = this.all().filter((e) => e.sessionId === sessionId)
    for (const e of doomed) {
      this.byId.delete(e.id)
      if (e.title) this.titleIndex.delete(e.title)
      for (const s of e.slots) {
        const set = this.slotIndex.get(s)
        if (set) {
          set.delete(e.id)
          if (set.size === 0) this.slotIndex.delete(s)
        }
      }
    }
    if (doomed.length > 0) this.persist()
    return doomed.length
  }
}
