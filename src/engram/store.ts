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

import { mkdirSync, readFileSync, writeFileSync, existsSync, renameSync, unlinkSync, copyFileSync, readdirSync } from 'node:fs'
import { basename } from 'node:path'
/** 进程内写锁（同步标志位）：persist 是同步函数，JS 单线程下同步代码天然不交错；tmp 每实例唯一已防跨实例冲突。 */
let fileLockHeld = false
function runWithFileLock(_file: string, critical: () => void): void {
  if (fileLockHeld) {
    // 重入（不可能发生于同步 persist 链），保守直接执行
    critical()
    return
  }
  fileLockHeld = true
  try {
    critical()
  } finally {
    fileLockHeld = false
  }
}

/** 判断载荷是否完好：无 NUL 主导 + 首行可解析（空文件视为完好）。 */
function isHealthyPayload(raw: string): boolean {
  if (raw.length === 0) return true
  const nulCount = (raw.match(/\0/g) ?? []).length
  if (nulCount / raw.length > 0.1) return false
  const first = raw.split('\n').find((l) => l.trim() !== '')
  if (first === undefined) return true
  try {
    JSON.parse(first)
    return true
  } catch {
    return false
  }
}

/** 备份文件列表（新 → 旧）。 */
function listBackups(file: string): string[] {
  try {
    return readdirSync(dirname(file))
      .filter((n) => n.startsWith(basename(file) + '.bak-'))
      .sort()
      .reverse()
      .map((n) => join(dirname(file), n))
  } catch {
    return []
  }
}

/** 备份剪枝（保留最近 keep 代）。 */
function pruneBackups(file: string, keep: number): void {
  for (const b of listBackups(file).slice(keep)) {
    try { unlinkSync(b) } catch { /* 忽略 */ }
  }
}
import { homedir } from 'node:os'
import { dirname, join, resolve } from 'node:path'

import { NgramHashAddressing, type HashResult } from './hash.js'

/** 记忆节点类型（统一，不预分轨；kind 仅作展示标签，非分层）。 */
export type EngramKind = 'fact' | 'decision' | 'event' | 'note'

/**
 * 记忆分层（预设骨架，归属由 AI 自主决策）——分层的本质 = 生命周期 × 可见范围：
 *  - global：全局持久，所有会话可见（长期事实/用户偏好）
 *  - project：项目持久，仅同工作目录（cwd）会话可见（项目约定/决策）
 * 层是**节点属性**（大一统图谱，不分家），不是物理分库。
 * （v0.3：session 层已删除——会话内冗余且与"跨会话记忆"定位矛盾；
 *  会话临时状态由 DSH 上下文承担，记忆只有 global/project 两层。）
 */
export type EngramLayer = 'global' | 'project'

/** 分层常量（工具 description 引用）。 */
export const ENGRAM_LAYERS: EngramLayer[] = ['global', 'project']

/**
 * 分层可见性判定（跨会话准入的单源逻辑，wake/tools/图谱 API 共用）。
 *  - global：所有会话可见；
 *  - project：仅 node.projectId === viewer.cwd 的会话。
 * 空 viewer（无 cwd）向后兼容全可见（生产路径总传 viewer，
 * 缺省仅测试/直接调用）。viewer.sessionId 已无分层作用（保留字段兼容）。
 */
export function isVisible(e: EngramNode, viewer: { sessionId?: string; cwd?: string }): boolean {
  if (viewer.cwd === undefined) return true
  switch (e.layer) {
    case 'global':
      return true
    case 'project':
      return e.projectId !== null && e.projectId === viewer.cwd
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
  /** 确认状态：pending=待确认（不参与检索/唤醒命中），confirmed=已确认（缺省；旧数据视为 confirmed）。 */
  status?: 'pending' | 'confirmed'
  /** 强化事件时间戳（写入/命中/展开/链接；类脑激活模型 B=ln(Σt^(-d)) 的输入）。旧数据缺省 [createdAt]。 */
  reinforces?: number[]
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
  /** 标题索引：title -> id[]（多值——跨项目同标题合法存在；解析时消歧）。 */
  private titleIndex = new Map<string, string[]>()

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
    // 恢复链（绝不丢记忆防线 2）：主文件 → 各代备份（新→旧），
    // 取第一个能解析出节点的来源；主文件损坏时从最近完好备份自动恢复。
    const sources = [this.file, ...listBackups(this.file)]
    let recoveredFrom: string | null = null
    for (const src of sources) {
      if (!existsSync(src)) continue
      let raw: string
      try {
        raw = readFileSync(src, 'utf8')
      } catch {
        continue
      }
      const cleaned = raw.replace(/\0+/g, '')
      let loaded = 0
      let corrupt = 0
      for (const line of cleaned.split('\n')) {
        if (line.trim() === '') continue
        try {
          const e = JSON.parse(line) as EngramNode
          // —— 旧数据迁移兜底（v0.2.0 跨会话分层前持久化的节点缺字段）——
          // v0.3：session 层删除——旧 session 节点归一化为 project（保留不丢）
          e.layer = (e as { layer: string }).layer === 'session' ? 'project' : (e.layer ?? 'project')
          e.projectId = e.projectId ?? null
          e.slots = Array.isArray(e.slots) ? e.slots : []
          e.links = Array.isArray(e.links) ? e.links : []
          e.causes = Array.isArray(e.causes) ? e.causes : []
          e.effects = Array.isArray(e.effects) ? e.effects : []
          e.importance = typeof e.importance === 'number' ? e.importance : 0
          e.hits = typeof e.hits === 'number' ? e.hits : 0
          e.createdAt = typeof e.createdAt === 'number' ? e.createdAt : 0
          e.reinforces = Array.isArray(e.reinforces) ? e.reinforces : [e.createdAt || Date.now()]
          this.byId.set(e.id, e)
          for (const s of e.slots) this.indexSlot(s, e.id)
          if (e.title) this.indexTitle(e.title, e.id)
          loaded++
        } catch {
          corrupt++
        }
      }
      if (loaded > 0) {
        if (src !== this.file) {
          recoveredFrom = src
          try {
            writeFileSync(this.file, cleaned, 'utf8')
          } catch { /* 写回失败不阻塞 */ }
        }
        return
      }
      // ⚠️ 主文件无节点但全部可解析 = 合法空库（如 clearSession 清空后），
      // 不是损坏——绝不回退备份（否则会把已删除的会话记忆恢复回来，
      // 实测「清空后重载残留」bug 的根因）。但含 NUL 的原始内容（写坏
      // 的文件）仍是损坏，必须回退备份恢复。
      if (src === this.file && corrupt === 0 && !raw.includes('\0')) {
        return
      }
      // 该来源全坏 → 下一个（更旧的备份）
    }
    if (recoveredFrom) {
      console.warn(`[engram-store] recovered ${this.byId.size} nodes from backup ${recoveredFrom}`)
    }
    // 全部来源都坏：留证（主文件损坏时）后以空库继续——绝不静默清空主文件
    if (existsSync(this.file)) {
      const raw = (() => { try { return readFileSync(this.file, 'utf8') } catch { return '' } })()
      if (raw.replace(/\s/g, '').length > 64) {
        try {
          renameSync(this.file, `${this.file}.corrupt-${Date.now()}`)
          console.warn(`[engram-store] all sources corrupt; main file preserved at corrupt backup`)
        } catch { /* 备份失败不阻塞 */ }
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

  /**
   * 原子持久化：写临时文件 + rename 替换。
   *
   * 背景：web 与 headless 两个 profile 可能同时装配本插件并写同一个
   * engrams.jsonl；热重载时同一进程内也会短暂存在两个 store 实例（旧
   * fiber dispose 前的最后一次 persist 与新实例并发）。tmp 必须**每实例
   * 唯一**（曾用 `${pid}` 导致同进程两实例共用同名 tmp → writeFileSync
   * 交错 → 整文件 NUL、记忆全丢），并加进程内写锁串行化 rename 竞态。
   * Windows 上 rename 覆盖已存在文件会失败，先 unlink 目标再 rename。
   */
  private persist(): void {
    mkdirSync(dirname(this.file), { recursive: true })
    const lines: string[] = []
    for (const e of this.byId.values()) lines.push(JSON.stringify(e))
    const payload = lines.join('\n') + '\n'
    // 写前快照（绝不丢记忆防线 1）：当前完好文件 → .bak-<ts>，保留 3 代。
    if (existsSync(this.file)) {
      try {
        const cur = readFileSync(this.file, 'utf8')
        if (isHealthyPayload(cur)) {
          copyFileSync(this.file, `${this.file}.bak-${Date.now()}`)
          pruneBackups(this.file, 3)
        }
      } catch { /* 快照失败不阻塞写入 */ }
    }
    const tmp = `${this.file}.tmp-${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
    writeFileSync(tmp, payload, 'utf8')
    // 进程内写锁：热重载窗口内两实例的 rename 串行（最后写入者胜，不交错）
    runWithFileLock(this.file, () => {
      try {
        renameSync(tmp, this.file)
      } catch {
        try {
          unlinkSync(this.file)
        } catch { /* 目标不存在等，忽略 */ }
        renameSync(tmp, this.file)
      }
    })
    // 写后校验（防线 3）：读回行数一致才算成功；不一致时上一代备份仍在。
    try {
      const back = readFileSync(this.file, 'utf8')
      const backLines = back.split('\n').filter((l) => l.trim() !== '').length
      if (backLines !== lines.length) {
        console.warn(`[engram-store] persist readback mismatch: wrote ${lines.length}, read ${backLines}`)
      }
    } catch { /* 校验失败不阻塞 */ }
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
      layer: input.layer ?? 'project',
      projectId: input.projectId ?? null,
      id: createEngramId(),
      createdAt: Date.now(),
      hits: 0,
      lastHitAt: null,
      slots,
      // 写入即第一次强化（类脑：刚记住时最容易被想起）
      reinforces: input.reinforces ?? [Date.now()],
    }
    this.byId.set(node.id, node)
    for (const s of slots) this.indexSlot(s, node.id)
    if (node.title) this.indexTitle(node.title, node.id)
    this.persist()
    return node
  }

  /** 标题索引登记（多值聚合，去重）。 */
  private indexTitle(title: string, id: string): void {
    const arr = this.titleIndex.get(title)
    if (arr) {
      if (!arr.includes(id)) arr.push(id)
    } else {
      this.titleIndex.set(title, [id])
    }
  }

  /** 标题索引移除（只摘该项，不影响同名其他节点）。 */
  private unindexTitle(title: string, id: string): void {
    const arr = this.titleIndex.get(title)
    if (!arr) return
    const i = arr.indexOf(id)
    if (i >= 0) arr.splice(i, 1)
    if (arr.length === 0) this.titleIndex.delete(title)
  }

  /** 按标题取节点（双向链接 [[title]] 解析）。同名消歧：最近写入优先。
   *  ⚠️ 曾有 Map<title,单id> 覆盖 bug——同名节点互相顶掉；多值后不再丢。 */
  byTitle(title: string): EngramNode | undefined {
    const arr = this.titleIndex.get(title)
    if (!arr || arr.length === 0) return undefined
    return this.byId.get(arr[arr.length - 1])
  }

  /** 按标题取全部同名节点（消歧/盘点用：跨项目同标题、版本链同主题）。 */
  byTitles(title: string): EngramNode[] {
    return (this.titleIndex.get(title) ?? [])
      .map((id) => this.byId.get(id))
      .filter((e): e is EngramNode => !!e)
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
        if (e && e.status !== 'pending') hits.push(e)
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
    const all = this.all().filter((e) => e.status !== 'pending')
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
    kind?: EngramKind
    since?: number
    until?: number
    limit?: number
    recent?: boolean
  } = {}): EngramNode[] {
    let list = this.all()
    if (filter.layer !== undefined) list = list.filter((e) => e.layer === filter.layer)
    if (filter.projectId !== undefined) list = list.filter((e) => e.projectId === filter.projectId)
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
    const counts: Record<EngramLayer, number> = { global: 0, project: 0 }
    for (const e of this.byId.values()) counts[e.layer] += 1
    return counts
  }

  /**
   * 提升/转层：改 layer 与 projectId（保留 id/因果/链接——引用不失效）。
   * project → global（跨项目共享真理）；global 不可降级。
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
      this.unindexTitle(e.title, e.id)
      e.title = patch.title
      if (e.title) this.indexTitle(e.title, e.id)
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

  /** 登记一次唤醒（LRU 衰减 + 激活强化：命中即复习，类脑巩固）。 */
  touch(id: string): void {
    const e = this.byId.get(id)
    if (!e) return
    e.hits += 1
    e.lastHitAt = Date.now()
    if (e.reinforces) e.reinforces.push(Date.now())
    this.persist()
  }

  /** 登记一次强化（展开/链接等深度使用——权重高于命中）。 */
  reinforce(id: string): void {
    const e = this.byId.get(id)
    if (!e) return
    e.reinforces = e.reinforces ?? [e.createdAt || Date.now()]
    e.reinforces.push(Date.now())
    this.persist()
  }

  /** 全部待确认节点（用户确认制管理面）。 */
  pending(): EngramNode[] {
    return this.all().filter((e) => e.status === 'pending')
  }

  /** 确认一个待确认节点（确认后才参与检索/唤醒命中）。幂等：已确认返回原节点。 */
  confirmNode(id: string): EngramNode | undefined {
    const e = this.byId.get(id)
    if (!e) return undefined
    if (e.status === 'pending') {
      e.status = 'confirmed'
      this.persist()
    }
    return e
  }

  /** 拒绝（删除）一个待确认节点。非 pending 节点不可拒绝（防误删已生效记忆）。 */
  rejectNode(id: string): boolean {
    const e = this.byId.get(id)
    if (!e || e.status !== 'pending') return false
    return this.remove(id)
  }

  remove(id: string): boolean {
    const e = this.byId.get(id)
    if (!e) return false
    this.byId.delete(id)
    if (e.title) this.unindexTitle(e.title, e.id)
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
}

