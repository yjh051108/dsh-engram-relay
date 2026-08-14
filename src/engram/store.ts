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

import { mkdirSync, readFileSync, writeFileSync, existsSync, renameSync, unlinkSync, copyFileSync, readdirSync, appendFileSync } from 'node:fs'
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
export type EngramKind = 'fact' | 'decision' | 'event' | 'note' | 'snapshot'

/**
 * 记忆归属（v0.4：不做分层分化——项目即标签，融会贯通）：
 *  - global：通用知识项目（原"全局层"降为平级标签：技术模式/平台坑/偏好）
 *  - project：具体项目（projectId = 工作目录）
 * 可见性：**全可见**（单用户本地系统，无多租户隐私需求；项目间通过
 * 记忆关联（link/causes 桥）自然融会贯通——"套娃"）。
 */
export type EngramLayer = 'global' | 'project'

/** 分层常量（工具 description 引用）。 */
export const ENGRAM_LAYERS: EngramLayer[] = ['global', 'project']

/**
 * 可见性判定（v0.4：全可见——项目不隔离，关联即桥）。
 * 保留函数签名（wake/tools/图谱 API 调用点不动），恒返回 true。
 */
export function isVisible(_e: EngramNode, _viewer: { sessionId?: string; cwd?: string }): boolean {
  return true
}

/** 默认标签（v0.4 迁移/兜底）：layer + projectId → 命名空间标签。 */
export function defaultTags(layer: EngramLayer, projectId: string | null): string[] {
  if (layer === 'global' || projectId === null || projectId === '') return ['全局']
  const parts = String(projectId).split(/[\\/]/)
  const name = parts[parts.length - 1] || '项目'
  return [`项目:${name}`]
}

/**
 * 沉睡判定（派生状态，实时计算）：最后强化 > 30 天前 且 hits < 3。
 * 被命中（touch）的节点刚强化，永不处于沉睡；沉默 >30 天的节点在
 * 渲染时降级为仅标题（不占注入预算），图谱仍可检索、命中即复苏。
 */
export function dormantOf(e: EngramNode, now: number = Date.now()): boolean {
  if (e.hits >= 3 && e.kind !== 'event') return false
  const last = e.reinforces && e.reinforces.length > 0 ? e.reinforces[e.reinforces.length - 1] : e.createdAt
  // ⚠️ event（过程性记忆）客观过时：创建超过 30 天直接沉睡——事件发生即
  // 结束，被检索命中（hits≥3）不改变"已过时"（用户实测：'重载失败'这类
  // 已解决的过程记录不该永久新鲜）；非 event 按最后强化判定（活的记忆
  // 靠使用保鲜）
  const anchor = e.kind === 'event' ? e.createdAt : last
  return (now - anchor) / 86400000 >= 30
}

/** 废止判定（版本链）：被新版本取代，退出检索/注入，可追溯。 */
export function isSuperseded(e: EngramNode): boolean {
  return e.supersededBy !== undefined && e.supersededBy !== ''
}

/** 教训判定（教训通道）：tags 含「教训:」前缀（命名空间约定，如 教训:代码）。
 * 教训类记忆在唤醒时走独立低阈值席位（见 wake.ts lesson channel）。 */
export function isLesson(e: EngramNode | undefined | null): boolean {
  if (!e) return false
  return (e.tags ?? []).some((t) => typeof t === 'string' && t.startsWith('教训:'))
}

/** 渐进披露层级。 */
export interface EngramNode {
  id: string
  kind: EngramKind
  /** 分层归属（AI 自主决策）：global=全局持久 / project=项目持久。 */
  layer: EngramLayer
  /**
   * 自由标签（v0.4：一节点多标签，自由分类——不建枚举 schema）：
   * 分类靠命名空间约定，如 '全局'（开发者喜好/全局要求）、
   * '项目:xxx'（项目自由命名）、'教训:代码' / '教训:思想'（教训自由子分类）。
   * layer/projectId 保留为系统内部字段（写入绑定/快照聚合），
   * tags 是用户可见的分类维度。
   */
  tags?: string[]
  /**
   * 巩固状态（双维度：可见性×巩固度，v0.3 引入）：
   *  - episodic：刚写入，事件性，细节丰富（情景记忆）
   *  - semantic：强化 ≥3 次，被归纳，去情景化（语义记忆）
   *  - dormant ：30 天无强化，低激活，退出注入入口层（沉睡）
   * 惰性迁移（touch/reinforce 时评估，无定时扫描）。
   */
  state?: 'episodic' | 'semantic' | 'dormant'
  /**
   * 版本链（v0.3，治理缺口①真理维护）：被哪个节点取代（id）。
   * 被取代 = 废止（superseded）：退出检索/注入（只注入"当前有效"），
   * 但 engram_open/byTitle 仍可追溯旧版（版本链可回溯，不删数据）。
   */
  supersededBy?: string
  /** 版本链：取代了哪些节点（id 列表，反向追溯）。 */
  supersedes?: string[]
  /** project 层标识（会话工作目录；global 层为 null）。 */
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
  /**
   * token 倒排索引（v0.5 纯算法语义关键修复）：token -> Set<nodeId>——
   * **词袋召回**（与 PCA 语义对齐）。哈希 n-gram 要求词组连续匹配，查询
   * 「pnpm 装包报错 EPERM」vs 记忆「pnpm EPERM 修复」词袋共享但窗口不
   * 连续 → 槽位交集 0 → 粗筛漏掉（实测根因）。倒排补上词袋维度。
   */
  private tokenIndex = new Map<string, Set<string>>()
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
          e.links = Array.isArray(e.links) ? e.links.map((l) => String(l).replace(/^\[\[|\]\]$/g, '').trim()).filter(Boolean) : []
          e.causes = Array.isArray(e.causes) ? e.causes : []
          e.effects = Array.isArray(e.effects) ? e.effects : []
          e.importance = typeof e.importance === 'number' ? e.importance : 0
          e.hits = typeof e.hits === 'number' ? e.hits : 0
          e.createdAt = typeof e.createdAt === 'number' ? e.createdAt : 0
          e.reinforces = Array.isArray(e.reinforces) ? e.reinforces : [e.createdAt || Date.now()]
          // v0.3 巩固状态归一化：旧数据默认 episodic（不丢、可迁移）
          e.state = (e.state === 'semantic' || e.state === 'dormant') ? e.state : 'episodic'
          // v0.4 标签迁移：旧数据无 tags → 从 layer/projectId 生成默认标签
          if (!Array.isArray(e.tags) || e.tags.length === 0) {
            e.tags = defaultTags(e.layer, e.projectId)
          }
          this.byId.set(e.id, e)
          for (const s of e.slots) this.indexSlot(s, e.id)
          this.indexTokens(this.textOfNode(e), e.id)
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

  /** token 倒排索引登记（词袋召回维度——与 PCA 语义对齐）。 */
  private indexTokens(text: string, id: string): void {
    for (const tok of this.hasher.normalize(text)) {
      let set = this.tokenIndex.get(tok)
      if (!set) {
        set = new Set()
        this.tokenIndex.set(tok, set)
      }
      set.add(id)
    }
  }

  /** token 倒排移除。 */
  private unindexTokens(text: string, id: string): void {
    for (const tok of this.hasher.normalize(text)) {
      const set = this.tokenIndex.get(tok)
      if (!set) continue
      set.delete(id)
      if (set.size === 0) this.tokenIndex.delete(tok)
    }
  }

  /** 节点检索文本（title + summary + content 的前部——token 倒排用）。 */
  private textOfNode(e: EngramNode): string {
    return `${e.title}：${e.summary} ${(e.content ?? '').slice(0, 100)}`
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
   * v0.4：支持 tags 多标签（缺省从 layer/projectId 生成默认标签）。
   */
  add(input: Omit<EngramNode, 'id' | 'createdAt' | 'hits' | 'lastHitAt' | 'slots' | 'layer' | 'projectId'>
    & { layer?: EngramLayer; projectId?: string | null; tags?: string[] }): EngramNode {
    const keyText = `${input.title} ${input.summary}`
    const result = this.hasher.hash(keyText)
    const slots = this.hasher.slotKeys(result)
    const layer = input.layer ?? 'project'
    const projectId = input.projectId ?? null
    const node: EngramNode = {
      ...input,
      layer,
      projectId,
      tags: Array.isArray(input.tags) && input.tags.length > 0
        ? [...new Set(input.tags)]
        : defaultTags(layer, projectId),
      id: createEngramId(),
      createdAt: Date.now(),
      hits: 0,
      lastHitAt: null,
      slots,
      // 写入即第一次强化（类脑：刚记住时最容易被想起）
      reinforces: input.reinforces ?? [Date.now()],
      // v0.3 巩固状态：写入即 episodic（刚记住，事件性最强）
      state: 'episodic',
    }
    this.byId.set(node.id, node)
    for (const s of slots) this.indexSlot(s, node.id)
    this.indexTokens(this.textOfNode(node), node.id)
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

  /** 按标题取节点（双向链接 [[title]] 解析）。同名消歧：**未废止（active）优先**，
   *  其次最近写入——否则链接解析会取到废止旧版（第七轮新 agent 实测：
   *  邻接标注错乱根因）。⚠️ 曾有 Map<title,单id> 覆盖 bug——多值后不再丢。 */
  byTitle(title: string): EngramNode | undefined {
    const arr = this.titleIndex.get(title)
    if (!arr || arr.length === 0) return undefined
    // active 优先：从后往前找第一个未废止的
    for (let i = arr.length - 1; i >= 0; i--) {
      const e = this.byId.get(arr[i])
      if (e && !isSuperseded(e)) return e
    }
    return this.byId.get(arr[arr.length - 1])
  }

  /** 按标题取全部同名节点（消歧/盘点用：跨项目同标题、版本链同主题）。 */
  byTitles(title: string): EngramNode[] {
    return (this.titleIndex.get(title) ?? [])
      .map((id) => this.byId.get(id))
      .filter((e): e is EngramNode => !!e)
  }

  /**
   * 工作快照（远景场景 6"继续昨天的工作"）：聚合 cwd 最近写入的进行中
   * 状态 → 快照节点（kind=snapshot，episodic 起步，每次更新即强化保持活跃）。
   * 幂等：同 cwd 已有快照且内容未变 → 不写盘；内容变化 → 原地更新 + 强化。
   * 快照是"当前状态"不是"事实"——原地更新，不建版本链。
   */
  upsertSnapshot(cwd: string, turn: number, sessionId: string | null = null): EngramNode | null {
    if (!cwd) return null
    const latest = this.query({ projectId: cwd, recent: true, limit: 6 })
      .filter((e) => e.kind !== 'snapshot' && !isSuperseded(e))
    if (latest.length === 0) return null
    const body = latest.map((e) => `- [[${e.title}]]：${e.summary.slice(0, 60)}`).join('\n')
    const summary = `进行中：${latest[0].title}${latest.length > 1 ? ` 等 ${latest.length} 项` : ''}`
    const existing = this.query({ kind: 'snapshot', projectId: cwd })
    if (existing.length > 0) {
      const snap = existing[existing.length - 1]
      if (snap.content !== body) {
        this.update(snap.id, {
          summary,
          content: body,
          // ⚠️ 修复（2026-08 用户反馈"中间一个大圆形"）：links 曾**累积追加**
          // （旧 links ∪ 最新 6 条）——快照每轮 +6 条涨到 172 条，把所有
          // 近期节点串成巨型连通分量（89%）→ 改为**替换**为当前最近 6 条
          // （快照 = 此刻工作状态，links = 此刻相关节点）
          links: latest.map((e) => e.title),
        })
        this.reinforce(snap.id) // 每次更新 = 强化（快照不沉睡）
      }
      return snap
    }
    const base = String(cwd).split(/[\\/]/).pop() || '项目'
    return this.add({
      kind: 'snapshot',
      layer: 'project',
      projectId: cwd,
      title: `工作快照·${base}`,
      summary,
      content: body,
      links: latest.map((e) => e.title),
      sessionId,
      turn,
      causes: [],
      effects: [],
      importance: 0.9,
    })
  }

  /** 按文本哈希寻址 + token 倒排词袋召回（去重，按关联度降序；不含废止节点）。 */
  lookup(text: string, limit = 8): EngramNode[] {
    const result = this.hasher.hash(text)
    return this.lookupHash(result, text, limit)
  }

  /** token 倒排查询（v0.6 共现扩展粗筛用）：含任一 token 的记忆并集
   *  ——与词袋语义对齐，支持"查询词共现邻居"召回。 */
  lookupTokens(tokens: string[], limit = 16): EngramNode[] {
    const ids = new Set<string>()
    for (const t of tokens) {
      const s = this.tokenIndex.get(t)
      if (s) for (const id of s) ids.add(id)
    }
    const out: EngramNode[] = []
    for (const id of ids) {
      const e = this.byId.get(id)
      if (e && e.status !== 'pending' && !isSuperseded(e)) out.push(e)
    }
    out.sort((a, b) => b.importance - a.importance)
    return out.slice(0, limit)
  }

  /** 按已计算的哈希结果寻址（避免重复哈希）；text 供 token 倒排词袋召回。 */
  lookupHash(result: HashResult, text = '', limit = 8): EngramNode[] {
    const keys = this.hasher.slotKeys(result)
    const seen = new Set<string>()
    const hits: EngramNode[] = []
    // ⚠️ 规模化（10 万级 benchmark）：高频词槽候选可爆炸到全库（max 10 万）——
    // 必须提前截断，不能全量收集再排序（O(N) 遍历）。
    // **双来源均衡采样**（v0.5 修复）：
    //  ① 哈希槽位（精确短语——词组连续匹配）
    //  ② token 倒排（词袋召回——与 PCA 语义对齐；查询「pnpm 装包报错
    //     EPERM」vs 记忆「pnpm EPERM 修复」词袋共享但 n-gram 窗口不连续
    //     → 哈希槽交集 0，靠倒排兜住——实测"查不到刚写的记忆"根因）
    // 每来源每轮取 1 个轮转，防止高频来源独占预算。
    const iters: Array<Iterator<string>> = []
    for (const k of keys) {
      const s = this.slotIndex.get(k)
      if (s) iters.push(s.values())
    }
    // token 倒排：查询 token 的并集迭代器（先收集 id 集合再取迭代器）
    const tokenIds = new Set<string>()
    for (const tok of this.hasher.normalize(text)) {
      const s = this.tokenIndex.get(tok)
      if (s) for (const id of s) tokenIds.add(id)
    }
    if (tokenIds.size > 0) iters.push(tokenIds.values())
    const budget = Math.max(limit, 64)
    let round = 0
    outer:
    while (hits.length < budget && round < 64) {
      let allDone = true
      for (const it of iters) {
        if (hits.length >= budget) break outer
        const step = it.next()
        if (step.done) continue
        allDone = false
        const id = step.value
        if (seen.has(id)) continue
        seen.add(id)
        const e = this.byId.get(id)
        // ⚠️ 版本链：废止节点不参与检索（只注入"当前有效"；追溯走 byTitle/open）
        if (e && e.status !== 'pending' && !isSuperseded(e)) {
          hits.push(e)
        }
      }
      round++
      // ⚠️ 终止条件：所有迭代器耗尽才算完——不能按"本轮无新增"退出
      if (allDone) break
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

  /** 巩固状态统计（status 工具用；旧数据默认 episodic；dormant 派生）。 */
  stateCounts(): Record<'episodic' | 'semantic' | 'dormant', number> {
    const counts: Record<'episodic' | 'semantic' | 'dormant', number> = { episodic: 0, semantic: 0, dormant: 0 }
    for (const e of this.byId.values()) {
      counts[dormantOf(e) ? 'dormant' : (e.state === 'semantic' ? 'semantic' : 'episodic')] += 1
    }
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

  /**
   * 版本链（真理维护核心）：newId 取代 oldId——
   *  - old.supersededBy = newId（退出检索/注入，只注入"当前有效"）
   *  - new.supersedes += oldId（反向追溯版本链）
   * 数据不删（可追溯），因果/链接继承由调用方迁移。
   */
  supersede(oldId: string, newId: string): boolean {
    const old = this.byId.get(oldId)
    const neu = this.byId.get(newId)
    if (!old || !neu) return false
    old.supersededBy = newId
    neu.supersedes = [...(neu.supersedes ?? []), oldId]
    this.persist()
    return true
  }

  /** 修正节点字段（title 变更会同步标题索引；层变更用 promote；tags 覆盖设置）。 */
  update(id: string, patch: Partial<Pick<EngramNode, 'title' | 'summary' | 'content' | 'links' | 'causes' | 'effects' | 'importance' | 'tags'>>): EngramNode | undefined {
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
    if (patch.tags !== undefined) e.tags = [...new Set(patch.tags)]
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
    this.evolveState(e)
    this.persist()
  }

  /** 登记一次强化（展开/链接等深度使用——权重高于命中）。 */
  reinforce(id: string): void {
    const e = this.byId.get(id)
    if (!e) return
    e.reinforces = e.reinforces ?? [e.createdAt || Date.now()]
    e.reinforces.push(Date.now())
    this.evolveState(e)
    this.persist()
  }

  /**
   * 惰性状态迁移（命中/强化时评估，无定时扫描）：
   *  - hits ≥ 3 → semantic（强化历史充足，去情景化，持久固化）
   *  - dormant 是**派生状态**（见 isDormant）：由强化历史实时计算——
   *    被 touch 的节点刚强化，永不处于沉睡；沉默 >30 天的节点在
   *    渲染/候选时降级为仅标题，不占注入预算。
   */
  private evolveState(e: EngramNode): void {
    if (e.hits >= 3) e.state = 'semantic'
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
    this.unindexTokens(this.textOfNode(e), e.id)
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

  /** 归档文件（被淘汰节点保留可恢复，不删数据）。 */
  private archiveFile(): string {
    return join(this.dir, 'archived.jsonl')
  }

  /**
   * 归档一个节点：JSON 追加到 archived.jsonl（带 archivedAt），然后从主库移除。
   * 归档 = 完全退出检索/注入/图谱，但可手动恢复（读回 archived.jsonl）。
   */
  archiveNode(id: string): boolean {
    const e = this.byId.get(id)
    if (!e) return false
    const record = { ...e, archivedAt: Date.now() }
    try {
      appendFileSync(this.archiveFile(), `${JSON.stringify(record)}\n`, 'utf8')
    } catch { /* 归档写失败不阻塞主库操作 */ }
    return this.remove(id)
  }

  /**
   * 硬上限（v0.5）：count > maxNodes 时按「留着最没用」顺序淘汰归档：
   *  ① superseded（已废止——真理已由当前版承载）
   *  ② dormant（沉睡——长期未用，不占注入预算仍占存储）
   *  ③ 低激活 episodic（强化少 + 创建久）
   * 返回淘汰数。惰性触发（写入后调用），不阻塞。
   */
  enforceLimit(maxNodes: number): number {
    if (maxNodes <= 0) return 0
    const excess = this.count() - maxNodes
    if (excess <= 0) return 0
    const now = Date.now()
    const valueKey = (e: EngramNode): number => {
      if (isSuperseded(e)) return 0
      if (dormantOf(e, now)) return 1
      // 低激活：强化次数少 + 久未用
      const last = e.reinforces && e.reinforces.length > 0 ? e.reinforces[e.reinforces.length - 1] : e.createdAt
      return 2 + (e.reinforces?.length ?? 0) / 10 + (now - last) / 1e12
    }
    const doomed = this.all()
      .sort((a, b) => valueKey(a) - valueKey(b))
      .slice(0, excess)
    let archived = 0
    for (const e of doomed) {
      if (this.archiveNode(e.id)) archived++
    }
    return archived
  }

  /** 归档节点数（status 显示）。 */
  archivedCount(): number {
    try {
      return readFileSync(this.archiveFile(), 'utf8').split('\n').filter((l) => l.trim() !== '').length
    } catch {
      return 0
    }
  }
}

