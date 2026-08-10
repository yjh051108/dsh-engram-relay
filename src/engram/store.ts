/**
 * EngramStore — 外置 engram 条件记忆表（JSONL 持久化）。
 *
 * 对应 DeepSeek Engram 论文的「静态记忆表」：哈希槽 → 记忆条目。
 *  - 每条 engram 由 N-gram 哈希寻址（NgramHashAddressing）写入多个槽位
 *    （多头：n2h0..n3h3），读取时按当前上下文的哈希命中间接取回；
 *  - 槽位映射（slot → engram ids）与条目本体（engrams.jsonl）分离，
 *    槽位表是派生索引，条目是事实源；
 *  - 记忆种类（本会话内）：
 *      fact / decision / event / preference（会话内折叠的历史）
 *
 * 定位：**单次会话上下文增强**——折叠本会话早期历史进记忆表、需要时
 * 模型原生回忆；会话结束即弃，不做跨会话记忆沉淀。
 */

import { mkdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs'
import { homedir } from 'node:os'
import { dirname, join, resolve } from 'node:path'

import { NgramHashAddressing, type HashResult } from './hash.js'

export type EngramKind = 'fact' | 'decision' | 'event' | 'preference'

export interface Engram {
  id: string
  kind: EngramKind
  /** 记忆正文（蒸馏后的精炼文本，唤醒时注入）。 */
  text: string
  /** 唤醒时的精简标签（一行）。 */
  label: string
  /** 归属（保留字段：当前统一 null，单会话场景无跨会话归属）。 */
  scope: string | null
  /** 来源会话 id（本会话内折叠）。 */
  sessionId: string | null
  /** 来源回合序号。 */
  turn: number
  /** 因果边：导致本痕迹的 engram id 集（会话内事件链）。 */
  causes: string[]
  /** 因果边：本痕迹导致的 engram id 集。 */
  effects: string[]
  /** 创建时间（epoch ms）。 */
  createdAt: number
  /** 重要度 0-1（蒸馏时小模型打分），用于稀疏截断。 */
  importance: number
  /** 被唤醒次数（LRU 衰减）。 */
  hits: number
  /** 最后唤醒时间。 */
  lastHitAt: number | null
  /** 该条记忆对应的哈希槽位（写入时固化，重哈希可重建）。 */
  slots: string[]
}

let seq = 0

export function createEngramId(): string {
  seq += 1
  return `e${Date.now().toString(36)}-${seq.toString(36)}`
}

export class EngramStore {
  readonly dir: string
  private file: string
  private byId = new Map<string, Engram>()
  /** 槽位索引：slotKey -> Set<engramId>（派生索引，写入/加载时构建）。 */
  private slotIndex = new Map<string, Set<string>>()

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
        const e = JSON.parse(line) as Engram
        this.byId.set(e.id, e)
        for (const s of e.slots) this.indexSlot(s, e.id)
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
   * 写入一条 engram：自动按文本哈希寻址，把条目挂到命中的槽位。
   * 显式传入 text 之外的 keyText 时可让「标签文本」决定寻址（例如
   * 规则记忆按规则主题寻址）。
   */
  add(input: Omit<Engram, 'id' | 'createdAt' | 'hits' | 'lastHitAt' | 'slots'>, keyText?: string): Engram {
    const text = keyText ?? input.text
    const result = this.hasher.hash(text)
    const slots = this.hasher.slotKeys(result)
    const engram: Engram = {
      ...input,
      id: createEngramId(),
      createdAt: Date.now(),
      hits: 0,
      lastHitAt: null,
      slots,
    }
    this.byId.set(engram.id, engram)
    for (const s of slots) this.indexSlot(s, engram.id)
    this.persist()
    return engram
  }

  /** 按文本哈希寻址，返回命中槽位的候选记忆（去重，按重要度降序）。 */
  lookup(text: string, limit = 8): Engram[] {
    const result = this.hasher.hash(text)
    return this.lookupHash(result, limit)
  }

  /** 按已计算的哈希结果寻址（避免重复哈希）。 */
  lookupHash(result: HashResult, limit = 8): Engram[] {
    const keys = this.hasher.slotKeys(result)
    const seen = new Set<string>()
    const hits: Engram[] = []
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

  get(id: string): Engram | undefined {
    return this.byId.get(id)
  }

  getMany(ids: string[]): Engram[] {
    const out: Engram[] = []
    for (const id of ids) {
      const e = this.byId.get(id)
      if (e) out.push(e)
    }
    return out
  }

  all(): Engram[] {
    return [...this.byId.values()]
  }

  byKind(kind: EngramKind): Engram[] {
    return this.all().filter((e) => e.kind === kind)
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
   * 会话隔离：清空某会话的全部 engram（单会话上下文增强——会话结束即弃）。
   * 这是「不做跨会话记忆沉淀」的执行面：会话终结时调用，记忆不跨会话残留。
   */
  clearSession(sessionId: string): number {
    const doomed = this.all().filter((e) => e.sessionId === sessionId)
    for (const e of doomed) {
      this.byId.delete(e.id)
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
