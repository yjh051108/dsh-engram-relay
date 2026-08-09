/**
 * EngramStore — 外置 engram 条件记忆表（JSONL 持久化）。
 *
 * 对应 DeepSeek Engram 论文的「静态记忆表」：哈希槽 → 记忆条目。
 *  - 每条 engram 由 N-gram 哈希寻址（NgramHashAddressing）写入多个槽位
 *    （多头：n2h0..n3h3），读取时按当前上下文的哈希命中间接取回；
 *  - 槽位映射（slot → engram ids）与条目本体（engrams.jsonl）分离，
 *    槽位表是派生索引，条目是事实源；
 *  - 记忆种类覆盖「传统+创新双轨」：
 *      fact / decision / event / preference（对话蒸馏，全量）
 *      global / project / rule（跨会话：全局记忆/项目记忆/规则）
 *
 * 确定性寻址的意义：相同模式永远命中相同槽位——不需要向量相似度
 * 近似，是 O(1) 精确查找。
 */

import { mkdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs'
import { homedir } from 'node:os'
import { dirname, join, resolve } from 'node:path'

import { NgramHashAddressing, type HashResult } from './hash.js'

export type EngramKind = 'fact' | 'decision' | 'event' | 'preference' | 'global' | 'project' | 'rule'

export interface Engram {
  id: string
  kind: EngramKind
  /** 记忆正文（蒸馏后的精炼文本，唤醒时注入）。 */
  text: string
  /** 唤醒时的精简标签（一行）。 */
  label: string
  /** 归属：null = 全局；否则为项目路径/会话 id。 */
  scope: string | null
  /** 来源会话 id（可空 = 手工/跨会话写入）。 */
  sessionId: string | null
  /** 来源回合序号。 */
  turn: number
  /** 因果边：导致本痕迹的 engram id 集（跨会话记忆的因果链）。 */
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
}
