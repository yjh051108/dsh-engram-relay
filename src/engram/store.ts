/**
 * EngramStore — 外置 engram 持久化存储（JSONL）。
 *
 * 每个 engram 是一条结构化记忆痕迹：
 *  - 类型：fact（事实）/ decision（决策）/ event（事件）/ preference（偏好）
 *  - 内容：小模型蒸馏后的摘要文本
 *  - 因果：causes（导致它的原因 id 集）/ effects（它导致的后果 id 集），
 *    构成因果图的边
 *  - 元数据：来源会话、时间戳、重要度、访问计数（LRU 相关）
 *
 * 存储位于 harness 之外（默认 ~/.dsh/engram-relay/），JSONL 追加写 +
 * 启动时全量加载，简单可靠、零外部依赖。
 */

import { mkdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs'
import { homedir } from 'node:os'
import { dirname, join, resolve } from 'node:path'

export type EngramKind = 'fact' | 'decision' | 'event' | 'preference'

export interface Engram {
  id: string
  kind: EngramKind
  /** 小模型蒸馏后的摘要文本（唤醒时注入的正文）。 */
  text: string
  /** 唤醒时的精简标签（一行）。 */
  label: string
  /** 来源会话 id（可空 = 全局）。 */
  sessionId: string | null
  /** 来源回合序号。 */
  turn: number
  /** 因果边：导致本痕迹的 engram id 集。 */
  causes: string[]
  /** 因果边：本痕迹导致的 engram id 集。 */
  effects: string[]
  /** 创建时间（epoch ms）。 */
  createdAt: number
  /** 重要度 0-1（小模型打分），用于稀疏截断。 */
  importance: number
  /** 被唤醒次数（LRU 衰减用）。 */
  hits: number
  /** 最后唤醒时间。 */
  lastHitAt: number | null
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

  constructor(storeDir: string) {
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
      } catch {
        // 单条损坏跳过，不拖垮整个存储
      }
    }
  }

  private persist(): void {
    mkdirSync(dirname(this.file), { recursive: true })
    const lines: string[] = []
    for (const e of this.byId.values()) lines.push(JSON.stringify(e))
    writeFileSync(this.file, lines.join('\n') + '\n', 'utf8')
  }

  add(input: Omit<Engram, 'id' | 'createdAt' | 'hits' | 'lastHitAt'>): Engram {
    const engram: Engram = {
      ...input,
      id: createEngramId(),
      createdAt: Date.now(),
      hits: 0,
      lastHitAt: null,
    }
    this.byId.set(engram.id, engram)
    this.persist()
    return engram
  }

  get(id: string): Engram | undefined {
    return this.byId.get(id)
  }

  /** 按 id 集批量取回（保序）。 */
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

  count(): number {
    return this.byId.size
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
    const ok = this.byId.delete(id)
    if (ok) this.persist()
    return ok
  }
}
