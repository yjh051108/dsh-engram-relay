/**
 * PythonEngramClient — spawn Python 魔改模型服务（JSON 行协议）。
 *
 * 对接 python/engram_model/server.py：stdin/stdout JSON 行 RPC。
 * 能力：load / generate / distill / write_memory / status。
 *
 * 模型未就绪/服务启动失败时：所有调用返回 null（插件降级为纯哈希
 * 路由 + 文本注入，不阻塞主流程）。
 */

import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))

export interface DistillEntry {
  kind: 'fact' | 'decision' | 'event' | 'preference'
  label: string
  text: string
  importance: number
  causes: string[]
}

export class PythonEngramClient {
  private proc: ChildProcessWithoutNullStreams | null = null
  private pending = new Map<number, (resp: unknown) => void>()
  private seq = 0
  private buffer = ''
  private ready: Promise<void> | null = null
  private failed = false

  constructor(private pythonPath = 'python', private modelId = 'Qwen/Qwen3-0.6B') {}

  /** 启动服务（幂等；失败记录后所有调用返回 null）。 */
  start(): void {
    if (this.proc || this.failed) return
    const pythonRoot = join(__dirname, '..', '..', 'python')
    const serverModule = 'engram_model.server'
    if (!existsSync(join(pythonRoot, 'engram_model', 'server.py'))) {
      this.failed = true
      return
    }
    try {
      this.proc = spawn(this.pythonPath, ['-m', serverModule], {
        cwd: pythonRoot,
        stdio: ['pipe', 'pipe', 'pipe'],
        windowsHide: true,
      })
      this.proc.stdout.setEncoding('utf8')
      this.proc.stderr.setEncoding('utf8')
      this.proc.stdout.on('data', (chunk: string) => {
        this.buffer += chunk
        let nl: number
        while ((nl = this.buffer.indexOf('\n')) >= 0) {
          const line = this.buffer.slice(0, nl)
          this.buffer = this.buffer.slice(nl + 1)
          if (line.trim() === '') continue
          try {
            const resp = JSON.parse(line)
            const id = resp?.id
            const cb = this.pending.get(id)
            if (cb) {
              this.pending.delete(id)
              cb(resp)
            }
          } catch {
            // 非 JSON 行（stderr 混入？）忽略
          }
        }
      })
      this.proc.stderr.on('data', () => { /* 日志可接入 ctx.logger */ })
      this.proc.on('exit', () => {
        this.proc = null
        // 所有挂起请求失败
        for (const cb of this.pending.values()) cb({ ok: false, error: 'service exited' })
        this.pending.clear()
      })
      this.proc.on('error', () => { this.failed = true })
    } catch {
      this.failed = true
    }
  }

  private request<T>(op: string, payload: Record<string, unknown>): Promise<T | null> {
    if (!this.proc || this.failed) return Promise.resolve(null)
    const id = ++this.seq
    return new Promise((resolve) => {
      this.pending.set(id, (resp) => {
        if (resp && (resp as { ok?: boolean }).ok) resolve((resp as { result: T }).result)
        else resolve(null)
      })
      this.proc!.stdin.write(JSON.stringify({ id, op, ...payload }) + '\n')
    })
  }

  async load(): Promise<{ loaded: boolean; stats?: { entries: number; slots: number } } | null> {
    this.start()
    return this.request('load', { model: this.modelId })
  }

  async generate(text: string, maxNewTokens = 64, temperature = 0.2): Promise<{ text: string } | null> {
    return this.request('generate', { text, max_new_tokens: maxNewTokens, temperature })
  }

  async distill(conversation: string): Promise<{ raw: string; parsed: DistillEntry | null } | null> {
    return this.request('distill', { conversation })
  }

  async writeMemory(entries: Array<{ text: string }>): Promise<{ written: number; slots: number[] } | null> {
    return this.request('write_memory', { entries })
  }

  async status(): Promise<{ loaded: boolean; entries?: number; slots?: number } | null> {
    return this.request('status', {})
  }

  stop(): void {
    if (this.proc) {
      this.proc.stdin.end()
      this.proc.kill()
      this.proc = null
    }
  }
}
