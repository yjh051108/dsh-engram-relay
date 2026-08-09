/**
 * installEngramTools — 模型面工具注册。
 *
 * 工具：
 *  - engram_recall：主动查询记忆（带因果链展开）
 *  - engram_store：显式写入一条记忆（绕过自动蒸馏）
 *  - engram_status：查看存储统计与唤醒情况
 */

import type { Context as CordisContext } from 'cordis'
import type ToolRegistry from '@deepseek-ai/dsh-tools'
import { defineTool } from '@deepseek-ai/dsh-tools'
import z from 'schemastery'

import { EngramRelay } from './relay.js'

type ToolsContext = CordisContext & { tools: ToolRegistry }

const TEXT_OUTPUT = {
  schema: { type: 'string' as const },
  render: (_args: unknown, value: unknown) => [{ type: 'text' as const, text: String(value) }],
}

export function installEngramTools(ctx: ToolsContext, relay: EngramRelay): () => void {
  const disposers: Array<() => void> = []

  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_recall',
    description: '查询外置 engram 记忆（带因果链展开）。当你需要回忆之前会话里的事实、决策、事件时调用；返回按因果激活排序的记忆痕迹。',
    parameters: {
      query: {
        type: 'string',
        required: true,
        description: '要回忆的内容描述（越具体召回越准）',
      },
      limit: {
        type: 'number',
        description: '最多返回条数（默认 3）',
      },
    },
    output: TEXT_OUTPUT,
    isConcurrencySafe: () => true,
    execute: async (args) => {
      const hit = await relay.recall(String(args.query), Number(args.limit ?? 3))
      if (hit.engrams.length === 0) return `（无相关记忆，reason=${hit.reason}）`
      return hit.engrams.map((e) => `- [${e.kind}] ${e.label}: ${e.text}`).join('\n')
    },
  })))

  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_store',
    description: '显式写入一条 engram 记忆（事实/决策/事件/偏好），绕过自动蒸馏。适合需要立即固化、不想等蒸馏节奏的内容。',
    parameters: {
      kind: {
        type: 'string',
        required: true,
        description: '记忆类型：fact（事实）/ decision（决策）/ event（事件）/ preference（偏好）',
      },
      label: {
        type: 'string',
        required: true,
        description: '一句话标签（唤醒时展示）',
      },
      text: {
        type: 'string',
        required: true,
        description: '记忆正文（小模型可蒸馏的精炼描述）',
      },
      causes: {
        type: 'array',
        items: { type: 'string' },
        description: '可选：导致本条记忆的已有 engram id 列表（因果边）',
      },
    },
    output: TEXT_OUTPUT,
    isConcurrencySafe: () => true,
    execute: async (args) => {
      const kind = String(args.kind)
      const label = String(args.label)
      const text = String(args.text)
      const causes = Array.isArray(args.causes) ? args.causes.map(String) : []
      if (!['fact', 'decision', 'event', 'preference'].includes(kind)) {
        return `错误：kind 必须是 fact/decision/event/preference（收到 ${kind}）`
      }
      const e = relay.store.add({ kind: kind as 'fact', label, text, sessionId: null, turn: 0, causes, effects: [], importance: 1 })
      // 因果边入图
      for (const causeId of causes) {
        relay.graph.addEdge(causeId, e.id, 'causes', 1)
      }
      return `已写入 engram ${e.id}（${kind}）`
    },
  })))

  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_status',
    description: '查看 engram 存储统计与唤醒情况（条数、图边数、模型状态、预算）。',
    parameters: {},
    output: TEXT_OUTPUT,
    isConcurrencySafe: () => true,
    execute: async () => {
      const s = await relay.status()
      return JSON.stringify(s, null, 2)
    },
  })))

  return () => disposers.forEach((d) => d())
}
