/**
 * installEngramTools — 模型面工具注册。
 *
 * 工具：
 *  - engram_recall：主动查询外置 engram 记忆（N-gram 哈希寻址 + 因果链）
 *  - engram_store：显式写入一条记忆（全局/项目/规则，绕过自动蒸馏）
 *  - engram_status：查看存储统计、槽位占用与唤醒情况
 */

import type { Context as CordisContext } from 'cordis'
import type ToolRegistry from '@deepseek-ai/dsh-tools'
import { defineTool } from '@deepseek-ai/dsh-tools'
import z from 'schemastery'

import { EngramRelay } from './relay.js'
import type { EngramKind } from './engram/store.js'

type ToolsContext = CordisContext & { tools: ToolRegistry }

const TEXT_OUTPUT = {
  schema: { type: 'string' as const },
  render: (_args: unknown, value: unknown) => [{ type: 'text' as const, text: String(value) }],
}

const KINDS: EngramKind[] = ['fact', 'decision', 'event', 'preference']

export function installEngramTools(ctx: ToolsContext, relay: EngramRelay): () => void {
  const disposers: Array<() => void> = []

  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_recall',
    description: '查询外置 engram 条件记忆（N-gram 哈希确定性寻址 + 因果链展开）。当你需要回忆之前会话的事实、决策、事件，或跨会话的全局记忆/项目记忆/规则时调用；返回按哈希命中 + 因果激活排序的记忆。',
    parameters: {
      query: {
        type: 'string',
        required: true,
        description: '要回忆的内容（越具体命中越准，与写入时的表述一致最好）',
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
      if (hit.engrams.length === 0) return `（无命中，reason=${hit.reason}）`
      return hit.engrams.map((e) => `- [${e.kind}] ${e.label}: ${e.text}`).join('\n')
    },
  })))

  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_store',
    description: '显式写入一条 engram（本会话内，会话结束即弃）。用于把当前会话的要点固化进记忆表，供模型原生回忆；不做跨会话沉淀。',
    parameters: {
      kind: {
        type: 'string',
        required: true,
        description: '记忆类型：fact/decision/event/preference',
      },
      label: {
        type: 'string',
        required: true,
        description: '一句话标签（唤醒时展示）',
      },
      text: {
        type: 'string',
        required: true,
        description: '记忆正文（写入后按此文本哈希寻址，相同主题永远命中）',
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
      if (!KINDS.includes(kind as EngramKind)) {
        return `错误：kind 必须是 ${KINDS.join('/')}（收到 ${kind}）`
      }
      const e = relay.store.add({
        kind: kind as EngramKind,
        label,
        text,
        scope: null,
        sessionId: relay.currentSessionId,
        turn: relay.lastTurnAt,
        causes,
        effects: [],
        importance: 1,
      })
      for (const causeId of causes) {
        relay.graph.addEdge(causeId, e.id, 'causes', 1)
      }
      return `已写入 engram ${e.id}（${kind}，哈希槽位 ${e.slots.length} 个）`
    },
  })))

  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_status',
    description: '查看 engram 条件记忆表状态：条目数、哈希槽位数、因果图边数、模型状态、注入预算。',
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
