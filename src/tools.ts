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

const KINDS: EngramKind[] = ['fact', 'decision', 'event', 'note']

export function installEngramTools(ctx: ToolsContext, relay: EngramRelay): () => void {
  const disposers: Array<() => void> = []

  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_recall',
    description: '主动唤醒记忆图谱入口（本会话内）。按当前查询匹配入口节点（[[标题]] + 摘要 + 因果邻接），需要详情时用 engram_open 展开。',
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
      return hit.engrams.map((e) => `- [[${e.title}]]: ${e.summary}`).join('\n')
    },
  })))

  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_store',
    description: '写入一个记忆节点（本会话内，会话结束即弃）。大一统记忆图谱：title 是入口锚点，summary 是一句话摘要（入口层展示），content 是完整正文（展开层），links 是双向关联 [[标题]]。',
    parameters: {
      kind: {
        type: 'string',
        required: true,
        description: '记忆类型：fact/decision/event/note',
      },
      title: {
        type: 'string',
        required: true,
        description: '入口锚点标题（如 [[部署端口决策]]；唤醒列表展示）',
      },
      summary: {
        type: 'string',
        required: true,
        description: '一句话摘要（渐进披露第一层）',
      },
      content: {
        type: 'string',
        description: '完整正文（渐进披露第二层，展开时给）',
      },
      links: {
        type: 'array',
        items: { type: 'string' },
        description: '可选：关联节点的标题（Obsidian 风格双向链接 [[标题]]）',
      },
      causes: {
        type: 'array',
        items: { type: 'string' },
        description: '可选：导致本条记忆的已有节点 id 列表（因果边）',
      },
    },
    output: TEXT_OUTPUT,
    isConcurrencySafe: () => true,
    execute: async (args) => {
      const kind = String(args.kind)
      const title = String(args.title)
      const summary = String(args.summary)
      const content = String(args.content ?? '')
      const links = Array.isArray(args.links) ? args.links.map(String) : []
      const causes = Array.isArray(args.causes) ? args.causes.map(String) : []
      if (!KINDS.includes(kind as EngramKind)) {
        return `错误：kind 必须是 ${KINDS.join('/')}（收到 ${kind}）`
      }
      const e = relay.store.add({
        kind: kind as EngramKind,
        title,
        summary,
        content,
        links,
        sessionId: relay.currentSessionId,
        turn: relay.lastTurnAt,
        causes,
        effects: [],
        importance: 1,
      })
      for (const causeId of causes) {
        relay.graph.addEdge(causeId, e.id, 'causes', 1)
      }
      // 双向链接：为每个 [[标题]] 建关联（Obsidian 风格）
      for (const t of links) {
        const target = relay.store.byTitle(t)
        if (target && target.id !== e.id && !target.links.includes(title)) {
          target.links.push(title)
          relay.store.add({ ...target, links: target.links }) // 持久化更新
        }
      }
      return `已写入记忆节点 [[${e.title}]]（${kind}，哈希槽位 ${e.slots.length} 个，链接 ${links.length} 条）`
    },
  })))

  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_open',
    description: '展开一个记忆节点（渐进披露第二层）：返回完整正文 + 双向链接 + 因果前因/后果。当你看到 [[标题]] 入口需要详情时调用。',
    parameters: {
      title: {
        type: 'string',
        required: true,
        description: '要展开的节点标题（入口列表里的 [[标题]]）',
      },
    },
    output: TEXT_OUTPUT,
    isConcurrencySafe: () => true,
    execute: async (args) => {
      const title = String(args.title)
      const node = relay.store.byTitle(title)
      if (!node) return `未找到节点 [[${title}]]（可能已随会话结束清理）`
      relay.store.touch(node.id)
      const causes = relay.store.getMany(node.causes)
      const effects = relay.store.getMany(node.effects)
      const linked = relay.store.getMany(
        node.links.map((t) => relay.store.byTitle(t)?.id ?? '').filter(Boolean),
      )
      const parts: string[] = []
      parts.push(`# [[${node.title}]] (${node.kind})`)
      parts.push(node.summary)
      if (node.content) parts.push(`\n${node.content}`)
      if (causes.length > 0) parts.push(`\n**前因**（因果 ↑）：${causes.map((c) => `[[${c.title}]]`).join('、')}`)
      if (effects.length > 0) parts.push(`**后果**（因果 ↓）：${effects.map((c) => `[[${c.title}]]`).join('、')}`)
      if (linked.length > 0) parts.push(`**关联**（双向链接）：${linked.map((c) => `[[${c.title}]]`).join('、')}`)
      return parts.join('\n')
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
