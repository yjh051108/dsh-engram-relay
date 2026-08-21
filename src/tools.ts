/**
 * installEngramTools — 模型面工具注册（跨会话分层记忆版）。
 *
 * 工具集（大一统记忆图谱 + 分层 + 因果链接）：
 *  - engram_recall：按需唤醒检索（跨会话分层准入 + 因果邻接）
 *  - engram_store：写入一条记忆（**AI 自主决策分层** + 因果前因/后果）
 *  - engram_open：展开入口（渐进披露第二层：正文/链接/因果）
 *  - engram_search：检索记忆图谱（分层/项目/类型/关键词，维护回顾）
 *  - engram_link：显式连接节点（因果/双向链接——织图谱）
 *  - engram_update：修正节点字段
 *  - engram_remove：删除节点
 *  - engram_promote：提升层（session→project/global，会话结束前转长期）
 *  - engram_status：记忆服务状态（分层统计/索引/模型）
 *
 * 可见性边界（跨会话分层）：global 所有会话 / project 同工作目录 / session
 * 本会话。工具 execute 从 exec.agent 取 sessionId + cwd 作为查看者视角。
 */

import type { Context as CordisContext } from 'cordis'
import type ToolRegistry from '@deepseek-ai/dsh-tools'
import { defineTool } from '@deepseek-ai/dsh-tools'

import { EngramRelay } from './relay.js'
import type { VerifyMark } from './types.js'
import { ENGRAM_LAYERS, isVisible, type EngramKind, type EngramLayer, type EngramNode } from './engram/store.js'
import type { CausalEdgeKind } from './engram/causal.js'

type ToolsContext = CordisContext & { tools: ToolRegistry }

const TEXT_OUTPUT = {
  schema: { type: 'string' as const },
  render: (_args: unknown, value: unknown) => [{ type: 'text' as const, text: String(value) }],
}

const KINDS: EngramKind[] = ['fact', 'decision', 'event', 'note']

/** 查看者视角（跨会话可见性边界）：从工具执行上下文解析。 */
function viewerOf(exec: unknown): { sessionId?: string; cwd?: string } {
  const agent = (exec as { agent?: { session?: { id?: string; header?: { cwd?: string } } } })?.agent
  return {
    sessionId: agent?.session?.id,
    cwd: agent?.session?.header?.cwd,
  }
}

/** 节点引用解析：支持 id 或 [[标题]]/标题。 */
function resolveNode(relay: EngramRelay, ref: string): EngramNode | undefined {
  const t = String(ref).replace(/^\[\[|\]\]$/g, '').trim()
  return relay.store.byTitle(t) ?? relay.store.get(t)
}

/**
 * 织网推荐：bge 语义余弦 × 时序归一化加权，返回 top-3 关联候选（供 AI 决策）。
 * 语义门槛 0.40（推荐可比自动唤醒略宽——决策权在 AI）；时序权重：近 20 回合
 * 内加权，远期收敛（1 / (1 + Δturn/20)）。
 */
async function recommendLinks(relay: EngramRelay, text: string, excludeId: string): Promise<string> {
  const others = relay.store.all().filter((e) => e.id !== excludeId && e.status !== 'pending')
  if (others.length === 0) return ''
  const scores = await relay.model.embed(text.slice(0, 300), others).catch(() => null)
  if (!scores || scores.size === 0) return ''
  const curTurn = relay.lastTurnAt
  const ranked = others
    .map((e) => {
      const cosine = scores.get(e.id) ?? 0
      const turnDist = Math.abs(curTurn - (typeof e.turn === 'number' ? e.turn : 0))
      const recency = 1 / (1 + turnDist / 20)
      return { e, cosine, score: cosine * (0.7 + 0.3 * recency) }
    })
    .filter((x) => x.cosine >= 0.4)
    .sort((a, b) => b.score - a.score)
    .slice(0, 3)
  return ranked
    .map((x, i) => `${i + 1}. [[${x.e.title}]]（语义 ${x.cosine.toFixed(2)} × 时序 ${x.score.toFixed(2)}）${x.e.summary.slice(0, 40)}`)
    .join('\n')
}

/** 入口行渲染（[[标题]][层] 摘要；待确认节点带 ⏳ 标记）。 */
function entryLine(e: EngramNode): string {
  const pendingMark = e.status === 'pending' ? ' ⏳' : ''
  return `- [[${e.title}]][${e.layer}]${pendingMark} ${e.summary}`
}

export function installEngramTools(ctx: ToolsContext, relay: EngramRelay): () => void {
  const disposers: Array<() => void> = []

  // ---- engram_recall：按需唤醒检索（跨会话分层准入） ----
  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_recall',
    description: '主动唤醒记忆图谱入口（跨会话分层）。按当前查询匹配入口节点（[[标题]] + 层 + 摘要 + 因果邻接）。缺省召回 global + 本目录 project + 本会话 session；标注含义：✓已锚定=灵枢确认 / ?图谱外=灵枢未锚定（诚实边界）。看到 [[标题]] 后由你判断——需要详情就 engram_open 展开，不需要就直接用摘要作答。',
    parameters: {
      query: {
        type: 'string',
        required: true,
        description: '要回忆的内容（越具体命中越准，与写入时的表述一致最好）',
      },
      layer: {
        type: 'string',
        description: '可选：只召回指定层（逗号分隔如 "global,project"；缺省=可见层全部）',
      },
      limit: {
        type: 'number',
        description: '最多返回条数（默认 3）',
      },
    },
    output: TEXT_OUTPUT,
    isConcurrencySafe: () => true,
    execute: async (args, exec) => {
      const hit = await relay.recall(String(args.query), Number(args.limit ?? 3), viewerOf(exec), String(args.layer ?? ''))
      if (hit.engrams.length === 0) return `（无命中，reason=${hit.reason}——诚实边界：图谱无相关记忆）`
      // 诚实标注：精确命中（标题含查询词）vs 近似召回（纯语义/图命中）
      const qk = String(args.query).replace(/[，。！？,.!?\s]/g, '')
      hit.engrams.forEach((e) => {
        const tag = e.title.includes(qk) || (qk.length >= 2 && e.title.includes(qk.slice(0, 2)))
          ? '' : ' ~近似'
        ;(e as unknown as { _tag?: string })._tag = tag
      })
      // 融合：灵枢白箱验证标注透出（✓已锚定 / ~部分 / ?图谱外）
      const vmarks = (hit as { verify?: Record<string, VerifyMark> }).verify
      const markOf = (e: { id: string }): string => {
        const m = vmarks?.[e.id]
        if (!m) return ''
        if (m.status === 'anchored') return ' ✓已锚定'
        if (m.status === 'partial') return ' ~部分锚定'
        if (m.status === 'unverified') return ' ?图谱外'
        return ''
      }
      return hit.engrams.map((e) => entryLine(e) + markOf(e) + ((e as unknown as { _tag?: string })._tag ?? '')).join('\n')
    },
  })))

  // ---- engram_store：写入记忆（AI 自主决策分层 + 因果前因/后果） ----
  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_store',
    description: '写入一个记忆节点（跨会话分层，**AI 自主决策层归属**）。大一统记忆图谱：title 入口锚点、summary 一句话摘要（入口层）、content 完整正文（展开层）、links 双向关联 [[标题]]、causes 因果前因、effects 因果后果。**撰写规范**：① title ≤12 字、具体可辨认（如「路由残留自愈方案」，忌泛化如「更新」「总结」）；② summary ≤30 字、**不看正文也能判断相关性**（含关键实体/结论）；③ content ≤200 字、只写增量（关键参数/结论/上下文，不重复摘要）；④ **织边方法（可选，你自主决定）**：想关联已有记忆时直接写——links 填 [[标题]]（双向引用）、causes/effects 填 [[标题]] 或 id（因果前因/后果，标题自动解析）；想不起来或不确定就不写，系统会自动基于语义×时序推荐关联候选，届时你再决定采纳（engram_link 建边）/展开确认（engram_open）/跳过——选择权始终在你；⑤ 同主题多处小更新优先 engram_update 修订原节点而非新增。**layer 决策准则**：跨会话长期有价值（事实/偏好/通用约定）→ global；仅本项目相关（决策/踩坑/架构约定）→ project（自动绑定当前工作目录，跨会话持久）；仅本次会话相关（临时进度/过程）→ session（会话结束清理，重要事后 engram_promote 转长期）。',
    parameters: {
      layer: {
        type: 'string',
        required: true,
        description: `记忆分层（AI 自主决策）：${ENGRAM_LAYERS.join('/')}（见 description 决策准则）`,
      },
      kind: {
        type: 'string',
        required: true,
        description: `记忆类型：${KINDS.join('/')}`,
      },
      title: {
        type: 'string',
        required: true,
        description: '入口锚点标题（如 [[部署端口决策]]；唤醒列表展示，Obsidian 风格）',
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
        description: '可选：关联节点的标题（Obsidian 双向链接 [[标题]]）',
      },
      causes: {
        type: 'array',
        items: { type: 'string' },
        description: '可选：导致本条记忆的已有节点（id 或 [[标题]]，标题自动解析）',
      },
      effects: {
        type: 'array',
        items: { type: 'string' },
        description: '可选：本条记忆导致的已有节点（id 或 [[标题]]，标题自动解析）',
      },
    },
    output: TEXT_OUTPUT,
    isConcurrencySafe: () => true,
    execute: async (args, exec) => {
      const viewer = viewerOf(exec)
      const layer = String(args.layer) as EngramLayer
      if (!ENGRAM_LAYERS.includes(layer)) {
        return `错误：layer 必须是 ${ENGRAM_LAYERS.join('/')}（收到 ${layer}）`
      }
      const kind = String(args.kind)
      if (!KINDS.includes(kind as EngramKind)) {
        return `错误：kind 必须是 ${KINDS.join('/')}（收到 ${kind}）`
      }
      const title = String(args.title)
      const summary = String(args.summary)
      const content = String(args.content ?? '')
      const links = Array.isArray(args.links) ? args.links.map(String) : []
      // causes/effects 支持 id 或 [[标题]]（标题自动解析成 id，与蒸馏 causesOf 一致）
      const resolveRef = (ref: string): string | null => {
        const clean = ref.replace(/^\[\[|\]\]$/g, '').trim()
        if (relay.store.get(clean)) return clean // 已是 id
        const byTitle = relay.store.byTitle(clean)
        return byTitle ? byTitle.id : null
      }
      const causes = (Array.isArray(args.causes) ? args.causes.map(String) : [])
        .map(resolveRef).filter((x): x is string => x !== null)
      const effects = (Array.isArray(args.effects) ? args.effects.map(String) : [])
        .map(resolveRef).filter((x): x is string => x !== null)
      // 分层归属校验：project 层需要当前工作目录；session 层需要会话 id
      if (layer === 'project' && !viewer.cwd) {
        return `错误：project 层需要当前工作目录（无 cwd 的会话不能写项目记忆——建议改用 global 或 session）`
      }
      if (layer === 'session' && !viewer.sessionId) {
        return `错误：session 层需要会话上下文（无会话视角——建议改用 global）`
      }
      const e = relay.store.add({
        kind: kind as EngramKind,
        layer,
        projectId: layer === 'project' ? viewer.cwd! : null,
        title,
        summary,
        content,
        links,
        sessionId: viewer.sessionId ?? relay.currentSessionId,
        turn: relay.lastTurnAt,
        causes,
        effects,
        importance: 1,
      })
      // 因果边：causes（前因 → 本节点）/ effects（本节点 → 后果）
      for (const causeId of causes) {
        const c = relay.store.get(causeId)
        if (c) {
          relay.graph.addEdge(c.id, e.id, 'causes', 1)
          if (!c.effects.includes(e.id)) relay.store.update(c.id, { effects: [...c.effects, e.id] })
        }
      }
      for (const effectId of effects) {
        const t = relay.store.get(effectId)
        if (t) {
          relay.graph.addEdge(e.id, t.id, 'causes', 1)
          if (!t.causes.includes(e.id)) relay.store.update(t.id, { causes: [...t.causes, e.id] })
        }
      }
      // 双向链接：为每个 [[标题]] 建关联（Obsidian 风格）
      for (const t of links) {
        const target = relay.store.byTitle(t)
        if (target && target.id !== e.id && !target.links.includes(title)) {
          target.links.push(title)
          relay.store.add({ ...target, links: target.links }) // 持久化更新
        }
      }
      // 织网推荐：AI 没带任何边时，触发一次「bge 语义 + 时序」推荐（不自动建边，
      // 由 AI 决策——认识的直接选，不认识的展开正文再定或跳过）。
      if (causes.length === 0 && effects.length === 0 && links.length === 0) {
        const rec = await recommendLinks(relay, `${title}：${summary}`, e.id)
        const base = `已写入记忆节点 [[${e.title}]]（${layer}·${kind}，哈希槽位 ${e.slots.length} 个，无因果/链接）`
        if (rec) {
          return `${base}\n\n📎 推荐关联（bge 语义 × 时序加权，未自动建边）：\n${rec}\n\n处理建议：① 标题熟悉且相关 → engram_link 直接采纳（建因果/引用边）；② 标题陌生但想确认 → 先 engram_open 展开正文再定；③ 不相关 → 跳过即可，不影响本条记忆。`
        }
        return base + '\n（当前无显著关联候选）'
      }
      return `已写入记忆节点 [[${e.title}]]（${layer}·${kind}，哈希槽位 ${e.slots.length} 个，链接 ${links.length} 条，因果 ↑${causes.length} ↓${effects.length}）`
    },
  })))

  // ---- engram_propose：提议写入（用户确认制：pending 不参与检索，确认后才生效） ----
  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_propose',
    description: '提议一条记忆节点（**用户确认制**）：写入后为待确认状态（⏳），不参与 recall/唤醒命中；用户确认后（engram_confirm）才生效。用于模型自动沉淀/不确定该不该记的内容。参数与 engram_store 相同；分层/类型准则见 engram_store 描述。',
    parameters: {
      layer: {
        type: 'string',
        required: true,
        description: `记忆分层（AI 自主决策）：${ENGRAM_LAYERS.join('/')}（见 engram_store 描述）`,
      },
      kind: {
        type: 'string',
        required: true,
        description: `记忆类型：${KINDS.join('/')}`,
      },
      title: {
        type: 'string',
        required: true,
        description: '入口锚点标题（如 [[部署端口决策]]；唤醒列表展示，Obsidian 风格）',
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
        description: '可选：关联节点的标题（Obsidian 双向链接 [[标题]]）',
      },
      causes: {
        type: 'array',
        items: { type: 'string' },
        description: '可选：导致本条记忆的已有节点（id 或 [[标题]]，标题自动解析）',
      },
      effects: {
        type: 'array',
        items: { type: 'string' },
        description: '可选：本条记忆导致的已有节点（id 或 [[标题]]，标题自动解析）',
      },
    },
    output: TEXT_OUTPUT,
    isConcurrencySafe: () => true,
    execute: async (args, exec) => {
      const viewer = viewerOf(exec)
      const layer = String(args.layer) as EngramLayer
      if (!ENGRAM_LAYERS.includes(layer)) {
        return `错误：layer 必须是 ${ENGRAM_LAYERS.join('/')}（收到 ${layer}）`
      }
      const kind = String(args.kind)
      if (!KINDS.includes(kind as EngramKind)) {
        return `错误：kind 必须是 ${KINDS.join('/')}（收到 ${kind}）`
      }
      const title = String(args.title)
      const summary = String(args.summary)
      const content = String(args.content ?? '')
      const links = Array.isArray(args.links) ? args.links.map(String) : []
      // causes/effects 支持 id 或 [[标题]]（标题自动解析成 id，与蒸馏 causesOf 一致）
      const resolveRef = (ref: string): string | null => {
        const clean = ref.replace(/^\[\[|\]\]$/g, '').trim()
        if (relay.store.get(clean)) return clean
        const byTitle = relay.store.byTitle(clean)
        return byTitle ? byTitle.id : null
      }
      const causes = (Array.isArray(args.causes) ? args.causes.map(String) : [])
        .map(resolveRef).filter((x): x is string => x !== null)
      const effects = (Array.isArray(args.effects) ? args.effects.map(String) : [])
        .map(resolveRef).filter((x): x is string => x !== null)
      if (layer === 'project' && !viewer.cwd) {
        return `错误：project 层需要当前工作目录（无 cwd 的会话不能写项目记忆——建议改用 global 或 session）`
      }
      if (layer === 'session' && !viewer.sessionId) {
        return `错误：session 层需要会话上下文（无会话视角——建议改用 global）`
      }
      const e = relay.store.add({
        kind: kind as EngramKind,
        layer,
        projectId: layer === 'project' ? viewer.cwd! : null,
        title,
        summary,
        content,
        links,
        sessionId: viewer.sessionId ?? relay.currentSessionId,
        turn: relay.lastTurnAt,
        causes,
        effects,
        importance: 1,
        status: 'pending',
      })
      return `已提议记忆节点 [[${e.title}]]（${layer}·${kind}·⏳待确认，哈希槽位 ${e.slots.length} 个）——用户确认（engram_confirm）后才会参与检索/唤醒`
    },
  })))

  // ---- engram_confirm：确认待确认节点（确认后参与检索/唤醒） ----
  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_confirm',
    description: '确认一个待确认（⏳）记忆节点：确认后参与 recall/唤醒命中。参数为节点引用（id 或 [[标题]]）。已确认节点调用幂等无副作用。',
    parameters: {
      ref: {
        type: 'string',
        required: true,
        description: '节点引用（id 或 [[标题]]）',
      },
    },
    output: TEXT_OUTPUT,
    isConcurrencySafe: () => true,
    execute: async (args, exec) => {
      const node = resolveNode(relay, String(args.ref))
      if (!node) return `未找到节点 [[${args.ref}]]`
      const viewer = viewerOf(exec)
      if (!isVisible(node, viewer)) {
        return `错误：只能确认当前会话可见的节点（global + 本目录 project + 本会话 session）`
      }
      if (node.status !== 'pending') {
        return `[[${node.title}]] 不是待确认状态（当前 ${node.status ?? 'confirmed'}），无需确认`
      }
      relay.store.confirmNode(node.id)
      return `已确认 [[${node.title}]]（${node.layer}·${node.kind}）——现在参与检索与唤醒`
    },
  })))

  // ---- engram_reject：拒绝（删除）待确认节点 ----
  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_reject',
    description: '拒绝（删除）一个待确认（⏳）记忆节点。**只能删除 pending 节点**（已确认节点不可拒删，防止误删已生效记忆——如需删除请用 engram_remove）。',
    parameters: {
      ref: {
        type: 'string',
        required: true,
        description: '节点引用（id 或 [[标题]]）',
      },
    },
    output: TEXT_OUTPUT,
    isConcurrencySafe: () => true,
    execute: async (args, exec) => {
      const node = resolveNode(relay, String(args.ref))
      if (!node) return `未找到节点 [[${args.ref}]]`
      const viewer = viewerOf(exec)
      if (!isVisible(node, viewer)) {
        return `错误：只能拒绝当前会话可见的节点（global + 本目录 project + 本会话 session）`
      }
      if (node.status !== 'pending') {
        return `[[${node.title}]] 不是待确认状态（当前 ${node.status ?? 'confirmed'}）——拒绝仅对 ⏳待确认 节点生效；已确认节点如需删除请用 engram_remove`
      }
      relay.store.rejectNode(node.id)
      return `已拒绝并删除待确认节点 [[${node.title}]]`
    },
  })))

  // ---- engram_open：展开入口（渐进披露第二层） ----
  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_open',
    description: '展开一个记忆节点（渐进披露第二层）：返回完整正文 + 层 + 双向链接 + 因果前因/后果。当你看到 [[标题]] 入口需要详情时调用。',
    parameters: {
      title: {
        type: 'string',
        required: true,
        description: '要展开的节点标题（入口列表里的 [[标题]]）',
      },
    },
    output: TEXT_OUTPUT,
    isConcurrencySafe: () => true,
    execute: async (args, exec) => {
      const node = resolveNode(relay, String(args.title))
      if (!node) return `未找到节点 [[${args.title}]]（可 engram_search 检索）`
      // 可见性：只能展开自己可见层的节点
      if (!isVisible(node, viewerOf(exec))) return `无权展开 [[${node.title}]]（${node.layer} 层对当前会话不可见）`
      // 展开 = 深度使用 → 强化（激活模型 B 增量）
      relay.store.reinforce(node.id)
      relay.activation?.update(node.id, relay.store.get(node.id)?.reinforces)
      relay.store.touch(node.id)
      const causes = relay.store.getMany(node.causes)
      const effects = relay.store.getMany(node.effects)
      const linked = relay.store.getMany(
        node.links.map((t) => relay.store.byTitle(t)?.id ?? '').filter(Boolean),
      )
      const parts: string[] = []
      parts.push(`# [[${node.title}]] (${node.kind} · ${node.layer}${node.projectId ? ` · ${node.projectId}` : ''})`)
      parts.push(node.summary)
      if (node.content) parts.push(`\n${node.content}`)
      if (causes.length > 0) parts.push(`\n**前因**（因果 ↑）：${causes.map((c) => `[[${c.title}]]`).join('、')}`)
      if (effects.length > 0) parts.push(`**后果**（因果 ↓）：${effects.map((c) => `[[${c.title}]]`).join('、')}`)
      if (linked.length > 0) parts.push(`**关联**（双向链接）：${linked.map((c) => `[[${c.title}]]`).join('、')}`)
      return parts.join('\n')
    },
  })))

  // ---- engram_search：检索图谱（分层/项目/类型/关键词） ----
  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_search',
    description: '检索记忆图谱（回顾/维护）：按层/项目/类型/关键词过滤，遵守跨会话可见性（global + 本目录 project + 本会话 session）。返回入口列表（[[标题]] + 摘要）。用于盘点已沉淀的记忆、找要 update/remove/promote/link 的目标。',
    parameters: {
      query: {
        type: 'string',
        description: '可选：标题/摘要关键词（子串匹配，大小写不敏感）',
      },
      layer: {
        type: 'string',
        description: `可选：只查指定层（${ENGRAM_LAYERS.join('/')}）`,
      },
      kind: {
        type: 'string',
        description: `可选：只查指定类型（${KINDS.join('/')}）`,
      },
      projectId: {
        type: 'string',
        description: '可选：只查指定项目（project 层；缺省=当前工作目录）',
      },
      limit: {
        type: 'number',
        description: '最多返回条数（默认 20）',
      },
      recent: {
        type: 'boolean',
        description: '可选：按最近创建排序（缺省按重要度）',
      },
    },
    output: TEXT_OUTPUT,
    isConcurrencySafe: () => true,
    execute: async (args, exec) => {
      const viewer = viewerOf(exec)
      const layer = String(args.layer ?? '')
      const kind = String(args.kind ?? '')
      const projectId = args.projectId !== undefined && String(args.projectId) !== ''
        ? String(args.projectId) : (viewer.cwd ?? undefined)
      let nodes = relay.store.query({
        layer: layer && ENGRAM_LAYERS.includes(layer as EngramLayer) ? layer as EngramLayer : undefined,
        projectId: projectId !== undefined ? projectId : undefined,
        kind: kind && KINDS.includes(kind as EngramKind) ? kind as EngramKind : undefined,
        limit: Number(args.limit ?? 20),
        recent: args.recent === true,
      })
      // 可见性：只返回当前会话可见的
      nodes = nodes.filter((e) => isVisible(e, viewer))
      const q = String(args.query ?? '').trim().toLowerCase()
      if (q) nodes = nodes.filter((e) => e.title.toLowerCase().includes(q) || e.summary.toLowerCase().includes(q))
      if (nodes.length === 0) return '（无匹配记忆）'
      const lines = nodes.map((e) => entryLine(e))
      return `记忆图谱（${nodes.length} 条）：\n${lines.join('\n')}`
    },
  })))

  // ---- engram_link：显式连接节点（织图谱） ----
  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_link',
    description: '显式连接两个记忆节点（把记忆织成图谱）。kind=causes 表示 from 是 to 的前因（因果边）；depends-on 表示 from 依赖 to；references 表示单纯引用。bidirectional=true 同时建立双向 [[标题]] 链接。',
    parameters: {
      from: {
        type: 'string',
        required: true,
        description: '源节点（id 或 [[标题]]）',
      },
      to: {
        type: 'string',
        required: true,
        description: '目标节点（id 或 [[标题]]）',
      },
      kind: {
        type: 'string',
        description: '边类型：causes（默认）/depends-on/references',
      },
      bidirectional: {
        type: 'boolean',
        description: '可选：同时建立双向链接（Obsidian 风格）',
      },
    },
    output: TEXT_OUTPUT,
    isConcurrencySafe: () => true,
    execute: async (args, exec) => {
      const from = resolveNode(relay, String(args.from))
      const to = resolveNode(relay, String(args.to))
      if (!from) return `未找到源节点 [[${args.from}]]`
      if (!to) return `未找到目标节点 [[${args.to}]]`
      if (from.id === to.id) return '错误：不能连接节点自身'
      const viewer = viewerOf(exec)
      if (!isVisible(from, viewer) || !isVisible(to, viewer)) {
        return '错误：只能连接当前会话可见的节点（global + 本目录 project + 本会话 session）'
      }
      const kind = (String(args.kind ?? 'causes') as CausalEdgeKind)
      relay.graph.addEdge(from.id, to.id, kind, 1)
      // 织网 = 深度使用 → 强化两端（激活模型 B 增量）
      relay.store.reinforce(from.id)
      relay.activation?.update(from.id, relay.store.get(from.id)?.reinforces)
      relay.store.reinforce(to.id)
      relay.activation?.update(to.id, relay.store.get(to.id)?.reinforces)
      // 同步 store 的因果数组（graph 从 store rebuild，保持一致）
      if (kind === 'causes') {
        if (!from.effects.includes(to.id)) relay.store.update(from.id, { effects: [...from.effects, to.id] })
        if (!to.causes.includes(from.id)) relay.store.update(to.id, { causes: [...to.causes, from.id] })
      } else {
        relay.graph.addEdge(from.id, to.id, 'references', 1)
      }
      if (args.bidirectional === true) {
        if (!to.links.includes(from.title)) {
          relay.store.update(to.id, { links: [...to.links, from.title] })
        }
        if (!from.links.includes(to.title)) {
          relay.store.update(from.id, { links: [...from.links, to.title] })
        }
      }
      return `已连接：[[${from.title}]] --${kind}--> [[${to.title}]]${args.bidirectional === true ? '（双向链接）' : ''}`
    },
  })))

  // ---- engram_update：修正节点 ----
  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_update',
    description: '修正一个记忆节点（摘要/正文/链接/因果/重要度/标题）。用于记忆过时或写错后订正。',
    parameters: {
      ref: {
        type: 'string',
        required: true,
        description: '要修正的节点（id 或 [[标题]]）',
      },
      title: {
        type: 'string',
        description: '可选：新标题',
      },
      summary: {
        type: 'string',
        description: '可选：新摘要',
      },
      content: {
        type: 'string',
        description: '可选：新正文',
      },
      links: {
        type: 'array',
        items: { type: 'string' },
        description: '可选：替换整个关联列表',
      },
      importance: {
        type: 'number',
        description: '可选：重要度 0-1（唤醒排序权重）',
      },
    },
    output: TEXT_OUTPUT,
    isConcurrencySafe: () => true,
    execute: async (args, exec) => {
      const node = resolveNode(relay, String(args.ref))
      if (!node) return `未找到节点 [[${args.ref}]]`
      if (!isVisible(node, viewerOf(exec))) return `无权修正 [[${node.title}]]（${node.layer} 层对当前会话不可见）`
      const patch: Parameters<typeof relay.store.update>[1] = {}
      if (args.title !== undefined) patch.title = String(args.title)
      if (args.summary !== undefined) patch.summary = String(args.summary)
      if (args.content !== undefined) patch.content = String(args.content)
      if (args.links !== undefined) patch.links = (args.links as string[]).map(String)
      if (args.importance !== undefined) patch.importance = Number(args.importance)
      const updated = relay.store.update(node.id, patch)
      return updated ? `已修正 [[${updated.title}]]` : '修正失败（节点不存在）'
    },
  })))

  // ---- engram_remove：删除节点 ----
  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_remove',
    description: '删除一个记忆节点（**谨慎：删除不可恢复**）。用于记忆确已无用/错误。删除后其因果边与链接不再可追踪。',
    parameters: {
      ref: {
        type: 'string',
        required: true,
        description: '要删除的节点（id 或 [[标题]]）',
      },
    },
    output: TEXT_OUTPUT,
    isConcurrencySafe: () => true,
    execute: async (args, exec) => {
      const node = resolveNode(relay, String(args.ref))
      if (!node) return `未找到节点 [[${args.ref}]]`
      if (!isVisible(node, viewerOf(exec))) return `无权删除 [[${node.title}]]（${node.layer} 层对当前会话不可见）`
      const ok = relay.store.remove(node.id)
      return ok ? `已删除记忆节点 [[${node.title}]]（${node.layer}·${node.kind}）` : '删除失败'
    },
  })))

  // ---- engram_promote：提升层（session→project/global） ----
  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_promote',
    description: '提升记忆层：session→project/global（会话结束前把临时记忆转长期跨会话持久）或 project→global。层只能升不能降。',
    parameters: {
      ref: {
        type: 'string',
        required: true,
        description: '要提升的节点（id 或 [[标题]]）',
      },
      layer: {
        type: 'string',
        required: true,
        description: '提升到哪一层（project/global）',
      },
    },
    output: TEXT_OUTPUT,
    isConcurrencySafe: () => true,
    execute: async (args, exec) => {
      const viewer = viewerOf(exec)
      const node = resolveNode(relay, String(args.ref))
      if (!node) return `未找到节点 [[${args.ref}]]`
      if (!isVisible(node, viewer)) return `无权提升 [[${node.title}]]（${node.layer} 层对当前会话不可见）`
      const target = String(args.layer) as EngramLayer
      if (!ENGRAM_LAYERS.includes(target) || target === 'session') {
        return '错误：目标层必须是 project 或 global（不能降级回 session）'
      }
      // 只能升不能降：global 已是最高
      if (node.layer === 'global') return `[[${node.title}]] 已是 global 层（最高），无需提升`
      if (node.layer === target) return `[[${node.title}]] 已在 ${target} 层`
      if (target === 'project' && !viewer.cwd) {
        return `错误：提升到 project 层需要当前工作目录（无 cwd——只能提升到 global）`
      }
      const promoted = relay.store.promote(node.id, target, viewer.cwd ?? null)
      return promoted
        ? `已提升 [[${promoted.title}]]：${node.layer} → ${target}${promoted.projectId ? `（项目 ${promoted.projectId}）` : ''}，跨会话持久`
        : '提升失败'
    },
  })))

  // ---- engram_status：服务状态 ----
  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_status',
    description: '查看 engram 记忆服务状态：分层统计（global/project/session 条数）、哈希槽位、因果图边数、模型状态、注入预算。',
    parameters: {
      verbose: {
        type: 'boolean',
        description: '可选：是否输出详细信息（默认 false）',
      },
    },
    output: TEXT_OUTPUT,
    isConcurrencySafe: () => true,
    execute: async (args: { verbose?: boolean } = {}) => {
      const s = await relay.status()
      return JSON.stringify(s, null, 2)
    },
  })))

  // ---- engram_verify：灵枢白箱验证（外置大脑 · 验证闸门） ----
  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_verify',
    description: '灵枢（Lingshu）白箱验证一个知识主张：图谱锚定判定——✓已锚定（可溯源）/ ?图谱外（证据不足不裁决，诚实边界，附最近候选卡与 D_norm）。回答前对拿不准的陈述调用；**验证不通过会自动补卡（异步 ~15s 生效，稍后即有）**。避免把错误知识当事实。',
    parameters: {
      claim: {
        type: 'string',
        required: true,
        description: '要验证的知识主张（如：量子纠缠不能用来超光速通信）',
      },
    },
    output: TEXT_OUTPUT,
    isConcurrencySafe: () => true,
    execute: async (args) => {
      const claim = String(args.claim).trim()
      if (!claim) return '（空主张）'
      const m = await relay.verifyClaim(claim)
      if (!m) return '（灵枢融合未启用：lingshuVerifyUrl 未配置）'
      if (m.status === 'error') return `（灵枢验证不可用：${m.note ?? ''}）`
      if (m.status === 'anchored') return `✓已锚定（${m.note ?? '图谱确认'}）：该主张与灵枢知识图谱一致，可溯源`
      if (m.status === 'partial') return `~部分锚定：${m.note ?? '部分主张命中图谱'}`
      return `?图谱外：${m.note ?? '灵枢无法确认该主张（不裁决，诚实边界）'}`
    },
  })))

  // ---- engram_respond：灵枢知识出招（外置大脑 · 知识之书） ----
  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_respond',
    description: '向灵枢（Lingshu）知识之书出招查询：条件/问题 → 命中的学科卡（名称 + 出招动作 + 层级）。用于知识问答、学科归属、跨学科分析；**无命中会自动补卡（当场学习）**。',
    parameters: {
      condition: {
        type: 'string',
        required: true,
        description: '条件/问题（如：铁门放外面久了为什么生锈）',
      },
      limit: {
        type: 'number',
        description: '最多返回条数（默认 3）',
      },
    },
    output: TEXT_OUTPUT,
    isConcurrencySafe: () => true,
    execute: async (args) => {
      const condition = String(args.condition).trim()
      if (!condition) return '（空条件）'
      const r = (await relay.lingshuRespond(condition, Number(args.limit ?? 3))) as { results?: Array<{ name?: string; score?: number; action?: string; level?: number; status?: string }>; error?: string }
      if (r.error) return `（灵枢出招不可用：${r.error}）`
      const results = r.results ?? []
      if (results.length === 0) return '（图谱无命中——灵枢诚实边界：不知道就说不知道）'
      return results.map((h, i) => `${i + 1}. ${h.name ?? '?'}（score=${(h.score ?? 0).toFixed(2)}, L${h.level ?? '?'}, ${h.status ?? '?'}）：${h.action ?? ''}`.slice(0, 300)).join('\n')
    },
  })))

  return () => disposers.forEach((d) => d())
}
