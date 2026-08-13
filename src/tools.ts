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
import { ENGRAM_LAYERS, isVisible, isSuperseded, type EngramKind, type EngramLayer, type EngramNode, type EngramStore } from './engram/store.js'
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
 * 织网推荐：纯算法语义（词汇 × 时序）加权，返回 top-3 关联候选（供 AI 决策）。
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
    .map((x, i) => {
      const proj = x.e.projectId ? `·${String(x.e.projectId).split(/[\\/]/).pop()}` : ''
      return `${i + 1}. [[${x.e.title}]][${x.e.layer}${proj}]（语义 ${x.cosine.toFixed(2)} × 时序 ${x.score.toFixed(2)}）${x.e.summary.slice(0, 40)}`
    })
    .join('\n')
}

/**
 * 写前查重（P2 写入管线）：哈希候选（**跨项目**——同一真理不分项目，
 * 语义高置信同主题即修订，v0.4）→ 语义打分 → 最高分 ≥0.6 视为同主题
 * （返回旧节点，供自动修订）；打分器不可用时退化为标题精确匹配。高置信
 * 阈值防误判；版本链 superseded 可追溯 = 自动修订的天然回滚安全网。
 * 返回 { dup, detailed }——detailed（通道分解）复用给自动织网。
 */
async function findDuplicate(relay: EngramRelay, text: string, _layer: EngramLayer): Promise<{ dup: EngramNode | null; detailed: Map<string, { score: number; lexical: number; graph: number; cooc: number }> | null }> {
  const cands = relay.store.lookup(text, 64)
  if (cands.length === 0) return { dup: null, detailed: null }
  const detailed = relay.model.semanticScores(text.slice(0, 300), cands)
  if (detailed && detailed.size > 0) {
    let best: EngramNode | null = null
    let bestScore = 0
    for (const e of cands) {
      const s = detailed.get(e.id)?.score ?? 0
      if (s > bestScore) { bestScore = s; best = e }
    }
    if (best && bestScore >= 0.6) return { dup: best, detailed }
    return { dup: null, detailed }
  }
  // 打分不可用：精确标题匹配（保守，防误修订）
  const exact = relay.store.byTitle(text.split('：')[0] ?? text)
  return { dup: exact ?? null, detailed: null }
}

/**
 * 自动织网（v0.4 核心原则：**怎么索引就怎么推荐，可逆才可解释**）：
 * 写入即织网——**lexical 通道 ≥ 0.5 判同主题**（词汇重叠是织网的可靠
 * 信号，比融合分阈值稳——融合分含图/PCA 分量会稀释词汇信号），排序
 * 用融合分（三维度加权，与检索同款）。因果强关系仍由 AI/蒸馏提供
 * （causes/effects），系统自动织的是弱关系（link）——写入织的边 =
 * 未来检索召回的理由（可逆）。
 */
async function weaveLinks(relay: EngramRelay, node: EngramNode, detailed: Map<string, { score: number; lexical: number; graph: number; cooc: number }>): Promise<number> {
  const ranked = [...detailed.entries()]
    .filter(([id, s]) => {
      const target = relay.store.get(id)
      // ⚠️ 排除自身与废止节点（修订路径中旧版已被 supersede——给它织链接无意义）
      return id !== node.id && target !== undefined && !isSuperseded(target) && s.lexical >= 0.5
    })
    .sort((a, b) => b[1].score - a[1].score)
  let woven = 0
  const newLinks: string[] = [...(node.links ?? [])]
  for (const [id] of ranked.slice(0, 2)) {
    const target = relay.store.get(id)
    if (!target || target.id === node.id) continue
    if (target.links.includes(node.title)) continue
    relay.store.update(target.id, { links: [...target.links, node.title] })
    if (!newLinks.includes(target.title)) newLinks.push(target.title)
    woven++
  }
  if (woven > 0) relay.store.update(node.id, { links: newLinks })
  return woven
}

/** 入口行渲染（[[标题]][层] 摘要 + **因果/链接邻接**——渐进披露入口层
 *  就给导航信息：↑因/↓果（因果链）→ 关联（双向链接），模型看到即可
 *  决定顺着哪条边探究（engram_open 展开正文），不用盲目逐个 open。 */
function entryLine(e: EngramNode, store: EngramStore): string {
  const pendingMark = e.status === 'pending' ? ' ⏳' : ''
  // 状态标注（新 agent 判断可信度：semantic=被反复巩固的知识，episodic=新写事件）
  const stateMark = e.state === 'semantic' ? '[语义]' : ''
  // 废止标注（防御性：wake 已过滤，但传播路径可能带入）
  const supersededMark = isSuperseded(e) ? '（已废止）' : ''
  return `- [[${e.title}]][${e.layer}]${stateMark}${pendingMark}${supersededMark} ${e.summary}${neighborsOf(e, store)}`
}

/** 邻接摘要：↑因/↓果/关联（id 解析标题 + 按重要度排序 + 截断——入口
 *  层只给导航线索，别让因果链淹没摘要；展开留给 engram_open）。
 *  ⚠️ 废止标注紧跟各自链接（第六轮新 agent 实测：行尾统一标注无法判断
 *  属于哪个链接）。 */
function neighborsOf(e: EngramNode, store: EngramStore): string {
  const parts: string[] = []
  const nodeOf = (id: string): EngramNode | undefined => store.get(id)
  const titlesOf = (ids: string[], max: number): string[] => {
    const withImp = ids
      .map((id) => store.get(id))
      .filter((n): n is EngramNode => !!n)
      .sort((a, b) => b.importance - a.importance)
      .map((n) => n.title)
    return withImp.slice(0, max)
  }
  // 渲染链接时对废止目标紧跟标注（不是行尾）
  const render = (titles: string[]): string => titles
    .map((t) => {
      const target = store.byTitle(t)
      return `[[${t}]]${target && isSuperseded(target) ? '（已废止）' : ''}`
    })
    .join(' ')
  const causes = titlesOf(e.causes ?? [], 4)
  const effects = titlesOf(e.effects ?? [], 4)
  const links = (e.links ?? []).slice(0, 3)
    .map((l) => String(l).replace(/^\[\[|\]\]$/g, '').trim())
  if (causes.length > 0) parts.push(`↑因:${render(causes)}${(e.causes?.length ?? 0) > 4 ? '…' : ''}`)
  if (effects.length > 0) parts.push(`↓果:${render(effects)}${(e.effects?.length ?? 0) > 4 ? '…' : ''}`)
  if (links.length > 0) parts.push(`→:${render(links)}`)
  return parts.length > 0 ? `\n    ${parts.join(' | ')}` : ''
}

export function installEngramTools(ctx: ToolsContext, relay: EngramRelay): () => void {
  const disposers: Array<() => void> = []

  // ---- engram_recall：按需唤醒检索（跨会话分层准入） ----
  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_recall',
    description: '主动唤醒记忆图谱入口（跨会话全可见，项目即标签）。按当前查询匹配入口节点（[[标题]] + 层 + 摘要 + **因果/链接邻接**：↑因=前因/↓果=后果/→=双向链接）。**入口图例**：[[标题]][层]（[语义]=已固化知识）摘要。看到 [[标题]] 后由你判断：需要详情就 engram_open 展开，顺着邻接可继续探究（递归导航），不需要就直接用摘要作答。',
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
      const query = String(args.query)
      const hit = await relay.recall(query, Number(args.limit ?? 3), viewerOf(exec), String(args.layer ?? ''))
      if (hit.engrams.length === 0) {
        // 新 agent 实测：无命中时需区分"没有这条记忆" vs "检索没召回到"
        const hint = hit.reason === 'no-hash-hit' || hit.reason === 'short-query'
          ? '（图谱中没有与查询重叠的记忆——可能确实没记过，或用 engram_store 写入后重试）'
          : `（未召回相关记忆，reason=${hit.reason}）`
        return hint
      }
      // 弱命中检测（第四轮新 agent 实测）：无相关记忆时 wake 仍可能返回
      // 弱相关（词汇低重叠）条目——用语义打分判定 top 条目的置信度，
      // 低置信则明示"未找到强相关记忆"，避免把噪声当命中。
      let weakHint = ''
      try {
        const detailed = relay.model.semanticScores(query.slice(0, 300), hit.engrams)
        const top = [...detailed.entries()].sort((a, b) => b[1].score - a[1].score)[0]
        if (top && top[1].score < 0.3) {
          weakHint = `\n（⚠️ 以上条目相关性较弱（最高 ${top[1].score.toFixed(2)}）——图谱中可能没有与查询强相关的记忆，可换关键词重试或 engram_store 写入）`
        }
      } catch { /* 弱命中检测失败不阻塞 */ }
      return hit.engrams.map((e) => entryLine(e, relay.store)).join('\n') + weakHint
    },
  })))

  // ---- engram_store：写入记忆（AI 自主决策分层 + 因果前因/后果） ----
  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_store',
    description: '写入一个记忆节点（跨会话分层，**AI 自主决策层归属**）。大一统记忆图谱：title 入口锚点、summary 一句话摘要（入口层）、content 完整正文（展开层）、links 双向关联 [[标题]]、causes 因果前因、effects 因果后果。**撰写规范**：① title ≤12 字、具体可辨认（如「路由残留自愈方案」，忌泛化如「更新」「总结」）；② summary ≤30 字、**不看正文也能判断相关性**（含关键实体/结论）；③ content ≤200 字、只写增量（关键参数/结论/上下文，不重复摘要）；④ **因果必织（关键）**：写前先想『什么导致了这条记忆』（causes 前因）与『这条记忆会导致什么』（effects 后果）——**已知的因果链必须写入**（causes/effects 填 [[标题]] 或 id，标题自动解析）；因果边是唤醒因果传播（补盲召回）的路径，只写 links 会漏掉因果维度。确实没有已知因果时留空，系统会推荐关联候选供你采纳（engram_link 建边）/展开确认（engram_open）/跳过——选择权始终在你；⑤ 同主题多处小更新优先 engram_update 修订原节点而非新增（**系统也会自动查重**：语义高置信同主题写入时自动修订——新增当前版 + 旧版废止可追溯，检索只命中当前版）；⑥ **价值门槛（判别——先问：这条值得写吗？）**：值得写——可复用知识/决策/踩坑/约定/规律（换个时间还有用）；**不值得写——寒暄/过程流水/一次性琐事/对话里已说清且不会再用的细节**（事件洪水会淹没可复用知识）。拿不准就写（宁多勿漏，系统有查重/归档兜底），但过程性流水账不要写；**layer 决策（v0.4 项目即标签，融会贯通）**：默认 **project**（归属当前工作目录项目）；**通用知识写 global**（开发者喜好/全局要求——换个项目还有用的）；**跨项目关联用 causes/links 织桥**（如"本项目的 X 方案引用 Y 项目的做法"→ links 填 [[Y 项目记忆标题]]，两项目自动关联）。**tags 自由分类（三类命名空间）**：`全局`（开发者喜好/全局要求）、`项目:xxx`（项目自由命名）、`教训:xxx`（教训自由子分类：代码/思想/流程…）——一节点可多标签（如 `["项目:dsh", "教训:代码"]`），自由创建不设枚举。',
    parameters: {
      layer: {
        type: 'string',
        required: true,
        description: `记忆分层（AI 自主决策）：${ENGRAM_LAYERS.join('/')}（见 description 决策准则）`,
      },
      tags: {
        type: 'array',
        items: { type: 'string' },
        description: '可选：**自由多标签**（一节点多标签，自由分类，命名空间约定）——「全局」（开发者喜好/全局要求）、「项目:xxx」（项目自由命名，如 项目:engram）、「教训:xxx」（教训自由子分类，如 教训:代码 / 教训:思想）。缺省自动生成（project→项目:<当前目录名>，global→全局）。',
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
      // v0.4 自由多标签（命名空间约定：全局/项目:xxx/教训:xxx）
      const tags = Array.isArray(args.tags) ? args.tags.map(String).filter((t) => t.trim() !== '') : []
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
      // 分层归属校验：project 层需要当前工作目录
      if (layer === 'project' && !viewer.cwd) {
        return `错误：project 层需要当前工作目录（无 cwd 的会话不能写项目记忆——建议改用 global）`
      }
      // P2 写入管线统一：写前查重（语义高置信 ≥0.6 同主题）→ 自动修订
      // （新增当前版 + 旧版 superseded + 因果/链接迁移），不重复堆节点。
      // detailed 复用于自动织网（不重复打分）。
      const { dup: dedupe, detailed } = await findDuplicate(relay, `${title}：${summary}`, layer)
      const e = relay.store.add({
        kind: kind as EngramKind,
        layer,
        projectId: layer === 'project' ? viewer.cwd! : null,
        title,
        summary,
        content,
        links,
        tags,
        sessionId: viewer.sessionId ?? relay.currentSessionId,
        turn: relay.lastTurnAt,
        causes,
        effects,
        importance: 1,
      })
      // 修订路径：旧版因果/链接继承 + 邻居指针迁移 + 版本链（旧版废止可追溯）
      let revisedOld: string | null = null
      let conflictNote = ''
      if (dedupe && dedupe.id !== e.id) {
        revisedOld = dedupe.title
        const old = dedupe
        const mergedCauses = [...new Set([...(e.causes ?? []), ...(old.causes ?? [])])]
        const mergedEffects = [...new Set([...(e.effects ?? []), ...(old.effects ?? [])])]
        // ⚠️ 反向矛盾边（第七轮新 agent 实测）：继承边与本次显式 causes
        // 指向同一节点时，同一节点会同时出现在 causes 和 effects（双向
        // 矛盾）——causes 优先，从 effects 移除重叠，回执提示
        const overlap = mergedCauses.filter((id) => mergedEffects.includes(id))
        if (overlap.length > 0) {
          conflictNote = `（合并时检测到 ${overlap.length} 条反向矛盾边，已按 causes 优先保留）`
        }
        const cleanEffects = mergedEffects.filter((id) => !mergedCauses.includes(id))
        relay.store.update(e.id, {
          causes: mergedCauses,
          effects: cleanEffects,
          links: [...new Set([...(e.links ?? []), ...(old.links ?? [])])],
        })
        for (const c of old.causes ?? []) {
          const n = relay.store.get(c)
          if (n) relay.store.update(c, { effects: [...n.effects.filter((x) => x !== old.id), e.id] })
        }
        for (const ef of old.effects ?? []) {
          const n = relay.store.get(ef)
          if (n) relay.store.update(ef, { causes: [...n.causes.filter((x) => x !== old.id), e.id] })
        }
        relay.graph.rebuild()
        relay.store.supersede(old.id, e.id)
      }
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
      // ⚠️ 用 update 持久化（add 会重新生成 id → 制造重复节点——隐藏 bug）
      for (const t of links) {
        const target = relay.store.byTitle(t)
        if (target && target.id !== e.id && !target.links.includes(title)) {
          relay.store.update(target.id, { links: [...target.links, title] })
        }
      }
      // 自动织网（v0.4：边是一开始就该有的——写入即织网，不依赖 AI 自觉）：
      // 用检索同款三维度加权（τ_sem·cos + τ_time·z(激活) + τ_cause·因果可达）
      // 选高置信邻居建双向链接——怎么索引就怎么推荐，可逆可解释。
      let woven = 0
      if (detailed && detailed.size > 0) {
        woven = await weaveLinks(relay, e, detailed)
      }
      const linkNote = woven > 0 ? `，自动织网 ${woven} 条（三维度加权）` : ''
      if (revisedOld !== null) {
        return `已修订记忆 [[${e.title}]]（${layer}·${kind}）：同主题旧版 [[${revisedOld}]] 已废止（版本链可追溯），因果/链接已继承——检索只命中当前版${conflictNote}${linkNote}`
      }
      // 织网推荐（因果边仍由 AI 决策：自动织网只做弱关系 link，因果是强语义）
      // ⚠️ 条件只查 causes/effects：links 已带但漏因果（常见偷懒）也必须推荐。
      if (causes.length === 0 && effects.length === 0) {
        const rec = await recommendLinks(relay, `${title}：${summary}`, e.id)
        const base = `已写入记忆节点 [[${e.title}]]（${layer}·${kind}，已进入检索索引，链接 ${links.length}${linkNote ? `+${woven}` : ''} 条，因果 ↑${causes.length} ↓${effects.length}）${linkNote}`
        if (rec) {
          return `${base}\n\n📎 推荐因果候选（纯算法语义 × 时序加权，未自动建边——可作 causes/effects）：\n${rec}\n\n处理建议：① 标题熟悉且相关 → engram_link 直接采纳（建因果边）；② 标题陌生但想确认 → 先 engram_open 展开正文再定；③ 不相关 → 跳过即可。`
        }
        return base + '\n（当前无显著因果候选）'
      }
      return `已写入记忆节点 [[${e.title}]]（${layer}·${kind}，已进入检索索引，链接 ${links.length}${linkNote ? `+${woven}` : ''} 条，因果 ↑${causes.length} ↓${effects.length}）${linkNote}`
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
      tags: {
        type: 'array',
        items: { type: 'string' },
        description: '可选：自由多标签（见 engram_store 描述：全局/项目:xxx/教训:xxx 命名空间约定）',
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
      // v0.4 自由多标签（命名空间约定：全局/项目:xxx/教训:xxx）
      const tags = Array.isArray(args.tags) ? args.tags.map(String).filter((t) => t.trim() !== '') : []
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
        return `错误：project 层需要当前工作目录（无 cwd 的会话不能写项目记忆——建议改用 global）`
      }
      const e = relay.store.add({
        kind: kind as EngramKind,
        layer,
        projectId: layer === 'project' ? viewer.cwd! : null,
        title,
        summary,
        content,
        links,
        tags: Array.isArray(args.tags) ? args.tags.map(String).filter((t) => t.trim() !== '') : [],
        sessionId: viewer.sessionId ?? relay.currentSessionId,
        turn: relay.lastTurnAt,
        causes,
        effects,
        importance: 1,
        status: 'pending',
      })
      return `已提议记忆节点 [[${e.title}]]（${layer}·${kind}·⏳待确认，已进入检索索引）——用户确认（engram_confirm）后才会参与检索/唤醒`
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
        return `错误：只能确认当前会话可见的节点（global + 本目录 project）`
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
        return `错误：只能拒绝当前会话可见的节点（global + 本目录 project）`
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
    description: '展开一个记忆节点（渐进披露第二层）：返回完整正文 + 层 + **前因/后果（因果）/关联（双向链接）/依赖引用（depends-on 与 references 边）**，空邻接显式（无）。当你看到 [[标题]] 入口需要详情时调用；展开后可顺着邻接的 [[标题]] 继续递归探究（记忆图谱导航）。',
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
      const titleRef = String(args.title)
      const node = resolveNode(relay, titleRef)
      if (!node) return `未找到节点 [[${titleRef}]]（可 engram_search 检索）`
      // 同名多版本提示（第五轮新 agent 实测：同名双节点 byTitle 取最近，
      // open 与 recall 可能解析到不同实例 → 因果展示矛盾）
      const allVersions = relay.store.byTitles(node.title)
      const dupNote = allVersions.length > 1
        ? `\n（⚠️ 标题「${node.title}」有 ${allVersions.length} 个节点——当前展开最近写入的（id ${node.id.slice(-6)}）；`
        + '用 engram_search 可盘点全部，同主题可经 engram_store 写入触发自动修订合并）'
        : ''
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
      // v0.5 第四轮新 agent 实测：depends-on/references 边织了看不见 →
      // open 邻接补「依赖/引用」展示（graph 层查询）
      const deps = relay.graph.depsOf(node.id)
      const parts: string[] = []
      const supMark = isSuperseded(node) ? '（已废止）' : ''
      parts.push(`# [[${node.title}]] (${node.kind} · ${node.layer}${node.projectId ? ` · ${node.projectId}` : ''} · ${node.state ?? 'episodic'})${supMark}`)
      parts.push(node.summary)
      if (node.content) parts.push(`\n${node.content}`)
      // ⚠️ 邻接契约：无因果/链接时显式占位（新 agent 实测：段落缺席被误判为 bug）
      if (causes.length > 0) parts.push(`\n**前因**（因果 ↑）：${causes.map((c) => `[[${c.title}]]`).join('、')}`)
      else parts.push('\n**前因**（因果 ↑）：（无）')
      if (effects.length > 0) parts.push(`**后果**（因果 ↓）：${effects.map((c) => `[[${c.title}]]`).join('、')}`)
      else parts.push('**后果**（因果 ↓）：（无）')
      if (linked.length > 0) parts.push(`**关联**（双向链接）：${linked.map((c) => `[[${c.title}]]`).join('、')}`)
      else parts.push('**关联**（双向链接）：（无）')
      if (deps.length > 0) parts.push(`**依赖/引用**：${deps.map((c) => `[[${c.title}]]`).join('、')}`)
      return parts.join('\n') + dupNote
    },
  })))

  // ---- engram_search：检索图谱（分层/项目/类型/关键词） ----
  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_search',
    description: '盘点记忆图谱（回顾/维护）：**先按关键词/层/项目/类型过滤全库，再排序截断**（关键词子串匹配标题/摘要，大小写不敏感）。返回入口列表（[[标题]] + 摘要 + 邻接；同名多版本折叠为 ×N，废止标「已废止」）。用于盘点已沉淀的记忆、找要 update/remove/promote/link 的目标。',
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
      // ⚠️ 第六轮新 agent 实测：带 query 时 global 层被静默排除——projectId
      // 默认取 cwd 会滤掉 global 节点（projectId=null）。修正：**仅显式传
      // projectId 才过滤项目**——缺省全层全项目（与描述"缺省=全部"一致）
      const projectId = args.projectId !== undefined && String(args.projectId) !== ''
        ? String(args.projectId)
        : undefined
      // ⚠️ P0 修复（第三轮新 agent 实测）：先过滤后截断——先 query 会截掉
      // 匹配目标再子串过滤 → 精确子串搜不到（上轮误判为"盘点/检索不一致"，
      // 实为 limit 顺序 bug）
      let nodes = relay.store.query({
        layer: layer && ENGRAM_LAYERS.includes(layer as EngramLayer) ? layer as EngramLayer : undefined,
        projectId: projectId !== undefined ? projectId : undefined,
        kind: kind && KINDS.includes(kind as EngramKind) ? kind as EngramKind : undefined,
        // 不过滤先取全量（过滤在下方子串/可见性后，limit 最后截断）
        limit: 0,
        recent: args.recent === true,
      })
      // 可见性 + 关键词过滤（全库）
      nodes = nodes.filter((e) => isVisible(e, viewer))
      const q = String(args.query ?? '').trim().toLowerCase()
      if (q) nodes = nodes.filter((e) => e.title.toLowerCase().includes(q) || e.summary.toLowerCase().includes(q))
      const limit = Number(args.limit ?? 20)
      // 排序（重要度优先，除非 recent）
      nodes = [...nodes].sort((a, b) => (args.recent === true ? b.createdAt - a.createdAt : b.importance - a.importance))
      nodes = nodes.slice(0, limit)
      if (nodes.length === 0) return '（无匹配记忆）'
      // v0.5 盘点去重（新 agent 实测：同名多版本重复显示观感杂乱）：
      // 同标题只显示当前版（未废止、重要度最高），重复折叠为 ×N 标注；
      // 废止旧版在盘点中显式标「已废止」（盘点=全状态视图，检索只当前版）
      const byTitle = new Map<string, EngramNode[]>()
      for (const e of nodes) {
        const arr = byTitle.get(e.title) ?? []
        arr.push(e)
        byTitle.set(e.title, arr)
      }
      const lines: string[] = []
      let shown = 0
      for (const [title, group] of byTitle) {
        if (shown >= Number(args.limit ?? 20)) break
        const active = group.filter((e) => !isSuperseded(e))
        const current = (active.length > 0 ? active : group).sort((a, b) => b.importance - a.importance)[0]
        const dupMark = group.length > 1 ? ` ×${group.length}` : ''
        const supersededMark = isSuperseded(current) ? '（已废止）' : ''
        lines.push(`${entryLine(current, relay.store)}${dupMark}${supersededMark}`)
        shown++
      }
      const total = nodes.length
      return `记忆图谱（${byTitle.size} 个唯一标题 / 共 ${total} 节点）：\n${lines.join('\n')}`
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
        return '错误：只能连接当前会话可见的节点（global + 本目录 project）'
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
    description: '修正一个记忆节点（摘要/正文/链接/因果/重要度/标题）。**就地修改，不走版本链**（旧版不保留）；如需保留历史版本，用 engram_store 写同主题（自动修订=旧版废止可追溯）。用于记忆过时或写错后订正。',
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
      if (!updated) return '修正失败（节点不存在）'
      const changed = Object.keys(patch).join('、')
      return `已修正 [[${updated.title}]]（就地修改，不走版本链）：更新了 ${changed}`
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

  // ---- engram_promote：提升层（project→global） ----
  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_promote',
    description: '提升记忆层：project→global（项目记忆升级为跨项目共享真理）。层只能升不能降。',
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
      if (!ENGRAM_LAYERS.includes(target)) {
        return '错误：目标层必须是 project 或 global'
      }
      // 只能升不能降：global 已是最高；此处 node 必为 project、target 必为 global
      if (node.layer === 'global') return `[[${node.title}]] 已是 global 层（最高），无需提升`
      if (node.layer === target) return `[[${node.title}]] 已在 ${target} 层`
      const fromLayer = node.layer // ⚠️ promote 原地修改，先存旧层（否则显示 "global → global"）
      const promoted = relay.store.promote(node.id, target, viewer.cwd ?? null)
      return promoted
        ? `已提升 [[${promoted.title}]]：${fromLayer} → ${target}${promoted.projectId ? `（项目 ${promoted.projectId}）` : ''}，跨会话持久`
        : '提升失败'
    },
  })))

  // ---- engram_status：服务状态 ----
  disposers.push(ctx.tools.register(defineTool({
    name: 'engram_status',
    description: '查看 engram 记忆服务状态：分层统计（global/project）、巩固状态（episodic/semantic/dormant）、归档数/硬上限、哈希槽位、因果图边数、语义引擎状态、注入预算。',
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
