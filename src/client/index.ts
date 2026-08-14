/**
 * dsh-engram-relay — client entry：注册会话页「图谱」Tab。
 *
 * 记忆图谱可视化：host 端 /engram-relay/api/graph 提供分层准入后的
 * 节点+边数据，本 Tab 用确定性力导向布局渲染 SVG（节点=记忆·颜色分层，
 * 实线=因果边，虚线=双向链接），点击节点展开详情（渐进披露第二层）。
 *
 * 装配：探测 graph API 存在（host 加载成功）才注册 Tab；label 绑定词典。
 */
import { GraphView } from './GraphView.tsx'
import { en, zh } from './locales.ts'

/** 插件显示名（诊断用）。 */
export const name = 'dsh-engram-relay-client'

/** 依赖服务：客户端 slots 注册表 + locale 词典服务。 */
export const inject = ['slots', 'locale']

/** 客户端文案命名空间。 */
export const LOCALE_NS = 'engram.relay'

/**
 * ClientContext 本地类型镜像（staging source checkout 未构建 lib/types，
 * 不直接 import @deepseek-ai/dsh-client-runtime——接口与 cordis 客户端
 * 上下文一致，只取本项目用到的面）。
 */
type ClientContext = {
  effect(fn: () => unknown, name?: string): unknown
  inject(deps: string[], callback: (ctx: Record<string, unknown>) => unknown): unknown
}

/**
 * SlotsService 本地类型镜像（staging source checkout 的 ui-slots 未构建
 * lib/types，不直接 import——接口与官方 slots 注册语义一致，只取本项目
 * 用到的面）。
 */
type SlotsService = {
  inject(slot: string, register: () => unknown): () => void
  register(
    meta: { name: string; id: string; order?: number; label?: () => string; locale?: string },
    render: (props: Record<string, unknown>) => unknown,
  ): unknown
}

type LocaleService = {
  register(ns: string, dict: { zh: Record<string, string>; en: Record<string, string> }): unknown
  bind(ns: string): (key: string, params?: Record<string, unknown>) => string
}

export function apply(ctx: ClientContext & { slots: SlotsService; locale: LocaleService }): void {
  // 先注册词典。
  ctx.effect(() => ctx.locale.register(LOCALE_NS, { zh, en }), 'dsh-engram-relay: dictionaries')

  const t = ctx.locale.bind(LOCALE_NS)

  let cancelled = false
  let disposeTab: (() => void) | undefined
  let probeTimer: ReturnType<typeof setTimeout> | undefined

  // 探测 graph API：host 加载成功才注册「图谱」Tab（与 memory-evolve 的
  // 各 Tab 同模式——API 404 即插件未装配，Tab 保持隐藏）。
  // ⚠️ 启动竞态修复（2026-08-14 用户反馈"装了插件但图谱不显示"）：client
  // bundle 执行可能早于 host 的 webServer 路由注册完成（新装/冷启动），
  // 一次探测失败就永久隐藏 Tab。改为**指数退避重试**：1s/2s/4s/8s/16s/
  // 30s/60s（约 2 分钟窗口），期间 host 就绪即注册；窗口耗尽才保持隐藏。
  const RETRY_DELAYS_MS = [1000, 2000, 4000, 8000, 16000, 30000, 60000]
  const registerTab = (): void => {
    disposeTab = ctx.slots.inject('conversation.view', () =>
      ctx.slots.register({
        name: 'conversation.view',
        id: 'engram-graph',
        order: 24,
        label: () => t('graphTab.label'),
        locale: LOCALE_NS,
      }, (props) => {
        // strict-session slot：props 自带 sessionId（host 端据此解析 cwd
        // 做分层准入）。
        const sessionId = (props as { sessionId?: string }).sessionId
        return GraphView({ t, sessionId })
      }))
  }
  const probe = (attempt: number): void => {
    void fetch('/engram-relay/api/graph')
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then(() => {
        if (cancelled) return
        registerTab()
      })
      .catch(() => {
        if (cancelled) return
        const delay = RETRY_DELAYS_MS[attempt]
        if (delay === undefined) return // 窗口耗尽：保持隐藏（host 端不可用）
        probeTimer = setTimeout(() => probe(attempt + 1), delay)
      })
  }
  probe(0)

  ctx.effect(() => () => {
    cancelled = true
    if (probeTimer !== undefined) clearTimeout(probeTimer)
    disposeTab?.()
  }, 'dsh-engram-relay: graph tab')
}
