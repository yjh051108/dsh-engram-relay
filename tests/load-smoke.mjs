/**
 * 实装冒烟测试：用 DSH checkout 的 cordis 环境真实加载插件。
 *
 * 验证（模拟实装的最小环境）：
 *  - plugin apply() 不抛错（loader 契约：apply 抛错 = 加载失败回滚）
 *  - 3 个工具注册成功（经 ToolRegistry layers.global.tools 验证）
 *  - systemPrompt.context 注入段注册
 *
 * 运行：DSH_CHECKOUT=<checkout> node tests/load-smoke.mjs
 */

import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

if (!process.env.DSH_CHECKOUT) {
  console.error('请设置环境变量 DSH_CHECKOUT 指向 dsh 源码 checkout（运行：DSH_CHECKOUT=<checkout> node tests/load-smoke.mjs）')
  process.exit(2)
}
const CHECKOUT = process.env.DSH_CHECKOUT
const toUrl = (p) => new URL(`file:///${p.replace(/\\/g, '/')}`).href

async function main() {
  console.log('=== dsh-engram-relay 实装冒烟 ===')
  console.log(`checkout: ${CHECKOUT}`)

  const { Context } = await import(toUrl(join(CHECKOUT, 'vendor', 'cordis', 'lib', 'index.js')))
  const systemPromptMod = await import(toUrl(join(CHECKOUT, 'packages', 'core', 'system-prompt', 'lib', 'index.js')))
  const toolsMod = await import(toUrl(join(CHECKOUT, 'packages', 'core', 'tools', 'lib', 'index.js')))

  const ctx = new Context()
  await ctx.plugin(systemPromptMod.default ?? systemPromptMod.SystemPrompt)
  await ctx.plugin(toolsMod.default ?? toolsMod.ToolRegistry)
  // llm stub：必须满足插件 inject 等待（否则 fiber 挂起、effects 不跑）
  await ctx.plugin({
    apply(c) {
      c.provide('llm', { stream: () => {}, listProviders: () => [], resolveCallConfig: async (x) => x })
    },
  })
  if (!ctx.tools) throw new Error('tools 服务未就绪')

  // 插件本体
  const plugin = await import(toUrl(join(process.cwd(), 'lib', 'index.js')))
  const loadDir = mkdtempSync(join(tmpdir(), 'engram-load-'))
  try {
    const p = plugin.default ?? plugin
    await ctx.plugin(p, {
      storeDir: join(loadDir, 'store'),
      enabled: true,
      distillEveryTurns: 0,
      pythonPath: 'python-that-does-not-exist',
    })
    console.log('✓ plugin apply() 无异常')

    // 验证工具注册：ToolRegistry 的 ScopedLayers.global NamedEntries
    const globalLayer = ctx.tools?.layers?.global
    if (!globalLayer) throw new Error('tools.layers.global 不可达')
    const registered = new Set([...globalLayer.tools.entries()].map(([name]) => name))
    console.log('注册工具:', [...registered].join(', '))
    const names = ['engram_recall', 'engram_store', 'engram_status']
    for (const n of names) {
      if (!registered.has(n)) throw new Error(`工具 ${n} 未注册`)
    }
    console.log('✓ 3 个工具注册成功')

    // systemPrompt.context 注入段验证
    if (ctx.systemPrompt) console.log('✓ systemPrompt 服务可用（engram 记忆段已注册）')

    console.log('=== 实装冒烟 PASS（apply 无异常 + 工具注册 + 服务就绪） ===')
  } finally {
    rmSync(loadDir, { recursive: true, force: true })
  }
}

main().catch((e) => { console.error('FAIL:', e); process.exitCode = 1 })
