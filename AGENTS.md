# AGENTS.md — dsh-engram-relay 插件开发规范

> 本项目是 **DSH 外部插件**（profile bundle）。开发、实装、验证全部遵循官方文档；
> 本文件把官方教程要点纳入项目，作为开发基准。**官方文档是权威**，本文件只做
> 索引与项目内约定，冲突时以官方为准。

## 官方文档索引（DSH checkout 内）

| 主题 | 文档 |
|---|---|
| 插件教程（7 章） | `<checkout>/docs/cordis-tutorial/01-first-plugin.md` → `07-into-the-harness.md` |
| 插件形态（函数/对象/类 + apply） | `01-first-plugin.md` |
| 生命周期与 effect | `02-lifecycle-and-effects.md` |
| 服务（inject） | `03-services.md` |
| 事件 | `04-events.md` |
| 配置（Config + schemastery） | `05-config.md` |
| 组合与 HMR | `06-composition-and-hmr.md` |
| 进入 harness（工具注册/执行） | `07-into-the-harness.md` |
| 工具开发 | `<checkout>/docs/user/develop/basic/tool.md` |
| 插件配置 | `<checkout>/docs/user/develop/basic/config.md` |
| **打包与实装（bundle/profile）** | `<checkout>/docs/user/develop/basic/publish.md` |
| 扩展点地图（feature → mechanism） | `<checkout>/docs/cookbook/extension-cookbook.md` |
| 系统架构/能力接缝 | `<checkout>/docs/architecture.md`、`docs/capability-seams.md` |
| profile/bundle 装配语义 | `<checkout>/packages/boot/app-boot/README.md` |

## 插件形态（官方教程核心）

```ts
import type { Context } from 'cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'

export const name = 'my-plugin'                    // 显示名（诊断用）
export const inject = ['tools', 'systemPrompt']    // 依赖的服务，必须显式声明
export const Config = Schema.object({ ... })        // schemastery schema + 默认值
export function apply(ctx: Context, config: Config) { ... }
```

- `apply` 抛错 = **加载失败回滚**（loud failure，不是跳过）；
- 模块无法解析 = 通过 logger 报告（不崩溃，boot 时可能丢失）——新增 entry 无效果先查拼写；
- 工具注册必须 `ctx.effect(() => ctx.tools.register(...))`（effect 生命周期，卸载自动注销）；
- `inject` 的服务未就绪时插件保持 PENDING，直到服务可用。

## 工具规范（官方 tool.md / 07 章）

```ts
ctx.tools.register(defineTool({
  name: 'my_tool',
  description: '...',
  parameters: { /* JSON Schema */ },
  output: {
    schema: { type: 'string' },
    render: (_args, value) => [{ type: 'text', text: value }],  // 必须有 render
  },
  async execute(args) { return 'result' },
}))
```

- `output` 必须声明 `{ schema, render }`（缺 render 会 TypeError）；
- `defineTool` 自动把 parameters 转 JSON Schema 给模型 + 校验参数；
- 工具注册在 `ctx.tools.layers.global.tools` 可见（无公开 list API，验证用 entries()）。

## 配置规范（官方 config.md）

- 导出 `Config` 类型 + 同名 schemastery schema，默认值写在 schema 字段上；
- 不要导出普通对象作为 Config（不实现 Standard Schema 接口）；
- 插件配置经 `cordis.yml` / patch 层的 `config:` 传入，schema 校验 + 填默认值。

## 打包与实装（官方 publish.md —— 本项目的关键流程）

### 两个概念

- **bundle**：npm 包，声明 `dsh.bundle.patch`，携带配置层（cordis.patch.yml）；
- **profile**：`$DSH_HOME/profiles/<name>/`，`dsh.profile.bundles` 有序列表 + 用户 patch 层。

### bundle 清单

```json
{
  "name": "@dsh-external/dsh-engram-relay",
  "version": "0.1.0",
  "type": "module",
  "main": "./lib/index.js",
  "files": ["lib", "cordis.patch.yml"],
  "dsh": { "bundle": { "patch": "./cordis.patch.yml" } }
}
```

### bundle 的 patch 层（**格式必须正确**）

```yaml
- insert:
    - id: dsh-engram-relay
      name: '@dsh-external/dsh-engram-relay'
      config: {}
```

⚠️ **顶层必须是 `- insert:` 包裹**（`- id:` 顶层 = id-targeted patch，会报
`entry not found`——本项目踩过的坑）。

### 实装命令（官方流程，非手改配置）

```sh
# 在插件仓库根目录执行（官方推荐；自动 link + 加 bundles + 报错指引）
dsh plugin --profile web add .

# 验证装配（应看到 "# == @dsh-external/dsh-engram-relay" 层）
dsh --profile web --dump-config

# 重启 web 生效（web 长驻进程需用户重启）
```

- `dsh plugin --profile <name> add .` = pnpm 转发 + 自动 append 到 `dsh.profile.bundles`；
- git 安装（`add github:you/repo`）只取源码不构建 → 作者需 `prepare` 脚本，用户需在
  profile 的 `pnpm-workspace.yaml` 加 `allowBuilds`（执行包代码 = 安装期信任！）；
- 无 `dsh.bundle` 声明的包仍可装，但只是普通依赖，不激活层（库的正确形态）。

### 层顺序（publish.md §loading order）

1. 每个 bundle patch（bundles 列表顺序）→ 2. profile 的 cordis.patch.yml →
3. home 级 `$DSH_HOME/cordis.patch.yml` → 4. `--patch` 覆盖 → 5. launcher flag patches。

后来层按行覆盖；patch 替换整行 config（非深合并）。

## 本项目约定

- **Node 侧零第三方运行时依赖**：只 import 闭包内包（cordis / @deepseek-ai/* /
  schemastery）+ node: 内置。模型推理全在 Python 服务（spawn `python -m engram_model.server`），
  不要往 Node 侧加 npm 依赖（实装时 checkout 闭包不包含它们）。
- **Python 侧**（`python/engram_model/`）：torch 魔改 Qwen3-0.6B（Engram 条件记忆 ×
  DSA 稀疏路由），JSON 行协议服务；依赖见 `python/requirements.txt`。
- **与官方 compact 共存**：不阻止、不替代 `dsh-compact-basic`（它负责腾 KV）；
  engram 在 `agent/turn-stopping` 实时蒸馏留底（细节保真，可唤醒找回）。
- 工具 contributes 声明（dsh.plugin.json / dshx.contributes）必须与注册一致（契约校验）。
- 测试：`npm test`（Node 单测）+ `npm run test:python` + `npm run sim:1m` / `sim:causal`；
  实装冒烟：`DSH_CHECKOUT=<checkout> node tests/load-smoke.mjs`。

## 扩展点（本项目用到的）

| 能力 | 机制 |
|---|---|
| 请求前唤醒 | `llm/stream` waterfall 旁路观察（必须返回 `next()`） |
| 记忆注入 | `ctx.systemPrompt.context()`（装配时渲染，变化检测重装配） |
| 回合后蒸馏 | `agent/turn-stopping`（serial，返回 void 不 veto；payload 带 agent → `agent.session.deriveMessages()`） |
| 工具 | `ctx.tools.register(defineTool(...))` |
| 上下文折叠 | 官方 `ctx.compact`（`dsh-compact-basic`），不替代 |
| HMR | 所有注册都是 ctx.effect → vendored HMR 自动生效 |
