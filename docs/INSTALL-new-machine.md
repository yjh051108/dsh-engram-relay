# 换机安装指南（新电脑从云端部署）

> 断电寄修/换机场景：全部代码与模型都在云端，新电脑按此文档装。

## 0. 前置

- Node.js 22+（`node -v`）
- Git（配好 SSH 或 HTTPS）
- 网络：GitHub 可达（代理/直连均可）

## 1. 拉取仓库（自研插件，全部私有）

```powershell
mkdir F:\dsh\01-memory; mkdir F:\dsh\02-web-ui; mkdir F:\dsh\03-dev-infra
cd F:\dsh\01-memory
git clone https://github.com/yjh051108/dsh-engram-relay.git
cd F:\dsh\03-dev-infra
git clone https://github.com/yjh051108/dsh-super-injector.git
cd F:\dsh\02-web-ui
git clone https://github.com/yjh051108/dsh-browser-panel.git
git clone https://github.com/yjh051108/dsh-code-map.git
```

## 2. 依赖 + 构建（每个仓库）

```powershell
cd F:\dsh\01-memory\dsh-engram-relay
npm install --legacy-peer-deps   # 依赖精确锁定（package-lock.json 已入库）
npm run build                    # 一次产出全量产物：host（lib/index.js）+ client bundle（lib/client.js）
# build 自动定位 DSH：DSH_CHECKOUT（源码 checkout）或 PATH 上的 dsh（npm 全局安装）
cd F:\dsh\03-dev-infra\dsh-super-injector
npm install --legacy-peer-deps
cd F:\dsh\02-web-ui\dsh-browser-panel
npm install
```

> ⚠️ **必须 `npm run build` 而不是只跑 tsc**：`lib/client.js`（client 占位 entry，boot 注入依赖）
> bundle）由 tsdown 构建，`build.sh` 已内置该步骤。若只编译 host 就装配，
> client-modules 在 `dsh.client` 声明下找不到 `lib/client.js`，会导致整个
> `window.__DSH_BOOT__` 注入失败——WebUI 所有插件面（含官方）首装即丢。

## 3. 安装 DSH + 装配插件

```powershell
npm install -g @deepseek-ai/dsh    # 官方渠道
dsh --profile web                  # 启动一次生成 profile
```

装配（两条路，任选）：

**A. 官方 `dsh plugin add`**（推荐）：
```powershell
cd F:\dsh\01-memory\dsh-engram-relay
dsh plugin --profile web add .        # pnpm 会跑 prepare（= npm run build）自动构建
cd F:\dsh\03-dev-infra\dsh-super-injector
dsh plugin --profile web add .
# browser-panel / code-map 同理
```

从 GitHub 仓库直接装（云仓库）：
```powershell
dsh plugin --profile web add "git+https://github.com/yjh051108/dsh-engram-relay.git"
# git 依赖的 prepare 脚本被 pnpm 默认拦截：按 pnpm 打印的提示，把 allowBuilds
# 键写进 C:\Users\<你>\.dsh\profiles\web\pnpm-workspace.yaml 后重跑即可
```

**B. 注入器热装配**（super-injector 先装好再用）：
```powershell
# 启动 web 后，让 Agent 调用：
# dev_install_package(dir=F:\dsh\01-memory\dsh-engram-relay)
# dev_install_package(dir=F:\dsh\03-dev-infra\dsh-super-injector)
# 注：dev_inject_plugin / dev_install_package 会预检 lib/client.js，
# 缺失时报错并提示先 npm run build（防 WebUI 注入整体丢失）
```

## 4. 模型（已随仓库分发，零下载）

- `dsh-engram-relay/model/bge-small-zh/`：int8 ONNX（23MB）已入库；
- embedModel **留空 = 纯算法语义匹配（SemanticScorer 主路径，零模型）**——默认；
- 需 bge 对比验证时显式配置 `embedModel` 指向 `model/bge-small-zh`；
- fp32 高精度版（96MB）：本地旧机 `engram-trial/bge-small-zh-onnx/model.onnx`，或重新导出（`python/tests/` 导出脚本）。

## 5. 旧记忆迁移（可选）

- 旧电脑 `~/.dsh/engram-relay/` 下的 `engrams.jsonl`（+ `.bak-*` 快照）复制到新电脑同路径即可；
- 旧记忆 reinforces 字段缺省自动补 `[createdAt]`（类脑激活模型兜底）。

## 6. 验证

```powershell
# 启动 web 后让 Agent 执行：
engram_status    # engramCount > 0、semanticEngine=纯算法 SemanticScorer（默认）
engram_recall "缓存命中率"   # 应召回相关记忆（bge 工作）
dev_plugin_status  # super-injector active
```

## 7. 分支说明

- `master`：重型路线（向量融合 + 类脑激活——正在实施）；
- `lite`：轻量版（哈希 + bge，零重型依赖）——低配机器 `git clone -b lite` 后用 lite。
