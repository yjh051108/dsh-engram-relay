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
# 构建（脚本会自行定位 DSH checkout 的 tsc）：
# 若无 DSH_CHECKOUT，先装好 DSH 全局包再 build
cd F:\dsh\03-dev-infra\dsh-super-injector
npm install --legacy-peer-deps
cd F:\dsh\02-web-ui\dsh-browser-panel
npm install
```

## 3. 安装 DSH + 装配插件

```powershell
npm install -g @deepseek-ai/dsh    # 官方渠道
dsh --profile web                  # 启动一次生成 profile
```

装配（两条路，任选）：

**A. 官方 `dsh plugin add`**（推荐）：
```powershell
cd F:\dsh\01-memory\dsh-engram-relay
dsh plugin --profile web add .
cd F:\dsh\03-dev-infra\dsh-super-injector
dsh plugin --profile web add .
# browser-panel / code-map 同理
```

**B. 注入器热装配**（super-injector 先装好再用）：
```powershell
# 启动 web 后，让 Agent 调用：
# dev_install_package(dir=F:\dsh\01-memory\dsh-engram-relay)
# dev_install_package(dir=F:\dsh\03-dev-infra\dsh-super-injector)
```

## 4. 模型（已随仓库分发，零下载）

- `dsh-engram-relay/model/bge-small-zh/`：int8 ONNX（23MB）已入库——**开箱即用**；
- embedModel 配置**留空**即自动使用包内模型（`resolveEmbedModel` 已实现）；
- fp32 高精度版（96MB）：本地旧机 `engram-trial/bge-small-zh-onnx/model.onnx`，或重新导出（`python/tests/` 导出脚本）。

## 5. 旧记忆迁移（可选）

- 旧电脑 `~/.dsh/engram-relay/` 下的 `engrams.jsonl`（+ `.bak-*` 快照）复制到新电脑同路径即可；
- 旧记忆 reinforces 字段缺省自动补 `[createdAt]`（类脑激活模型兜底）。

## 6. 验证

```powershell
# 启动 web 后让 Agent 执行：
engram_status    # engramCount > 0、embedModel 指向包内 model
engram_recall "缓存命中率"   # 应召回相关记忆（bge 工作）
dev_plugin_status  # super-injector active
```

## 7. 分支说明

- `master`：重型路线（向量融合 + 类脑激活——正在实施）；
- `lite`：轻量版（哈希 + bge，零重型依赖）——低配机器 `git clone -b lite` 后用 lite。
