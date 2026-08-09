# dsh-engram-relay

> 外置 engram 转接模型插件：内置 <1B 本地模型，作为 DSH 主模型与外部记忆之间的转接层。

## 是什么

`dsh-engram-relay` 为 DeepSeek Harness 主模型提供一个 **<1B 参数的内置转接模型**（transformers.js/ONNX 本地推理，无需外部服务），它负责：

1. **外置 engram 存储**：把会话中值得记住的事实、决策、事件，经小模型蒸馏为结构化记忆痕迹（engram），持久化到 harness 之外的本地存储；
2. **超长记忆**：通过 engram 转接层将主模型 100k 上下文**等效延展至少 10 倍**——历史不再全部塞进上下文，而是沉淀为外置 engram，按需唤醒；
3. **超稀疏精准主动唤醒**：每次请求前，转接层用小模型判断「现在该唤醒哪些记忆」，沿 **engram 因果图**传播激活（谁导致了谁、谁依赖谁），只注入极少数最相关的痕迹——比纯向量索引更强的因果检索；
4. **模型 API 底层转接**：经 `llm/stream` waterfall 拦截每个模型调用，在请求发出前注入唤醒的记忆，在回合结束后蒸馏新记忆，零核心改动。

## 安装

```bash
# 方式一：dshx（marisa）外部插件管理器
dshx install dsh-engram-relay https://github.com/dsh-external/dsh-engram-relay.git

# 方式二：plugin-registry
dsh registry install ./dsh-engram-relay
dsh registry enable @dsh-external/dsh-engram-relay
```

首次启用时插件自动下载 <1B 转接模型（约 300–500MB，缓存于 `~/.dsh/engram-relay/models/`），随后完全本地运行。

## 用法

安装启用后无需额外操作：engram 转接层自动工作。模型侧可用的工具：

| 工具 | 作用 |
|---|---|
| `engram_recall` | 主动查询记忆（带因果链展开） |
| `engram_store` | 显式写入一条记忆（绕过自动蒸馏） |
| `engram_status` | 查看 engram 存储统计与唤醒情况 |

## 配置（`~/.dsh/config.yaml`）

| 键 | 默认 | 说明 |
|---|---|---|
| `modelId` | `onnx-community/Qwen2.5-0.5B-Instruct` | 内置转接模型（<1B） |
| `dtype` | `q8` | ONNX 量化档 |
| `injectBudgetTokens` | `600` | 单次唤醒注入的 token 预算（超稀疏） |
| `maxWakePerTurn` | `3` | 每回合最多唤醒的 engram 条数 |
| `storeDir` | `~/.dsh/engram-relay/` | engram 持久化目录 |

## 目录

```
src/
├── index.ts          # 插件入口：llm/stream 转接 + systemPrompt 注入 + 工具注册
├── relay.ts          # 转接核心：请求前唤醒 / 回合后蒸馏
├── engram/
│   ├── store.ts      # engram 持久化（JSONL）
│   ├── causal.ts     # 因果图：节点=engram，边=导致/依赖/引用
│   └── wake.ts       # 超稀疏因果唤醒（激活传播）
├── model/
│   └── local.ts      # transformers.js/ONNX <1B 模型封装（蒸馏/打分/嵌入）
└── tools.ts          # engram_recall / engram_store / engram_status
```

## 设计要点

- **因果性 > 向量索引**：唤醒不是「语义相似度 top-k」，而是从当前上下文的小模型打分结果出发，沿因果图传播激活分数，能召回「导致当前问题的前因」和「依赖当前结论的后果」。
- **超稀疏**：唤醒注入预算默认 600 token（相对 100k+ 上下文 <1%），且每回合唤醒条数有上限，防止记忆污染主上下文。
- **零核心改动**：只使用公开 seam（`llm/stream` waterfall、`systemPrompt.context`、`tools.register`），不碰 DSH 源码。

## 收录

本仓库按 [dsh-external/hub LOOP.md](https://github.com/dsh-external/hub/blob/main/LOOP.md) 流程收录：`marisa-plugin` topic + `catalog.source.json` 登记（category: `plugin`）。
