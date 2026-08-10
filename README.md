# dsh-engram-relay

> 单会话上下文增强：魔改 <1B 模型（Engram 条件记忆 × DSA 稀疏路由），折叠本会话早期历史进外置记忆表、模型原生回忆，让单次会话等效承载 1M+ token；会话结束即弃，不做跨会话记忆沉淀。

## 是什么

`dsh-engram-relay` 融合 DeepSeek 开源的两项技术，做一个**模型级**的外置记忆转接层：

1. **[Engram（条件记忆，[deepseek-ai/Engram](https://github.com/deepseek-ai/Engram)）](https://arxiv.org/abs/2601.07372)**：N-gram 哈希 O(1) 确定性寻址 → 巨大静态记忆表 → gate 融合。为 Transformer 补上「知识查找原语」（MoE 扩计算容量，Engram 扩静态记忆容量）。
2. **[DSA（DeepSeek Sparse Attention，[deepseek-ai/DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)）](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v32/modular_deepseek_v32.py)**：Lightning Indexer（轻量多头打分器，`ReLU(q·k)` 门控聚合）+ Token Selector（top-K 稀疏选择）。把 O(L²) 注意力降到 O(L·k)。

**融合点**：用 DSA 的 Lightning Indexer 做 **KV 参与 engram 寻址后的路由**——当前上下文的 KV/query 表示对 engram 候选（哈希寻址结果）打分，top-K 超稀疏选择，只有被选中的记忆参与 gate 融合。哈希粗筛（精确模式匹配）+ KV 精筛（学习过的打分器）+ top-K 超稀疏 = **比普通向量索引更强**。

## 架构

```
┌────────────────────────────────────────────────────────┐
│ 云端 API 主模型（100k 上下文，KV 保持小）                 │
│   ↑ 超稀疏文本注入（systemPrompt 记忆段，预算 600 token）  │
├────────────────────────────────────────────────────────┤
│ Node 插件（llm/stream 转接）                            │
│  ├─ 请求前：哈希寻址 → 唤醒 → 注入                      │
│  ├─ 回合后：蒸馏 → 写入 engram 表（实时留底）            │
│  └─ 与官方 compact 共存：官方折叠腾 KV，engram 保细节    │
├────────────────────────────────────────────────────────┤
│ Python 魔改 <1B 模型（Qwen3-0.6B torch）                │
│  ├─ EngramModule：哈希记忆表 + gate 融合（hidden states 级）│
│  └─ DSA Indexer：KV 打分路由 + top-K 稀疏选择            │
└────────────────────────────────────────────────────────┘
```

- **大 engram**：外置记忆表（哈希槽 → 记忆嵌入），容量无限、可运行时写入；
- **与官方 compact 共存**：DSH 自带的 `dsh-compact-basic` 是成熟的有损总结式折叠（LLM 生成 checkpoint 替换 surface），负责腾 KV 空间；engram 在每回合结束时**先实时蒸馏留底**（`agent/turn-stopping` → `session.deriveMessages()`），官方折叠丢失的细节随时经「哈希寻址 → KV 路由 → 稀疏注入」找回——100k 等效延展 ≥10 倍；
- **双轨注入**：本地魔改模型内部 gate 融合（hidden states 级）+ 云端 API 模型超稀疏文本注入（systemPrompt 级）。

## 安装

```bash
# dshx（marisa）外部插件管理器
dshx install dsh-engram-relay https://github.com/dsh-external/dsh-engram-relay.git
```

依赖：Node ≥ 18 + Python 3.10+（torch、transformers）。首次启用自动下载 Qwen3-0.6B（约 1.2GB，缓存于 `~/.dsh/engram-relay/models/`）。

## 工具

| 工具 | 作用 |
|---|---|
| `engram_recall` | 主动查询记忆（哈希寻址 + 因果链展开） |
| `engram_store` | 显式写入（fact/decision/event/preference，本会话内，会话结束即弃） |
| `engram_status` | 查看记忆表状态（条目/槽位/模型/预算） |

## 配置（`~/.dsh/config.yaml`）

| 键 | 默认 | 说明 |
|---|---|---|
| `modelId` | `Qwen/Qwen3-0.6B` | 魔改基座模型（<1B） |
| `dtype` | `bfloat16` | 模型精度 |
| `pythonPath` | `python` | Python 解释器（spawn 魔改模型服务） |
| `injectBudgetTokens` | `600` | 单次唤醒注入预算（超稀疏，<1%） |
| `maxWakePerTurn` | `3` | 每回合唤醒条数上限 |
| `storeDir` | `~/.dsh/engram-relay/` | engram 持久化目录 |

## 目录

```
src/                    # Node 插件（llm/stream 转接 + 工具）
python/engram_model/    # Python 魔改模型
├── hash.py             # N-gram 哈希寻址（论文移植）
├── engram_module.py    # EngramMemory + DSA Indexer + Gate 融合
├── model.py            # Qwen3 decoder layer 注入
└── server.py           # JSON 行协议服务（Node spawn 对接）
python/tests/           # 融合模块/哈希数学测试
```

## 设计要点

- **确定性寻址**：相同模式永远命中相同槽位（精确匹配，非相似度近似）；
- **KV 路由**：indexer 打分器是学习过的（`ReLU(q·k)` 多头加权），哈希粗筛 + KV 精筛 + top-K；
- **超稀疏**：唤醒注入预算 600 token（100k 上下文的 <1%），每回合条数有上限；
- **零核心改动**：只使用公开 seam（`llm/stream`、`systemPrompt.context`、`tools.register`、`agent/turn-stopping`、`agent/pre-step`）。

## 收录

按 [dsh-external/hub LOOP.md](https://github.com/dsh-external/hub/blob/main/LOOP.md) 收录：`marisa-plugin` topic + `catalog.source.json`（category: `plugin`）。
