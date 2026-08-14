# dsh-engram-relay

> **本仓库是公开镜像**（上游：`dsh-external/dsh-engram-relay`，BSD-3-Clause）。
> 安装请用下方公开地址；上游转为公开后本镜像仍保持同步。

> 跨会话分层记忆：大一统记忆图谱（Obsidian 式双向链接 + 因果双向追溯 + 自组织聚类），
> N-gram 哈希确定性寻址 × **三通道纯算法语义精排（词汇/图/共现，零外部模型）** × 因果传播的超稀疏主动唤醒，渐进披露
> （入口 = `[[标题]]` + 摘要，按需展开正文）；**分层归属由 AI 自主决策**——
> global 全局持久 / project 项目持久 / session 会话临时，跨会话沉淀与召回。

## 是什么

`dsh-engram-relay` 把 DeepSeek Engram 的「N-gram 哈希 O(1) 确定性寻址」思想做成**系统级**
外置记忆转接层（无需训练任何小模型）：

1. **[Engram（条件记忆，[deepseek-ai/Engram](https://github.com/deepseek-ai/Engram)）](https://arxiv.org/abs/2601.07372)**：N-gram 哈希 O(1) 确定性寻址 → 巨大静态记忆表。为 Transformer 补上「知识查找原语」（MoE 扩计算容量，Engram 扩静态记忆容量）。
2. **三通道纯算法语义引擎（SemanticScorer，零嵌入模型）**：对哈希粗筛候选做语义精排——
   词汇通道（n-gram Jaccard 重叠 + 词频）、图通道（因果/链接邻居语义传播）、
   共现通道（PCA 谱分解语义桥，跨词共现如「缓存↔cache」）；确定性、可解释、
   **零外部模型、零联网、零 Python**。修掉哈希的 mode-level 跨主题误命中
   （同 80 查询实测：纯哈希精确率 54% → 混合 95%，三通道承接了当年 bge 实验的增益）。
3. **因果图**：节点间 `causes/effects` 双向边，唤醒时沿因果链双向传播激活——「什么导致了它 / 它导致了什么」，这是普通向量索引做不到的。

**混合检索管线**：哈希粗筛（精确、O(1)、确定性）→ 三通道纯算法精排（修误命中，零模型）→
因果传播（召回前因后果）→ top-K 超稀疏注入 = **精确 + 语义 + 因果**。

## 架构

```
┌────────────────────────────────────────────────────────┐
│ 云端 API 主模型（100k 上下文，KV 保持小）                 │
│   ↑ 超稀疏文本注入（systemPrompt 记忆段，预算 600 token）  │
├────────────────────────────────────────────────────────┤
│ Node 插件（核心零第三方运行时依赖）                       │
│  ├─ 请求前：哈希粗筛 → 分层准入 → 三通道纯算法精排 →      │
│  │           因果传播 → 稀疏注入（global 全可见 /          │
│  │           project 同 cwd / session 本会话）             │
│  ├─ 写入：模型调 engram_store 落节点（自主分层 + 因果）     │
│  ├─ 维护：search/link/update/remove/promote（织图谱/转层） │
│  └─ 会话结束：agent/disposed → 只清 session 层临时记忆     │
│            （global/project 跨会话持久）                   │
├────────────────────────────────────────────────────────┤
│ 可选向量缓存增强（非必需；核心语义引擎零模型）            │
│  ├─ TS ONNX：包内 int8 bge 模型（transformers.js 懒加载） │
│  └─ Python 转接服务（遗留回退，JSON 行协议）：            │
│      embed op：本地 fp32 bge-small-zh-v1.5 编码（懒加载） │
└────────────────────────────────────────────────────────┘
```

- **大一统记忆图谱**：节点 = `{layer, title, summary, content, links[], causes[], effects[]}`，
  `[[标题]]` 双向链接（Obsidian 风格）+ 因果双向追溯；**层是节点属性（不分家）**，
  主题结构由链接/因果密度自组织成簇（连通分量），唤醒时给出簇概览；
- **预设分层 + AI 自主决策**：3 层 = `global`（全局持久·所有会话）/ `project`（项目持久·
  按工作目录隔离）/ `session`（会话临时·结束清理）；模型经 `engram_store` 写入时**自主决策**
  归属层（跨会话长期→global、本项目→project、仅本次→session），可在会话结束前
  `engram_promote` 把 session 临时记忆转长期；
- **跨会话准入**：唤醒/检索按查看者视角过滤——global 所有会话可见、project 仅同工作目录、
  session 仅本会话；session 层随会话结束清理，global/project 跨会话沉淀与召回；
- **渐进披露**：唤醒只注入入口（`[[标题]]` + 层 + 摘要 + ↑因/↓果），模型可经 `engram_open`
  按需展开正文与因果邻居——超稀疏且不丢细节；
- **与官方 compact 共存**：DSH 自带的 `dsh-compact-basic` 负责腾 KV（有损总结式折叠）；
  engram 在折叠前实时留底（`agent/turn-stopping`），细节可唤醒找回。

## 安装

```bash
# dshx（marisa）外部插件管理器（公开镜像地址）
dshx install dsh-engram-relay https://github.com/yjh051108/dsh-engram-relay.git
```

依赖：Node ≥ 18。**核心语义引擎零模型、零依赖、开箱即用**——三通道纯算法
（词汇 n-gram + 图语义传播 + PCA 共现），无需下载嵌入模型、无需 Python、无联网。
可选增强（向量缓存补分用，非必需）：包内自带 int8 量化 bge-small-zh
（`model/bge-small-zh/`，~24MB，TS ONNX 懒加载）或 Python 3.10+
（sentence-transformers）加载本地 fp32 bge-small-zh-v1.5 目录
（配置 `embedModel` 或环境变量 `ENGRAM_EMBED_MODEL`）。

## 工具

| 工具 | 作用 |
|---|---|
| `engram_recall` | 按需唤醒检索（跨会话分层准入 + 因果邻接，可过滤层） |
| `engram_store` | 写入记忆（**AI 自主决策分层** + 因果前因/后果 + 双向链接） |
| `engram_propose` | 提议记忆（写入 ⏳pending，用户确认后才参与召回，确认制沉淀） |
| `engram_confirm` | 确认一个待确认节点（确认后参与 recall/唤醒命中） |
| `engram_reject` | 拒绝（删除）一个待确认节点（仅 pending 可拒） |
| `engram_open` | 展开入口（正文 + 层 + 链接 + 因果邻居，渐进披露第二层） |
| `engram_search` | 盘点记忆图谱（按层/项目/类型/关键词检索，遵守可见性） |
| `engram_link` | 显式连接节点（因果/依赖/引用边，双向链接织图谱） |
| `engram_update` | 修正节点（摘要/正文/链接/重要度/标题） |
| `engram_remove` | 删除节点（谨慎，不可恢复） |
| `engram_promote` | 提升层（session→project/global，会话结束前转长期） |
| `engram_weave` | 织网清洗（孤立节点语义配对自动织双向链接） |
| `engram_status` | 查看服务状态（分层统计/槽位/因果边/模型/预算） |

## 配置（profile patch，如 `~/.dsh/profiles/web/cordis.patch.yml`）

```yaml
- id: dsh-engram-relay
  name: '@dsh-external/dsh-engram-relay'
  config:
    # 可选增强（向量缓存）：本地 fp32 bge 模型目录；
    # 缺省留空 = 核心三通道纯算法引擎（推荐，零模型零配置）
    embedModel: '/path/to/bge-small-zh-v1.5'
```

| 键 | 默认 | 说明 |
|---|---|---|
| `embedModel` | `` | 可选增强：bge 嵌入模型目录（向量缓存补分）；空 = 核心三通道纯算法（零模型），包内 int8（TS ONNX）→ Python 服务 `ENGRAM_EMBED_MODEL` 依次可选 |
| `modelId` | `` | 遗留：0.6B 蒸馏模型目录（已移除；空 = 不加载） |
| `pythonPath` | `python` | Python 解释器（可选增强服务） |
| `injectBudgetTokens` | `600` | 单次唤醒注入预算（超稀疏，<1%） |
| `maxWakePerTurn` | `3` | 每回合唤醒条数上限 |
| `storeDir` | `~/.dsh/engram-relay/` | engram 持久化目录 |

## 目录

```
src/                    # Node 插件（llm/stream 转接 + 工具）
├── engram/hash.ts      # N-gram 多头哈希寻址（论文移植，确定性）
├── engram/causal.ts    # 因果图（causes/effects 双向传播）
├── engram/store.ts     # 大一统记忆图谱（节点/链接/聚类/持久化）
├── engram/wake.ts      # 唤醒管线（哈希粗筛 → 三通道纯算法精排 → 因果 → 稀疏截断）
├── engram/semantic-scorer.ts  # 三通道纯算法语义引擎（词汇/图/PCA 共现，零模型）
└── model/              # 可选向量缓存：TS ONNX（包内 int8）+ Python 服务客户端
python/engram_model/    # Python 转接服务（可选/遗留）
├── hash.py             # N-gram 哈希寻址（Python 移植，与 TS 同构）
├── server.py           # JSON 行协议服务（embed op：bge 编码；遗留 0.6B op）
python/embed_compare.py # 哈希 vs bge vs 混合 检索质量对比（95% 精确率，历史实验）
python/tests/           # 哈希/融合模块数学测试
```

## 设计要点

- **确定性寻址**：相同 N-gram 模式永远命中相同槽位（精确匹配，非相似度近似）；
- **三通道纯算法语义精排**：哈希粗筛保证精确命中保底，词汇（n-gram Jaccard + 词频）/
  图（因果链接邻居传播）/ 共现（PCA 谱分解语义桥）三通道修跨主题误命中——
  零嵌入模型、确定性、可解释；因果传播再召回前因后果；
- **渐进披露**：入口超稀疏（title + summary），正文/链接按需展开；
- **零核心改动**：只使用公开 seam（`llm/stream`、`systemPrompt.context`、`tools.register`、`agent/turn-stopping`、`agent/disposed`）。

## 收录

按 [dsh-external/hub LOOP.md](https://github.com/dsh-external/hub/blob/main/LOOP.md) 收录：`marisa-plugin` topic + `catalog.source.json`（category: `plugin`）。
