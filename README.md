# dsh-engram-relay

> 单会话上下文增强：大一统记忆图谱（Obsidian 式双向链接 + 因果双向追溯 + 自组织聚类），
> N-gram 哈希确定性寻址 × bge 语义精排 × 因果传播的超稀疏主动唤醒，渐进披露
> （入口 = `[[标题]]` + 摘要，按需展开正文）；会话结束即弃，不做跨会话记忆沉淀。

## 是什么

`dsh-engram-relay` 把 DeepSeek Engram 的「N-gram 哈希 O(1) 确定性寻址」思想做成**系统级**
外置记忆转接层（无需训练任何小模型）：

1. **[Engram（条件记忆，[deepseek-ai/Engram](https://github.com/deepseek-ai/Engram)）](https://arxiv.org/abs/2601.07372)**：N-gram 哈希 O(1) 确定性寻址 → 巨大静态记忆表。为 Transformer 补上「知识查找原语」（MoE 扩计算容量，Engram 扩静态记忆容量）。
2. **bge-small-zh-v1.5（专用嵌入模型，~95MB）**：对哈希粗筛候选做语义精排，修掉哈希的 mode-level 跨主题误命中（同 80 查询实测：纯哈希精确率 54% → 混合 95%）。
3. **因果图**：节点间 `causes/effects` 双向边，唤醒时沿因果链双向传播激活——「什么导致了它 / 它导致了什么」，这是普通向量索引做不到的。

**混合检索管线**：哈希粗筛（精确、O(1)、确定性）→ bge 语义精排（修误命中）→
因果传播（召回前因后果）→ top-K 超稀疏注入 = **精确 + 语义 + 因果**。

## 架构

```
┌────────────────────────────────────────────────────────┐
│ 云端 API 主模型（100k 上下文，KV 保持小）                 │
│   ↑ 超稀疏文本注入（systemPrompt 记忆段，预算 600 token）  │
├────────────────────────────────────────────────────────┤
│ Node 插件（llm/stream 转接，零第三方运行时依赖）           │
│  ├─ 请求前：哈希粗筛 → bge 精排 → 因果传播 → 稀疏注入      │
│  ├─ 写入：模型调 engram_store 落节点（标题/摘要/正文/因果） │
│  ├─ 回合后：自动蒸馏（遗留 0.6B 轨，已移除时跳过）         │
│  └─ 会话结束：agent/disposed → 清空该会话全部 engram      │
├────────────────────────────────────────────────────────┤
│ Python 转接服务（JSON 行协议）                            │
│  └─ embed op：bge-small-zh-v1.5 编码（懒加载，本地离线）   │
└────────────────────────────────────────────────────────┘
```

- **大一统记忆图谱**：节点 = `{title, summary, content, links[], causes[], effects[]}`，
  `[[标题]]` 双向链接（Obsidian 风格）+ 因果双向追溯；**不硬编码分层**——主题结构由
  链接/因果密度自组织成簇（连通分量），唤醒时给出簇概览；
- **渐进披露**：唤醒只注入入口（`[[标题]]` + 摘要 + ↑因/↓果），模型可经 `engram_open`
  按需展开正文与因果邻居——超稀疏且不丢细节；
- **与官方 compact 共存**：DSH 自带的 `dsh-compact-basic` 负责腾 KV（有损总结式折叠）；
  engram 在折叠前实时留底（`agent/turn-stopping`），细节可唤醒找回；
- **会话级**：只做单次会话上下文增强，`agent/disposed` 清空该会话记忆，无跨会话沉淀。

## 安装

```bash
# dshx（marisa）外部插件管理器
dshx install dsh-engram-relay https://github.com/dsh-external/dsh-engram-relay.git
```

依赖：Node ≥ 18 + Python 3.10+（sentence-transformers、torch）。
嵌入模型本地离线加载（无 HF 联网需求）：先下载 bge-small-zh-v1.5 到本地目录，
再在 profile patch 里配置 `embedModel` 指向该目录（或设环境变量 `ENGRAM_EMBED_MODEL`）。

## 工具

| 工具 | 作用 |
|---|---|
| `engram_recall` | 主动查询记忆（混合检索 + 因果链展开） |
| `engram_store` | 显式写入（fact/decision/event/note，本会话内，会话结束即弃） |
| `engram_open` | 展开入口（正文 + 链接 + 因果邻居，渐进披露第二层） |
| `engram_status` | 查看记忆表状态（条目/槽位/模型/预算） |

## 配置（profile patch，如 `~/.dsh/profiles/web/cordis.patch.yml`）

```yaml
- id: dsh-engram-relay
  name: '@dsh-external/dsh-engram-relay'
  config:
    embedModel: 'F:/dsh/engram-trial/bge-small-zh'   # bge 本地目录（必配才能语义精排）
```

| 键 | 默认 | 说明 |
|---|---|---|
| `embedModel` | `` | bge 嵌入模型目录（本地路径；空 = 服务端 `ENGRAM_EMBED_MODEL`，再空则禁用精排） |
| `modelId` | `` | 遗留：0.6B 蒸馏模型目录（已移除；空 = 不加载） |
| `pythonPath` | `python` | Python 解释器（spawn 转接服务） |
| `injectBudgetTokens` | `600` | 单次唤醒注入预算（超稀疏，<1%） |
| `maxWakePerTurn` | `3` | 每回合唤醒条数上限 |
| `storeDir` | `~/.dsh/engram-relay/` | engram 持久化目录 |

## 目录

```
src/                    # Node 插件（llm/stream 转接 + 工具）
├── engram/hash.ts      # N-gram 多头哈希寻址（论文移植，确定性）
├── engram/causal.ts    # 因果图（causes/effects 双向传播）
├── engram/store.ts     # 大一统记忆图谱（节点/链接/聚类/持久化）
├── engram/wake.ts      # 唤醒管线（哈希粗筛 → 精排 → 因果 → 稀疏截断）
└── model/              # Python 服务客户端（embed/distill/status）
python/engram_model/    # Python 转接服务
├── hash.py             # N-gram 哈希寻址（Python 移植，与 TS 同构）
├── server.py           # JSON 行协议服务（embed op：bge 编码；遗留 0.6B op）
python/embed_compare.py # 哈希 vs bge vs 混合 检索质量对比（95% 精确率）
python/tests/           # 哈希/融合模块数学测试
```

## 设计要点

- **确定性寻址**：相同 N-gram 模式永远命中相同槽位（精确匹配，非相似度近似）；
- **混合检索**：哈希粗筛保证精确命中保底，bge 精排修跨主题误命中，因果传播召回前因后果；
- **渐进披露**：入口超稀疏（title + summary），正文/链接按需展开；
- **零核心改动**：只使用公开 seam（`llm/stream`、`systemPrompt.context`、`tools.register`、`agent/turn-stopping`、`agent/disposed`）。

## 收录

按 [dsh-external/hub LOOP.md](https://github.com/dsh-external/hub/blob/main/LOOP.md) 收录：`marisa-plugin` topic + `catalog.source.json`（category: `plugin`）。
