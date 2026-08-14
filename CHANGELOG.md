# Changelog

本项目版本与仓库提交对应，格式参照 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## 0.1.0 开源发布（2026-08-13）

- 仓库公开（`dsh-external/dsh-engram-relay`）：去 `private`、补 repository/homepage/bugs/keywords
- 开源卫生：清除全部开发者本机绝对路径（`cordis.patch.yml` / `src/index.ts` /
  python 脚本 / tests 的默认值改为环境变量或包内相对路径）；内部装机文档移出仓库
- 开箱即用：`embedModel` 留空默认走包内 int8 bge 模型（TS ONNX，免 Python），
  配置/环境变量可覆盖为本地 fp32 模型
- 元数据对齐：`dsh.plugin.json` 与 `dshx.contributes.tools` 补齐全部 13 个工具
  （新增 `engram_propose` / `engram_confirm` / `engram_reject` / `engram_weave`）
- 图谱 UI：碰撞边界硬分离（位置级分离 + 输出前多轮扫描兜底，修节点重叠）

## 未发布（2026-08-13 更新）

### 真实历史数据仿真（`simulate_real.py`）

- 97 条真实记忆 × 161 条真实会话查询 × 真实 bge 全链路：相关召回 **82.9%**、平均注入 **100 tok/轮**（动态分级生效）
- ground truth 关键词粗标注（诚实声明：语义相关但词表不匹配的查询被误标无关，误召指标被高估，召回可信）
- 动态注入预算：按相关度分级渲染（最高分完整入口/次之标题摘要/其余仅标题）
- engram_store 撰写规范（标题/摘要/正文字数与质量、反模式、同主题修订优先）

### 唤醒系统数学建模（`1c6ba25`）

- **真实 bge 分布校准**：`python/tests/sample_bge_dist.py`（86 条记忆 × 8 查询，sentence-transformers 加载 bge-small-zh）——相关对 N(0.516, 0.088)、无关对 N(0.293, 0.099)，两分布重叠显著
- **数学建模报告**：`docs/simulation-model.md`（问题重述 / 假设 / 符号 / 召回-成本模型 / ROC 求解 / 灵敏度 / 多重比较 / 局限诚实声明）
- **多重比较自适应阈值**：固定阈值下 n 条候选"至少一条误过"≈ 1−(1−q)^n → θ_eff = θ + 0.03·log₂(n/16)
- **1M 上下文仿真**：`tests/simulate-1m-context.mjs`（8:2 输入输出比、800k 输入长跑、注入占比/缓存命中/召回精度）
- **PID 式调参仿真**：`tests/simulate-tune.mjs`（增益/成本/数量扫描，网格寻优）
- 顺带修复：注入预算严格截断（头部+因果注全计入，之前 200 预算实际渲染 400+）、lookup 候选 64→256、estimateTokens 校准（中文 0.7 tok/字）

### 唤醒语义门槛与入口级注入（`70935af`、`97858e5`）

- **语义阈值 0.42**（可配 `semanticMinScore`）：bge 余弦低于阈值不注入；embedder 不可用时宁缺毋滥（删除"重要度垫底全激活"路径）——无关记忆零注入
- **注入预算 600 → 200**（入口级：[[标题]]+摘要+因果注，渐进披露第一层）
- **蒸馏自动因果**：prompt 提供已有记忆入口列表并要求 `causesOf` 引用 → 解析后自动建因果边 + effects 双写（零主代理负担）

### 修复

- **中英混合词哈希寻址失效**：`"主题0"` / `"browser-panel操控"` 类混合词被整词成单 token（无 n-gram）→ 永远无法被哈希召回；normalize 改为逐字/逐段拆分（`17a37d9`）

## 历史（2026-08-12 及更早）

- 记忆注入缓存友好化：system 段固定化 + 动态召回改消息尾注入（`8d5ca16`）
- engram_status compact 探测改 ctx.get + 记忆段 order 9997 尾部化（`d18bfa4`）
- 存储原子写 + 损坏自愈备份（engrams.jsonl 曾全 NUL 损坏，双实例非原子并发写）
- 蒸馏 reasoningEffort off（max 思考会吃光 800 token 预算导致输出为空）

