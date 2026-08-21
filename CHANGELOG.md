# Changelog

本项目版本与仓库提交对应，格式参照 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## 未发布（2026-08-15 更新）

### 图谱可视化 Obsidian 化（两阶段布局 + 交互）

- **布局引擎重写**（`src/client/force.ts`）：连通分量两阶段递归布局——阶段 1 组件级排布（虚拟空间按组件尺寸放大，组件从结构上不可能互相穿插），阶段 2 成员级 FR 力导向展开（k²/d 斥力 + d²/k 弹簧 + 温度冷却 alpha 衰减），碰撞分离后处理兜底零重叠；布局保持自然尺度，缩放交给视图 fitTransform
- **交互**（`src/client/GraphView.tsx`）：滚轮缩放（以光标为锚）、拖拽平移、双击复位、fit-to-content；悬停高亮邻接网络（其余淡化）+ 原生 tooltip；过滤切换节点平滑过渡（CSS transform）；标签贪心防重叠 + 缩放足够深才批量显示（悬停常显）；边随距离淡出；半径随 importance（7–16）
- **质量回归**：`tests/layout-quality.mjs`（95 节点/21 分量压力图：重叠对 0、聚团比 4.9、边距≈115）；`tests/force.test.mjs` 补零重叠/组件聚类断言

### 云仓库首装不再丢 WebUI 注入（安装链路修复）

- **`scripts/build.sh` 全量构建**：一次 `npm run build` = host tsc + client tsdown + 产物自检（lib/index.js + lib/client.js 缺失即失败）——此前只编译 host，`lib/client.js` 缺失会让 client-modules 的整个 `__DSH_BOOT__` 注入失败（WebUI 所有插件面丢失）
- **双布局依赖解析**：插件自身 node_modules 优先，DSH 源码 checkout（DSH_CHECKOUT）/ npm 全局包（dsh on PATH / npm root -g / Windows 全局目录扫描）回退链接，缺 tsc/tsdown 给明确指引
- **`package.json` 加 `prepare`**：`dsh plugin add <git|file>` 安装时 pnpm 自动构建（git 依赖需按提示加 allowBuilds）
- **docs/INSTALL-new-machine.md 更新**：单命令全量构建 + 云仓库直接安装流程

### 修复

- **store 清空复活**：`load()` 对"主文件存在且完全可解析（含合法空库）"不再回退备份——此前 clearSession/remove 清空后重载会从旧 .bak 复活已删记忆；NUL 损坏（有真实事故）仍走备份恢复链（`tests/session.test.mjs` + `store-recovery.test.mjs` 双绿）
- **测试对齐 2026-08-12 语义门槛改造**：wake 测试补桩 embedder（无 embedder 时宁缺毋滥=零注入是新有意语义）；graph-api 测试的 fake server 键名 httpServer→webServer（改名后测试未跟上）；hybrid 测试对齐阈值/分数团规则——全量 61/61 通过

## 历史（2026-08-13 及更早）

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

### 2026-08-12 及更早

- 记忆注入缓存友好化：system 段固定化 + 动态召回改消息尾注入（`8d5ca16`）
- engram_status compact 探测改 ctx.get + 记忆段 order 9997 尾部化（`d18bfa4`）
- 存储原子写 + 损坏自愈备份（engrams.jsonl 曾全 NUL 损坏，双实例非原子并发写）
- 蒸馏 reasoningEffort off（max 思考会吃光 800 token 预算导致输出为空）

