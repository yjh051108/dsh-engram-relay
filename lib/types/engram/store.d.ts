/**
 * EngramStore — 大一统记忆图谱（JSONL 持久化）。
 *
 * 模型（参考 Obsidian 双向链接 + skill 渐进式披露）：
 *  - **节点**：统一记忆（不预分轨/不硬编码分层）。每条记忆 =
 *      title（入口锚点）+ summary（一句话摘要，渐进披露第一层）
 *      + content（完整正文，按需展开）+ links（双向链接 [[title]]）
 *      + causes/effects（因果边，双向可追溯）
 *  - **索引**：N-gram 哈希寻址（NgramHashAddressing）→ 槽位 → 节点，
 *    确定性 O(1) 匹配当前上下文；
 *  - **自组织**：不手动分层——链接密度/主题关联自然形成结构，
 *    唤醒按关联度排序（类 Obsidian 图谱的局部密度）。
 *
 * 定位：**单次会话上下文增强**——本会话记忆写入、入口唤醒、渐进展开、
 * 因果双向追溯；会话结束即弃（clearSession），不做跨会话沉淀。
 */
import { NgramHashAddressing, type HashResult } from './hash.js';
/** 记忆节点类型（统一，不预分轨；kind 仅作展示标签，非分层）。 */
export type EngramKind = 'fact' | 'decision' | 'event' | 'note' | 'snapshot';
/**
 * 记忆归属（v0.4：不做分层分化——项目即标签，融会贯通）：
 *  - global：通用知识项目（原"全局层"降为平级标签：技术模式/平台坑/偏好）
 *  - project：具体项目（projectId = 工作目录）
 * 可见性：**全可见**（单用户本地系统，无多租户隐私需求；项目间通过
 * 记忆关联（link/causes 桥）自然融会贯通——"套娃"）。
 */
export type EngramLayer = 'global' | 'project';
/** 分层常量（工具 description 引用）。 */
export declare const ENGRAM_LAYERS: EngramLayer[];
/**
 * 可见性判定（v0.4：全可见——项目不隔离，关联即桥）。
 * 保留函数签名（wake/tools/图谱 API 调用点不动），恒返回 true。
 */
export declare function isVisible(_e: EngramNode, _viewer: {
    sessionId?: string;
    cwd?: string;
}): boolean;
/** 默认标签（v0.4 迁移/兜底）：layer + projectId → 命名空间标签。 */
export declare function defaultTags(layer: EngramLayer, projectId: string | null): string[];
/**
 * 沉睡判定（派生状态，实时计算）：最后强化 > 30 天前 且 hits < 3。
 * 被命中（touch）的节点刚强化，永不处于沉睡；沉默 >30 天的节点在
 * 渲染时降级为仅标题（不占注入预算），图谱仍可检索、命中即复苏。
 */
export declare function dormantOf(e: EngramNode, now?: number): boolean;
/** 废止判定（版本链）：被新版本取代，退出检索/注入，可追溯。 */
export declare function isSuperseded(e: EngramNode): boolean;
/** 渐进披露层级。 */
export interface EngramNode {
    id: string;
    kind: EngramKind;
    /** 分层归属（AI 自主决策）：global=全局持久 / project=项目持久。 */
    layer: EngramLayer;
    /**
     * 自由标签（v0.4：一节点多标签，自由分类——不建枚举 schema）：
     * 分类靠命名空间约定，如 '全局'（开发者喜好/全局要求）、
     * '项目:xxx'（项目自由命名）、'教训:代码' / '教训:思想'（教训自由子分类）。
     * layer/projectId 保留为系统内部字段（写入绑定/快照聚合），
     * tags 是用户可见的分类维度。
     */
    tags?: string[];
    /**
     * 巩固状态（双维度：可见性×巩固度，v0.3 引入）：
     *  - episodic：刚写入，事件性，细节丰富（情景记忆）
     *  - semantic：强化 ≥3 次，被归纳，去情景化（语义记忆）
     *  - dormant ：30 天无强化，低激活，退出注入入口层（沉睡）
     * 惰性迁移（touch/reinforce 时评估，无定时扫描）。
     */
    state?: 'episodic' | 'semantic' | 'dormant';
    /**
     * 版本链（v0.3，治理缺口①真理维护）：被哪个节点取代（id）。
     * 被取代 = 废止（superseded）：退出检索/注入（只注入"当前有效"），
     * 但 engram_open/byTitle 仍可追溯旧版（版本链可回溯，不删数据）。
     */
    supersededBy?: string;
    /** 版本链：取代了哪些节点（id 列表，反向追溯）。 */
    supersedes?: string[];
    /** project 层标识（会话工作目录；global 层为 null）。 */
    projectId: string | null;
    /** 入口锚点（唤醒列表展示；如 Obsidian 的页面标题）。 */
    title: string;
    /** 一句话摘要（渐进披露第一层——入口列表只给这个）。 */
    summary: string;
    /** 完整正文（渐进披露第二层——展开时给）。 */
    content: string;
    /** 双向链接：关联节点的 title 集（Obsidian 风格 [[title]]）。 */
    links: string[];
    /** 因果边（前因）：导致本节点的节点 id 集。 */
    causes: string[];
    /** 因果边（后果）：本节点导致的节点 id 集。 */
    effects: string[];
    /** 来源会话 id（本会话内）。 */
    sessionId: string | null;
    /** 来源回合序号。 */
    turn: number;
    /** 创建时间（epoch ms）。 */
    createdAt: number;
    /** 关联度 0-1（唤醒排序用；自组织：链接越多/被引用越多越高）。 */
    importance: number;
    /** 被唤醒次数（LRU 衰减）。 */
    hits: number;
    /** 最后唤醒时间。 */
    lastHitAt: number | null;
    /** 该节点对应的哈希槽位（写入时固化，重哈希可重建）。 */
    slots: string[];
    /** 确认状态：pending=待确认（不参与检索/唤醒命中），confirmed=已确认（缺省；旧数据视为 confirmed）。 */
    status?: 'pending' | 'confirmed';
    /** 强化事件时间戳（写入/命中/展开/链接；类脑激活模型 B=ln(Σt^(-d)) 的输入）。旧数据缺省 [createdAt]。 */
    reinforces?: number[];
}
/** 渐进披露视图。 */
export interface EngramEntry {
    id: string;
    title: string;
    summary: string;
    kind: EngramKind;
    /** 因果邻接摘要（入口层展示：前因/后果标题）。 */
    causeTitles: string[];
    effectTitles: string[];
    /** 双向链接标题。 */
    linkTitles: string[];
}
export declare function createEngramId(): string;
export declare class EngramStore {
    private hasher;
    readonly dir: string;
    private file;
    private byId;
    /** 槽位索引：slotKey -> Set<nodeId>（派生索引，写入/加载时构建）。 */
    private slotIndex;
    /**
     * token 倒排索引（v0.5 纯算法语义关键修复）：token -> Set<nodeId>——
     * **词袋召回**（与 PCA 语义对齐）。哈希 n-gram 要求词组连续匹配，查询
     * 「pnpm 装包报错 EPERM」vs 记忆「pnpm EPERM 修复」词袋共享但窗口不
     * 连续 → 槽位交集 0 → 粗筛漏掉（实测根因）。倒排补上词袋维度。
     */
    private tokenIndex;
    /** 标题索引：title -> id[]（多值——跨项目同标题合法存在；解析时消歧）。 */
    private titleIndex;
    constructor(storeDir: string, hasher?: NgramHashAddressing);
    private load;
    private indexSlot;
    /** token 倒排索引登记（词袋召回维度——与 PCA 语义对齐）。 */
    private indexTokens;
    /** token 倒排移除。 */
    private unindexTokens;
    /** 节点检索文本（title + summary + content 的前部——token 倒排用）。 */
    private textOfNode;
    /**
     * 原子持久化：写临时文件 + rename 替换。
     *
     * 背景：web 与 headless 两个 profile 可能同时装配本插件并写同一个
     * engrams.jsonl；热重载时同一进程内也会短暂存在两个 store 实例（旧
     * fiber dispose 前的最后一次 persist 与新实例并发）。tmp 必须**每实例
     * 唯一**（曾用 `${pid}` 导致同进程两实例共用同名 tmp → writeFileSync
     * 交错 → 整文件 NUL、记忆全丢），并加进程内写锁串行化 rename 竞态。
     * Windows 上 rename 覆盖已存在文件会失败，先 unlink 目标再 rename。
     */
    private persist;
    /**
     * 写入/更新一个记忆节点：按 title+summary 哈希寻址，挂到命中槽位。
     * 渐进披露：title/summary 是入口层，content 是展开层。
     * v0.4：支持 tags 多标签（缺省从 layer/projectId 生成默认标签）。
     */
    add(input: Omit<EngramNode, 'id' | 'createdAt' | 'hits' | 'lastHitAt' | 'slots' | 'layer' | 'projectId'> & {
        layer?: EngramLayer;
        projectId?: string | null;
        tags?: string[];
    }): EngramNode;
    /** 标题索引登记（多值聚合，去重）。 */
    private indexTitle;
    /** 标题索引移除（只摘该项，不影响同名其他节点）。 */
    private unindexTitle;
    /** 按标题取节点（双向链接 [[title]] 解析）。同名消歧：**未废止（active）优先**，
     *  其次最近写入——否则链接解析会取到废止旧版（第七轮新 agent 实测：
     *  邻接标注错乱根因）。⚠️ 曾有 Map<title,单id> 覆盖 bug——多值后不再丢。 */
    byTitle(title: string): EngramNode | undefined;
    /** 按标题取全部同名节点（消歧/盘点用：跨项目同标题、版本链同主题）。 */
    byTitles(title: string): EngramNode[];
    /**
     * 工作快照（远景场景 6"继续昨天的工作"）：聚合 cwd 最近写入的进行中
     * 状态 → 快照节点（kind=snapshot，episodic 起步，每次更新即强化保持活跃）。
     * 幂等：同 cwd 已有快照且内容未变 → 不写盘；内容变化 → 原地更新 + 强化。
     * 快照是"当前状态"不是"事实"——原地更新，不建版本链。
     */
    upsertSnapshot(cwd: string, turn: number, sessionId?: string | null): EngramNode | null;
    /** 按文本哈希寻址 + token 倒排词袋召回（去重，按关联度降序；不含废止节点）。 */
    lookup(text: string, limit?: number): EngramNode[];
    /** token 倒排查询（v0.6 共现扩展粗筛用）：含任一 token 的记忆并集
     *  ——与词袋语义对齐，支持"查询词共现邻居"召回。 */
    lookupTokens(tokens: string[], limit?: number): EngramNode[];
    /** 按已计算的哈希结果寻址（避免重复哈希）；text 供 token 倒排词袋召回。 */
    lookupHash(result: HashResult, text?: string, limit?: number): EngramNode[];
    /** 渐进披露入口视图：摘要级 + 因果/链接邻接摘要。 */
    entry(node: EngramNode): EngramEntry;
    /** 批量入口视图。 */
    entries(nodes: EngramNode[]): EngramEntry[];
    /**
     * 自组织聚类：按连接密度（links + causes/effects）自然成簇——不预定义
     * 主题、不硬编码分层。连通分量即簇；每簇选「代表节点」（连接度最高者）
     * 作为唤醒入口。类似 Obsidian 图谱的视觉密度：密集连接处自然成团。
     */
    clusters(): Array<{
        label: string;
        members: string[];
        representative: string;
    }>;
    get(id: string): EngramNode | undefined;
    getMany(ids: string[]): EngramNode[];
    all(): EngramNode[];
    count(): number;
    slotCount(): number;
    /**
     * 分层统一查询（维护/检索入口）：按层/项目/会话/类型/时间过滤。
     * 缺省按 importance 降序；recent=true 按创建时间倒序。
     */
    query(filter?: {
        layer?: EngramLayer;
        projectId?: string | null;
        kind?: EngramKind;
        since?: number;
        until?: number;
        limit?: number;
        recent?: boolean;
    }): EngramNode[];
    /** 分层统计（status 工具用）。 */
    layerCounts(): Record<EngramLayer, number>;
    /** 巩固状态统计（status 工具用；旧数据默认 episodic；dormant 派生）。 */
    stateCounts(): Record<'episodic' | 'semantic' | 'dormant', number>;
    /**
     * 提升/转层：改 layer 与 projectId（保留 id/因果/链接——引用不失效）。
     * project → global（跨项目共享真理）；global 不可降级。
     */
    promote(id: string, layer: EngramLayer, projectId?: string | null): EngramNode | undefined;
    /**
     * 版本链（真理维护核心）：newId 取代 oldId——
     *  - old.supersededBy = newId（退出检索/注入，只注入"当前有效"）
     *  - new.supersedes += oldId（反向追溯版本链）
     * 数据不删（可追溯），因果/链接继承由调用方迁移。
     */
    supersede(oldId: string, newId: string): boolean;
    /** 修正节点字段（title 变更会同步标题索引；层变更用 promote；tags 覆盖设置）。 */
    update(id: string, patch: Partial<Pick<EngramNode, 'title' | 'summary' | 'content' | 'links' | 'causes' | 'effects' | 'importance' | 'tags'>>): EngramNode | undefined;
    /** 清空一个项目（project 层全部节点；项目移除/归档时）。 */
    clearProject(projectId: string): number;
    /** 登记一次唤醒（LRU 衰减 + 激活强化：命中即复习，类脑巩固）。 */
    touch(id: string): void;
    /** 登记一次强化（展开/链接等深度使用——权重高于命中）。 */
    reinforce(id: string): void;
    /**
     * 惰性状态迁移（命中/强化时评估，无定时扫描）：
     *  - hits ≥ 3 → semantic（强化历史充足，去情景化，持久固化）
     *  - dormant 是**派生状态**（见 isDormant）：由强化历史实时计算——
     *    被 touch 的节点刚强化，永不处于沉睡；沉默 >30 天的节点在
     *    渲染/候选时降级为仅标题，不占注入预算。
     */
    private evolveState;
    /** 全部待确认节点（用户确认制管理面）。 */
    pending(): EngramNode[];
    /** 确认一个待确认节点（确认后才参与检索/唤醒命中）。幂等：已确认返回原节点。 */
    confirmNode(id: string): EngramNode | undefined;
    /** 拒绝（删除）一个待确认节点。非 pending 节点不可拒绝（防误删已生效记忆）。 */
    rejectNode(id: string): boolean;
    remove(id: string): boolean;
    /** 归档文件（被淘汰节点保留可恢复，不删数据）。 */
    private archiveFile;
    /**
     * 归档一个节点：JSON 追加到 archived.jsonl（带 archivedAt），然后从主库移除。
     * 归档 = 完全退出检索/注入/图谱，但可手动恢复（读回 archived.jsonl）。
     */
    archiveNode(id: string): boolean;
    /**
     * 硬上限（v0.5）：count > maxNodes 时按「留着最没用」顺序淘汰归档：
     *  ① superseded（已废止——真理已由当前版承载）
     *  ② dormant（沉睡——长期未用，不占注入预算仍占存储）
     *  ③ 低激活 episodic（强化少 + 创建久）
     * 返回淘汰数。惰性触发（写入后调用），不阻塞。
     */
    enforceLimit(maxNodes: number): number;
    /** 归档节点数（status 显示）。 */
    archivedCount(): number;
}
