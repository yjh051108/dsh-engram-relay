/**
 * EngramRelay — 转接核心：大 engram 小 KV 的落地实现。
 *
 * 链路（全部挂在公开 seam 上，零核心改动）：
 *
 * 1. 请求前唤醒（读取）：`llm/stream` 旁路观察当前请求 → N-gram 哈希
 *    寻址外置 engram 表 → 门控打分 → 因果传播 → 超稀疏注入
 *    （systemPrompt 记忆段，预算默认 600 token）。
 *
 * 2. 回合后蒸馏（写入）：`agent/turn-stopping`（回合关闭边界）→ 从
 *    `agent.session.deriveMessages()` 提取最近回合文本 → <1B 模型蒸馏
 *    为 engram 条目写入外置表。**实时留底**：在官方 compact 折叠之前
 *    细节已进记忆表，折叠后仍可唤醒找回。
 *
 * 3. 与官方 compact 共存：不阻止、不替代官方折叠（它负责腾 KV，是
 *    成熟的有损总结式压缩）。engram 的职责在官方折叠之前完成——细节
 *    保真（可检索、带因果），官方负责空间（surface 替换）。
 */
import type { Context as CordisContext } from 'cordis';
import type LlmService from '@deepseek-ai/dsh-llm';
import type SystemPrompt from '@deepseek-ai/dsh-system-prompt';
import type ToolRegistry from '@deepseek-ai/dsh-tools';
import type CompactionEngine from '@deepseek-ai/dsh-compaction';
import { EngramStore } from './engram/store.js';
import { CausalGraph } from './engram/causal.js';
import { NgramHashAddressing } from './engram/hash.js';
import { EngramWakeEngine, type WakeViewer } from './engram/wake.js';
import { RelayModel } from './model/relay-model.js';
import type { EngramRelayConfig } from './types.js';
export interface EngramRelayDeps {
    llm: LlmService;
    systemPrompt: SystemPrompt;
    tools: ToolRegistry;
    compaction?: CompactionEngine;
}
/** 唤醒结果：本次请求注入的记忆痕迹（哈希命中 + 因果激活，超稀疏）。 */
export interface WakeResult {
    engrams: import('./engram/store.js').EngramNode[];
    reason: string;
    injectedTokens: number;
}
export declare class EngramRelay {
    private ctx;
    private config;
    readonly store: EngramStore;
    readonly graph: CausalGraph;
    readonly hasher: NgramHashAddressing;
    readonly wake: EngramWakeEngine;
    readonly model: RelayModel;
    /** 类脑激活缓存（B=ln(Σt^-d)，强化事件驱动；wake 阶段 3 接入排序）。 */
    readonly activation: import('./engram/activation.js').ActivationCache;
    /** 向量索引（int8 粗筛 + fp32 精筛双表；prefilter 候选来源）。 */
    readonly vectorIndex: import('./engram/vector-index.js').BruteForceIndex;
    /** τ 融合权重（v0.4：写入织网与检索召回共用同一加权——可逆可解释）。 */
    get fusionTau(): {
        sem: number;
        time: number;
        cause: number;
    };
    private disposers;
    constructor(ctx: CordisContext, config: EngramRelayConfig);
    /**
     * 向量预筛（prefilter 钩子）：查询向量 → int8 全量内积 top-50 → 候选 id。
     * 含懒补 ensure：新记忆未入向量表时差量 embed 补入；embedder 不可用返回 null（哈希兜底）。
     */
    private vectorPrefilter;
    /** 挂载所有 seam。 */
    install(): () => void;
    private renderMemorySection;
    /** 异步触发训练模型的原生回忆（由 llm/stream 旁路调用，缓存结果）。 */
    maybeRecall(query: string): Promise<void>;
    private lastRecallText;
    /** 回合后蒸馏：LLM 把最近回合内容提取为 engram（⏳待确认，用户确认后生效）。 */
    private maybeDistill;
    /** 蒸馏排查日志（写入图谱目录 distill-debug.log）。 */
    private debugLog;
    private lastConversationText;
    /** 最近一次模型调用的路由（llm/stream 拦截时捕获；LLM 蒸馏复用）。 */
    private lastLlmRoute;
    /** 当前会话 id（工具写入时归属；会话结束清理用）。 */
    currentSessionId: string | null;
    /** 当前回合号（工具写入时归属）。 */
    lastTurnAt: number;
    /** 当前工作目录（分层准入：project 层按 cwd 过滤；turn-stopping 持续追踪）。 */
    currentCwd: string | null;
    /**
     * 供工具使用的唤醒查询入口。
     * @param viewer - 查看者视角（分层准入：{ sessionId, cwd }）。
     * @param layer - 可选层过滤（逗号分隔如 'global,project'；缺省不过滤，
     *   由 viewer 准入决定可见层）。
     */
    recall(query: string, limit?: number, viewer?: WakeViewer, layer?: string): Promise<WakeResult>;
    status(): Promise<Record<string, unknown>>;
}
