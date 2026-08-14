/**
 * RelayModel — 转接模型门面。
 *
 * 双轨：
 *  - 语义轨（v3 核心）：bge 专用嵌入模型（sentence-transformers）对
 *    hash 粗筛候选做余弦精排（混合检索：确定性寻址 + 语义重排）；
 *  - 蒸馏轨（遗留）：原 0.6B 魔改模型（Engram 条件记忆 × DSA 路由）
 *    的蒸馏/打分/原生回忆。模型目录未配置/缺失时全部优雅返回 null。
 *
 * 模型不可用（Python 缺失/模型未配置/服务崩溃）时自动降级：
 *  蒸馏 → 跳过；打分 → 重要度；记忆写入 → 无操作。插件始终可用。
 */
import type { Context as CordisContext } from 'cordis';
import type { EngramRelayConfig } from '../types.js';
import { PythonEngramClient } from './python-client.js';
import { SemanticScorer } from '../engram/semantic-scorer.js';
import type { EngramStore, EngramNode } from '../engram/store.js';
import type { CausalGraph } from '../engram/causal.js';
export declare class RelayModel {
    private ctx;
    private config;
    readonly python: PythonEngramClient;
    /** v0.5 纯算法语义打分器（词汇 + 图通道，零模型——替代 embedding 精排）。 */
    readonly scorer: SemanticScorer;
    private loadError;
    constructor(ctx: CordisContext, config: EngramRelayConfig, store: EngramStore);
    /** 预热：启动 Python 服务并加载模型（失败不抛出，记录后降级）。 */
    warmup(): Promise<void>;
    private warmupDone;
    /** 蒸馏：把对话转为记忆节点写入存储（模型不可用时跳过）。 */
    distillTurn(store: EngramStore, graph: CausalGraph, conversation: string, sessionId: string, turn: number): Promise<EngramNode[]>;
    /**
     * 原始向量嵌入（向量缓存/检索用）：返回查询向量 + 候选向量（ONNX 优先，Python 回退）。
     */
    embedRaw(query: string, texts: string[]): Promise<{
        query_vec: number[];
        vectors: number[][];
    } | null>;
    /**
     * 语义打分（v0.5：纯算法 SemanticScorer——词汇 n-gram Jaccard + 词频
     * + 图语义传播 + PCA 共现谱分解，零 embedding 模型；确定性、可解释）。
     * 返回「候选 id → 融合分数 [0,1]」（0.6 阈值语义沿用；上层无需改动）。
     */
    embed(query: string, candidates: EngramNode[]): Promise<Map<string, number> | null>;
    /** 详细语义分（通道分解——查重/织网用 lexical 阈值，比融合分更稳）。 */
    semanticScores(query: string, candidates: EngramNode[]): Map<string, {
        score: number;
        lexical: number;
        graph: number;
        cooc: number;
    }>;
    private cosineScores;
    /** 门控打分（遗留 0.6B 轨；模型不可用时返回空 Map（上层降级重要度）。 */
    score(query: string, candidates: EngramNode[]): Promise<Map<string, number>>;
    /**
     * 原生回忆：让训练好的记忆模型直接生成答案（forward 自动融合记忆表）。
     * 这是「回忆是模型行为」的对外接口——主模型转接层把回忆结果注入上下文。
     * 模型不可用/未训练时返回 null（调用方降级为纯 engram 文本注入）。
     */
    recall(query: string, maxNewTokens?: number): Promise<string | null>;
    describe(): Promise<Record<string, unknown>>;
    stop(): void;
}
