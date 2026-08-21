/** 共享配置类型：由插件 Config（schemastery）填充，供各模块消费。 */
/** 灵枢（Lingshu）白箱验证标注（融合）：唤醒的 engram 过 auto_verify 后的结果。 */
export type VerifyMark = {
    status: 'anchored' | 'partial' | 'unverified' | 'error';
    note?: string;
};
export interface EngramRelayConfig {
    modelId: string;
    dtype: string;
    storeDir: string;
    injectBudgetTokens: number;
    maxWakePerTurn: number;
    distillEveryTurns: number;
    gapDailyLimit: number;
    enabled: boolean;
    /** Python 解释器路径（spawn 转接服务用）。 */
    pythonPath: string;
    /** Python 模型服务超时（预热等）。 */
    pythonTimeoutMs: number;
    /** 训练好的原生 engram checkpoint（engram.pt 路径；空 = 未训练随机表）。 */
    checkpoint: string;
    /** bge 嵌入模型目录（本地路径；空 = 服务端 ENGRAM_EMBED_MODEL 或禁用语义精排）。 */
    embedModel: string;
    /**
     * 蒸馏产物是否需用户确认才生效。
     * true = 写 pending（⏳，不参与检索/唤醒，确认后生效）；
     * false（默认）= 无确认模式，蒸馏直接写 confirmed，立即参与检索——
     * Obsidian 式开箱即用，不用每回合确认。
     */
    distillRequireConfirm: boolean;
    /** 唤醒语义阈值：bge 余弦相似度下限（低于此值不注入，宁缺毋滥）。 */
    semanticMinScore: number;
    /** 融合：灵枢（Lingshu）白箱验证服务地址（如 http://127.0.0.1:18766）。
     *  非空时唤醒标注 ✓锚定/?图谱外 + 知识之书注入 + 写入闸门；空 = 关闭融合。 */
    lingshuVerifyUrl: string;
}
