/** 共享配置类型：由插件 Config（schemastery）填充，供各模块消费。 */

export interface EngramRelayConfig {
  modelId: string
  dtype: string
  storeDir: string
  injectBudgetTokens: number
  maxWakePerTurn: number
  distillEveryTurns: number
  enabled: boolean
  /** Python 解释器路径（spawn 转接服务用）。 */
  pythonPath: string
  /** Python 模型服务超时（预热等）。 */
  pythonTimeoutMs: number
  /** 训练好的原生 engram checkpoint（engram.pt 路径；空 = 未训练随机表）。 */
  checkpoint: string
  /** bge 嵌入模型目录（本地路径；空 = 服务端 ENGRAM_EMBED_MODEL 或禁用语义精排）。 */
  embedModel: string
  /**
   * 蒸馏产物是否需用户确认才生效。
   * true = 写 pending（⏳，不参与检索/唤醒，确认后生效）；
   * false（默认）= 无确认模式，蒸馏直接写 confirmed，立即参与检索——
   * Obsidian 式开箱即用，不用每回合确认。
   */
  distillRequireConfirm: boolean
  /** 唤醒语义阈值：bge 余弦相似度下限（低于此值不注入，宁缺毋滥）。 */
  semanticMinScore: number
  /** 时序 recency 加权强度（排序乘法项 1+w·e^(-Δturn/20)；仿真显示方向需
   *  数据驱动——0 关闭，负数反转"旧更相关"；默认 0.25 保守）。 */
  recencyWeight: number
  /** 唤醒采样日志：记录查询/候选分数/注入选择到 storeDir/wake-samples.jsonl
   *  （实战样本积累，供融合权重离线拟合；默认开）。 */
  wakeSampleLog: boolean
  /** τ 加法融合权重（建模 §3.1/§3.5）：排序分数 = τ_sem·z(语义) +
   *  τ_time·z(激活) + τ_cause·因果可达。默认 [1,0,0] = 纯语义（与旧行为
   *  等价）；fit-tau.mjs 拟合后更新生效（样本驱动调参闭环）。 */
  tauSem: number
  tauTime: number
  tauCause: number
  /** 硬上限（v0.5）：主库节点数超过该值触发归档淘汰（superseded → dormant
   *  → 低激活；归档到 archived.jsonl 可恢复）。0 = 无限（默认 10000）。 */
  maxNodes: number
  /** 教训通道阈值（v0.6）：tags 含「教训:」的节点在自动唤醒时用更低阈值
   * （默认 0.42，低于主阈值 ≈0.50）独立占席——同类操作时踩坑提醒必达；
   * 0 = 关闭教训通道。 */
  lessonMinScore: number
  /** 教训通道注入预算（独立于 injectBudgetTokens，教训提醒不挤占普通记忆）。 */
  lessonBudgetTokens: number
}
