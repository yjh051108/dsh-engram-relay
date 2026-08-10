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
}
