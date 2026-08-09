/** 共享配置类型：由插件 Config（schemastery）填充，供各模块消费。 */

export interface EngramRelayConfig {
  modelId: string
  dtype: string
  storeDir: string
  injectBudgetTokens: number
  maxWakePerTurn: number
  distillEveryTurns: number
  enabled: boolean
  /** Python 解释器路径（spawn 魔改模型服务用）。 */
  pythonPath: string
  /** Python 模型服务超时（预热等）。 */
  pythonTimeoutMs: number
}
