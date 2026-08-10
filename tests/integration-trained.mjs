/**
 * 集成验证：Node PythonEngramClient → 训练好的记忆模型 → 回忆链路。
 *
 * 模拟插件的真实调用序列（与 relay-model 一致）：
 *   load(checkpoint) → write_memory(新知识) → generate(提问) → 验证回忆
 *
 * 运行：node tests/integration-trained.mjs
 */

const CHECKPOINT = process.env.ENGRAM_CHECKPOINT
  ?? 'F:/dsh/dsh-engram-relay/python/checkpoints/engram-trained/engram.pt'
const MODEL = process.env.ENGRAM_MODEL
  ?? 'F:/dsh/engram-trial/qwen3-model'

async function main() {
  console.log('=== 集成验证：训练好的原生 engram 记忆模型 ===')
  const { PythonEngramClient } = await import('../lib/model/python-client.js')
  const client = new PythonEngramClient('python', MODEL, CHECKPOINT)

  // 1. load（带 checkpoint）
  const load = await client.load()
  if (!load) throw new Error('load 失败（Python 服务不可用？）')
  console.log('✓ load:', JSON.stringify(load))
  if (!load.lora) console.log('⚠ LoRA 未加载（回忆能力受限）')

  // 2. 写入新知识（训练后动态记忆）
  const wm = await client.writeMemory([{ text: '项目部署端口是 8080' }])
  console.log('✓ write_memory:', JSON.stringify(wm))

  // 3. 蒸馏能力
  const dist = await client.distill('用户说：CI 跑在 Jenkins 上，触发条件是 push 到 main。')
  console.log('✓ distill parsed:', JSON.stringify(dist?.parsed))

  // 4. 回忆查询（模型从记忆表回忆，非提示词注入）
  const gen = await client.generate('部署端口是多少？', 12, 0)
  console.log('✓ 回忆查询「部署端口是多少？」→', JSON.stringify(gen?.text))
  const hit = gen?.text?.includes('8080')
  console.log(hit ? '✓ 模型回忆出 8080' : '⚠ 未直接命中 8080（可能答出候选列表）')

  client.stop()
  console.log(hit ? '=== 集成验证 PASS（模型原生回忆） ===' : '=== 集成验证部分通过（加载+写入正常，回忆需更多训练） ===')
  process.exitCode = hit ? 0 : 1
}

main().catch((e) => { console.error('FAIL:', e.message); process.exitCode = 1 })
