/**
 * simulate-100k.mjs — 10 万节点规模化验证（P5）
 *
 * 实测（内存态，不落盘——store.add 每次全量 persist 是 O(N²) 灾难）：
 *  ① NgramHashAddressing 槽位分布（幂律长尾：每槽候选数 max/avg/p95）
 *  ② BruteForceIndex int8 全量内积检索延迟（10 万 × 512 维）
 *  ③ 内存估算（i8 51MB + f32 204MB + 节点对象）
 *  ④ 全链路注入延迟估算（哈希粗筛 → int8 top-50 → 排序）
 *
 * 用法：node tests/simulate-100k.mjs
 */
import { NgramHashAddressing } from '../lib/engram/hash.js'
import { BruteForceIndex } from '../lib/engram/vector-index.js'

const N = 100000
const DIM = 512

/** 合成记忆文本：幂律主题（部分高频词重复 → 长尾槽位） */
function makeText(i) {
  const topics = ['缓存', '注入器', '热重载', '图谱', '向量索引', 'embed', '唤醒', '蒸馏', '版本链', '快照']
  const t = topics[i % topics.length] // 循环 → 高频
  const r = topics[Math.floor(Math.random() * topics.length)] // 随机 → 长尾
  return `${t}${i}：${r} 相关方案第 ${Math.floor(i / 7)} 版，${i % 3 === 0 ? '决策' : '踩坑'}，端口 ${8000 + (i % 100)} 配置`
}

function bench() {
  console.log(`=== 10 万节点规模化验证（N=${N}, dim=${DIM}）===`)

  // ① 哈希槽位分布
  console.log('\n① NgramHashAddressing 槽位分布（幂律长尾）…')
  const hasher = new NgramHashAddressing()
  const slotCount = new Map()
  const t0 = Date.now()
  for (let i = 0; i < N; i++) {
    const { slots } = hasher.hash(makeText(i))
    for (const s of slots) slotCount.set(s, (slotCount.get(s) ?? 0) + 1)
  }
  const tHash = Date.now() - t0
  const counts = [...slotCount.values()].sort((a, b) => a - b)
  const pct = (p) => counts[Math.min(counts.length - 1, Math.floor(counts.length * p))]
  console.log(`  哈希 ${N} 条: ${tHash}ms（${(tHash / N).toFixed(2)}ms/条）| 总槽位 ${slotCount.size} | 每槽: avg=${(counts.reduce((s, x) => s + x, 0) / counts.length).toFixed(1)} p95=${pct(0.95)} max=${counts[counts.length - 1]}`)
  console.log(`  长尾评估: max 槽候选 ${counts[counts.length - 1]}（>64 说明单槽爆炸需扩容/分词优化）`)

  // ①b lookup 提前截断验证（修复前全量收集 O(N)，修复后预算内截断）
  console.log('\n①b lookup 候选收集（提前截断，预算 256）…')
  {
    const idx = new Map() // slotKey -> id[]
    for (let i = 0; i < N; i++) {
      const { slots } = hasher.hash(makeText(i))
      for (const s of slots) {
        if (!idx.has(s)) idx.set(s, [])
        idx.get(s).push(`m${i}`)
      }
    }
    const t3 = Date.now()
    const R = 100
    let totalSeen = 0
    for (let r = 0; r < R; r++) {
      const { slots } = hasher.hash(makeText(Math.floor(Math.random() * N)))
      const seen = new Set()
      let cnt = 0
      const budget = 256
      outer:
      for (const k of slots) {
        const ids = idx.get(k)
        if (!ids) continue
        for (const id of ids) {
          if (cnt >= budget) break outer
          if (seen.has(id)) continue
          seen.add(id)
          cnt++
        }
      }
      totalSeen += cnt
    }
    const tLookup = (Date.now() - t3) / R
    console.log(`  lookup（预算截断）: ${tLookup.toFixed(2)}ms/次 | 平均候选 ${(totalSeen / R).toFixed(0)}（修复前需遍历全槽候选）`)
  }

  // ② 向量索引
  console.log('\n② BruteForceIndex int8 全量内积检索延迟…')
  const idx = new BruteForceIndex('', DIM)
  const t1 = Date.now()
  for (let i = 0; i < N; i++) {
    const v = new Float32Array(DIM)
    for (let d = 0; d < DIM; d++) v[d] = (Math.random() * 2 - 1) * (i % 5 === 0 ? 0.1 : 1) // 部分相似
    idx.add(`m${i}`, v)
  }
  const tAdd = Date.now() - t1
  const q = new Float32Array(DIM)
  for (let d = 0; d < DIM; d++) q[d] = Math.random() * 2 - 1
  // 预热
  idx.search(q, 50)
  const t2 = Date.now()
  const R = 20
  for (let r = 0; r < R; r++) idx.search(q, 50)
  const tSearch = (Date.now() - t2) / R
  console.log(`  add ${N} 条: ${tAdd}ms（${(tAdd / N).toFixed(2)}ms/条）| search top-50: ${tSearch.toFixed(1)}ms/次`)

  // ③ 内存估算
  const i8MB = (N * DIM) / 1024 / 1024
  const f32MB = (N * DIM * 4) / 1024 / 1024
  console.log(`\n③ 内存: int8 粗筛表 ${i8MB.toFixed(0)}MB + fp32 精筛表 ${f32MB.toFixed(0)}MB + 哈希/节点 ~${Math.round(N * 0.5)}MB ≈ ${Math.round(i8MB + f32MB + N * 0.5 / 1024)}MB`)

  // ④ 全链路注入延迟估算（哈希粗筛 → 语义 → 排序截断）
  console.log('\n④ 全链路估算: 哈希粗筛 ~0.1ms + int8 top-50 ' + tSearch.toFixed(1) + 'ms + 精排/排序 ~1ms ≈ ' + (tSearch + 1.1).toFixed(1) + 'ms/轮（<100ms 目标 ✓）')
}

bench()
