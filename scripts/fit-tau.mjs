/**
 * fit-tau.mjs — 融合权重拟合（P4：样本驱动调参闭环）
 *
 * 数据：wake-samples.jsonl（每次唤醒的 query/候选 cos·act·importance/picked）
 *       + engrams.jsonl（reinforces 反馈）
 * 特征：f_sem = cos（样本内 z-score）；f_time = 激活（z-score）；
 *       f_cause = 因果图 1 跳可达（0/1）
 * 弱监督：picked 为系统注入选择（自举）；后续 reinforces 增长 = 弱正反馈
 *       （被 open/再命中）→ 该样本权重 2，无反馈权重 1
 * 目标：网格扫 (τ_sem, τ_time, τ_cause) 最大化 picked 的 NDCG@3
 * 输出：τ 建议 + 与现配置对比 + 样本统计；样本 < 200 时仅报告不拟合。
 *
 * 用法：node scripts/fit-tau.mjs [--synthetic]
 */
import { readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'

const STORE_DIR = join(homedir(), '.dsh/engram-relay')
const SAMPLES = join(STORE_DIR, 'wake-samples.jsonl')
const ENGRAMS = join(STORE_DIR, 'engrams.jsonl')

function loadSamples() {
  const out = []
  for (const line of readFileSync(SAMPLES, 'utf8').split('\n')) {
    const t = line.trim()
    if (!t) continue
    try { out.push(JSON.parse(t)) } catch { /* 跳过坏行 */ }
  }
  return out
}

function loadNodes() {
  const out = new Map()
  for (const line of readFileSync(ENGRAMS, 'utf8').split('\n')) {
    const t = line.trim()
    if (!t) continue
    try {
      const n = JSON.parse(t)
      if (n.status !== 'pending') out.set(n.id, n)
    } catch { /* 跳过坏行 */ }
  }
  return out
}

/** 样本内 z-score */
function zscore(values) {
  const mean = values.reduce((s, x) => s + x, 0) / values.length
  const std = Math.sqrt(values.reduce((s, x) => s + (x - mean) ** 2, 0) / values.length) || 1
  return values.map((x) => (x - mean) / std)
}

/** NDCG@k：注入选择的排序质量（分数降序下 picked 的累积增益） */
function ndcgAt(scores, picked, k = 3) {
  const ranked = [...scores.entries()].sort((a, b) => b[1] - a[1]).slice(0, k)
  let dcg = 0
  let idcg = 0
  ranked.forEach(([id], i) => {
    const rel = picked.includes(id) ? 1 : 0
    dcg += rel / Math.log2(i + 2)
  })
  for (let i = 0; i < picked.length; i++) idcg += 1 / Math.log2(i + 2)
  return idcg > 0 ? dcg / idcg : 0
}

function fit(samples, nodes, label) {
  // 构建特征样本
  const feats = []
  for (const s of samples) {
    if (!Array.isArray(s.candidates) || s.candidates.length < 2) continue
    const picked = s.picked ?? []
    const ids = s.candidates.map((c) => c.id)
    const cos = s.candidates.map((c) => (typeof c.cos === 'number' ? c.cos : 0))
    const act = s.candidates.map((c) => (typeof c.act === 'number' ? c.act : 0))
    const semZ = zscore(cos)
    const timeZ = zscore(act)
    // 因果 1 跳可达（真实图）：候选 id → 邻居是否也在候选集（弱信号）
    const causeReach = ids.map((id) => {
      const n = nodes.get(id)
      if (!n) return 0
      const nb = [...(n.causes ?? []), ...(n.effects ?? [])]
      return nb.some((x) => ids.includes(x)) ? 1 : 0
    })
    // 弱反馈：picked 中样本时间后有 reinforces 增长的节点数
    const feedback = picked.filter((id) => {
      const n = nodes.get(id)
      if (!n) return false
      const after = (n.reinforces ?? []).filter((t) => t > s.time)
      return after.length > 0
    }).length
    const weight = feedback > 0 ? 2 : 1
    feats.push({ ids, picked, semZ, timeZ, causeReach, weight })
  }
  if (feats.length < 200) {
    console.log(`[${label}] 样本不足（${feats.length} < 200）——仅统计，不拟合 τ`)
    return null
  }
  // 网格扫
  let best = { ndcg: 0, tau: [1, 0, 0] }
  for (const ts of [0, 0.25, 0.5, 1, 1.5, 2]) {
    for (const tt of [-1, -0.5, 0, 0.25, 0.5, 1]) {
      for (const tc of [0, 0.25, 0.5, 1]) {
        let sum = 0, wsum = 0
        for (const f of feats) {
          const scores = new Map(f.ids.map((id, i) => [id, ts * f.semZ[i] + tt * f.timeZ[i] + tc * f.causeReach[i]]))
          sum += f.weight * ndcgAt(scores, f.picked)
          wsum += f.weight
        }
        const ndcg = sum / wsum
        if (ndcg > best.ndcg) best = { ndcg, tau: [ts, tt, tc] }
      }
    }
  }
  // 现配置对照（sem=1, time=0, cause=0）
  let cur = 0, wcur = 0
  for (const f of feats) {
    const scores = new Map(f.ids.map((id, i) => [id, f.semZ[i]]))
    cur += f.weight * ndcgAt(scores, f.picked)
    wcur += f.weight
  }
  console.log(`[${label}] 样本 ${feats.length} | 现配置 NDCG@3 = ${(cur / wcur).toFixed(4)}`)
  console.log(`[${label}] 最优 (τ_sem=${best.tau[0]}, τ_time=${best.tau[1]}, τ_cause=${best.tau[2]}) NDCG@3 = ${best.ndcg.toFixed(4)}（提升 ${(((best.ndcg - cur / wcur) / (cur / wcur)) * 100).toFixed(1)}%）`)
  console.log(`[${label}] 建议配置：recencyWeight≈${(best.tau[1] * 0.25).toFixed(2)}（τ_time 映射）`)
  return best
}

// ---- 合成验证：已知 τ 生成样本 → 拟合还原 ----
function synthetic() {
  const rng = (seed) => () => {
    seed = (seed * 1103515245 + 12345) % 2147483648
    return seed / 2147483648
  }
  const r = rng(42)
  const nodes = new Map()
  for (let i = 0; i < 40; i++) {
    nodes.set(`n${i}`, {
      id: `n${i}`, causes: i > 0 ? [`n${i - 1}`] : [], effects: i < 39 ? [`n${i + 1}`] : [],
      reinforces: [Date.now() - Math.floor(r() * 86400000)],
    })
  }
  const TRUE_TAU = [0.8, 0.4, 0.3] // sem, time, cause
  const samples = []
  for (let s = 0; s < 300; s++) {
    const cands = []
    for (let i = 0; i < 12; i++) {
      const id = `n${Math.floor(r() * 40)}`
      if (cands.some((c) => c.id === id)) continue
      cands.push({ id, cos: r(), act: r() })
    }
    const ids = cands.map((c) => c.id)
    const semZ = zscore(cands.map((c) => c.cos))
    const timeZ = zscore(cands.map((c) => c.act))
    const causeReach = ids.map((id) => {
      const n = nodes.get(id)
      return [...(n.causes ?? []), ...(n.effects ?? [])].some((x) => ids.includes(x)) ? 1 : 0
    })
    // 按真 τ 排序取 top-3 为 picked（模拟系统按正确权重选择）
    const scored = ids.map((id, i) => [id, TRUE_TAU[0] * semZ[i] + TRUE_TAU[1] * timeZ[i] + TRUE_TAU[2] * causeReach[i]])
    const picked = scored.sort((a, b) => b[1] - a[1]).slice(0, 3).map(([id]) => id)
    samples.push({ candidates: cands, picked, time: Date.now() })
  }
  const best = fit(samples, nodes, 'synthetic')
  if (best) {
    console.log(`[synthetic] 真值 τ = [${TRUE_TAU.join(', ')}]，拟合 τ = [${best.tau.join(', ')}]（scale 无关，看相对比例）`)
  }
}

const isSyn = process.argv.includes('--synthetic')
if (isSyn) {
  synthetic()
} else {
  const samples = loadSamples()
  const nodes = loadNodes()
  console.log(`样本 ${samples.length} 条 | 节点 ${nodes.size} 个`)
  fit(samples, nodes, 'real')
}
