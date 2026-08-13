/**
 * 三维度索引准确率 + 融合仿真（simulate-fusion.mjs）
 *
 * 语义边界（与设计文档一致）：
 *  - 知识通道 f_sem  ：内容相关性（embedding 余弦）——"讲的是不是这个知识"
 *  - 因果通道 f_cause：结构关系（causes/effects 图上传播）——"理解链路上是否连着"
 *  - 时序通道 f_time  ：时效价值（recency / 强化历史 / 幂律激活）——"现在值不值得想起"
 *
 * 数据：60 条真实记忆（~/.dsh/engram-relay/engrams.jsonl，含因果边/时序信号）
 *       + 真实 bge-small-zh（TS ONNX，包内模型，零 Python）
 * 查询：60 条真实记忆标题（自检索标定）+ 本会话真实用户消息
 * GT  ：自身 + 因果直接邻居 + 共享主题关键词（粗标注，诚实声明：自检索偏乐观，
 *       维度间相对比较有效）
 *
 * 输出：① 三个维度各自的索引准确率（Recall@k / 补盲增益 / 判别力 AUC）
 *       ② 融合权重网格搜索（τ_sem/τ_cause/τ_time）→ 对比纯语义基线 top-3 注入
 *
 * 用法：node --experimental-strip-types tests/simulate-fusion.mjs
 */
import { readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { embedWithOnnx } from '../src/model/onnx-embedder.ts'

// ---------- 1. 数据 ----------
const STORE = join(homedir(), '.dsh/engram-relay/engrams.jsonl')
const MODEL_DIR = decodeURIComponent(new URL('../model/bge-small-zh/', import.meta.url).pathname)
  .replace(/^\/([A-Za-z]:)/, '$1').replace(/\//g, '\\')

/** 主题关键词（GT 粗标注，沿用 simulate_real.py 的诚实声明） */
const TOPIC_KW = {
  '热重载': ['热重载', '重载', 'hmr', 'patch'],
  '注入': ['注入', 'inject', 'super-injector', '模组', 'junction', '挂载'],
  '缓存': ['缓存', '命中', 'cache', 'token', '预算'],
  '记忆': ['记忆', 'engram', '图谱', '唤醒', '蒸馏', 'evolve'],
  '浏览器': ['浏览器', 'browser', '面板', 'sidebar', 'sider'],
  '卸载': ['卸载', 'uninject', '卸载器'],
  '迁移': ['迁移', '打包', 'restore', '新电脑', 'zip'],
  '开源': ['开源', 'public', 'github', '仓库'],
  'solo': ['solo', 'code-server', 'vscode', '侧边栏'],
  '布局': ['布局', '图谱', '聚团', '力导向', '缩放', '节点', '圆形', '边'],
  '向量': ['向量', 'embed', 'bge', '索引', 'int8', 'fp32', '检索'],
}

function topicOf(text) {
  const hits = new Set()
  for (const [topic, kws] of Object.entries(TOPIC_KW)) {
    for (const kw of kws) if (text.includes(kw)) hits.add(topic)
  }
  return hits
}

// 真实用户消息（会话 session-12ca2121 导出，25 条——fork 插件→适配→优化的真实提问）
const USER_QUERIES = [
  '我的ghcli上有两大宝藏，一个superinjestor，一个engram memory，都是你的插件，我要你fork到本地分别两个文件夹，作为开发环境优化这两个插件',
  '反正是个超级注入器，你就找我的名义创建的东西，我GitHub上东西不多的！',
  '不是，是超级注入器！我放到external里面的，你能不能把所有名字列出来看',
  '是github一个叫做external的云仓库！',
  '你在干嘛？',
  '源码都在对吧？',
  '那就跑rc5适配',
  '先装injector然后调试，调试完就方便开发维护其它插件了',
  '都说了先把注入器完善！',
  '非要重启吗，注射器不就是用来支持插件热插拔的吗？',
  '开源不是打包了模型吗？为啥模型没了？',
  '说错了仓库不是打包了模型吗？为啥模型没了？',
  '但是为什么记忆插件的webui没了',
  '如果按照你的想法某个事情没有发生，要么你改让你觉得该这么做的提示词要么你让插件能按照你的想法实现',
  '不对，不是engram插件的问题，我再说一遍，注入器是为了让你热插拔和能够自主迭代插件的，懂我意思吗？',
  '很好，我问你，注射器还有问题吗？',
  '需要给这个工具的提示词注入加一句“注入器是为了让你热插拔和能够自主迭代插件的，若他无法实现此目的，优先修复注入器”',
  '提示词怎么注入的？',
  '你觉得应该这样吗？',
  '在哪里注入？',
  '告诉我，为了缓存你应该怎么注入提示词，多思考',
  '精简不需要，去雷就行，动态也要动态到尾，静态就静态到头明白吗？',
  '你想怎么精简？',
  '落地，落地完成你体验一下体感',
  '有没有歧义，你能不能做什么都知道是为了什么目的有什么预期结果，准确率如何？之前你都没法装插件的webui这件事她有没有负责？',
]

function loadMemories() {
  const out = []
  for (const line of readFileSync(STORE, 'utf8').split('\n')) {
    const t = line.trim()
    if (!t) continue
    try {
      const n = JSON.parse(t)
      if (n.status === 'pending' || !n.title) continue
      out.push({
        id: n.id, title: n.title, summary: n.summary ?? '', content: n.content ?? '',
        causes: n.causes ?? [], effects: n.effects ?? [], links: n.links ?? [],
        createdAt: n.createdAt ?? 0, hits: n.hits ?? 0, reinforces: n.reinforces ?? [],
      })
    } catch { /* 单条损坏跳过 */ }
  }
  return out
}

// ---------- 2. 时序特征 ----------
/** 幂律激活（ACT-R）：A = ln(Σ t_k^-d)，t 用小时 */
function activation(node, now) {
  const d = 0.5
  let sum = 0
  for (const t of node.reinforces) {
    const hours = Math.max(0.5, (now - t) / 3600000)
    sum += Math.pow(hours, -d)
  }
  if (sum === 0) sum = Math.pow(Math.max(0.5, (now - node.createdAt) / 3600000), -d)
  return Math.log(sum + 1e-9)
}

// ---------- 3. 维度评估 ----------
function recallAt(ranked, gt, k) {
  const top = ranked.slice(0, k)
  const hit = gt.filter((g) => top.includes(g)).length
  return hit / Math.max(1, gt.length)
}

/** Set 交集（⚠️ 不能用位运算 &，那对对象返回 0） */
function intersect(a, b) {
  const out = new Set()
  for (const x of a) if (b.has(x)) out.add(x)
  return out
}

async function main() {
  const now = Date.now()
  const memories = loadMemories()
  console.log(`记忆 ${memories.length} 条（store: ${STORE}）`)
  const memTexts = memories.map((m) => `${m.title}：${m.summary.slice(0, 200)}`)
  const memTopics = memTexts.map(topicOf)
  const byId = new Map(memories.map((m) => [m.id, m]))

  // 查询集：记忆标题 + 真实用户消息
  const queries = [
    ...memories.map((m, i) => ({ text: m.title, kind: 'self', selfIdx: i })),
    ...USER_QUERIES.map((q) => ({ text: q, kind: 'user', selfIdx: -1 })),
  ]
  console.log(`查询 ${queries.length} 条（自检索 ${memories.length} + 用户消息 ${USER_QUERIES.length}）`)

  // GT：自身 + 因果直接邻居 + 共享主题（title 查询）；共享主题（用户查询）
  const gtFor = (q) => {
    const set = new Set()
    if (q.selfIdx >= 0) {
      const m = memories[q.selfIdx]
      set.add(m.id)
      for (const c of [...m.causes, ...m.effects]) if (byId.has(c)) set.add(c)
    }
    const qTopic = topicOf(q.text)
    memories.forEach((m, i) => {
      if (qTopic.size > 0 && intersect(qTopic, memTopics[i]).size > 0) set.add(m.id)
    })
    return [...set]
  }
  const gts = queries.map(gtFor)

  // ---------- 嵌入（真实 bge，TS ONNX）----------
  console.log('嵌入记忆向量…')
  const memRes = await embedWithOnnx(memTexts, queries[0].text, MODEL_DIR)
  if (!memRes) { console.error('embed 失败'); return }
  const memVecs = memRes.vectors // [N][512]
  console.log(`记忆向量 ${memVecs.length} × ${memVecs[0]?.length ?? 0} 维`)

  const qVecs = []
  for (const q of queries) {
    const r = await embedWithOnnx([], q.text, MODEL_DIR)
    qVecs.push(r?.query_vec ?? null)
  }
  const cos = (a, b) => {
    let s = 0
    for (let i = 0; i < a.length; i++) s += a[i] * b[i]
    return s
  }

  // ---------- 4. 三个维度 ----------
  const semScores = [] // [q][memIdx]
  for (let qi = 0; qi < queries.length; qi++) {
    const qv = qVecs[qi]
    semScores.push(memVecs.map((v) => (qv ? cos(qv, v) : 0)))
  }

  // 知识通道：Recall@k
  console.log('\n=== ① 知识通道（纯语义 cosine 排序）===')
  for (const k of [1, 3, 5, 10]) {
    const r = queries.map((_, qi) => {
      const ranked = [...memories.keys()].sort((a, b) => semScores[qi][b] - semScores[qi][a])
      return recallAt(ranked.map((i) => memories[i].id), gts[qi], k)
    })
    const mean = r.reduce((s, x) => s + x, 0) / r.length
    console.log(`  Recall@${k}: ${(mean * 100).toFixed(1)}%`)
  }

  // 因果通道：从语义 top-50 候选做 2 跳 BFS 扩展 → 补盲增益 + 精确率
  console.log('\n=== ② 因果通道（causes/effects 图传播）===')
  {
    let extraHits = 0, expandedTotal = 0, semTopHit = 0, causeReachGT = 0, gtTotal = 0
    for (let qi = 0; qi < queries.length; qi++) {
      const ranked = [...memories.keys()].sort((a, b) => semScores[qi][b] - semScores[qi][a])
      const semTop = new Set(ranked.slice(0, 50).map((i) => memories[i].id))
      const gt = new Set(gts[qi])
      gtTotal += gt.size
      semTopHit += [...gt].filter((g) => semTop.has(g)).length
      // BFS 2 跳
      const reached = new Set(semTop)
      const queue = [...semTop]
      for (let depth = 0; depth < 2 && queue.length > 0; depth++) {
        const next = []
        for (const id of queue) {
          const m = byId.get(id)
          if (!m) continue
          for (const c of [...m.causes, ...m.effects]) {
            if (byId.has(c) && !reached.has(c)) { reached.add(c); next.push(c) }
          }
        }
        queue.length = 0; queue.push(...next)
      }
      const newHits = [...gt].filter((g) => reached.has(g) && !semTop.has(g))
      extraHits += newHits.length
      causeReachGT += [...gt].filter((g) => reached.has(g)).length
      expandedTotal += reached.size - semTop.size
    }
    console.log(`  语义 top-50 覆盖 GT: ${semTopHit}/${gtTotal} (${(semTopHit / gtTotal * 100).toFixed(1)}%)`)
    console.log(`  因果 2 跳补盲: +${extraHits} 条 GT（语义漏掉、因果捞回）`)
    console.log(`  因果可达总 GT: ${causeReachGT}/${gtTotal} (${(causeReachGT / gtTotal * 100).toFixed(1)}%)`)
    console.log(`  传播扩展精确率（扩展节点中 GT 占比）: ${(extraHits / Math.max(1, expandedTotal) * 100).toFixed(1)}%`)
  }

  // 时序通道：对每个查询的候选集，相关 vs 无关记忆的时序特征判别力（条件 AUC）
  console.log('\n=== ③ 时序通道（recency / 强化 / 激活）===')
  {
    // 只用用户查询（自检索 GT 含自身，流行度无区分度）
    const uIdx = queries.map((q, i) => (q.kind === 'user' ? i : -1)).filter((i) => i >= 0)
    const feats = memories.map((m, i) => ({
      id: m.id, idx: i,
      recencyHours: (now - m.createdAt) / 3600000,
      reinforces: m.reinforces.length,
      hits: m.hits,
      act: activation(m, now),
    }))
    const auc = (pos, neg) => {
      if (pos.length === 0 || neg.length === 0) return NaN
      let score = 0
      for (const p of pos) for (const n of neg) if (p > n) score++
      return score / (pos.length * neg.length)
    }
    // 条件判别：同一查询内，GT 候选 vs 非 GT 候选的特征分离（只对有 GT 的用户查询）
    for (const [name, f] of [['recency(负=新)', (x) => -x.recencyHours], ['强化次数', (x) => x.reinforces], ['hits', (x) => x.hits], ['激活 A', (x) => x.act]]) {
      const aucs = []
      let pairs = 0
      for (const qi of uIdx) {
        const gt = new Set(gts[qi])
        if (gt.size === 0) continue
        // 语义候选池（top-50）内的相关 vs 无关
        const ranked = [...memories.keys()].sort((a, b) => semScores[qi][b] - semScores[qi][a]).slice(0, 50)
        const pos = ranked.filter((i) => gt.has(memories[i].id)).map((i) => f(feats[i]))
        const neg = ranked.filter((i) => !gt.has(memories[i].id)).map((i) => f(feats[i]))
        const a = auc(pos, neg)
        if (!Number.isNaN(a)) { aucs.push(a); pairs += pos.length * neg.length }
      }
      if (aucs.length > 0) {
        const mean = aucs.reduce((s, x) => s + x, 0) / aucs.length
        console.log(`  ${name}: 条件 AUC(候选池内相关 vs 无关) = ${mean.toFixed(3)}（${aucs.length} 查询 × ${pairs} 对）`)
      }
    }
    console.log(`  有强化历史的记忆: ${feats.filter((x) => x.reinforces > 0).length}/${feats.length}`)
  }

  // ---------- 5. 融合（logistic 网格扫）----------
  console.log('\n=== ④ 融合：τ_sem·f_sem + τ_cause·f_cause + τ_time·f_time ===')
  const N = memories.length
  // 特征矩阵
  const F = []
  for (let qi = 0; qi < queries.length; qi++) {
    const ranked = [...memories.keys()].sort((a, b) => semScores[qi][b] - semScores[qi][a])
    const semTop = new Set(ranked.slice(0, 50).map((i) => memories[i].id))
    // 因果得分：语义候选的 2 跳传播（先算好）
    const reached = new Set(semTop)
    const queue = [...semTop]
    for (let depth = 0; depth < 2 && queue.length > 0; depth++) {
      const next = []
      for (const id of queue) {
        const m = byId.get(id)
        if (!m) continue
        for (const c of [...m.causes, ...m.effects]) {
          if (byId.has(c) && !reached.has(c)) { reached.add(c); next.push(c) }
        }
      }
      queue.length = 0; queue.push(...next)
    }
    // 特征归一化（z-score，用全部记忆的均值和 std）
    const semMean = semScores[qi].reduce((s, x) => s + x, 0) / N
    const semStd = Math.sqrt(semScores[qi].reduce((s, x) => s + (x - semMean) ** 2, 0) / N) || 1
    const timeFeats = memories.map((m) => activation(m, now))
    const tMean = timeFeats.reduce((s, x) => s + x, 0) / N
    const tStd = Math.sqrt(timeFeats.reduce((s, x) => s + (x - tMean) ** 2, 0) / N) || 1
    F.push(memories.map((m, i) => ({
      sem: (semScores[qi][i] - semMean) / semStd,
      cause: reached.has(m.id) ? 1 : 0,
      time: (timeFeats[i] - tMean) / tStd,
    })))
  }

  // 注入质量：recall@3（注入的 top-3 中 GT 占比）+ 无 GT 查询的误注入条数
  const hitRate = (tauSem, tauCause, tauTime) => {
    const uIdx = queries.map((q, i) => (q.kind === 'user' ? i : -1)).filter((i) => i >= 0)
    let recSum = 0, hasGt = 0, misSum = 0, noGt = 0
    for (const qi of uIdx) {
      const gt = new Set(gts[qi])
      const scored = memories.map((m, i) => ({
        id: m.id, s: tauSem * F[qi][i].sem + tauCause * F[qi][i].cause + tauTime * F[qi][i].time,
      })).sort((a, b) => b.s - a.s).slice(0, 3)
      if (gt.size > 0) {
        hasGt++
        recSum += scored.filter((x) => gt.has(x.id)).length / Math.min(3, gt.size)
      } else {
        noGt++
        misSum += scored.length
      }
    }
    return { rec: recSum / Math.max(1, hasGt), mis: misSum / Math.max(1, noGt * 3), hasGt, noGt }
  }

  // 基线：纯语义 top-3（z-score 阈值 0 ≈ cos 均值；生产用 cos≥0.42）
  const base = hitRate(1, 0, 0)
  console.log(`基线（纯语义 top-3，仅用户查询）: recall@3 = ${(base.rec * 100).toFixed(1)}%（${base.hasGt} 查询有 GT）| 无GT查询误注入 ${(base.mis * 100).toFixed(1)}%（${base.noGt} 条）`)

  // 网格扫（τ_time 允许负值——时序信号可能是"旧更相关"）
  let best = { rec: 0, tau: [0, 0, 0] }
  for (const ts of [0, 0.5, 1, 2]) {
    for (const tc of [0, 0.5, 1, 2]) {
      for (const tt of [-2, -1, -0.5, 0, 0.5, 1]) {
        const r = hitRate(ts, tc, tt)
        if (r.rec > best.rec) best = { rec: r.rec, mis: r.mis, tau: [ts, tc, tt] }
      }
    }
  }
  console.log(`融合最优 (τ_sem=${best.tau[0]}, τ_cause=${best.tau[1]}, τ_time=${best.tau[2]}): recall@3 = ${(best.rec * 100).toFixed(1)}% | 误注入 ${(best.mis !== undefined ? (best.mis * 100).toFixed(1) : '?')}%`)
  console.log(`增益: ${((best.rec - base.rec) * 100).toFixed(1)}pp`)

  // 两两组合（τ_time 用数据驱动方向）
  for (const [name, t] of [['语义+因果', [1, 1, 0]], ['语义+时序', [1, 0, -1]], ['因果+时序', [0, 1, -1]]]) {
    const r = hitRate(t[0], t[1], t[2])
    console.log(`  ${name}: recall@3 = ${(r.rec * 100).toFixed(1)}%`)
  }
}

main().catch((e) => { console.error(e); process.exit(1) })
