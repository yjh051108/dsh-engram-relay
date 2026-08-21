#!/usr/bin/env node
/* 风识套装自检：装配后 30 秒确认与本地一致 */
const CLOUD = 'http://127.0.0.1:18766'
const WEB = 'http://127.0.0.1:3080'
const steps = []
async function check(name, fn) {
  try { const ok = await fn(); steps.push({ name, ok, detail: ok ? '✓' : '✗' }) }
  catch (e) { steps.push({ name, ok: false, detail: String(e).slice(0, 80) }) }
}
async function post(path, body) {
  const res = await fetch(CLOUD + path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: AbortSignal.timeout(5000) })
  return res.json()
}
const main = async () => {
  // 1) 灵枢服务健康（GET /dex/status）
  await check('灵枢服务', async () => {
    const res = await fetch(CLOUD + '/dex/status', { signal: AbortSignal.timeout(4000) })
    if (!res.ok) throw new Error('http ' + res.status)
    const s = await res.json()
    const n = s.total_entries ?? s.entries
    if (!n) throw new Error('status 无条目数')
    console.log('  灵枢条目:', n)
    return true
  })
  // 2) 卡库可用（respond）
  await check('知识出招', async () => {
    const r = await post('/dex/query', { op: 'respond', params: { condition: '信息差是什么', limit: 1 } })
    const hit = r.results?.[0]
    if (!hit || (hit.score ?? 0) < 0.02) throw new Error('respond 无命中')
    console.log('  出招:', hit.name, hit.score)
    return true
  })
  // 3) 验证闸门（add_card 端点在，verify 可走）
  await check('验证闸门', async () => {
    const r = await post('/dex/query', { op: 'auto_verify', params: { knowledge: '1+1=2', limit: 3, threshold: 0.5 } })
    if (!r.results?.judgment) throw new Error('verify 无判定')
    console.log('  判定:', r.results.judgment)
    return true
  })
  // 4) 记忆图谱 API（插件已装配的证据）
  await check('记忆图谱', async () => {
    const res = await fetch(WEB + '/engram-relay/api/graph', { signal: AbortSignal.timeout(4000) })
    const g = await res.json()
    if (!Array.isArray(g.nodes)) throw new Error('graph 无节点')
    console.log('  记忆节点:', g.total ?? g.nodes.length)
    return true
  })
  // 5) 检索端点
  await check('记忆检索', async () => {
    const res = await fetch(WEB + '/engram-relay/api/search?q=' + encodeURIComponent('热重载') + '&limit=1', { signal: AbortSignal.timeout(4000) })
    const s = await res.json()
    if (!Array.isArray(s.items)) throw new Error('search 异常')
    console.log('  检索:', s.reason, s.total)
    return true
  })
  // 报告
  console.log('')
  for (const s of steps) console.log((s.ok ? '✅' : '❌') + ' ' + s.name + (s.ok ? '' : ' — ' + s.detail))
  const pass = steps.every((s) => s.ok)
  console.log('')
  console.log(pass ? '风识套装自检通过 ✓（与本地一致）' : '风识套装自检失败 ✗——见上方 ❌ 项')
  process.exit(pass ? 0 : 1)
}
main().catch((e) => { console.error('自检异常:', e); process.exit(1) })
