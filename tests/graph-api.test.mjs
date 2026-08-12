/**
 * 图谱 Web API 测试：分层准入的 graph/node 接口。
 *
 * 覆盖：/graph 按 viewer（sessionId→cwd）过滤（global 全可见 / project 同
 * cwd / session 本会话）；边聚合（因果 + 双向链接）；/node 详情 + 可见性
 * 403 + 不存在 404。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { EngramStore } from '../lib/engram/store.js'
import { NgramHashAddressing } from '../lib/engram/hash.js'
import { installGraphApi } from '../lib/graph-api.js'

function tempDir() {
  return mkdtempSync(join(tmpdir(), 'dsh-engram-graphapi-'))
}

function makeStore(dir) {
  return new EngramStore(dir, new NgramHashAddressing({ seed: 0 }))
}

/** fake relay：真实 store + 最小 status。 */
function fakeRelay(dir) {
  const store = makeStore(dir)
  return {
    store,
    status: async () => ({ engramCount: store.count() }),
  }
}

function setupStore(store) {
  store.add({ kind: 'fact', layer: 'global', projectId: null, title: '全局偏好', summary: '喜欢简洁文档', content: '全局正文', links: [], sessionId: null, turn: 1, causes: [], effects: [], importance: 0.8 })
  store.add({ kind: 'decision', layer: 'project', projectId: '/proj-a', title: '项目A决策', summary: 'A 用 pnpm', content: 'A 项目正文', links: ['全局偏好'], sessionId: null, turn: 2, causes: [], effects: [], importance: 0.9 })
  store.add({ kind: 'event', layer: 'session', projectId: null, title: 'A临时', summary: '正在调试端口', content: '临时正文', links: [], sessionId: 'sess-a', turn: 3, causes: [], effects: [], importance: 0.6 })
}

/** 收集路由的 fake httpServer。 */
function fakeHttpServer() {
  const routes = []
  return {
    httpServer: { register: (route) => { routes.push(route); return () => { routes.splice(routes.indexOf(route), 1) } } },
    routes,
  }
}

function fakeReqRes(method, url) {
  const res = { status: 0, body: '', ended: false }
  res.writeHead = (s) => { res.status = s }
  res.end = (t) => { res.body = t; res.ended = true }
  return { req: { method, url }, res }
}

async function callRoute(route, method, url) {
  const { req, res } = fakeReqRes(method, url)
  await route.handler(req, res)
  return { status: res.status, body: res.body === '' ? null : JSON.parse(res.body) }
}

test('graph: 分层准入——session A 见 global+本project+本session，session B 见 global+本project 但无 A 的 session 层', async () => {
  const dir = tempDir()
  const store = makeStore(dir)
  setupStore(store)
  const relay = fakeRelay(dir)
  const { httpServer, routes } = fakeHttpServer()
  const ctx = {
    get: (key) => key === 'agents'
      ? { get: (id) => ({ session: { header: { cwd: id === 'sess-a' ? '/proj-a' : '/proj-b' } } }) }
      : undefined,
  }
  installGraphApi({ ...ctx, httpServer }, relay)
  const route = routes[0]

  // 会话 A（cwd=/proj-a）：全三层的节点可见
  const a = await callRoute(route, 'GET', '/engram-relay/api/graph?sessionId=sess-a')
  assert.equal(a.status, 200)
  const aTitles = a.body.nodes.map((n) => n.title)
  assert.ok(aTitles.includes('全局偏好'))
  assert.ok(aTitles.includes('项目A决策'))
  assert.ok(aTitles.includes('A临时'))

  // 会话 B（cwd=/proj-b）：global 可见；A 的 project/session 不可见
  const b = await callRoute(route, 'GET', '/engram-relay/api/graph?sessionId=sess-b')
  const bTitles = b.body.nodes.map((n) => n.title)
  assert.ok(bTitles.includes('全局偏好'))
  assert.ok(!bTitles.includes('项目A决策'), '他人项目 project 层不可见')
  assert.ok(!bTitles.includes('A临时'), '他人会话 session 层不可见')

  // 无 sessionId（无视角）：只看 global
  const anon = await callRoute(route, 'GET', '/engram-relay/api/graph')
  const anonTitles = anon.body.nodes.map((n) => n.title)
  assert.deepEqual(anonTitles, ['全局偏好'])
  rmSync(dir, { recursive: true, force: true })
})

test('graph: 边聚合（因果边 + 双向链接边）', async () => {
  const dir = tempDir()
  const store = makeStore(dir)
  const a = store.add({ kind: 'decision', layer: 'global', projectId: null, title: '根因', summary: '磁盘满了', content: '', links: ['故障'], sessionId: null, turn: 1, causes: [], effects: [], importance: 0.8 })
  const b = store.add({ kind: 'event', layer: 'global', projectId: null, title: '故障', summary: '服务挂了', content: '', links: ['根因'], sessionId: null, turn: 2, causes: [a.id], effects: [], importance: 0.8 })
  const relay = fakeRelay(dir)
  const { httpServer, routes } = fakeHttpServer()
  installGraphApi({ get: () => undefined, httpServer }, relay)
  const route = routes[0]
  const g = await callRoute(route, 'GET', '/engram-relay/api/graph')
  assert.equal(g.status, 200)
  assert.equal(g.body.nodes.length, 2)
  // 因果边（b 的 causes=[a.id]）+ 双向链接边（links 相互）
  const kinds = g.body.edges.map((e) => e.kind)
  assert.ok(kinds.includes('causes'))
  assert.ok(kinds.includes('link'))
  // 因果边方向：from 根因 → to 故障
  const causeEdge = g.body.edges.find((e) => e.kind === 'causes')
  assert.equal(causeEdge.from, a.id)
  assert.equal(causeEdge.to, b.id)
  rmSync(dir, { recursive: true, force: true })
})

test('node: 详情返回 + 可见性 403 + 不存在 404', async () => {
  const dir = tempDir()
  const store = makeStore(dir)
  setupStore(store)
  const relay = fakeRelay(dir)
  const { httpServer, routes } = fakeHttpServer()
  const ctx = {
    get: (key) => key === 'agents'
      ? { get: (id) => ({ session: { header: { cwd: id === 'sess-a' ? '/proj-a' : '/proj-b' } } }) }
      : undefined,
  }
  installGraphApi({ ...ctx, httpServer }, relay)
  const route = routes[0]

  // 会话 A 展开自己 project 层节点：详情含 content + 因果/链接
  const ok = await callRoute(route, 'GET', `/engram-relay/api/node/${encodeURIComponent('项目A决策')}?sessionId=sess-a`)
  assert.equal(ok.status, 200)
  assert.equal(ok.body.content, 'A 项目正文')
  assert.equal(ok.body.layer, 'project')

  // 会话 B 展开 A 的 project 节点：403 不可见
  const forbidden = await callRoute(route, 'GET', `/engram-relay/api/node/${encodeURIComponent('项目A决策')}?sessionId=sess-b`)
  assert.equal(forbidden.status, 403)

  // 不存在：404
  const missing = await callRoute(route, 'GET', `/engram-relay/api/node/${encodeURIComponent('不存在')}`)
  assert.equal(missing.status, 404)
  rmSync(dir, { recursive: true, force: true })
})
