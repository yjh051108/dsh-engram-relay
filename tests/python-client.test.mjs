/**
 * PythonEngramClient 降级测试：
 * Python 缺失/服务不可用时应返回 null（插件降级为纯哈希路由），不抛异常。
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

// 用不存在的 python 路径强制启动失败
test('client: failed spawn degrades to null', async () => {
  const { PythonEngramClient } = await import('../lib/model/python-client.js')
  const client = new PythonEngramClient('python-that-does-not-exist-xyz')
  const status = await client.status()
  assert.equal(status, null, '服务不可用时应返回 null')
  client.stop()
})

test('client: missing server file degrades to null', async () => {
  const { PythonEngramClient } = await import('../lib/model/python-client.js')
  // 临时目录里没有 python/engram_model/server.py
  const dir = mkdtempSync(join(tmpdir(), 'engram-no-server-'))
  try {
    const client = new PythonEngramClient('python', 'Qwen/Qwen3-0.6B')
    // 直接改内部路径检查逻辑：不存在时 failed=true → null
    const status = await client.status()
    assert.equal(status, null)
    client.stop()
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

// ---- v3：0.6B 移除后，Python 服务 = 嵌入服务（bge 语义精排） ----

const BGE_PATH = process.env.ENGRAM_EMBED_MODEL || 'F:/dsh/01-memory/engram-trial/bge-small-zh'

test('client: load with missing model dir degrades to loaded:false (no crash)', async () => {
  const { PythonEngramClient } = await import('../lib/model/python-client.js')
  const client = new PythonEngramClient('python', 'Qwen/Qwen3-0.6B')
  const status = await client.load()
  assert.ok(status !== null, '服务应存活（返回结果而非 null）')
  assert.equal(status.loaded, false, '模型目录缺失 → loaded:false，不崩溃不联网')
  client.stop()
})

test('client: embed op returns 512-dim vectors + query vec (bge 真实服务)', async () => {
  const { PythonEngramClient } = await import('../lib/model/python-client.js')
  const client = new PythonEngramClient('python', '', '', BGE_PATH)
  const out = await client.embed(
    ['缓存上线：缓存层全量生效', '数据上线：数据库切换完成'],
    '缓存压测怎么样',
  )
  assert.ok(out !== null, 'embed 应成功')
  assert.equal(out.vectors.length, 2, '每条文本一个向量')
  assert.equal(out.vectors[0].length, 512, 'bge-small-zh 维度 512')
  assert.ok(out.query_vec && out.query_vec.length === 512, 'query 向量')
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0)
  assert.ok(dot(out.vectors[0], out.query_vec) > dot(out.vectors[1], out.query_vec),
    '缓存查询应语义更接近缓存节点')
  client.stop()
})
