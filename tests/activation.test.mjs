/** 类脑激活模型单测：B=ln(Σt^(-d)) 的行为断言。 */
import { test } from 'node:test'
import assert from 'node:assert'
import { ActivationCache } from '../lib/engram/activation.js'

const MIN = 60000
const NOW = Date.now()

test('新记忆激活高、久未强化激活低（遗忘曲线）', () => {
  const fresh = ActivationCache.baseActivation([NOW], NOW)
  const old = ActivationCache.baseActivation([NOW - 30 * 24 * 60 * MIN], NOW)
  assert.ok(fresh > old, `新记忆 B=${fresh} 应大于 30 天前 B=${old}`)
})

test('多次强化 > 单次强化（间隔重复巩固）', () => {
  const once = ActivationCache.baseActivation([NOW - 10 * MIN], NOW)
  const twice = ActivationCache.baseActivation([NOW - 10 * MIN, NOW - MIN], NOW)
  assert.ok(twice > once, `两次强化 B=${twice} 应大于一次 B=${once}`)
})

test('时间越近贡献越大（幂律）', () => {
  const near = ActivationCache.baseActivation([NOW - MIN], NOW)
  const far = ActivationCache.baseActivation([NOW - 100 * MIN], NOW)
  assert.ok(near > far, `1 分钟前 B=${near} 应大于 100 分钟前 B=${far}`)
})

test('缓存：rebuild 全量 + update 增量 + remove', () => {
  const cache = new ActivationCache(0.5)
  const nodes = [
    { id: 'a', status: 'confirmed', reinforces: [NOW - MIN] },
    { id: 'b', status: 'confirmed', reinforces: [NOW - 30 * 24 * 60 * MIN] },
    { id: 'p', status: 'pending', reinforces: [NOW] }, // pending 不参与
  ]
  cache.rebuild(nodes)
  assert.ok(cache.get('a') > cache.get('b'), 'a（近）激活应高于 b（远）')
  assert.equal(cache.get('p'), 0, 'pending 节点不应有激活')
  // 增量：b 被强化后激活应上升
  const bBefore = cache.get('b')
  cache.update('b', [NOW - 30 * 24 * 60 * MIN, NOW - MIN])
  assert.ok(cache.get('b') > bBefore, '强化后 b 激活应上升')
  // 移除
  cache.remove('a')
  assert.equal(cache.get('a'), 0)
  assert.equal(cache.size, 1)
})
