/**
 * extractRecentTurn 测试：从会话消息投影提取最近回合文本。
 * relay.ts 的 extractRecentTurn 未导出，这里通过复制逻辑验证行为，
 * 或用 e2e 方式（构造消息数组 → 断言提取结果）。
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'

// 复制 extractRecentTurn 逻辑（与 src/relay.ts 保持一致）
function extractRecentTurn(messages, _turn) {
  let start = 0
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i]?.role === 'user') {
      start = i
      break
    }
  }
  const parts = []
  for (let i = start; i < messages.length; i += 1) {
    const m = messages[i]
    if (!m) continue
    const role = m.role ?? 'unknown'
    const content = m.content
    if (typeof content === 'string') {
      parts.push(`[${role}] ${content}`)
    } else if (Array.isArray(content)) {
      const text = content
        .map((b) => (typeof b === 'object' && b !== null && 'text' in b ? String(b.text) : ''))
        .filter((t) => t !== '')
        .join(' ')
      if (text !== '') parts.push(`[${role}] ${text}`)
    }
  }
  return parts.join('\n').slice(0, 4000)
}

function msg(role, content) {
  return { role, content }
}

test('extract: 提取最后 user 回合（含 assistant 回复）', () => {
  const messages = [
    msg('user', '第一回合问题'),
    msg('assistant', '第一回合回答'),
    msg('user', '第二回合问题'),
    msg('assistant', '第二回合回答'),
  ]
  const out = extractRecentTurn(messages, 2)
  assert.ok(out.includes('[user] 第二回合问题'), '应包含最后 user 消息')
  assert.ok(out.includes('[assistant] 第二回合回答'), '应包含其后 assistant 回复')
  assert.ok(!out.includes('第一回合'), '不应包含早期回合')
})

test('extract: 块状 content（text blocks）', () => {
  const messages = [
    msg('user', [{ type: 'text', text: '块一' }, { type: 'text', text: '块二' }]),
    msg('assistant', '回答'),
  ]
  const out = extractRecentTurn(messages, 1)
  assert.ok(out.includes('[user] 块一 块二'), '应拼接文本块')
  assert.ok(out.includes('[assistant] 回答'))
})

test('extract: 空消息安全', () => {
  const out = extractRecentTurn([], 0)
  assert.equal(out, '')
  const out2 = extractRecentTurn([msg('assistant', '只有回答')], 0)
  assert.ok(out2.includes('[assistant] 只有回答'))
})

test('extract: 长度上限 4000', () => {
  const long = 'x'.repeat(6000)
  const out = extractRecentTurn([msg('user', long)], 1)
  assert.ok(out.length <= 4000 + 7, '应截断到 ~4000')
})
