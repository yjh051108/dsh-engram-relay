/** 存储恢复链测试：写坏主文件后，新实例应从 .bak 快照自动恢复（绝不丢记忆）。 */
import { test } from 'node:test'
import assert from 'node:assert'
import { mkdtempSync, rmSync, readFileSync, writeFileSync, existsSync, readdirSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { NgramHashAddressing } from '../lib/engram/hash.js'
import { EngramStore } from '../lib/engram/store.js'

test('主文件 NUL 损坏 → 从最近完好备份恢复全部节点', () => {
  const dir = mkdtempSync(join(tmpdir(), 'engram-recover-'))
  try {
    const hasher = new NgramHashAddressing({ seed: 0 })
    const store = new EngramStore(dir, hasher)
    const titles = ['第一', '第二', '第三'].map((t, i) => {
      const e = store.add({
        kind: 'fact', layer: 'project', title: `节点${t}`, summary: `摘要${t}`,
        content: `内容${t}`, links: [], sessionId: null, turn: i, causes: [], effects: [], importance: 0.5,
      })
      return e.title
    })
    assert.equal(store.count(), 3)

    // 再写一次（制造第二份快照）
    store.add({
      kind: 'fact', layer: 'project', title: '节点第四', summary: '摘要第四',
      content: '内容第四', links: [], sessionId: null, turn: 3, causes: [], effects: [], importance: 0.5,
    })

    // 模拟损坏：主文件写入全 NUL
    const main = join(dir, 'engrams.jsonl')
    const good = readFileSync(main, 'utf8')
    writeFileSync(main, '\0'.repeat(30000), 'utf8')

    // 新实例加载：应从备份恢复（≥ 备份代次的节点）
    const recovered = new EngramStore(dir, hasher)
    console.log(`恢复节点数: ${recovered.count()}（备份存在: ${listBaks(dir)}）`)
    assert.ok(recovered.count() >= 3, `应恢复至少 3 条（上一代快照），实际 ${recovered.count()}`)
    // 至少包含第一代快照的 3 条（第四可能不在最早快照，允许）
    assert.ok(titles.every((t) => recovered.all().some((n) => n.title === t)), '第一代 3 条应全部恢复')
    // 主文件已被修复（非 NUL）
    const after = readFileSync(main, 'utf8')
    assert.ok(!/\0/.test(after.slice(0, 100)), '主文件应已从备份修复')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

function listBaks(dir) {
  return readdirSync(dir).filter((n) => n.includes('.bak-'))
}
