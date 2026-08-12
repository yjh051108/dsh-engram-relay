import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { NgramHashAddressing } from './lib/engram/hash.js'
import { EngramStore } from './lib/engram/store.js'
const dir = mkdtempSync(join(tmpdir(), 'engram-dbg-'))
const hasher = new NgramHashAddressing({ seed: 0 })
const store = new EngramStore(dir, hasher)
const e = store.add({ kind: 'event', layer: 'project', title: '主题0配置0', summary: '主题0第1条记录', content: '主题0细节0', links: [], sessionId: 'sim', turn: 0, causes: [], effects: [], importance: 0.5 })
console.log('added:', e.id, 'slots:', e.slots.length)
console.log('count:', store.count())
const hits = store.lookup('查询主题0相关的记忆内容', 64)
console.log('lookup hits:', hits.length, hits.map(h => h.title))
const h2 = store.lookup('主题0', 64)
console.log('lookup 主题0:', h2.length)
const r = hasher.hash('主题0')
console.log('hash 主题0 keys:', hasher.slotKeys(r).slice(0, 8))
const r2 = hasher.hash('查询主题0相关的记忆内容')
console.log('hash query keys:', hasher.slotKeys(r2).slice(0, 8))
