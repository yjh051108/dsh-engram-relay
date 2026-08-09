/**
 * NgramHashAddressing — DeepSeek Engram 风格的多头 N-gram 哈希寻址。
 *
 * 论文（Conditional Memory via Scalable Lookup）核心：对 token 序列做
 * 2/3-gram 多项式哈希（multi-head，多素数取模），O(1) 确定性寻址到
 * 巨大记忆表的槽位。确定性寻址 = 相同模式永远命中相同槽位，无相似度
 * 检索的近似性——这是「比普通向量索引更强」的根源。
 *
 * 本实现按论文 demo（engram_demo_v1.py）的逻辑移植到 TypeScript：
 *  - token 归一化（大小写折叠 + 空白归一，对应论文 CompressedTokenizer）
 *  - 逐层独立随机奇数乘子（对应论文 layer_multipliers）
 *  - 多头：每 n-gram 长度多个素数模数（对应论文 head_vocab_sizes）
 *
 * 与论文差异：论文的记忆表是训练出的 embedding table；本插件的外置
 * engram 表是「哈希槽 → 记忆条目」（JSONL 持久化），由 <1B 模型蒸馏
 * 写入、请求前按当前上下文的哈希寻址唤醒。
 */

/** 多头哈希配置。 */
export interface HashConfig {
  /** 最大 n-gram 长度（论文 max_ngram_size=3 → 2-gram、3-gram 两级）。 */
  maxNgramSize: number
  /** 每级 n-gram 的头数（论文 n_head_per_ngram=8）。 */
  headsPerNgram: number
  /** 每级 n-gram 的槽位数（论文 engram_vocab_size，这里用 2 的幂便于取模）。 */
  slotsPerNgram: number
  /** 随机种子（决定乘子与素数序列，固定后寻址确定）。 */
  seed: number
  /** 归一化时折叠大小写。 */
  lowercase: boolean
}

export const DEFAULT_HASH_CONFIG: HashConfig = {
  maxNgramSize: 3,
  headsPerNgram: 4,
  slotsPerNgram: 4096,
  seed: 0,
  lowercase: true,
}

/** 一次哈希寻址的结果：每 (n-gram 长度 × 头) 一个槽位 id。 */
export interface HashResult {
  /** 每个 n-gram 长度下的多头槽位：ngramLen -> head -> slotId */
  slots: Array<Array<number>>
  /** 归一化后的 token 序列（调试/日志用）。 */
  tokens: string[]
}

/** 确定性伪随机数生成器（mulberry32）——保证跨进程/跨平台一致。 */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** 确定性素数判定 + 取下一个素数（论文用 sympy.isprime，这里用 Miller-Rabin）。 */
function isPrime(n: number): boolean {
  if (n < 2) return false
  for (const p of [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]) {
    if (n % p === 0) return n === p
  }
  // 确定性 Miller-Rabin（适用于 < 2^32）
  let d = n - 1
  let s = 0
  while (d % 2 === 0) { d /= 2; s += 1 }
  for (const a of [2, 3, 5, 7, 11, 13, 17]) {
    if (a >= n) continue
    let x = modPow(a, d, n)
    if (x === 1 || x === n - 1) continue
    let composite = true
    for (let r = 1; r < s; r += 1) {
      x = (x * x) % n
      if (x === n - 1) { composite = false; break }
    }
    if (composite) return false
  }
  return true
}

function modPow(base: number, exp: number, mod: number): number {
  let result = 1
  base %= mod
  while (exp > 0) {
    if (exp % 2 === 1) result = (result * base) % mod
    base = (base * base) % mod
    exp = Math.floor(exp / 2)
  }
  return result
}

/** 下一素数（跳过已用）。 */
function nextPrime(start: number, seen: Set<number>): number {
  let c = start + 1
  while (true) {
    if (isPrime(c) && !seen.has(c)) return c
    c += 1
  }
}

export class NgramHashAddressing {
  private config: HashConfig
  /** 每层（n-gram 长度）的随机奇数乘子（对应论文 layer_multipliers）。 */
  private multipliers: number[][]
  /** 每层每头的素数模数（对应论文 head_vocab_sizes）。 */
  private primesPerHead: number[][]

  constructor(config: Partial<HashConfig> = {}) {
    this.config = { ...DEFAULT_HASH_CONFIG, ...config }
    const { maxNgramSize, headsPerNgram, slotsPerNgram, seed } = this.config

    // 乘子：每 n-gram 长度一个随机奇数（种子固定 → 确定）。
    const rng = mulberry32(seed)
    this.multipliers = []
    for (let n = 2; n <= maxNgramSize; n += 1) {
      this.multipliers.push(Array.from({ length: n }, () => {
        const r = Math.floor(rng() * 0x3fffffff)
        return r * 2 + 1
      }))
    }

    // 素数模数：每层每头一个素数，从 slotsPerNgram 附近开始递增。
    this.primesPerHead = []
    const seen = new Set<number>()
    for (let n = 2; n <= maxNgramSize; n += 1) {
      const heads: number[] = []
      let start = slotsPerNgram - 1
      for (let h = 0; h < headsPerNgram; h += 1) {
        const p = nextPrime(start, seen)
        seen.add(p)
        heads.push(p)
        start = p
      }
      this.primesPerHead.push(heads)
    }
  }

  /** 归一化文本 → token 数组（对应论文 CompressedTokenizer 的压缩）。 */
  normalize(text: string): string[] {
    let t = text
    if (this.config.lowercase) t = t.toLowerCase()
    // 空白归一 + 去首尾
    t = t.replace(/[ \t\r\n]+/g, ' ')
    t = t.trim()
    if (t === '') return []
    // 按空白切词，保留中英文词与数字；标点折叠（去重连续标点）
    const words = t.split(' ')
    const tokens: string[] = []
    for (const w of words) {
      if (w === '') continue
      // 中文按字拆（每个汉字是一个 token，保证 n-gram 有意义）
      const cjk = w.match(/[\u4e00-\u9fff]/g)
      if (cjk && cjk.length === w.length) {
        tokens.push(...cjk)
      } else {
        tokens.push(w.replace(/[^\w\u4e00-\u9fff-]+/g, ''))
      }
    }
    return tokens.filter((x) => x !== '')
  }

  /** 对 token 序列做多头 n-gram 哈希寻址。 */
  hashTokens(tokens: string[]): HashResult {
    const { maxNgramSize } = this.config
    const slots: Array<Array<number>> = []

    for (let n = 2; n <= maxNgramSize; n += 1) {
      const layerIdx = n - 2
      const mults = this.multipliers[layerIdx]
      const primes = this.primesPerHead[layerIdx]
      const heads: number[] = []
      for (let h = 0; h < this.config.headsPerNgram; h += 1) {
        const mod = primes[h]
        // 滚动多项式哈希：mix = Σ token[i] * mult[i]（论文用 xor 混合）
        let mix = 0
        for (let i = 0; i + n <= tokens.length; i += 1) {
          let hval = 0
          for (let k = 0; k < n; k += 1) {
            hval = (hval + hashStr(tokens[i + k]) * mults[k]) % 2147483647
          }
          mix = (mix + hval) % mod
        }
        heads.push(mix % mod)
      }
      slots.push(heads)
    }
    return { slots, tokens }
  }

  /** 便捷入口：文本 → 寻址。 */
  hash(text: string): HashResult {
    return this.hashTokens(this.normalize(text))
  }

  /** 把多头槽位折叠成一组可索引的键（用于外置表寻址）。 */
  slotKeys(result: HashResult): string[] {
    const keys: string[] = []
    for (let n = 2; n <= this.config.maxNgramSize; n += 1) {
      const layerIdx = n - 2
      for (let h = 0; h < this.config.headsPerNgram; h += 1) {
        keys.push(`n${n}h${h}:${result.slots[layerIdx][h]}`)
      }
    }
    return keys
  }
}

/** FNV-1a 字符串哈希（32 位，确定性）。 */
function hashStr(s: string): number {
  let h = 0x811c9dc5
  for (let i = 0; i < s.length; i += 1) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}
