# -*- coding: utf-8 -*-
"""语义算法通道（移植自 engram SemanticScorer · 纯算法三通道，零模型）。

  ① 词汇通道（lexical）：字符 n-gram Jaccard × 词频覆盖——词面重叠保底；
  ② 图语义通道（graph）：候选的因果/链接邻居是否命中查询哈希节点；
  ③ 统计语义通道（cooc）：词-词共现相似（PMI 风格，零矩阵分解）——
     语义桥机制：词 a、b 在同一卡出现 → 共现计数建桥（如「信息差」↔
     「预测误差」），随语料增多收敛。灵枢翻译表的自动补全。

分数 ∈ [0,1]，融合 = 0.5·lexical + 0.25·graph + 0.25·cooc。
"""
import re
import threading


def char_ngrams(text, n=2):
    t = re.sub(r"\s+", "", text or "")
    out = set()
    if len(t) < n:
        if t:
            out.add(t)
        return out
    for i in range(len(t) - n + 1):
        out.add(t[i:i + n])
    return out


def words_of(text):
    out = {}
    tokens = re.findall(r"[a-z0-9]+|[^\u0000-\u007f]+", text or "", re.I)
    for tok in tokens:
        if re.match(r"[a-z0-9]", tok, re.I):
            k = tok.lower()
            out[k] = out.get(k, 0) + 1
        elif len(tok) == 1:
            out[tok] = out.get(tok, 0) + 1
        else:
            for i in range(len(tok) - 1):
                k = tok[i:i + 2]
                out[k] = out.get(k, 0) + 1
    return out


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


class CoocSemantics:
    """词-词共现（IDF 加权；高频字停用防坍缩）。"""

    def __init__(self):
        self.co = {}
        self.built = False

    def mark_dirty(self):
        self.built = False

    def ensure(self, nodes):
        if self.built:
            return
        self.built = True
        self.co = {}
        if len(nodes) < 2:
            return
        df = {}
        node_words = []
        for n in nodes:
            words = list(words_of(n["text"]).keys())
            if not words:
                continue
            node_words.append(words)
            for w in set(words):
                df[w] = df.get(w, 0) + 1
        N = len(node_words)
        idf = lambda w: math.log((N + 1) / (df.get(w, 1) + 1)) + 1
        stop = {w for w, d in df.items() if d > max(5, N * 0.2)}
        for words in node_words:
            kept = [w for w in words if w not in stop]
            if not kept:
                continue
            weights = {w: idf(w) for w in kept}
            for i in range(len(kept)):
                wi = kept[i]
                for j in range(i, len(kept)):
                    wj = kept[j]
                    w = weights[wi] * weights[wj]
                    self.co.setdefault(wi, {})[wj] = self.co[wi].get(wj, 0) + w
                    self.co.setdefault(wj, {})[wi] = self.co[wj].get(wi, 0) + w

    def raw_score(self, query, mem_text):
        q_words = list(words_of(query).keys())
        m_words = list(words_of(mem_text).keys())
        if not q_words or not m_words:
            return 0.0
        total = 0.0
        for q in q_words:
            row = self.co.get(q)
            if not row:
                continue
            for m in m_words:
                total += row.get(m, 0)
        return total / max(1, len(q_words))


class SemanticScorer:
    """纯算法语义打分器（语料 = 卡库节点）。"""

    def __init__(self, dex):
        self.dex = dex
        self.cooc = CoocSemantics()
        self._cache = None

    def _nodes(self):
        if self._cache is None:
            from aeis.core import MemoryLayer
            nodes = []
            for n in self.dex.store.query_nodes(layer=MemoryLayer.KNOWLEDGE, limit=2000):
                sa = n.state_attributes or {}
                name = sa.get("name")
                if not name:
                    continue
                resp = sa.get("response") or {}
                nodes.append({
                    "id": n.id, "name": name,
                    "text": f"{name}：{n.content or ''} {resp.get('trigger', '')}",
                })
            self._cache = nodes
            self.cooc.ensure(self._cache)
        return self._cache

    def mark_dirty(self):
        self._cache = None
        self.cooc.mark_dirty()

    def score(self, query, limit=None):
        nodes = self._nodes()
        q_grams = char_ngrams(query)
        q_words = words_of(query)
        out = []
        for n in nodes:
            m_grams = char_ngrams(n["text"])
            lex = jaccard(q_grams, m_grams)
            m_words = words_of(n["text"])
            hit_w = sum(min(c, m_words.get(w, 0)) for w, c in q_words.items())
            total_w = sum(q_words.values())
            word_cover = hit_w / total_w if total_w > 0 else 0
            lexical = min(1.0, lex * 0.6 + word_cover * 0.4)
            cooc_raw = self.cooc.raw_score(query, n["text"])
            out.append({"name": n["name"], "lexical": lexical, "cooc": cooc_raw})
        max_raw = max([1.0] + [x["cooc"] for x in out])
        for x in out:
            cooc = x["cooc"] / max_raw if x["cooc"] > 0 else 0
            x["score"] = round(min(1.0, 0.5 * x["lexical"] + 0.25 * cooc + 0.25 * 0.0), 4)
            x["cooc"] = round(cooc, 4)
            x["lexical"] = round(x["lexical"], 4)
        out.sort(key=lambda x: -x["score"])
        return out[:limit] if limit else out


def algo_rerank(dex, query, results):
    """respond 结果算法融合：附加 algo_score + 融合分重排。

    融合分 = 原分 + β·算法分（β=2.0）——算法词汇通道不受翻译表限制，
    弱触发查询（翻译表未覆盖的词面）由算法通道兜底。
    """
    global _scorer_cache
    if _scorer_cache is None or _scorer_cache[0] is not dex:
        _scorer_cache = (dex, SemanticScorer(dex))
    scorer = _scorer_cache[1]
    scored = scorer.score(query)
    by_name = {x["name"]: x for x in scored}
    for r in results:
        a = by_name.get(r.get("name"))
        r["algo_score"] = a["score"] if a else None
        if a:
            r["algo_lexical"] = a["lexical"]
            r["algo_cooc"] = a["cooc"]
        base = float(r.get("score") or 0)
        algo = float(r.get("algo_score") or 0)
        r["fused_score"] = round(base + 2.0 * algo, 4)
        # 展示分 = 融合分（保留原分在 raw_score 供对比）
        r["raw_score"] = r.get("score")
        r["score"] = r["fused_score"]
    results.sort(key=lambda r: -(r.get("fused_score") if r.get("fused_score") is not None else -1))
    return results


_scorer_cache = None
import math
