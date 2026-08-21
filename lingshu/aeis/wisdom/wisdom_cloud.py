# -*- coding: utf-8 -*-
"""
智慧之书 · mock 云服务（零依赖 http.server）
==========================================
端点：
  POST /dex/query   {op, params}          七操作查询（读公开 · 任何灵枢智能体）
  POST /dex/upload  {entry, contributor}  上传已验证条目（verified 闸门 + 贡献记账）
  GET  /dex/ledger?contributor=X          贡献账本
  GET  /dex/status                        图谱元信息

运行：python wisdom_cloud.py [port]
"""
import json
import os
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from wisdom_book import ConditionDex, _default_cs  # noqa: E402

CLOUD_DB = os.path.join(HERE, "wisdom-book-cloud.db")


def _cs_from_dict(d):
    from aeis.core import ConditionSpace
    if not d:
        return _default_cs()
    try:
        tw = tuple(d.get("time_window") or (0.0, 9999999999.0))
    except Exception:
        tw = (0.0, 9999999999.0)
    return ConditionSpace(
        observation_position=d.get("observation_position", ""),
        observation_tool=d.get("observation_tool", ""),
        time_window=tw,
        existence_constraint=d.get("existence_constraint", ""))


class DexHandler(BaseHTTPRequestHandler):
    cloud = None  # ConditionDex 实例（由 run_server 注入）

    def log_message(self, *args):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n).decode("utf-8"))

    # ---------------- GET ----------------

    def do_GET(self):
        path = self.path.split("?")[0]
        qs = {}
        if "?" in self.path:
            qs = urllib.parse.parse_qs(self.path.split("?", 1)[1])
        if path in ("/", "/ui", "/index.html", "/ui/index.html", "/wisdom_ui.html"):
            self._send_html()
        elif path == "/dex/status":
            self._send(self._status())
        elif path == "/dex/ledger":
            c = qs.get("contributor", [None])[0]
            self._send(self._ledger(c))
        else:
            self._send({"error": "not_found"}, 404)

    def _send_html(self):
        """人类学习/搜索界面（零依赖单页）"""
        html_path = os.path.join(HERE, "wisdom_ui.html")
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                body = f.read().encode("utf-8")
        except OSError:
            self._send({"error": "ui_not_found"}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---------------- POST ----------------

    def do_POST(self):
        try:
            body = self._read()
        except Exception:
            self._send({"error": "bad_json"}, 400)
            return
        path = self.path.split("?")[0]
        if path == "/dex/query":
            self._send(self._query(body))
        elif path == "/dex/upload":
            self._send(self._upload(body))
        else:
            self._send({"error": "not_found"}, 404)

    # ---------------- 实现 ----------------

    # 学习统计（模块级：收敛观测仪表盘）
    _LEARN_STATS = {"respond": 0, "strong": 0, "weak": 0, "miss": 0,
                    "clusters": 0, "cards": 0}
    # 自动补词网状态（模块级：服务进程内持续统计）
    _WEAK_HITS = {}  # 查询→{count, top_card, last_at}

    def _auto_cluster(self, condition, results):
        """弱命中统计与自动建簇：
        同一查询多次弱命中（top<0.5）→ 从查询提取俗语词，
        建簇映射到已命中卡（规范词=卡名）→ 写 clusters_ext.json + 动态更新翻译表。
        """
        import time as _t
        q = (condition or "").strip()
        if len(q) < 4 or not results:
            return None
        top = results[0] if results else {}
        top_score = top.get("score") or top.get("algo_score") or 0
        top_algo = top.get("algo_score") or 0
        top_name = top.get("name")
        if not top_name or float(top_score) >= 0.5:
            return None  # 强命中无需补
        # 相关性门槛：algo_score（词面关联）低于噪声线 → top 卡是重叠噪声，不建簇
        if float(top_algo) < 0.05:
            return None
        key = q[:30]
        rec = self._WEAK_HITS.get(key)
        if rec is None:
            self._WEAK_HITS[key] = {"count": 1, "top_card": top_name, "last_at": _t.time()}
            return None
        rec["count"] += 1
        rec["last_at"] = _t.time()
        if rec["count"] < 3:
            return None
        # 触发建簇：提取查询词（≥2 字/ASCII≥3），排除已在翻译表的
        import semantic_translate as _st
        import json as _json, os as _os
        existing = set()
        for _cls in (_st.SYNONYM_CLUSTERS.values(), _st.DOMAIN_SYNONYM_CLUSTERS.values()):
            for _words in _cls:
                existing.update(_words)
        import re as _re
        words = set()
        for m in _re.finditer(r"[a-z][a-z0-9_]{2,}", q.lower()):
            words.add(m.group())
        for m in _re.finditer(r"[\u4e00-\u9fff]{2,}", q):
            s = m.group()
            if len(s) == 2:
                words.add(s)
            elif len(s) == 3:
                words.add(s[:2]); words.add(s[1:3])
            else:
                # 首+中+尾 2-gram（中间片段承载核心语义——「代码可读性」→ 代码/可读/读性；奇数长取中后段）
                _mid = (len(s) - 1) // 2
                words.add(s[:2]); words.add(s[_mid:_mid + 2]); words.add(s[-2:])
        _QW = set("怎么 什么 如何 为啥 为什么 哪些 哪个 多少 哪里 何时 是否 有没有 能不能 会不会 怎么样 怎样 这样 那样 一个 一种 一下 起来 以后 内容 东西".split())
        # 已覆盖词：翻译表 + 目标卡 trigger（从卡库查——避免建无用簇）
        from aeis.core import MemoryLayer as _ML
        _top_trig = set()
        try:
            for _n in self.cloud.store.query_nodes(layer=_ML.KNOWLEDGE, limit=2000):
                _sa = _n.state_attributes or {}
                if _sa.get("name") == top_name:
                    _top_trig = set(t.strip() for t in str((_sa.get("response") or {}).get("trigger") or "").split(",") if t.strip())
                    break
        except Exception:
            pass
        new_words = [w for w in words if w not in existing and w not in _top_trig and w not in top_name and w not in _QW]
        if not new_words:
            rec["count"] = 0  # 无新词可补，重置计数
            return None
        # 建簇：规范词=已命中卡名
        ext_path = _os.path.join(_os.path.dirname(_os.path.abspath(_st.__file__)), "clusters_ext.json")
        try:
            with open(ext_path, encoding="utf-8") as _f:
                ext = _json.load(_f)
        except Exception:
            ext = {}
        if top_name not in ext:
            ext[top_name] = []
        merged = list(dict.fromkeys(ext[top_name] + new_words))
        ext[top_name] = merged
        with open(ext_path, "w", encoding="utf-8") as _f:
            _json.dump(ext, _f, ensure_ascii=False, indent=1)
        # 动态更新翻译表（本进程立即生效）
        _st.DOMAIN_SYNONYM_CLUSTERS[top_name] = merged
        rec["count"] = 0
        return {"added": top_name, "words": new_words}

    def _query(self, body):
        op = body.get("op", "")
        params = body.get("params") or {}
        d = self.cloud
        try:
            if op == "filter":
                return {"op": op, "results": d.dex_filter(**params)}
            if op == "learn_stats":
                return {"op": op, "results": dict(self._LEARN_STATS)}
            if op == "add_card":
                # 自动补卡端点（relay 缺口闭环调用）：add_entry 写入知识卡
                name = params.get("name", "")
                if not name:
                    return {"op": op, "ok": False, "error": "name required"}
                # 学习统计：auto-gap 来源计数（补卡仪表盘）
                if str(params.get("source", "")) == "auto-gap":
                    self._LEARN_STATS["cards"] += 1
                # 同名查重（灵枢 _by_name upsert——重复写入会覆盖，先查）
                if hasattr(d, "_by_name") and name in d._by_name:
                    return {"op": op, "ok": True, "existed": True, "name": name}
                from aeis.core import ConditionSpace
                cs = ConditionSpace(
                    observation_position=params.get("obs_pos", "自动补卡"),
                    observation_tool="识别卡",
                    time_window=(0.0, 1e10),
                    existence_constraint=params.get("cons", "通用"))
                d.add_entry(
                    name=name,
                    domain=params.get("domain", "通用"),
                    claim=params.get("claim", ""),
                    cs=cs,
                    level=int(params.get("level", 2)),
                    status=params.get("status", "verified"),
                    response={
                        "trigger": params.get("trigger", ""),
                        "action": params.get("action", ""),
                        "counters": params.get("counters", ""),
                    },
                    tags=[f"domain:{params.get('domain', '通用')}"],
                    card2={"source": params.get("source", "auto")})
                return {"op": op, "ok": True, "existed": False, "name": name}
            if op == "respond":
                # 融合修复：传 translator——出招必须走翻译表+学科路由，
                # 纯字符重叠对日常话命中不了学科卡
                try:
                    import semantic_translate as _st
                except Exception:
                    _st = None
                results = d.dex_respond(
                    params.get("condition", ""),
                    limit=int(params.get("limit", 10)),
                    translator=_st)
                # 语义算法通道（移植 engram SemanticScorer 三通道：词汇/共现），
                # 对候选附加 algo_score 并按算法分重排；algo=0 关闭（对比用）
                if str(params.get("algo", "1")) not in ("0", "false", "False"):
                    try:
                        from semantic_algo import algo_rerank
                        results = algo_rerank(d, params.get("condition", ""), results)
                    except Exception:
                        pass
                # 自动补词网：弱命中（top<0.5）高频（≥3）→ 自动生成俗语簇（零 LLM）
                try:
                    cluster_hint = self._auto_cluster(params.get("condition", ""), results)
                except Exception as _e:
                    import traceback as _tb
                    try:
                        with open(r"D:\\aeis\\_cluster_err.log", "a", encoding="utf-8") as _f:
                            _f.write(_tb.format_exc() + "\n")
                    except Exception:
                        pass
                    cluster_hint = None
                # 学习统计（收敛观测）
                try:
                    self._LEARN_STATS["respond"] += 1
                    top_s = results[0].get("score") if results else 0
                    if top_s >= 0.5: self._LEARN_STATS["strong"] += 1
                    elif top_s >= 0.1: self._LEARN_STATS["weak"] += 1
                    else: self._LEARN_STATS["miss"] += 1
                    if cluster_hint: self._LEARN_STATS["clusters"] += 1
                except Exception:
                    pass
                # 学习趋势持久化（每 20 次响应快照——跨重启长期观测）
                try:
                    if self._LEARN_STATS["respond"] % 20 == 0:
                        with open(r"D:\\aeis\\learn_stats.log", "a", encoding="utf-8") as _f:
                            import datetime as _dt
                            _f.write(_dt.datetime.now().isoformat(timespec="seconds") + " " +
                                     json.dumps(self._LEARN_STATS, ensure_ascii=False) + "\n")
                except Exception:
                    pass
                return {"op": op, "results": results, "cluster": cluster_hint, "learn": dict(self._LEARN_STATS)}
            if op == "status_node":
                return {"op": op, "results": d.dex_status(params.get("node_id", ""))}
            if op == "cs":
                return {"op": op, "results": d.dex_cs(params.get("code", ""))}
            if op == "combine":
                return {"op": op, "results": d.dex_combine(params.get("a", ""), params.get("b", ""))}
            if op == "separate":
                return {"op": op, "results": d.dex_separate(params.get("node_id", ""))}
            if op == "invert":
                return {"op": op, "results": d.dex_invert(params.get("node_id", ""))}
            if op == "cycle":
                return {"op": op, "results": d.dex_cycle(params.get("node_id", ""))}
            if op == "analyze":
                return {"op": op, "results": d.dex_analyze(params.get("knowledge", ""))}
            if op == "predict":
                return {"op": op, "results": d.dex_predict(
                    params.get("knowledge", ""),
                    horizon=int(params.get("horizon", 2)),
                    limit=int(params.get("limit", 4)))}
            if op == "predict_compare":
                return {"op": op, "results": d.dex_predict_compare(
                    params.get("knowledge", ""),
                    params.get("theory", ""),
                    horizon=int(params.get("horizon", 2)),
                    limit=int(params.get("limit", 4)))}
            if op == "auto_verify":
                # 传 translator（与 respond 一致）——锚定走翻译表评分链路
                try:
                    import semantic_translate as _st
                except Exception:
                    _st = None
                return {"op": op, "results": d.dex_auto_verify(
                    params.get("knowledge", ""),
                    limit=int(params.get("limit", 5)),
                    threshold=float(params.get("threshold", 0.50)),
                    translator=_st)}
            if op == "compose":
                return {"op": op, "results": d.dex_compose(
                    params.get("knowledge", ""),
                    limit=int(params.get("limit", 5)),
                    max_anchors=int(params.get("max_anchors", 3)))}
            if op == "test":
                return {"op": op, "results": d.dex_test(params.get("knowledge", ""))}
            if op == "battle":
                return {"op": op, "results": d.dex_battle(params.get("a", ""), params.get("b", ""))}
            if op == "layer_trace":
                return {"op": op, "results": d.dex_layer_trace()}
            if op == "sandbox":
                return {"op": op, "results": d.dex_sandbox(
                    params.get("a", ""), params.get("b", ""),
                    params.get("disturbance", ""))}
            if op == "auto_test":
                return {"op": op, "results": d.dex_auto_test(
                    params.get("a", ""), params.get("b", ""))}
            if op == "usage":
                return {"op": op, "results": d.dex_usage()}
            if op == "homology":
                return {"op": op, "results": d.dex_homology(
                    params.get("entry", ""), params.get("strip_concepts"))}
            if op == "standard_battle":
                return {"op": op, "results": d.dex_standard_battle(
                    params.get("a", ""), params.get("b", ""))}
            return {"op": op, "error": "unknown_op"}
        except Exception as e:
            return {"op": op, "error": str(e)}

    def _upload(self, body):
        entry = body.get("entry") or {}
        contributor = body.get("contributor", "anonymous")
        # ---- 上传闸门：verified 且验证轨迹完整 ----
        if entry.get("status") != "verified":
            return {"ok": False, "reason": "upload_gate: status 必须为 verified",
                    "name": entry.get("name", "")}
        trail = entry.get("verification_trail") or {}
        if not trail.get("verified_by"):
            return {"ok": False, "reason": "upload_gate: verification_trail.verified_by 必填",
                    "name": entry.get("name", "")}
        if not entry.get("condition_space"):
            return {"ok": False, "reason": "upload_gate: 无明确条件空间不配加入图鉴（P17 收录判据）",
                    "name": entry.get("name", "")}
        d = self.cloud
        nid = d.add_entry(
            name=entry.get("name", "未命名"),
            domain=entry.get("domain", "未分类"),
            claim=entry.get("claim", ""),
            cs=_cs_from_dict(entry.get("condition_space")),
            level=int(entry.get("level", 2)),
            status="verified",
            response=entry.get("response"))
        now = time.time()
        d.store.conn.execute(
            "INSERT OR REPLACE INTO contributions (entry_id, contributor, verified_by, verified_at, weight) "
            "VALUES (?,?,?,?,?)",
            (nid, contributor, trail.get("verified_by"), now,
             float(entry.get("weight", 1.0))))
        d.store.conn.commit()
        cnt = d.store.conn.execute(
            "SELECT COUNT(*) FROM contributions WHERE contributor=?", (contributor,)).fetchone()[0]
        return {"ok": True, "entry_id": nid, "contributor": contributor,
                "contribution_count": cnt}

    def _ledger(self, contributor=None):
        conn = self.cloud.store.conn
        if contributor:
            rows = conn.execute(
                "SELECT entry_id, contributor, verified_by, verified_at, weight "
                "FROM contributions WHERE contributor=?", (contributor,)).fetchall()
        else:
            rows = conn.execute(
                "SELECT entry_id, contributor, verified_by, verified_at, weight "
                "FROM contributions ORDER BY verified_at").fetchall()
        return {"contributions": [
            {"entry_id": r[0], "contributor": r[1], "verified_by": r[2],
             "verified_at": r[3], "weight": r[4]} for r in rows]}

    def _status(self):
        from aeis.core import MemoryLayer
        d = self.cloud
        nodes = d.store.query_nodes(layer=MemoryLayer.KNOWLEDGE, limit=1000)
        total = len(nodes)
        verified = sum(1 for n in nodes
                       if n.state_attributes.get("status") == "verified")
        domains = {}
        for n in nodes:
            dom = n.state_attributes.get("domain", "未知")
            domains[dom] = domains.get(dom, 0) + 1
        contrib = d.store.conn.execute(
            "SELECT COUNT(*) FROM contributions").fetchone()[0]
        return {"total_entries": total, "verified": verified,
                "domains": domains, "contributions": contrib}


def run_server(port=0, db_path=None):
    """启动 mock 云（daemon 线程）。port=0 → 自动分配空闲端口。返回 (server, dex)。"""
    db = db_path or CLOUD_DB
    if os.path.exists(db):
        os.remove(db)  # v0 mock：每次全新
    dex = ConditionDex(db_path=db, fresh=True)
    dex.seed_base()
    dex.store.conn.execute(
        "CREATE TABLE IF NOT EXISTS contributions ("
        "entry_id TEXT PRIMARY KEY, contributor TEXT, verified_by TEXT, "
        "verified_at REAL, weight REAL)")
    dex.store.conn.commit()

    DexHandler.cloud = dex
    srv = ThreadingHTTPServer(("127.0.0.1", port), DexHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, dex


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18766
    srv, _dex = run_server(port=port)
    print(f"智慧之书 mock 云运行于 http://127.0.0.1:{srv.server_address[1]}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        srv.shutdown()
