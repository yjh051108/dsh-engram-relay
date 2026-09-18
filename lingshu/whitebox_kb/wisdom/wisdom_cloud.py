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
    """字典→ConditionSpace 还原（缺省字段落协议默认）。"""
    from aeis_core import ConditionSpace
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

    # ---- DSH 侧扩展（engram 融合用；上游 0.5.0 无此三处，同步时按此处回植）----
    # 学习统计（模块级：收敛观测仪表盘）
    _LEARN_STATS = {"respond": 0, "strong": 0, "weak": 0, "miss": 0,
                    "clusters": 0, "cards": 0}
    # 自动补词网状态（服务进程内持续统计）
    _WEAK_HITS = {}  # 查询→{count, top_card, last_at}

    def _auto_cluster(self, condition, results):
        """弱命中统计与自动建簇（零 LLM）：同一查询多次弱命中（top<0.5）→
        从查询提取俗语词，建簇映射到已命中卡（规范词=卡名）→
        写 clusters_ext.json + 动态更新翻译表。"""
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
                # 首+中+尾 2-gram（中间片段承载核心语义）
                _mid = (len(s) - 1) // 2
                words.add(s[:2]); words.add(s[_mid:_mid + 2]); words.add(s[-2:])
        _QW = set("怎么 什么 如何 为啥 为什么 哪些 哪个 多少 哪里 何时 是否 有没有 能不能 会不会 怎么样 怎样 这样 那样 一个 一种 一下 起来 以后 内容 东西".split())
        # 已覆盖词：翻译表 + 目标卡 trigger（从卡库查——避免建无用簇）
        from aeis_core import MemoryLayer as _ML
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

    def log_message(self, *args):
        """静默访问日志：默认逐请求刷屏，覆写为空。"""
        pass

    def _send(self, obj, code=200):
        """JSON 应答统一出口：编码·头·状态码收口。"""
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read(self):
        """读取请求体并按 UTF-8 JSON 解析。"""
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n).decode("utf-8"))

    # ---------------- GET ----------------

    def do_GET(self):
        """GET 三路分发：UI 页面·状态 API·数据端点。"""
        path = self.path.split("?")[0]
        qs = {}
        if "?" in self.path:
            qs = urllib.parse.parse_qs(self.path.split("?", 1)[1])
        if path in ("/", "/ui", "/index.html", "/ui/index.html", "/wisdom_ui.html"):
            self._send_html()
        elif path in ("/chat", "/chat.html", "/chat/index.html"):
            self._send_chat_html()
        elif path == "/dex/status":
            self._send(self._status())
        elif path == "/dex/ledger":
            c = qs.get("contributor", [None])[0]
            self._send(self._ledger(c))
        else:
            self._send({"error": "not_found"}, 404)

    def _send_chat_html(self):
        """普通人对话界面（H5 聊天式 · 第一智能入口）"""
        html_path = os.path.join(HERE, "chat.html")
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                body = f.read().encode("utf-8")
        except OSError:
            self._send({"error": "chat_ui_not_found"}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
        """POST 分发：UTF-8 JSON 解析后按 path 派发（坏包 400）。"""
        try:
            body = self._read()
        except Exception:
            self._send({"error": "bad_json"}, 400)
            return
        path = self.path.split("?")[0]
        if path == "/chat":
            self._send(self._chat(body))
        elif path == "/dex/query":
            self._send(self._query(body))
        elif path == "/dex/upload":
            self._send(self._upload(body))
        else:
            self._send({"error": "not_found"}, 404)

    # ---------------- 普通人对话 ----------------

    def _chat(self, body):
        """普通人对话端点：人话检索 + 情感 + 记忆 + 诚实边界（chat_engine）
        记忆挂在 cloud 实例上（handler 每请求新建，cloud 是单例 → 跨请求共享）"""
        try:
            import chat_engine as _ce
            cloud = self.cloud
            if not hasattr(cloud, "_chat_memory"):
                cloud._chat_memory = {}
            return _ce.chat(cloud, body.get("message", ""),
                            session_id=body.get("session_id", "default"),
                            memory=cloud._chat_memory)
        except Exception as e:
            return {"error": f"chat_failed: {e}", "reply": "我暂时没反应过来，稍等再试？",
                    "hits": [], "emotion": None}

    # ---------------- 实现 ----------------

    def _query(self, body):
        """查询参数解析：query/domain/limit 三键归一。"""
        op = body.get("op", "")
        params = body.get("params") or {}
        d = self.cloud
        try:
            if op == "filter":
                return {"op": op, "results": d.dex_filter(**params)}
            if op == "learn_stats":
                # DSH 侧：补卡/收敛观测仪表盘（engram 融合）
                return {"op": op, "results": dict(self._LEARN_STATS)}
            if op == "add_card":
                # DSH 侧：自动补卡端点（relay 缺口闭环调用）——add_entry 写入知识卡
                name = params.get("name", "")
                if not name:
                    return {"op": op, "ok": False, "error": "name required"}
                if str(params.get("source", "")) == "auto-gap":
                    self._LEARN_STATS["cards"] += 1
                # 同名查重（加载既有库时 _by_name 为空 → 卡库级扫描，重复卡真实防写入）
                try:
                    from aeis_core import MemoryLayer as _ML2
                    _dup = [n for n in d.store.query_nodes(layer=_ML2.KNOWLEDGE, limit=2000)
                            if (n.state_attributes or {}).get("name") == name]
                    if _dup:
                        return {"op": op, "ok": True, "existed": True, "name": name}
                except Exception:
                    pass
                from aeis_core import ConditionSpace
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
                # 知识翻译体系全链路（四路融合：语义指纹+学科路由+二元组+神经索引）
                # + DSH 侧增强：algo 三通道重排（零 LLM）+ 弱命中自动补词网 + 学习统计
                try:
                    import semantic_translate as _st
                except Exception:
                    _st = None
                try:
                    results = _st.graph_retrieve(d, params.get("condition", ""), limit=8)
                except Exception:
                    results = d.dex_respond(params.get("condition", ""), translator=_st)
                if str(params.get("algo", "1")) not in ("0", "false", "False"):
                    try:
                        from semantic_algo import algo_rerank
                        results = algo_rerank(d, params.get("condition", ""), results)
                    except Exception:
                        pass
                # DSH 侧字段规整：新翻译层会返「口语直答」条目（id=null · _from_daily ·
                # 只有 daily 文本、无 level/status/action）——engram 侧按
                # L{level}/{status}：{action} 渲染出招，不补则出招文本为空、信息被丢。
                for _h in (results or []):
                    if not _h.get("action"):
                        _h["action"] = _h.get("daily") or _h.get("direct_answer") or ""
                    if not _h.get("status"):
                        _h["status"] = "verified" if _h.get("id") else "daily"
                # 保底卡锚：整批都没有卡背书（全是口语直答）时，追卡库出招——
                # 保证「直答 + 可溯源卡」两全（engram 融合：知道出处才敢用）。
                # 选取即按「具体出招卡优先、模板学科卡兜底」排序：否则种子卡会把
                # 具体识别卡挤出名额（实测：生锈题只有 2 个名额，全被学科卡占）。
                if results and not any(x.get("id") for x in results):
                    try:
                        _extra = d.dex_respond(params.get("condition", ""), translator=_st) or []
                        _have = {x.get("name") for x in results}

                        def _rank(e):
                            _a = (e.get("action") or "").strip()
                            return (0 if (_a and not _a.endswith("知识点回应")) else 1,
                                    -(e.get("score") or 0))

                        for _e in sorted([x for x in _extra if x.get("name") not in _have],
                                         key=_rank)[:3]:
                            results.append(_e)
                    except Exception:
                        pass
                cluster_hint = None
                try:
                    cluster_hint = self._auto_cluster(params.get("condition", ""), results)
                except Exception:
                    cluster_hint = None
                try:
                    self._LEARN_STATS["respond"] += 1
                    top_s = results[0].get("score") if results else 0
                    if top_s >= 0.5:
                        self._LEARN_STATS["strong"] += 1
                    elif top_s >= 0.1:
                        self._LEARN_STATS["weak"] += 1
                    else:
                        self._LEARN_STATS["miss"] += 1
                    if cluster_hint:
                        self._LEARN_STATS["clusters"] += 1
                    # 学习趋势持久化（每 20 次响应快照——跨重启长期观测）
                    if self._LEARN_STATS["respond"] % 20 == 0:
                        try:
                            import datetime as _dt
                            _logdir = os.path.join(HERE, "audit_log")
                            os.makedirs(_logdir, exist_ok=True)
                            with open(os.path.join(_logdir, "learn_stats.log"), "a", encoding="utf-8") as _f:
                                _f.write(_dt.datetime.now().isoformat(timespec="seconds") + " " +
                                         json.dumps(self._LEARN_STATS, ensure_ascii=False) + "\n")
                        except Exception:
                            pass
                except Exception:
                    pass
                # DSH 侧排序（落库学科卡后加的护栏）：上游种子卡的 response.action 是
                # 模板句「以X知识点回应」——它是知识容器、不是出招，触发词又长，容易在
                # 通用词上与具体识别卡同分甚至更高，把可执行出招挤出 top3（实测：生锈题
                # 被「初中化学 6.40」顶掉「化学_氧化还原」）。规则：模板卡降权 0.5（保留
                # 相对次序 → 学科题如「量子力学是什么」仍能命中该卡），口语直答恒排首位。
                for _h in (results or []):
                    _a = (_h.get("action") or "").strip()
                    if _a.endswith("知识点回应"):
                        _h["templated"] = True
                        _h["score"] = round((_h.get("score") or 0) * 0.5, 4)
                try:
                    results.sort(key=lambda x: (0 if x.get("status") == "daily" else 1,
                                                -(x.get("score") or 0)))
                except Exception:
                    pass
                # 尊重调用方 limit（上游此处硬编码 8，无视 params.limit；落库后 8 条里
                # 后段多为同族学科卡，对「出招」是噪声）：默认仍 8，显式传则照传。
                try:
                    _lim = int(params.get("limit", 8) or 8)
                    if _lim > 0:
                        results = results[:_lim]
                except Exception:
                    pass
                return {"op": op, "results": results, "cluster": cluster_hint,
                        "learn": dict(self._LEARN_STATS)}
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
                return {"op": op, "results": d.dex_auto_verify(
                    params.get("knowledge", ""),
                    limit=int(params.get("limit", 5)),
                    threshold=float(params.get("threshold", 0.50)))}
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
            if op == "impact":
                return {"op": op, "results": d.dex_impact(
                    params.get("node_id", ""),
                    max_depth=int(params.get("max_depth", 3)))}
            if op == "chain":
                return {"op": op, "results": d.dex_chain(
                    params.get("node_id", ""),
                    max_depth=int(params.get("max_depth", 5)))}
            if op == "verify":
                return {"op": op, "results": d.dex_verify(
                    params.get("node_id", ""))}
            if op == "hot_paths":
                import os as _os
                hp = os.path.join(HERE, 'audit_log', 'chain_heat.json')
                if _os.path.exists(hp):
                    import json as _json
                    with open(hp, encoding='utf-8') as _f:
                        data = _json.load(_f)
                    chains = {k: v for k, v in data.items() if '→' in k}
                    singles = {k.replace('单卡:', ''): v
                               for k, v in data.items() if k.startswith('单卡:')}
                    return {"op": op, "results": {
                        "chains": sorted(chains.items(),
                                         key=lambda x: -x[1])[:10],
                        "singles": sorted(singles.items(),
                                          key=lambda x: -x[1])[:10],
                        "total": sum(singles.values())}}
                return {"op": op, "results": {"chains": [], "singles": [],
                                              "total": 0}}
            if op == "audit_danmaku":
                # 直播弹幕审核（三层判定：词表→信任上下文→终裁）
                try:
                    import danmaku_audit as _da
                    return {"op": op, "results": _da.audit(
                        params.get("text", ""))}
                except Exception:
                    return {"op": op, "error": "danmaku_audit 模块不可用"}
            if op == "audit_log_recent":
                import os as _os
                import json as _json
                log = os.path.join(HERE, 'audit_log', 'danmaku_audit.json')
                if _os.path.exists(log):
                    with open(log, encoding='utf-8') as _f:
                        data = _json.load(_f)
                    return {"op": op, "results": data[-10:]}
                return {"op": op, "results": []}
            return {"op": op, "error": "unknown_op"}
        except Exception as e:
            return {"op": op, "error": str(e)}

    def _upload(self, body):
        """云端上传处理：verified 状态与验证轨迹完整性闸门，不符即拒绝并说明原因。"""
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
        """台账快照：近期变更记录列表直出。"""
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
        """云库状态统计：知识层节点总量与 verified 占比汇总。"""
        from aeis_core import MemoryLayer
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


SEED_CARDS_DIR = os.path.join(os.path.dirname(HERE), "seed_knowledge", "wisdom_cards")


def _parse_card_md(path):
    """解析卡 md → (name, domain, edu, kp_dict)。"""
    import re as _re
    with open(path, encoding="utf-8") as f:
        text = f.read()
    name = os.path.basename(path).replace("·知识综述.md", "").replace(".md", "")
    domain = None
    m = _re.search(r'^- \*\*领域\*\*: (.+)$', text, _re.M)
    if m:
        domain = m.group(1).strip()
    edu = None
    m = _re.search(r'^- \*\*教育层级\*\*: (E\d)', text, _re.M)
    if m:
        edu = m.group(1).strip()
    kp_start = text.find("## 知识点内容（按骨架填充）")
    kps = []
    if kp_start >= 0:
        kp_end = len(text)
        nxt = text.find("\n## ", kp_start + 10)
        if nxt >= 0:
            kp_end = nxt
        seg = text[kp_start:kp_end]
        cur = None
        for ln in seg.split("\n"):
            s = ln.strip()
            if s.startswith("### "):
                cur = s[4:].strip()
                kps.append([cur, []])
            elif cur and s and not s.startswith("#"):
                kps[-1][1].append(s)
    else:
        # 旧格式兼容：知识分层（### E2/E3/E4 各节）→ 每节一句
        kp_start = text.find("## 知识分层")
        if kp_start >= 0:
            kp_end = len(text)
            nxt = text.find("\n## ", kp_start + 10)
            if nxt >= 0:
                kp_end = nxt
            seg = text[kp_start:kp_end]
            cur = None
            for ln in seg.split("\n"):
                s = ln.strip()
                if s.startswith("### "):
                    cur = s[4:].strip()
                    kps.append([cur, []])
                elif cur and s and not s.startswith("#"):
                    kps[-1][1].append(s)
    kp_dict = {k[0]: " ".join(k[1]) for k in kps if k[1]}
    return name, domain, edu, kp_dict


def _seed_cards(dex):
    """从本地打包的卡源重建知识卡（首启种子，离线可用）。返回新增数。"""
    from aeis_core import MemoryLayer, ConditionSpace
    # DSH 侧修正：查重窗口 500 → 5000。本仓现役库已有近千节点，500 窗口外的既有卡名
    # 查不到 → 会把同名卡再建一遍（重复卡污染 respond/verify 候选）。窗口必须盖住全库。
    existing = {n.state_attributes.get("name")
                for n in dex.store.query_nodes(layer=MemoryLayer.KNOWLEDGE, limit=5000)
                if n.state_attributes.get("name")}
    added = 0
    if not os.path.isdir(SEED_CARDS_DIR):
        return 0
    for fn in sorted(os.listdir(SEED_CARDS_DIR)):
        if not fn.endswith(".md"):
            continue
        path = os.path.join(SEED_CARDS_DIR, fn)
        name, domain, edu, kps = _parse_card_md(path)
        if not kps or name in existing:
            continue
        first_kp = next(iter(kps))
        claim = (f"{name}（知识卡源 {len(kps)} 知识点）——{kps[first_kp][:60]}……")
        cs = ConditionSpace(
            observation_position=f"{name} 外部观测位",
            observation_tool="知识卡源种子",
            time_window=(0.0, 9999999999.0),
            existence_constraint="通用现象/规律不受版权保护，开源非盈利知识库")
        response = {
            "trigger": f"涉及{name}议题（如：{'、'.join(list(kps)[:10])}）",
            "action": f"以{name}知识点回应",
        }
        level = {"E1": 1, "E2": 2, "E3": 3, "E4": 4, "E5": 5}.get(edu, 2)
        nid = dex.add_entry(name, domain or "未分类", claim, cs,
                            level=level, status="verified", response=response)
        node = dex.store.get_node(nid)
        node.state_attributes["edu_level"] = edu
        node.state_attributes["source_kind"] = "card_seed"
        node.content = claim + "\n" + "\n".join(
            f"{i+1}. {v}" for i, v in enumerate(kps.values()))
        dex.store.add_node(node)
        added += 1
    return added


def run_server(port=0, db_path=None):
    """启动智慧之书云（daemon 线程）。port=0 → 自动分配空闲端口。

    知识库策略：保留已有库（不删除）；缺卡时从本地种子重建（首启）。
    """
    db = db_path or CLOUD_DB
    if os.path.exists(db) and os.path.getsize(db) < 1024:
        os.remove(db)  # 空壳库（<1KB）重建
    if not os.path.exists(db):
        dex = ConditionDex(db_path=db, fresh=True)
        dex.seed_base()
    else:
        dex = ConditionDex(db_path=db, fresh=False)
    dex.store.conn.execute(
        "CREATE TABLE IF NOT EXISTS contributions ("
        "entry_id TEXT PRIMARY KEY, contributor TEXT, verified_by TEXT, "
        "verified_at REAL, weight REAL)")
    added = _seed_cards(dex)
    if added:
        dex.store.conn.commit()
        print(f"[seed] 首启重建 {added} 张知识卡（本地种子）")
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
