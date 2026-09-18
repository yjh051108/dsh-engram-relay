# -*- coding: utf-8 -*-
"""风识 · 灵枢校准器自愈启动（v0.5.0 形态 · 上游 whitebox_kb 布局）。

用法：python lingshu/start_lingshu.py [端口] [db路径]
服务：127.0.0.1:18766（默认），崩溃 1s 自动重启；首次启动自动播种卡库。

布局说明（2026-09-11 同步自 FuRongJun-1999/dsh-memory@main，tag 3.4-aeis-0.5.0 时期）：
  whitebox_kb/{wisdom,aeis_core} 同级互导（上游 md_cg/whitebox_kb 原样落位）；
  wisdom_cloud.py 内含 DSH 侧扩展（add_card / learn_stats / respond 增强），
  该三处在同步上游时须按注释回植——上游本体没有。
旧版 vendored 树保留在 lingshu/aeis/（回滚用，不再被本脚本引用）。
"""
import os, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))

# ── 黑匣子（2026-09-11 加）：监督器以 stdio:'ignore' 拉起本服务，输出全被丢弃——
# 服务"起得来、服务几拍后无声消失"时无从取证（实测踩过：进程没了 + watchdog 未留痕
# = 硬崩而非 Python 异常）。故把 stdout/stderr 与 faulthandler 都落盘：
# 段错误/栈溢出会在此写出 C 级栈，日志无痕则说明是被外部 TerminateProcess 杀掉。
_LOG_DIR = os.path.join(HERE, "audit_log")
try:
    os.makedirs(_LOG_DIR, exist_ok=True)
    _LOG = open(os.path.join(_LOG_DIR, "start.log"), "a", buffering=1, encoding="utf-8")
    import faulthandler
    faulthandler.enable(_LOG)
    sys.stdout = _LOG
    sys.stderr = _LOG
except Exception:
    _LOG = None


def _box(msg):
    """黑匣子写一行（带时间戳）。"""
    if _LOG is not None:
        try:
            _LOG.write("%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg))
        except Exception:
            pass

WB = os.path.join(HERE, "whitebox_kb")
# 上游布局：wisdom/ 与 aeis_core/ 同级互导（wisdom_book 从 aeis_core 导入）
for _p in (WB, os.path.join(WB, "wisdom")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DB = os.path.join(HERE, "engram-fusion-full.db")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18766
if len(sys.argv) > 2:
    DB = sys.argv[2]


def _say(msg):
    """UTF-8 安全打印：Windows GBK 管道打不了中文，别让一行日志打死服务（项目已记录该坑）。"""
    try:
        print(msg, flush=True)
    except Exception:
        try:
            sys.stdout.write(msg.encode("ascii", "replace").decode("ascii") + "\n")
            sys.stdout.flush()
        except Exception:
            pass


def build_dex():
    from wisdom_book import ConditionDex
    from aeis_core import MemoryLayer
    dex = ConditionDex(db_path=DB, fresh=False)
    # 瞬时锁韧性：WAL 库上并发/残留写事务会退化为 "database is locked"（实测：
    # 热重载打断写事务后，后续写全数立即失败）；busy_timeout 让写等待而非立刻失败。
    try:
        dex.store.conn.execute("PRAGMA busy_timeout = 5000")
    except Exception:
        pass
    # contributions 表（/dex/status 与 /dex/ledger 依赖）
    dex.store.conn.execute(
        "CREATE TABLE IF NOT EXISTS contributions ("
        "entry_id TEXT PRIMARY KEY, contributor TEXT, verified_by TEXT, "
        "verified_at REAL, weight REAL)")
    dex.store.conn.commit()
    # 首次启动播种：卡库为空时从知识卡 md 建库（非空库绝不重播种——保数据）
    names = [n.state_attributes.get("name") for n in
             dex.store.query_nodes(layer=MemoryLayer.KNOWLEDGE, limit=10)
             if (n.state_attributes or {}).get("name")]
    if not names:
        seed_cards(dex)
    return dex


def seed_cards(dex):
    """播种：knowledge/ 下所有 md（识别卡 + 补卡 + 理论）→ 卡库。"""
    import glob
    from aeis_core import ConditionSpace
    card_dir = os.path.join(HERE, "knowledge")
    for f in sorted(glob.glob(os.path.join(card_dir, "**", "*.md"), recursive=True)):
        try:
            text = open(f, encoding="utf-8").read()
        except Exception:
            continue
        import re as _re

        def grab(pat):
            m = _re.search(pat, text)
            return m.group(1).strip() if m else ""

        name = os.path.basename(f)[:-3]
        cs = ConditionSpace(observation_position="识别卡", observation_tool="识别卡",
                            time_window=(0.0, 1e10), existence_constraint="")
        dex.add_entry(
            name=name,
            domain=grab(r"\*\*领域\*\*: (.+)") or name,
            claim=grab(r"## 核心主张\s*\n\s*(.+)") or "（识别卡）",
            cs=cs,
            level=int(grab(r"\*\*层级\*\*: L(\d)") or 2),
            status=grab(r"\*\*状态\*\*: (\w+)") or "verified",
            response={"trigger": grab(r"\*\*触发\*\*: (.+)"),
                      "action": grab(r"\*\*行动\*\*: (.+)"),
                      "counters": grab(r"\*\*克制\*\*: (.+)")},
            tags=[f"domain:{grab(r'\*\*领域\*\*: (.+)') or name}"],
            card2={"source": "seed"})
    _say("seed: 识别卡播种完成")


def serve():
    from wisdom_book import ConditionDex
    from wisdom_cloud import DexHandler
    from http.server import ThreadingHTTPServer
    dex = build_dex()
    DexHandler.cloud = dex
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), DexHandler)
    _box("listening on %d db=%s (pid=%d)" % (PORT, DB, os.getpid()))
    _say(f"风识·灵枢校准器 on {PORT}（watchdog 自愈 · whitebox_kb 布局 · db={DB}）")
    srv.serve_forever()
    _box("serve_forever returned (服务循环退出)")


if __name__ == "__main__":
    _box("=== boot: argv=%r pid=%d ===" % (sys.argv[1:], os.getpid()))
    while True:
        try:
            serve()
        except KeyboardInterrupt:
            _box("KeyboardInterrupt → 退出")
            break
        except Exception:
            _box("CRASH（Python 异常，1s 后重启）:\n" + traceback.format_exc())
            traceback.print_exc()
            _say("校准器崩溃——1s 后重启")
            time.sleep(1)
