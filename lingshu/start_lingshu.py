# -*- coding: utf-8 -*-
"""风识 · 灵枢校准器自愈启动（发布包版）。

用法：python lingshu/start_lingshu.py [端口]
服务：127.0.0.1:18766（默认），崩溃 1s 自动重启；首次启动自动播种卡库。
"""
import os, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
AEIS = os.path.join(HERE, "aeis")
sys.path.insert(0, AEIS)
sys.path.insert(0, os.path.join(AEIS, "wisdom"))

DB = os.path.join(HERE, "engram-fusion-full.db")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18766


def build_dex():
    from wisdom_book import ConditionDex
    from aeis.core import MemoryLayer
    dex = ConditionDex(db_path=DB, fresh=False)
    # contributions 表（/dex/status 依赖）
    dex.store.conn.execute(
        "CREATE TABLE IF NOT EXISTS contributions ("
        "entry_id TEXT PRIMARY KEY, contributor TEXT, verified_by TEXT, "
        "verified_at REAL, weight REAL)")
    dex.store.conn.commit()
    # 首次启动播种：卡库为空时从知识卡 md 建库
    names = [n.state_attributes.get("name") for n in
             dex.store.query_nodes(layer=MemoryLayer.KNOWLEDGE, limit=10)
             if (n.state_attributes or {}).get("name")]
    if not names:
        seed_cards(dex)
    return dex


def seed_cards(dex):
    """播种：识别卡 md（26 张）→ 卡库；加 18 张实用卡种子。"""
    import glob
    from aeis.core import ConditionSpace
    card_dir = os.path.join(HERE, "knowledge")
    # 遍历 knowledge/ 下所有子目录（识别卡 + 补卡）
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
    print(f"seed: 识别卡播种完成", flush=True)


def serve():
    from wisdom_book import ConditionDex
    from wisdom_cloud import DexHandler
    from http.server import ThreadingHTTPServer
    dex = build_dex()
    DexHandler.cloud = dex
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), DexHandler)
    print(f"风识·灵枢校准器 on {PORT}（watchdog 自愈）", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    while True:
        try:
            serve()
        except KeyboardInterrupt:
            break
        except Exception:
            traceback.print_exc()
            print("校准器崩溃——1s 后重启", flush=True)
            time.sleep(1)
