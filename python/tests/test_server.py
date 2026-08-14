"""
server.py JSON 行协议集成测试：Node 插件对接的真实接口验证。

用子进程起 server，发 load / write_memory / generate / distill / status。
"""

import json
import os
import subprocess
import sys
import time

SERVER = os.path.join(os.path.dirname(__file__), "..", "engram_model", "server.py")
MODEL_PATH = os.environ.get("ENGRAM_MODEL_PATH", "")


def rpc(proc, req: dict, timeout=300):
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.flush()
    deadline = time.time() + timeout
    buf = ""
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.1)
            continue
        buf = line.strip()
        if buf:
            return json.loads(buf)
    raise TimeoutError(f"no response for {req['op']}")


def main():
    env = dict(os.environ)
    env["ENGRAM_MODEL_PATH"] = MODEL_PATH
    python_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # python/
    proc = subprocess.Popen(
        [sys.executable, "-m", "engram_model.server"],
        cwd=python_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        # 1. load（模型加载，首次约 10s）
        t0 = time.time()
        resp = rpc(proc, {"id": 1, "op": "load", "model": MODEL_PATH}, timeout=600)
        assert resp["ok"], resp
        print(f"✓ load in {time.time()-t0:.1f}s")

        # 2. status
        resp = rpc(proc, {"id": 2, "op": "status"})
        assert resp["ok"] and resp["result"]["loaded"]
        print("✓ status:", resp["result"])

        # 3. write_memory（蒸馏一条记忆进模型记忆表）
        resp = rpc(proc, {"id": 3, "op": "write_memory", "entries": [{"text": "项目部署端口是 8080"}]})
        assert resp["ok"] and resp["result"]["written"] == 1
        print("✓ write_memory slot:", resp["result"]["slots"])

        # 4. distill（对话 → 结构化记忆 JSON）
        resp = rpc(proc, {"id": 4, "op": "distill", "conversation": "用户说：数据库用 PostgreSQL，测试环境端口 8081。"})
        assert resp["ok"]
        print("✓ distill parsed:", resp["result"]["parsed"])

        # 5. generate（带 engram 注入的生成）
        resp = rpc(proc, {"id": 5, "op": "generate", "text": "你好", "max_new_tokens": 16})
        assert resp["ok"]
        print("✓ generate:", repr(resp["result"]["text"]))

        print("\n=== server integration PASS ===")
    finally:
        proc.stdin.close()
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
