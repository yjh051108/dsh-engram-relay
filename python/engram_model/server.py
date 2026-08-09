"""
Engram 转接服务（本地 JSON 行协议，供 Node 插件 spawn 对接）。

协议：stdin 读 JSON 行请求，stdout 写 JSON 行响应。
请求：
  {"id": 1, "op": "load", "model": "Qwen/Qwen3-0.6B", ...}
  {"id": 2, "op": "generate", "text": "...", "max_new_tokens": 64}
  {"id": 3, "op": "distill", "conversation": "...", "session": "s1", "turn": 3}
  {"id": 4, "op": "write_memory", "slots": [..], "embeds": [[..]]}
  {"id": 5, "op": "status"}
响应：
  {"id": 1, "ok": true, "result": {...}}
  {"id": 2, "ok": false, "error": "..."}

蒸馏（distill）：用魔改模型的生成能力把对话转为结构化记忆 JSON；
Node 插件把记忆文本经 write_memory 写入记忆表（文本 → 嵌入用模型
的 hidden states 编码，保证与查询在同一表示空间）。
"""

from __future__ import annotations

import json
import os
import sys

# 允许作为脚本直接运行（python server.py）
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

try:
    from .model import EngramQwen3
except ImportError:
    from model import EngramQwen3


class EngramServer:
    def __init__(self, model_id: str = "Qwen/Qwen3-0.6B", **kwargs):
        self.model: EngramQwen3 | None = None
        self.model_id = model_id
        self.kwargs = kwargs

    def handle(self, req: dict) -> dict:
        op = req.get("op")
        if op == "load":
            return self._load(req)
        if op == "generate":
            return self._generate(req)
        if op == "distill":
            return self._distill(req)
        if op == "write_memory":
            return self._write_memory(req)
        if op == "status":
            return self._status(req)
        return {"ok": False, "error": f"unknown op: {op}"}

    def _load(self, req: dict) -> dict:
        if self.model is not None:
            return {"ok": True, "result": {"loaded": True, "reused": True}}
        self.model = EngramQwen3(
            model_id=req.get("model", self.model_id),
            engram_layers=tuple(req.get("engram_layers", (1, 7))),
            **self.kwargs,
        )
        return {"ok": True, "result": {"loaded": True, "stats": self.model.memory_stats()}}

    def _generate(self, req: dict) -> dict:
        self._ensure()
        text = req["text"]
        inputs = self.model.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            inputs.input_ids,
            max_new_tokens=int(req.get("max_new_tokens", 64)),
            temperature=float(req.get("temperature", 0.2)),
        )
        generated = self.model.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return {"ok": True, "result": {"text": generated}}

    def _distill(self, req: dict) -> dict:
        """把对话蒸馏为结构化记忆 JSON（强约束 prompt + JSON 提取）。"""
        self._ensure()
        conversation = req["conversation"][:1500]
        prompt = (
            "你是记忆蒸馏器。把下面的对话提炼为记忆条目，只输出 JSON，不要其他文字。\n\n"
            '格式：{"kind":"fact|decision|event|preference","label":"15字内标签","text":"50字内正文","importance":0到1的数字,"causes":[]}\n\n'
            f"对话：\n{conversation}\n\nJSON："
        )
        inputs = self.model.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(inputs.input_ids, max_new_tokens=120, temperature=0.2)
        raw = self.model.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        parsed = extract_json(raw)
        return {"ok": True, "result": {"raw": raw, "parsed": parsed}}

    def _write_memory(self, req: dict) -> dict:
        """把记忆文本写入记忆表：文本 → 模型编码（hidden states 池化）→ 哈希槽位写入。"""
        self._ensure()
        entries = req["entries"]  # [{text, label?, kind?}, ...]
        slot_ids: list[int] = []
        embeds: list[list[float]] = []
        for entry in entries:
            text = entry["text"]
            inputs = self.model.tokenizer(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.model.model.embed_tokens(inputs.input_ids)
                vec = out.mean(dim=1).squeeze(0).cpu().tolist()
            slots = self.model.engram_slots_for(inputs.input_ids)
            # 取该文本第一个位置的槽位（多头哈希的首槽）
            slot_ids.append(int(slots[0, 0, 0].item()))
            embeds.append(vec)
        self.model.write_memory(
            slot_ids,
            torch.tensor(embeds, dtype=self.model.dtype, device=self.model.device),
        )
        return {"ok": True, "result": {"written": len(entries), "slots": slot_ids}}

    def _status(self, req: dict) -> dict:
        if self.model is None:
            return {"ok": True, "result": {"loaded": False}}
        return {"ok": True, "result": {"loaded": True, **self.model.memory_stats()}}

    def _ensure(self):
        if self.model is None:
            self.model = EngramQwen3(model_id=self.model_id, **self.kwargs)


def extract_json(raw: str) -> dict | None:
    """提取首个 {...} 并解析（0.5B 输出不稳定，容忍前后缀）。"""
    start = raw.find("{")
    if start < 0:
        return None
    # 括号配对
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def main():
    server = EngramServer()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"ok": False, "error": "bad json"}, ensure_ascii=False))
            sys.stdout.flush()
            continue
        try:
            resp = server.handle(req)
        except Exception as e:  # noqa: BLE001
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        resp["id"] = req.get("id")
        print(json.dumps(resp, ensure_ascii=False))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
