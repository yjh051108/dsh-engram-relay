"""
Engram 转接服务（本地 JSON 行协议，供 Node 插件 spawn 对接）。

协议：stdin 读 JSON 行请求，stdout 写 JSON 行响应。
请求：
  {"id": 1, "op": "load", "model": "Qwen/Qwen3-0.6B", ...}
  {"id": 2, "op": "generate", "text": "...", "max_new_tokens": 64}
  {"id": 3, "op": "distill", "conversation": "...", "session": "s1", "turn": 3}
  {"id": 4, "op": "write_memory", "slots": [..], "embeds": [[..]]}
  {"id": 5, "op": "status"}
  {"id": 6, "op": "embed", "texts": ["..."], "query": "..."}
响应：
  {"id": 1, "ok": true, "result": {...}}
  {"id": 2, "ok": false, "error": "..."}

两条能力线：
  - embed（现行核心）：专用小型嵌入模型（bge-small-zh-v1.5）做语义
    精排。模型路径取请求的 embed_model 或环境变量 ENGRAM_EMBED_MODEL；
    懒加载，首次 embed 才载入。
  - 蒸馏/生成（遗留）：原 0.6B 魔改模型（Engram 条件记忆 × DSA 路由）
    的蒸馏/回忆能力。模型未配置或路径不存在时 load 返回
    loaded:false，相关 op 全部优雅报错，Node 侧降级为纯图谱检索。
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
    def __init__(self, model_id: str = "", **kwargs):
        self.model: EngramQwen3 | None = None
        self.model_id = model_id
        self.kwargs = kwargs
        self.embed_model = None  # sentence-transformers 嵌入模型（懒加载）
        self.embed_model_id: str | None = None

    @staticmethod
    def _unwrap(model):
        """循环解包 PeftModel 到最底层（LoRA 包装后 embed_tokens 路径变化）。"""
        while hasattr(model, "get_base_model"):
            model = model.get_base_model()
        return model

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
        if op == "embed":
            return self._embed(req)
        if op == "status":
            return self._status(req)
        return {"ok": False, "error": f"unknown op: {op}"}

    def _load(self, req: dict) -> dict:
        if self.model is not None:
            return {"ok": True, "result": {"loaded": True, "reused": True}}
        model_id = req.get("model", self.model_id) or ""
        if model_id == "" or not os.path.isdir(model_id):
            return {"ok": True, "result": {
                "loaded": False,
                "reason": f"model dir not found: {model_id or '(not configured)'}",
            }}
        self.model = EngramQwen3(
            model_id=model_id,
            engram_layers=tuple(req.get("engram_layers", (1, 7))),
            **self.kwargs,
        )
        # 加载训练好的记忆模型（原生 engram：训练过的记忆表 + gate + LoRA）
        checkpoint = req.get("checkpoint") or os.environ.get("ENGRAM_CHECKPOINT")
        lora_loaded = False
        if checkpoint and os.path.exists(checkpoint):
            ckpt = torch.load(checkpoint, map_location=self.model.device, weights_only=False)
            self.model.memory.pool.data.copy_(ckpt["memory_pool"].to(self.model.memory.pool.dtype))
            self.model.memory.slot_index.copy_(ckpt["slot_index"])
            for k, sd in ckpt["engram_modules"].items():
                if k in self.model.engram_modules:
                    self.model.engram_modules[k].load_state_dict(sd)
            # LoRA 适配器
            lora_dir = os.path.join(os.path.dirname(checkpoint), "lora")
            if os.path.exists(lora_dir):
                try:
                    from peft import PeftModel
                    self.model.model = PeftModel.from_pretrained(self.model.model, lora_dir)
                    lora_loaded = True
                except Exception as e:
                    print(f"[engram-server] LoRA 加载失败（继续用 base）: {e}")
            result = {"loaded": True, "stats": self.model.memory_stats(), "checkpoint": checkpoint,
                      "lora": lora_loaded, "eval_acc": ckpt.get("eval_acc")}
        else:
            result = {"loaded": True, "stats": self.model.memory_stats()}
        return {"ok": True, "result": result}

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
        """把记忆文本写入记忆表：文本 → 模型编码（hidden states 池化）→ 哈希槽位写入。

        多头哈希的 16 个通道对应不同模数；查询时 fused module 会查全部
        通道，因此写入时把每条记忆写到前 WRITE_CHANNELS 个通道的槽位，
        保证「写入必可被查询命中」，同时分散同主题碰撞。
        """
        self._ensure()
        entries = req["entries"]  # [{text, label?, kind?}, ...]
        write_channels = int(req.get("write_channels", 4))
        slot_ids: list[int] = []
        embeds: list[list[float]] = []
        for entry in entries:
            text = entry["text"]
            inputs = self.model.tokenizer(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                base = self._unwrap(self.model.model)
                out = base.model.embed_tokens(inputs.input_ids)
                vec = out.mean(dim=1).squeeze(0).cpu().tolist()
            slots = self.model.engram_slots_for(inputs.input_ids)
            # 写到前 write_channels 个通道的槽位（每通道一个槽 id）
            for ch in range(min(write_channels, slots.shape[-1])):
                slot_ids.append(int(slots[0, 0, ch].item()))
                embeds.append(vec)
        self.model.write_memory(
            slot_ids,
            torch.tensor(embeds, dtype=self.model.dtype, device=self.model.device),
        )
        return {"ok": True, "result": {"written": len(entries), "slots": slot_ids[:len(entries)]}}

    def _embed(self, req: dict) -> dict:
        """文本编码（bge-small-zh-v1.5，语义精排用）。

        texts 与 query 一起编码；query 向量单独返回，Node 侧做余弦
        相似度重排（hash 粗筛候选内）。模型懒加载：路径取请求的
        embed_model，缺省用环境变量 ENGRAM_EMBED_MODEL；未配置时
        返回 ok:false（Node 侧降级为纯 hash + 重要度）。
        """
        model_id = req.get("embed_model") or os.environ.get("ENGRAM_EMBED_MODEL") or ""
        if model_id == "" or not os.path.isdir(model_id):
            return {"ok": False, "error": f"embed model not found: {model_id or '(ENGRAM_EMBED_MODEL unset)'}"}
        if self.embed_model is None or self.embed_model_id != model_id:
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer(model_id)
            self.embed_model_id = model_id
        texts = req.get("texts") or []
        if not texts:
            return {"ok": False, "error": "texts empty"}
        query = req.get("query") or ""
        all_texts = texts + ([query] if query else [])
        vecs = self.embed_model.encode(all_texts, normalize_embeddings=True)
        n = len(texts)
        result = {"vectors": [v.tolist() for v in vecs[:n]]}
        if query:
            result["query_vec"] = vecs[n].tolist()
        return {"ok": True, "result": result}

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
    # 强制 UTF-8 三流（Windows 管道默认按 GBK 编码，Node 侧按 utf8 收发）
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
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
