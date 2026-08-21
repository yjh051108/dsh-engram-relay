"""
严格质量评估：训练后的原生 engram 记忆到底提升多少？

三组对照（同一 32 问，同一 eval 知识）：
  A. 训练后 checkpoint + 知识写入记忆表（完整链路）
  B. 随机表（未训练 gate/indexer）+ 知识写入（隔离「训练」的贡献）
  C. 训练后 checkpoint + 空记忆表（隔离「记忆表」的贡献，= 模型泛化基线）

指标：
  - 精确命中率（答案出现在生成里）
  - 主题命中率（生成包含该模板的任一个真实值 = 模型确实在回忆记忆表）
  - 生成质量（平均 token 数、是否胡编）

用法：PYTHONIOENCODING=utf-8 python quality.py
"""

from __future__ import annotations

import os
import sys

import torch

from engram_model.model import EngramQwen3
from train import generate_knowledge

MODEL = os.environ.get("ENGRAM_MODEL_PATH", r"F:/dsh/engram-trial/qwen3-model")
CKPT = "checkpoints/engram-trained/engram.pt"

Q_PROMPTS = {
    0: "部署端口是多少来着？",
    1: "数据库用的什么？连接串呢？",
    2: "测试环境地址是什么来着？",
    3: "缓存用的什么方案？TTL 多少？",
    4: "CI 跑在哪？什么触发？",
    5: "日志级别是多少？输出到哪？",
    6: "告警阈值多少？通知渠道？",
    7: "密钥放哪？多久轮换？",
}

# 每模板的全部真实值（用于「主题命中」判定）
ALL_VALUES = {
    "port": ["8080", "9090", "3000", "8443"],
    "db": ["PostgreSQL", "MySQL", "Redis", "MongoDB"],
    "host": ["test.internal.example", "staging.example.com", "dev.local:8081", "qa.example.net"],
    "cache": ["Redis 集群", "本地 LRU", "CDN 边缘", "Memcached"],
    "ci": ["GitHub Actions", "Jenkins", "GitLab CI", "自建 runner"],
    "level": ["INFO", "DEBUG", "WARN", "ERROR"],
    "threshold": ["80%", "90%", "95%", "99%"],
    "vault": ["Vault", "KMS", "Secrets Manager", "本地加密文件"],
}
TPL_KEY = {0: "port", 1: "db", 2: "host", 3: "cache", 4: "ci", 5: "level", 6: "threshold", 7: "vault"}


def build_model(trained: bool) -> EngramQwen3:
    eng = EngramQwen3(model_id=MODEL, device="cuda", dtype=torch.bfloat16)
    if trained:
        ckpt = torch.load(CKPT, map_location="cuda", weights_only=False)
        eng.memory.pool.data.copy_(ckpt["memory_pool"].to(eng.memory.pool.dtype))
        for k, sd in ckpt["engram_modules"].items():
            eng.engram_modules[k].load_state_dict(sd)
        from peft import PeftModel
        eng.model = PeftModel.from_pretrained(eng.model, os.path.join(os.path.dirname(CKPT), "lora"))
    return eng


def prefill_knowledge(eng, items):
    """全通道写入 eval 知识。"""
    embed_model = eng.model
    while hasattr(embed_model, "get_base_model"):
        embed_model = embed_model.get_base_model()
    slot_ids, embeds = [], []
    for it in items:
        ids = eng.tokenizer(it["sentence"], return_tensors="pt").to(eng.device)
        with torch.no_grad():
            vec = embed_model.model.embed_tokens(ids.input_ids).mean(dim=1).squeeze(0)
        slots = eng.engram_slots_for(ids.input_ids)
        written = set()
        for ch in range(slots.shape[-1]):
            for pos in range(slots.shape[1]):
                s = int(slots[0, pos, ch].item())
                if s not in written:
                    written.add(s)
                    slot_ids.append(s)
                    embeds.append(vec)
    eng.memory.prefill_batch(torch.tensor(slot_ids, device=eng.device), torch.stack(embeds).to(eng.device))


def run_eval(eng, items, label: str):
    eng.eval()
    exact = 0
    topic = 0
    vals = []
    with torch.no_grad():
        for it in items:
            q = Q_PROMPTS[it["tpl"]]
            ids = eng.tokenizer(q, return_tensors="pt").to(eng.device)
            out = eng.model.generate(ids.input_ids, max_new_tokens=16, do_sample=False)
            gen = eng.tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
            vals.append(gen.strip())
            if it["answer"].lower() in gen.lower():
                exact += 1
            # 主题命中：生成包含该模板任一个真实值
            key = TPL_KEY[it["tpl"]]
            if any(v.lower() in gen.lower() for v in ALL_VALUES[key]):
                topic += 1
    n = len(items)
    print(f"  {label}: 精确命中 {exact}/{n} ({exact/n:.0%}) | 主题命中 {topic}/{n} ({topic/n:.0%})")
    return exact, topic, vals


def main():
    items = generate_knowledge(32, seed=7, eval_only=True)  # 值未参与训练，只能靠记忆表
    print(f"=== 质量评估（{len(items)} 问，eval 知识未参与训练） ===\n")

    print("[A] 训练后 + 记忆表（完整链路）")
    eng_a = build_model(trained=True)
    prefill_knowledge(eng_a, items)
    a_exact, a_topic, _ = run_eval(eng_a, items, "训练后+记忆")

    print("\n[B] 随机表 + 记忆表（隔离「训练」贡献）")
    eng_b = build_model(trained=False)
    prefill_knowledge(eng_b, items)
    b_exact, b_topic, _ = run_eval(eng_b, items, "随机表+记忆")

    print("\n[C] 训练后 + 空记忆表（隔离「记忆表」贡献 = 泛化基线）")
    eng_c = build_model(trained=True)
    c_exact, c_topic, _ = run_eval(eng_c, items, "训练后+空表")

    print("\n=== 结论 ===")
    print(f"训练贡献 (A-B): 精确 +{a_exact-b_exact} | 主题 +{a_topic-b_topic}")
    print(f"记忆贡献 (A-C): 精确 +{a_exact-c_exact} | 主题 +{a_topic-c_topic}")
    print(f"组合质量 (A):   精确 {a_exact/len(items):.0%} | 主题 {a_topic/len(items):.0%}")


if __name__ == "__main__":
    main()
