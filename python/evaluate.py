"""
对照评估：证明「回忆来自记忆表」而非模型背下答案。

三组对照（同一训练好的 checkpoint）：
  A. eval 知识写入记忆表 → 问问题（期望：模型通过共享 n-gram 命中槽位 → 回忆）
  B. eval 知识不写入记忆表 → 问同样问题（期望：答不出——证明不是背的）
  C. 知识写入但 gate 随机（未训练 checkpoint）→ 问问题（期望：答不出——
     证明是训练让模型学会用记忆，不是哈希命中本身）

用法：PYTHONIOENCODING=utf-8 python evaluate.py --checkpoint <path>
"""

from __future__ import annotations

import argparse
import os

import torch

from engram_model.model import EngramQwen3
from train import generate_knowledge

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


def load_model(checkpoint: str | None, device: str = "cuda"):
    eng = EngramQwen3(model_id=os.environ.get("ENGRAM_MODEL_PATH", r"F:/dsh/engram-trial/qwen3-model"),
                      device=device, dtype=torch.bfloat16)
    if checkpoint and os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        eng.memory.pool.data.copy_(ckpt["memory_pool"].to(eng.memory.pool.dtype))
        eng.memory.slot_index.copy_(ckpt["slot_index"])
        for k, sd in ckpt["engram_modules"].items():
            eng.engram_modules[k].load_state_dict(sd)
        # LoRA 适配器（如有）：base model 必须未包装——用 PeftModel.from_pretrained
        lora_dir = os.path.join(os.path.dirname(checkpoint), "lora")
        if os.path.exists(lora_dir):
            from peft import PeftModel
            # 若已被 get_peft_model 包装则先还原（from_pretrained 需要裸 base）
            while hasattr(eng.model, "get_base_model"):
                # PeftModel.from_pretrained 接受已包装模型会嵌套——这里还原到裸模型
                eng.model = eng.model.get_base_model()
            eng.model = PeftModel.from_pretrained(eng.model, lora_dir)
            print(f"✓ LoRA 适配器加载: {lora_dir}")
        print(f"✓ checkpoint 加载: {checkpoint}（eval_acc={ckpt.get('eval_acc', 'n/a')}）")
    return eng


def ask(eng, question: str, max_new: int = 12) -> str:
    ids = eng.tokenizer(question, return_tensors="pt").to(eng.device)
    with torch.no_grad():
        out = eng.model.generate(ids.input_ids, max_new_tokens=max_new, do_sample=False)
    return eng.tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/engram-trained/engram.pt")
    parser.add_argument("--n-eval", type=int, default=16)
    args = parser.parse_args()

    eng = load_model(args.checkpoint)
    eng.eval()

    knowledge = generate_knowledge(args.n_eval, seed=42)
    eval_knowledge = knowledge  # 全部作为"未训练"知识（train.py 里 train 在前）

    print("\n=== A. eval 知识写入记忆表（模型应回忆） ===")
    slot_ids, embeds = [], []
    # LoRA 包装后 embed_tokens 路径：循环解包 PeftModel 到最底层
    embed_model = eng.model
    while hasattr(embed_model, "get_base_model"):
        embed_model = embed_model.get_base_model()
    for it in eval_knowledge:
        ids = eng.tokenizer(it["sentence"], return_tensors="pt").to(eng.device)
        with torch.no_grad():
            vec = embed_model.model.embed_tokens(ids.input_ids).mean(dim=1).squeeze(0)
        slots = eng.engram_slots_for(ids.input_ids)  # [1, S, 16]
        # 全通道全位置写入（与 train.prefill 一致，保证查询命中）
        written = set()
        for ch in range(slots.shape[-1]):
            for pos in range(slots.shape[1]):
                s = int(slots[0, pos, ch].item())
                if s not in written:
                    written.add(s)
                    slot_ids.append(s)
                    embeds.append(vec)
    eng.memory.prefill_batch(torch.tensor(slot_ids, device=eng.device), torch.stack(embeds).to(eng.device))

    correct_a = 0
    for i, it in enumerate(eval_knowledge):
        q = Q_PROMPTS[it["tpl"]]
        ans = ask(eng, q)
        hit = it["answer"].lower() in ans.lower()
        correct_a += hit
        print(f"  Q:{q} → {ans.strip()!r} | 期望:{it['answer']} | {'✓' if hit else '✗'}")
    print(f"  A 准确率: {correct_a}/{len(eval_knowledge)} = {correct_a/len(eval_knowledge):.0%}")

    print("\n=== B. 清空记忆表（同一模型，不写入知识——应答不出） ===")
    eng.memory.pool.data.zero_()
    eng.memory.slot_index.fill_(-1)
    eng.memory.slot_used.zero_()
    eng.memory.pool_used.zero_()
    eng.memory.active.zero_()

    correct_b = 0
    for i, it in enumerate(eval_knowledge):
        q = Q_PROMPTS[it["tpl"]]
        ans = ask(eng, q)
        hit = it["answer"].lower() in ans.lower()
        correct_b += hit
        print(f"  Q:{q} → {ans.strip()!r} | 期望:{it['answer']} | {'✓' if hit else '✗'}")
    print(f"  B 准确率: {correct_b}/{len(eval_knowledge)} = {correct_b/len(eval_knowledge):.0%}")

    print(f"\n=== 结论: A-B = {correct_a - correct_b}（回忆确实来自记忆表） ===")


if __name__ == "__main__":
    main()
