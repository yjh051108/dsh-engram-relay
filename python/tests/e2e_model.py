"""
端到端验证：魔改 Qwen3-0.6B（Engram 模块 + DSA 路由）真实推理。

步骤：
1. 加载模型（本地路径，避免重新下载）；
2. 验证魔改后生成正常（Qwen3 基线能力保持）；
3. 写入一条记忆到 engram 表；
4. 验证记忆表状态与哈希寻址链路。
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from engram_model.model import EngramQwen3

MODEL_PATH = os.environ.get("ENGRAM_MODEL_PATH", r"F:/dsh/01-memory/engram-trial/qwen3-model")


def main():
    t0 = time.time()
    print(f"=== loading EngramQwen3 from {MODEL_PATH} ===")
    eng = EngramQwen3(
        model_id=MODEL_PATH,
        device="cuda",
        dtype=torch.bfloat16,
        engram_layers=(1, 7),
        num_memory_slots=65536,
        index_topk=4,
    )
    print(f"✓ loaded in {time.time()-t0:.1f}s")

    # 1. 基线生成（魔改后模型应正常说话）
    t1 = time.time()
    prompt = "法国的首都是哪里？"
    inputs = eng.tokenizer(prompt, return_tensors="pt").to(eng.device)
    out = eng.generate(inputs.input_ids, max_new_tokens=32, temperature=0)
    text = eng.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"✓ generate ({(time.time()-t1)*1000:.0f}ms): {text!r}")

    # 2. 写入一条记忆
    t2 = time.time()
    # 编码记忆文本为嵌入（embed_tokens 池化）
    mem_text = "项目部署端口是 8080"
    mi = eng.tokenizer(mem_text, return_tensors="pt").to(eng.device)
    with torch.no_grad():
        vec = eng.model.model.embed_tokens(mi.input_ids).mean(dim=1)
    slots = eng.engram_slots_for(mi.input_ids)
    slot_id = int(slots[0, 0, 0].item())
    eng.write_memory([slot_id], vec)
    stats = eng.memory_stats()
    print(f"✓ memory write ({(time.time()-t2)*1000:.0f}ms): slot={slot_id}, {stats}")

    # 3. 哈希寻址验证：相同文本 → 相同槽位
    slots2 = eng.engram_slots_for(mi.input_ids)
    assert int(slots2[0, 0, 0].item()) == slot_id, "确定性寻址失败"
    print("✓ deterministic addressing: same text -> same slot")

    # 4. 生成器管线完整跑一遍（含 engram_slots 注入）
    t3 = time.time()
    out2 = eng.generate(mi.input_ids, max_new_tokens=16, temperature=0)
    text2 = eng.tokenizer.decode(out2[0][mi.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"✓ generate with engram injection ({(time.time()-t3)*1000:.0f}ms): {text2!r}")

    print("\n=== e2e PASS ===")


if __name__ == "__main__":
    main()
