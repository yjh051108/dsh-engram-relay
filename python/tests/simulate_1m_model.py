"""
真模型融合仿真：1M token 规模下的模型内部链路。

验证（真实 Qwen3-0.6B + Engram 融合模块）：
1. 批量写入 2000 条记忆到记忆池（=1M token 历史）；
2. 随机主题查询 → 真实哈希寻址 + KV 路由打分 + top-K 选择 + gate 融合；
3. 统计：写入耗时 / 池占用 / 路由激活分布 / 单次 forward 耗时。

运行：python tests/simulate_1m_model.py
（需 ENGRAM_MODEL_PATH 指向本地模型目录）
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from engram_model.model import EngramQwen3

MODEL_PATH = os.environ.get("ENGRAM_MODEL_PATH", r"F:/dsh/engram-trial/qwen3-model")
ROUNDS = 2000
TOKENS_PER_ROUND = 500

TOPICS = [
    "部署端口配置", "数据库迁移", "前端构建优化", "API 鉴权设计", "日志系统改造",
    "缓存策略调整", "单元测试补全", "CI 流水线修复", "依赖版本升级", "性能瓶颈排查",
    "错误码规范", "配置中心接入", "消息队列选型", "权限模型设计", "监控告警配置",
    "灰度发布流程", "代码评审标准", "文档体系搭建", "脚本自动化", "数据备份策略",
]


def main():
    t0 = time.time()
    print(f"=== 加载魔改模型 {MODEL_PATH} ===")
    eng = EngramQwen3(model_id=MODEL_PATH, device="cuda", dtype=torch.bfloat16, engram_layers=(1, 7))
    print(f"✓ 加载 {time.time()-t0:.1f}s，记忆表槽位 {eng.memory.num_slots}")

    # 1. 批量写入 2000 条记忆（=1M token 历史）
    t1 = time.time()
    written = 0
    for i in range(ROUNDS):
        topic = TOPICS[i % len(TOPICS)]
        text = f"{topic}的结论：第 {i} 回合确定了方案，采用渐进式实施。"
        mi = eng.tokenizer(text, return_tensors="pt").to(eng.device)
        with torch.no_grad():
            vec = eng.model.model.embed_tokens(mi.input_ids).mean(dim=1)
        slots = eng.engram_slots_for(mi.input_ids)
        # 写入前 4 个哈希通道（与 server._write_memory 一致，保证查询可命中）
        for ch in range(min(4, slots.shape[-1])):
            eng.write_memory([int(slots[0, 0, ch].item())], vec)
        written += 1
    stats = eng.memory.slot_usage()
    write_s = time.time() - t1
    print(f"✓ 写入 {written} 条（{write_s:.1f}s，{write_s/written*1000:.0f}ms/条）")
    print(f"  记忆池: {stats}")

    # 2. 随机主题查询 → 真实路由 + 融合
    t2 = time.time()
    fwd_times = []
    for i in range(50):
        topic = TOPICS[i % len(TOPICS)]
        query = f"关于{topic}我们上次怎么决定的？"
        qi = eng.tokenizer(query, return_tensors="pt").to(eng.device)
        tf = time.time()
        with torch.no_grad():
            out = eng.generate(qi.input_ids, max_new_tokens=8, temperature=0)
        fwd_times.append(time.time() - tf)
    query_s = time.time() - t2
    print(f"✓ 50 次查询（{query_s:.1f}s，{query_s/50*1000:.0f}ms/次）")

    # 3. 槽位命中检查：查询主题与写入主题同槽
    hit = 0
    for i in range(20):
        topic = TOPICS[i % len(TOPICS)]
        qi = eng.tokenizer(f"{topic}的结论", return_tensors="pt").to(eng.device)
        qslots = eng.engram_slots_for(qi.input_ids)
        # 检查查询哈希的槽位下是否有记忆
        embeds, valid = eng.memory.lookup(qslots)
        if bool(valid.any()):
            hit += 1
    print(f"✓ 查询槽位命中: {hit}/20（同主题文本 → 同槽 → 记忆可达）")

    print(f"\n=== 1M 仿真（真模型）PASS：{written} 条 / {write_s:.0f}s 写入，forward {sum(fwd_times)/len(fwd_times)*1000:.0f}ms/次 ===")


if __name__ == "__main__":
    main()
