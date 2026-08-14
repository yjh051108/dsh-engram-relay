"""
原生 engram 记忆训练：让 0.6B 模型"学会"使用外置记忆表。

范式（对齐 DeepSeek Engram 论文 + engram-peft）：
1. 构造"知识语料"：一批含事实的句子（如「项目部署端口是 8080」）；
2. 把每条知识的嵌入**预填充**进对应哈希槽位（记忆表 = 知识库）；
3. 模型在语料上做 next-token prediction，**记忆表嵌入 + gate/indexer
   投影 + LoRA 适配器**可训练——模型学会「看到模式 → 查表 → 融合记忆
   → 生成更准的下一个 token」；
4. 评估：向记忆表写入**训练时没见过的新知识**，问模型相关问题，
   看它能否"回忆"（对比随机表/未训练基线）。

用法：
  python train.py --model <path> --epochs 3 --lr 5e-4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from engram_model.model import EngramQwen3

# ---------------------------------------------------------------- 数据

# 会话内折叠历史模板：每条 = 本会话早期对话中被折叠的一段事实。
# 模型要学：会话中段问起时，从记忆表（被折叠的早期历史）原生回忆。
KNOWLEDGE_TEMPLATES = [
    "这个项目我们之前定了部署端口用 {port}",
    "刚才讨论过，数据库用 {db}，连接串是 {conn}",
    "前面说过测试环境地址是 {host}",
    "我们刚定缓存用 {cache}，TTL 设 {ttl} 秒",
    "上面聊到 CI 跑在 {ci} 上，触发条件是 {trigger}",
    "之前确认日志级别是 {level}，输出到 {log_target}",
    "刚才定了告警阈值 {threshold}，通知走 {channel}",
    "前面说密钥放 {vault}，{rotate} 天轮换一次",
]

# 问题模板：会话中段/后段的提问（模型需从被折叠的早期历史回忆）
QUESTION_TEMPLATES = [
    "部署端口是多少来着？",
    "数据库用的什么？连接串呢？",
    "测试环境地址是什么来着？",
    "缓存用的什么方案？TTL 多少？",
    "CI 跑在哪？什么触发？",
    "日志级别是多少？输出到哪？",
    "告警阈值多少？通知渠道？",
    "密钥放哪？多久轮换？",
]

FILLERS = {
    "port": ["8080", "9090", "3000", "8443"],
    "db": ["PostgreSQL", "MySQL", "Redis", "MongoDB"],
    "conn": ["postgres://prod:8080/main", "mysql://app:3306/core", "redis://cache:6379/0", "mongodb://db:27017/app"],
    "host": ["test.internal.example", "staging.example.com", "dev.local:8081", "qa.example.net"],
    "cache": ["Redis 集群", "本地 LRU", "CDN 边缘", "Memcached"],
    "ttl": ["300", "600", "1800", "3600"],
    "ci": ["GitHub Actions", "Jenkins", "GitLab CI", "自建 runner"],
    "trigger": ["push 到 main", "PR 合并", "定时每日", "手动触发"],
    "level": ["INFO", "DEBUG", "WARN", "ERROR"],
    "log_target": ["stdout", "ELK", "Loki", "文件滚动"],
    "threshold": ["80%", "90%", "95%", "99%"],
    "channel": ["钉钉", "企业微信", "邮件", "Slack"],
    "vault": ["Vault", "KMS", "Secrets Manager", "本地加密文件"],
    "rotate": ["30", "60", "90", "180"],
}

# 评估专用值（**训练从未出现过**——模型只能靠记忆表回忆，无法背答案）
EVAL_ONLY_VALUES = {
    "port": ["13579", "24680"],
    "db": ["CockroachDB", "TiDB"],
    "conn": ["postgres://eval:13579/main", "mysql://eval:24680/core"],
    "host": ["eval-only.internal", "eval-2.example.net"],
    "cache": ["Caffeine 本地", "自研缓存层"],
    "ttl": ["135", "246"],
    "ci": ["Travis CI", "CircleCI"],
    "trigger": ["定时每两小时", "tag 发布"],
    "level": ["TRACE", "FATAL"],
    "log_target": ["Graylog", "DataDog"],
    "threshold": ["77%", "88%"],
    "channel": ["飞书", "Webhook"],
    "vault": ["AWS KMS 专用", "HashiCorp Vault 企业版"],
    "rotate": ["135", "246"],
}


def generate_knowledge(n: int, seed: int = 42, eval_only: bool = False) -> list[dict]:
    """生成 n 条知识：{sentence, question, answer}。

    eval_only=True 时用 EVAL_ONLY_VALUES（训练从未见过的值）——
    评估时模型唯一信息来源是记忆表，无法背训练答案。
    """
    rng = random.Random(seed)
    pool = EVAL_ONLY_VALUES if eval_only else FILLERS
    out = []
    for i in range(n):
        tpl_idx = i % len(KNOWLEDGE_TEMPLATES)
        tpl = KNOWLEDGE_TEMPLATES[tpl_idx]
        q_tpl = QUESTION_TEMPLATES[tpl_idx]
        # 填充每个占位符（同一条知识内固定，跨条随机）
        filled = {}
        for key, values in pool.items():
            placeholder = "{" + key + "}"
            if placeholder in tpl:
                filled[key] = rng.choice(values)
        sentence = tpl.format(**filled)
        # 答案 = 第一个占位符的值（问题的核心事实）
        first_key = next(k for k in pool if "{" + k + "}" in tpl)
        answer = filled[first_key]
        out.append({"sentence": sentence, "question": q_tpl, "answer": answer, "tpl": tpl_idx})
    return out


class KnowledgeDataset(Dataset):
    """训练样本：知识句子（LM 目标）+ 问答对（记忆回忆目标）。

    关键设计：模型要学到「问题 → 哈希命中知识槽位 → 记忆融合 → 答出答案」，
    因此样本 = 「知识句子」与「问题：答案」交替——前者让记忆嵌入学会
    表达知识，后者让 gate/indexer 学会在提问时激活记忆。
    """

    def __init__(self, items: list[dict], tokenizer, max_len: int = 64, qa_ratio: float = 0.5):
        self.samples: list[torch.Tensor] = []
        self.pad_id = tokenizer.pad_token_id or 0
        for it in items:
            # 1) 知识句子样本（LM）
            sent = tokenizer(it["sentence"], return_tensors="pt").input_ids[0]
            self.samples.append(sent[:max_len])
            # 2) 问答样本（记忆回忆）："Q：...\nA：answer"（A 之后是下一个 token 预测）
            qa = tokenizer(
                f"Q: {it['question']}\nA: {it['answer']}",
                return_tensors="pt",
            ).input_ids[0]
            self.samples.append(qa[:max_len])
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

    def collate(self, batch):
        """等长 padding（右侧 pad），labels 与 input_ids 相同（pad 位置忽略）。"""
        max_len = max(len(x) for x in batch)
        padded = torch.full((len(batch), max_len), self.pad_id, dtype=torch.long)
        for i, x in enumerate(batch):
            padded[i, : len(x)] = x
        return padded


# ---------------------------------------------------------------- 训练

def train(
    model_path: str,
    epochs: int = 3,
    lr: float = 5e-4,
    batch_size: int = 8,
    n_knowledge: int = 64,
    n_eval: int = 16,
    seed: int = 42,
    device: str = "cuda",
):
    torch.manual_seed(seed)
    t0 = time.time()

    # 1. 加载魔改模型（engram 模块随机初始化）
    print(f"=== 加载模型 {model_path} ===")
    eng = EngramQwen3(model_id=model_path, device=device, dtype=torch.bfloat16)
    print(f"✓ 加载 {time.time()-t0:.1f}s，记忆表 {eng.memory.num_slots} 槽")

    # 2. 训练知识（每 epoch 重新生成，见训练循环；eval 在评估段用 eval_only 值生成）
    train_knowledge = generate_knowledge(n_knowledge, seed)
    print(f"✓ 训练知识 {len(train_knowledge)} 条（每 epoch 重新实例化，值变化）")

    # 3. 预填充记忆表：训练知识的嵌入 → 对应哈希槽位。
    #    对齐 engram-peft：记忆表嵌入 = 知识句子嵌入的**带噪副本**
    #    （随机初始化会让 gate 无从学起；embed_tokens 初始化给一个
    #    合理的起点，训练中嵌入可继续演化）。
    def prefill(items: list[dict], keep_grad: bool = False, noise: float = 0.1):
        slot_ids: list[int] = []
        embeds: list[torch.Tensor] = []
        # LoRA 包装后 embed_tokens 路径：循环解包 PeftModel 到最底层
        embed_model = eng.model
        while hasattr(embed_model, "get_base_model"):
            embed_model = embed_model.get_base_model()
        for it in items:
            ids = eng.tokenizer(it["sentence"], return_tensors="pt").to(device)
            with torch.no_grad():
                vec = embed_model.model.embed_tokens(ids.input_ids).mean(dim=1).squeeze(0)
                # 带噪副本：避免记忆嵌入与句子嵌入完全同构（否则 gate 退化）
                if noise > 0:
                    vec = vec + torch.randn_like(vec) * noise
            if not keep_grad:
                vec = vec.detach()
            slots = eng.engram_slots_for(ids.input_ids)  # [1, S, 16]
            # 关键：写入**全部通道全部位置**的槽位（去重）——查询通过任意
            # 共享 n-gram 槽位命中；只写 [0,0,0] 会导致查询命中不到
            written = set()
            for ch in range(slots.shape[-1]):
                for pos in range(slots.shape[1]):
                    s = int(slots[0, pos, ch].item())
                    if s not in written:
                        written.add(s)
                        slot_ids.append(s)
                        embeds.append(vec)
        eng.memory.prefill_batch(torch.tensor(slot_ids, device=device), torch.stack(embeds).to(device))
        return slot_ids

    train_slots = prefill(train_knowledge, keep_grad=False)
    print(f"✓ 记忆表预填充 {len(train_slots)} 条（train 知识）")
    print(f"  记忆表占用: {eng.memory.slot_usage()}")

    # 4. 冻结 backbone，只训练 engram 参数 + LoRA（对齐论文 + engram-peft）
    #    - engram 参数（记忆池 + indexer + gate）直接训练
    #    - LoRA 适配器让 backbone 学会「从记忆融合中提取信息」（解码答案）
    for p in eng.model.parameters():
        p.requires_grad_(False)

    # LoRA：只适配注意力/MLP 投影（少参数，让 backbone 学会用记忆）
    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none",
    )
    eng.model = get_peft_model(eng.model, lora_cfg)

    # engram 参数可训练：记忆池 + indexer + gate（每层注入模块）
    trainable = []
    for name, p in eng.named_parameters():
        if "memory.pool" in name or "engram_modules" in name:
            p.requires_grad_(True)
            trainable.append(name)
    print(f"✓ 可训练参数（engram {len(trainable)} 组 + LoRA {eng.model.print_trainable_parameters()!r}）")
    for n in trainable[:4]:
        print(f"    {n}")

    # 5. 优化器（数据集/记忆表每 epoch 重建，见训练循环）
    opt = torch.optim.AdamW(
        [p for p in eng.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    # 6. 训练循环（next-token prediction）
    #    关键设计（逼出真回忆）：**每 epoch 重新生成实例 + 清空重填记忆表**。
    #    同一问题模板每 epoch 对应不同答案 → 模型无法背「Q→固定A」，
    #    只能学会「提问时查当前记忆表、融合、回忆出该实例的值」。
    eng.train()
    total_steps = 0
    for epoch in range(epochs):
        # 每 epoch 重新生成知识实例（同模板不同值）并重填记忆表
        epoch_knowledge = generate_knowledge(n_knowledge, seed + epoch * 1000)
        # 清空记忆表（保留训练中的参数，只重置槽位映射与池内容）
        eng.memory.pool.data.normal_(0, 0.02)
        eng.memory.slot_index.fill_(-1)
        eng.memory.slot_used.zero_()
        eng.memory.pool_used.zero_()
        eng.memory.active.zero_()
        prefill(epoch_knowledge, keep_grad=False)
        dataset = KnowledgeDataset(epoch_knowledge, eng.tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate)

        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            # labels: pad 位置 -100（不计算 loss）
            labels = batch.clone()
            labels[labels == dataset.pad_id] = -100
            out = eng.model(input_ids=batch, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in eng.parameters() if p.requires_grad], 1.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
            total_steps += 1
        print(f"epoch {epoch+1}/{epochs}: loss={epoch_loss/n_batches:.4f} ({time.time()-t0:.0f}s)")

    # 7. 评估：回忆测试（eval 知识用**训练从未见过的值**——模型只能靠记忆表）
    print("\n=== 评估：模型能否回忆 eval 知识（值未参与训练） ===")
    eval_knowledge = generate_knowledge(n_eval, seed + 9999, eval_only=True)
    eval_slots = prefill(eval_knowledge)  # 把 eval 知识也写入记忆表
    print(f"✓ eval 知识写入记忆表 {len(eval_slots)} 条（值均为训练未见过的）")
    eng.eval()
    correct = 0
    with torch.no_grad():
        for it in eval_knowledge:
            prompt = it["question"]
            ids = eng.tokenizer(prompt, return_tensors="pt").to(device)
            out = eng.model.generate(
                ids.input_ids, max_new_tokens=12, do_sample=False, temperature=0,
            )
            gen = eng.tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
            hit = it["answer"].lower() in gen.lower()
            if hit:
                correct += 1
            print(f"  Q: {prompt} | 答: {gen.strip()!r} | 期望: {it['answer']} | {'✓' if hit else '✗'}")

    acc = correct / len(eval_knowledge)
    print(f"\n=== 回忆准确率: {acc:.0%} ({correct}/{len(eval_knowledge)}) ===")

    # 8. 保存（engram 参数 + LoRA 适配器）
    save_dir = os.environ.get("ENGRAM_SAVE_DIR", "checkpoints/engram-trained")
    os.makedirs(save_dir, exist_ok=True)
    # LoRA 适配器（PeftModel）
    try:
        eng.model.save_pretrained(os.path.join(save_dir, "lora"))
        print(f"✓ LoRA 适配器已保存到 {save_dir}/lora")
    except Exception as e:
        print(f"⚠ LoRA 保存失败: {e}")
    torch.save({
        "memory_pool": eng.memory.pool.data,
        "slot_index": eng.memory.slot_index,
        "engram_modules": {k: v.state_dict() for k, v in eng.engram_modules.items()},
        "eval_acc": acc,
    }, os.path.join(save_dir, "engram.pt"))
    print(f"✓ 已保存到 {save_dir}/engram.pt")

    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", dest="model_path", default=os.environ.get("ENGRAM_MODEL_PATH", ""))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-knowledge", type=int, default=64)
    parser.add_argument("--n-eval", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(**vars(args))
