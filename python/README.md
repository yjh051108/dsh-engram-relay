# dsh-engram-relay Python 侧说明

魔改 <1B 模型（Engram 条件记忆 × DSA 稀疏路由）的 Python 实现。

## 依赖

```bash
pip install -r requirements.txt
# 可选 GPU（推荐）：
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

模型权重（Qwen/Qwen3-0.6B，约 1.2GB）首次加载时自动从 HuggingFace 下载，
或先手动下载：`huggingface-cli download Qwen/Qwen3-0.6B --local-dir <dir>`
然后用 `ENGRAM_MODEL_PATH=<dir>` 指向本地目录。

## 结构

| 文件 | 作用 |
|---|---|
| `engram_model/hash.py` | N-gram 哈希寻址（论文 NgramHashMapping 移植，确定性 O(1)） |
| `engram_model/engram_module.py` | EngramMemory（外置可写记忆表）+ DsaEngramIndexer（KV 打分路由，top-K）+ EngramGateModule（gate 融合） |
| `engram_model/model.py` | Qwen3 decoder layer 注入融合模块（包装 forward，零源码修改） |
| `engram_model/server.py` | JSON 行协议服务（Node 插件 spawn 对接） |

## 测试

```bash
# 纯逻辑（不加载大模型，秒级）
python tests/test_engram_module.py
python tests/test_hash.py

# 真模型 e2e（需 GPU/CPU + 模型权重，约 1-2 分钟）
python tests/e2e_model.py

# 服务协议集成（spawn server → RPC 全链路）
python tests/test_server.py
```

## 设计：融合点

```
input_ids ──► N-gram 哈希（O(1) 确定性寻址）──► 记忆表候选（粗筛）
                                                    │
hidden_states ──► DSA Indexer 打分（ReLU(q·k) 多头加权）◄── KV 参与路由（精筛）
                                                    │
                                              top-K 稀疏选择
                                                    │
                                              gate × value 融合注入
```

- **大 engram**：记忆表 65652 槽 × 2 容量 × 1024 维（bf16 ≈ 268MB），外置可写；
- **小 KV**：主注意力 KV cache 只装工作记忆，历史折叠进 engram 表；
- **KV 路由**：indexer 的 query 投影自当前 hidden states，对 engram 候选打分——
  「KV 负责优化 engram 寻址后的路由」；
- **超稀疏**：top-K（默认 4）条记忆参与融合，其余候选零开销。

## 与 DeepSeek 开源对照

- Engram 论文：`deepseek-ai/Engram`（N-gram 哈希 + 记忆表 + gate 融合）——
  本实现移植其 `NgramHashMapping` 与 gate 语义，记忆表改为外置可写；
- DSA：`deepseek-ai/DeepSeek-V3.2-Exp`（Lightning Indexer + Token Selector）——
  本实现移植 indexer 的 `ReLU(q·k)` 多头加权打分与 top-K 选择，
  打分目标从「历史 token」扩展为「engram 记忆候选」。
