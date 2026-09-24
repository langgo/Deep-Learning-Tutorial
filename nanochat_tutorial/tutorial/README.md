# nanochat 从零到一：从数据训练到聊天模型

> 面向 LLM 初学者的 nanochat 源码精读教程。教程独立编写，不隶属于 nanochat 上游作者或任何课程；所有实现描述以本仓库当前 `nanochat/` 源码为准。

这不是“几行 API 调用大模型”的指南，而是沿着当前源码的一条真实训练链路，从原始文本、token 和张量形状出发，理解一个 decoder-only 模型怎样训练、如何分别评估 base 与 chat 行为、再怎样变成可对话系统。写作顺序是**自底向上**：先建立直觉，再给必要公式，最后回到代码和形状；教程不冒充 nanochat 上游作者或任何课程。

## 这套教程讲什么

nanochat 将 tokenizer、base 预训练、评估、推理、SFT 和可选 RL 放在可阅读的代码路径中。它适合学习系统如何连接；运行成本、训练时长和结果质量仍取决于当前数据、硬件、配置和源码版本，教程不作能力或复现承诺。

读完后，你应能回答：文本为什么必须先变成 token；`(B,T)` 如何变为 `(B,T,V)`；右移 target 与交叉熵如何训练 base model；为什么 base 与 chat 评估不能混为一个分数；SFT 的 mask 怎样界定模型责任；KV cache 怎样支持 rollout；以及 RL 为什么依赖已经能生成的 inference 系统。

## 面向谁 / 需要什么基础

- **读者**：了解 GPT、Transformer 或 attention 名词、但尚未从源码和张量层面理解 LLM 的初学者。
- **需要**：能读 Python；知道参数、梯度、矩阵乘法的基本直觉。
- **不需要**：推导完整数学、GPU，或昂贵训练。教程的形状例和只读检查即可帮助理解；最后一章也只建议小规模流程验证。

建议一边读一边打开 `nanochat/` 对照，并把关键张量形状打印出来。一次只验证一个假设；不要把某次小实验的输出当成模型能力结论。

## 三种读法

不必第一次就从头精读约四千行内容。先根据目标选择路线：

| 读法 | 目标 | 推荐路线 |
|---|---|---|
| **极速（约 30 分钟）** | 只建立端到端数据流骨架 | 本页 → 第 01 章 → 第 02–13 章各自的“学习目标”和“小结” |
| **主线（约半天）** | 理解完整 base 闭环和 chat 阶段分工 | 第 01–06 章 → 第 07 章只读 §1、§4 和“小结” → 第 08–13 章 |
| **精读（数天）** | 对照源码理解实现与边界 | 第 01–13 章顺读；打开每章源码地图；运行 [Toy Checks](./toy-checks.md)；用[术语表](./glossary.md)回查 |

如果只想先跑出正反馈，从 `python3 tutorial/toy_checks.py` 开始；它不需要数据、checkpoint、PyTorch 或 GPU。

## 全景数据流与真实依赖

关键依赖不是“先有模型再找数据”：**原始文本先存在，tokenizer 从这批文本训练出来**；之后 base 训练才用这个 tokenizer 重新编码原始文本。base checkpoint 是 SFT 的输入；SFT checkpoint 必须先经 inference engine 生成，才能聊天、做 chat eval 或 rollout；RL 则把这个 rollout 的奖励重新变成训练信号。

```text
原始文本 / ClimbMix Parquet shard
        │  先下载；供 tokenizer 训练与 base 训练读取
        ▼
Tokenizer（BPE） ────────────────► token id / token_bytes
        │                                  │
        └──────────────────────────────────┘
                                           ▼
Dataset loader：文档 → (B, T+1) → x/y，各为 (B, T)
                                           ▼
GPT：token id → logits (B, T, V) → base checkpoint
              │                     │
              │                     ├── Base Eval：BPB、CORE、sample
              │                     │
              │                     ▼
              │              SFT：对话 ids/mask → SFT checkpoint
              │                                      │
              │                                      ▼
              │       Inference：prefill / KV cache / decode / rollout
              │                    │                         │
              │                    ├── Chat Eval             └── RL
              │                    │   分类或生成任务             奖励 → RL checkpoint
              │                    ▼
              └────────────────── chat CLI
```

**两类评估必须分开读：**第 08 章只讨论 base 模型的 held-out BPB、CORE 和 sample；第 11 章才讨论聊天协议下的 ARC/MMLU/GSM8K/HumanEval、ChatCORE 与 pass@k。它们的 prompt、评分方式和被测对象不同，不能互相替代。

当前 `runs/speedrun.sh` 的参考路径是“下载数据 → 训练 tokenizer → base → base eval → SFT → chat eval”；它**没有调用** `scripts/chat_rl.py`。RL 是需要 SFT checkpoint 和 inference rollout 的独立可选阶段。

## 13 章目录

| # | 章节 | 核心问题 | 对应源码 |
|---|---|---|---|
| 01 | [总览](./01-overview.md) | 完整链路、checkpoint 与真实依赖是什么？ | `runs/speedrun.sh`、`checkpoint_manager.py` |
| 02 | [Dataset：从互联网文本到训练批次](./02-dataset.md) | Parquet 文档怎样成为 `(B,T)` 的 `x/y`？ | `dataset.py`、`dataloader.py` |
| 03 | [Tokenizer：从字节到 Token](./03-tokenizer.md) | BPE 怎样从文本训练并编码文本？ | `tokenizer.py`、`tok_train.py` |
| 04 | [GPT attention：嵌入、注意力与 RoPE](./04-gpt-attention.md) | token 怎样读取前文？ | `gpt.py`、`flash_attention.py` |
| 05 | [GPT MLP / 现代技巧](./05-gpt-mlp-tricks.md) | MLP、残差流和完整 forward 怎样得到 logits？ | `gpt.py` |
| 06 | [Base Train：数据到损失与训练循环](./06-base-train-loop.md) | `x/y` 怎样得到交叉熵与一次更新？ | `base_train.py`、`gpt.py` |
| 07 | [Base Train：优化器与规模法则](./07-base-train-optim-scaling.md) | 参数怎样稳定更新、规模怎样设定？ | `optim.py`、`base_train.py` |
| 08 | [Base Eval：BPB、CORE 与 sample](./08-base-evaluation.md) | 怎样测 base model 的续写能力？ | `loss_eval.py`、`core_eval.py`、`base_eval.py` |
| 09 | [Chat SFT](./09-sft.md) | 怎样把续写模型校准为对话模型？ | `chat_sft.py`、`tokenizer.py` |
| 10 | [Inference](./10-inference-engine.md) | 怎样用 KV cache 生成和 rollout？ | `engine.py`、`chat_cli.py` |
| 11 | [Chat Eval](./11-chat-evaluation.md) | 怎样测聊天任务完成情况？ | `chat_eval.py`、`tasks/` |
| 12 | [RL](./12-rl.md) | 怎样从 rollout 的结果奖励继续更新？ | `chat_rl.py`、`gsm8k.py` |
| 13 | [端到端](./13-end-to-end.md) | 脚本、产物和低风险验证怎样接起来？ | `speedrun.sh`、`runcpu.sh` |

## 配套附录

- [附录 A：零数据 Toy Checks](./toy-checks.md)：六个只用 Python 标准库、带 `assert` 的最小实验。
- [附录 B：术语与符号速查](./glossary.md)：形状符号和关键术语的一句话定义与章节入口。

## 学习路径

1. **首次通读**：按 01 → 13 顺序。02 和 03 共同打好“数据先存在、tokenizer 再训练”的地基；04–05 解释模型；06–07 解释 base 训练。
2. **先完成 base 闭环**：读到第 08 章即可形成“数据 → base train → base eval”的完整闭环。BPB、CORE 和 sample 只回答 base 模型的续写问题。
3. **再完成 chat 闭环**：09 定义对话数据与监督边界，10 提供实际生成，11 在这些前提下评估聊天任务。不要拿第 08 章的数代替第 11 章的任务结果。
4. **最后读 RL 与整合**：12 的 rollout、奖励和 pass@k 定义以前三步为前提；13 回到当前脚本，先做只读或小规模检查，避免不必要的昂贵训练。

准备好后，从[第 01 章：总览](./01-overview.md)开始。
