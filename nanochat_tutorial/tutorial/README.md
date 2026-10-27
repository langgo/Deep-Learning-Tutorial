# nanochat 从零到一

> 沿源码与张量形状理解 LLM 的训练、评估、推理与强化学习

从一段原始文本开始，亲手追踪它如何变成 token、训练 batch、模型 logits、checkpoint 和最终回答。本书不把大语言模型描述成一个黑箱 API，而是沿 nanochat 的真实代码路径，建立一套可以在纸上推导、在源码中定位、在实验里验证的完整认识。

> **版本说明**　本书对应上游项目 [karpathy/nanochat](https://github.com/karpathy/nanochat) 的固定 revision `92d63d4`。书中实现细节以该版本源码和测试为准。

---

## 内容提要

一本完整的 LLM 教程需要回答的不只是“Transformer 如何计算 attention”，还要解释数据、训练、评估和部署如何真正连接起来。本书围绕一条贯穿始终的主线展开：

```text
原始文本 → token → 训练张量 → logits → loss → checkpoint
         → 对话数据 → 生成回答 → 任务评估 → 可选的结果奖励
```

沿着这条主线，你将依次理解：

- 原始文本如何经过 BPE tokenizer，成为模型可读取的整数序列；
- 不同长度的文档如何被打包成 `(B,T)` 的输入与目标；
- decoder-only Transformer 如何把 token id 变成 `(B,T,V)` 的 logits；
- 交叉熵、反向传播、Muon/AdamW 和规模法则如何完成基础模型（base model）预训练；
- BPB、CORE 与生成样本分别能说明什么，不能说明什么；
- SFT 如何通过对话协议和 loss mask，把续写模型校准为聊天模型；
- KV cache 如何把自回归生成拆成 prefill 与 decode；
- 聊天评估如何用分类、数值验证和代码执行评估真实回答；
- 简化的 GRPO/REINFORCE 如何把 rollout 的结果奖励重新变成梯度。

本书始终追问五个问题：**数据是什么形状，谁产生它，谁消费它，它如何参与训练或评估，以及它的实现边界在哪里。**

## 本书的定位与边界

本书是一部面向学习者的源码精读教程，而不是：

- 几行 API 即可调用大模型的应用开发指南；
- nanochat 上游作者发布的官方文档；
- 对训练成本、完成时间或模型能力的复现承诺；
- 将某次小规模实验外推为通用结论的基准报告。

nanochat 的数据、训练配方和代码仍会演进。书中所有数字和实现判断都绑定到上述 revision；讨论其他版本时，应重新核对源码、测试和运行脚本。

---

## 读者对象与预备知识

本书适合以下读者：

- 听说过 GPT、Transformer 和 attention，但尚未从源码理解完整 LLM 流水线；
- 能阅读 Python，希望建立张量形状、训练目标和工程模块之间的联系；
- 想从“知道概念”进阶到“能够定位代码、解释行为并设计验证实验”。

阅读前只需要：

- Python 基础；
- 参数、梯度和矩阵乘法的基本直觉；
- 愿意在阅读时手写形状、运行小实验并对照源码。

不要求读者拥有 GPU，也不要求先掌握完整的概率论或深度学习数学。没有 GPU 时，仍可完成全部概念推导、源码阅读，并运行自检脚本确认本书对源码的描述仍然成立；昂贵训练只是可选实践。

## 如何使用本书

### 每章的统一结构

全书十三章采用同一套模板，让你进入任意一章都很快定位：

1. **章首元数据**：本章定位、中心问题、范围、先决条件、本章产物、源码入口；
2. **「本章讲什么」**：开场直说这章在讲什么、跟前后章如何链接；
3. **预备词汇**：昨章未用过的术语在这里一次说清，正文直接用；
4. **贯穿章的小例子**：所有其后小节都回到同一组具体数字；
5. **正文**：按「先讲动机 → 再给公式 → 再给代码」的三段式排布，关键公式都有具体数字验证；
6. **三层分类表**（核心原理 / 工程优化 / 实验调参）：帮你区分哪些是模型必然、哪些是本仓库的具体选择；
7. **常见误区**：正反对比式列举，至少把最容易踩的坑按在显眼处；
8. **小结 + 练习**：练习里至少有一道手算或具体数字题；
9. **前后章导航**：指向上下文。

### 读到复杂章节时

第 04 / 07 / 10 / 12 章比较长，都在章首给了**读法建议**（如“第一遍只读 §1–4、§6 和小结，其他内容第二遍再读”）。不需要一次读完所有细节。

### 阅读时的一个习惯

建议边读边打开源码，对重要对象记四件事：

```text
名称 / dtype / shape / 生产者与消费者
```

例如：

```text
x       / torch.long  / (B, T)      / dataloader 产生，GPT.forward 消费
logits  / float32     / (B, T, V)   / lm_head 产生，cross_entropy 消费
```

**一次只验证一个假设**。Toy 模型跑通、某条 sample 读起来通顺、一次 loss 下降——**都不能单独证明模型能力提升**。

### 路径约定

本教程仓库中，`nanochat/` 是上游项目的固定源码副本。因此：

- 本书写 `nanochat/gpt.py` 时，指相对于**上游项目根目录**的路径；
- 从本教程仓库根目录看，实际文件位于 `nanochat/nanochat/gpt.py`；
- `scripts/...`、`runs/...`、`tasks/...` 同样相对于上游 `nanochat/` 目录；
- 除非命令另有说明，上游训练和评估命令应从 `nanochat/` 目录执行。

---

## 全书结构

全书由一个序篇和五个部分组成。章节顺序体现教学上的认知递进；程序运行时的硬依赖将在下一节单独说明。

### 序篇：建立全局地图

**[第 01 章　总览：从零到聊天模型的完整链路](./01-overview.md)**
先认识 tokenizer、base 预训练、SFT、推理、评估、RL 与 checkpoint 的分工，建立后续阅读所需的系统地图。

### 第一部分　数据与表示

这一部分回答：自然语言如何成为模型可以训练的整数张量？教程先介绍数据链路，再打开其中的 tokenizer 黑箱；实际运行时，原始文本会先训练 tokenizer，再由 loader 使用它重新编码文本。

**[第 02 章　数据集：从互联网文本到训练批次](./02-dataset.md)**
理解 ClimbMix、Parquet、row group、分布式分片、BOS-aligned best-fit packing，以及 `(B,T+1) → x/y` 的过程。

**[第 03 章　分词器：从字节到 Token](./03-tokenizer.md)**
理解字节级 BPE、特殊 token、对话渲染、监督 mask 与 `token_bytes.pt`。

### 第二部分　训练并评估基础模型

这一部分完成第一个完整闭环：定义 decoder-only Transformer，用原始文本训练它，再以独立协议评估基础模型 checkpoint。

**[第 04 章　GPT 注意力：嵌入、因果注意力与 RoPE](./04-gpt-attention.md)**
沿 `idx(B,T) → x(B,T,C)` 追踪 Q/K/V、多头注意力、GQA、RoPE、滑动窗口与 FA3/SDPA。

**[第 05 章　GPT 的 MLP、残差流与现代改进](./05-gpt-mlp-tricks.md)**
补齐 MLP、RMSNorm、残差连接和完整 forward，并辨认 value embedding、smear、backout 与 softcap 等增量。

**[第 06 章　基础模型训练：从数据到损失与梯度](./06-base-train-loop.md)**
把右移目标、交叉熵、反向传播、梯度累积和 checkpoint 串成一次完整训练更新。

**[第 07 章　基础模型训练：优化器、精度与规模法则](./07-base-train-optim-scaling.md)**
理解 AdamW/Muon 参数分组、学习率日程、低精度、多卡通信和 `--depth` 复杂度旋钮。

**[第 08 章　基础模型评估：BPB、CORE 与生成样本](./08-base-evaluation.md)**
区分语言建模损失、任务能力和人工样本检查，完成基础模型的训练—评估闭环。

### 第三部分　从基础模型到聊天模型

这一部分回答：一个擅长续写的基础模型，如何学会按对话协议回答，并通过真实推理系统接受评估？

**[第 09 章　监督微调：教会模型对话](./09-sft.md)**
理解 conversation、special token、loss mask、任务混合和 base → SFT checkpoint 的交接。

**[第 10 章　推理引擎：KV 缓存与高效生成](./10-inference-engine.md)**
理解朴素生成、prefill/decode、KV cache、多样本生成、采样策略和工具状态机。

**[第 11 章　聊天评估：聊天模型会做题了吗？](./11-chat-evaluation.md)**
理解受限分类、生成后验证、GSM8K、HumanEval、ChatCORE、pass@k 及执行安全边界。

### 第四部分　可选的强化学习

**[第 12 章　强化学习：用结果奖励改进推理](./12-rl.md)**
从 SFT checkpoint 出发，理解 rollout、组内 advantage、masked log probability 和简化的 on-policy 策略梯度。只关心基础训练或聊天系统的读者可以跳过本章。

### 第五部分　端到端实践

**[第 13 章　端到端：从脚本到可验证实验](./13-end-to-end.md)**
回到 `speedrun.sh` 与 `runcpu.sh`，整理环境、命令、产物、checkpoint 交接和低风险实验方法。本章不依赖读者实际运行 RL。

---

## 系统全景：运行时依赖与产物

下面的图只表达**硬依赖和主要产物**，不表示章节阅读顺序。评估是从 checkpoint 分出的质量检查，不会产生下一训练阶段必须读取的新模型。

```text
原始文本（ClimbMix Parquet）
    │
    ├──训练 BPE──→ tokenizer.pkl / token_bytes.pt ──┐
    │                                               │
    └───────────────────────────────────────────────┘
                            │
                            ▼
                    基础模型 DataLoader
               文档 → token → (B,T+1) → x/y
                            │
                            ▼
                       基础模型训练
                            │
                            ▼
                  base checkpoint
                    │             │
                    │             └──基础模型评估：BPB / CORE / sample
                    ▼
                    SFT
                    │
                    ▼
                  SFT checkpoint
                    │
                    ├──推理引擎──→ Chat CLI
                    ├──推理引擎──→ 聊天评估：accuracy / ChatCORE / pass@k
                    └──推理引擎生成 rollout → reward → RL checkpoint
                                                        │
                                                        └──推理引擎 → CLI / 聊天评估
```

### 四条关键依赖

1. **原始文本先于 tokenizer。** tokenizer 必须先从文本学习词表；base loader 随后用同一 tokenizer 重新编码文本。
2. **Tokenizer 与 checkpoint 必须成套使用。** 即使词表大小相同，替换 tokenizer 也会改变 token id 与 embedding 行的含义。
3. **Checkpoint 是训练阶段的接力棒。** base checkpoint 是 SFT 的输入；SFT checkpoint 是聊天、聊天评估和 RL rollout 的起点；RL checkpoint 是可选的后续产物。
4. **推理引擎是聊天评估与 RL 的共同前提。** 二者都必须先让模型实际生成回答；前者只计分，后者再将结果奖励转化为训练信号。

### 评估边界

基础模型评估与聊天评估不能互相替代：

| 评估阶段 | 被测对象 | 主要指标 | 回答的问题 |
|---|---|---|---|
| [基础模型评估](./08-base-evaluation.md) | base checkpoint | BPB、CORE、sample | 模型是否更会预测和续写文本？ |
| [聊天评估](./11-chat-evaluation.md) | SFT 或 RL checkpoint + Engine | 任务 accuracy、ChatCORE、pass@k | 模型能否按聊天协议完成任务？ |

当前 `runs/speedrun.sh` 的默认参考路径为：

```text
数据下载 → Tokenizer → 基础模型训练 → 基础模型评估 → SFT → 聊天评估
```

它不执行第 12 章 RL。RL 是从 SFT checkpoint 分出的独立可选支线。

---

## 阅读路线

### 路线 A：完整学习

按第 01–13 章顺序阅读。适合第一次系统理解 LLM 完整生命周期的读者。

### 路线 B：只完成基础模型闭环

```text
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```

读完即可理解“数据 → 模型 → 训练 → 基础模型评估”，无需进入聊天和 RL。

### 路线 C：聚焦聊天系统

如果已经掌握基础 Transformer 和语言模型训练，可阅读：

```text
01 → 03（对话 token）→ 05（完整 forward）→ 09 → 10 → 11 → 13
```

第 12 章 RL 可按需选读。

### 路线 D：源码与研究精读

顺读全书，并同时：

- 对照每章的源码地图；
- 运行 `python3 tutorial/source_checks.py`，确认书里引用的源码事实在当前 revision 上仍成立；
- 阅读 `tests/` 中对应的不变量；
- 阅读 `dev/LOG.md` 中成功与失败的实验；
- 使用附录 C 的记录模板设计对照实验。

### 没有 GPU 时怎么读？

无需 GPU 即可完成：第 01–13 章的全部概念阅读、公式推导、源码检查、全部附录，以及源码契约回归检查。GPU 只影响是否能经济地运行完整训练、FA3/FP8 和真实性能 benchmark，不影响理解主线。

---

## 配套附录与代码

### 学习附录

- **[附录 A　术语与符号速查](./glossary.md)**：集中查询形状符号、训练、评估、推理和 RL 术语。
- **[附录 C　性能、排行榜与研究方法](./performance-research.md)**：解释 Time-to-GPT-2、miniseries、MFU/MBU、TTFT/TPOT 和负结果实验。

### 版本核验与维护资料

- **[附录 B　源码主题与文件索引](./source-index.md)**：以常见工程问题为入口，指向对应源码文件、关键符号和本书章节。
- `tutorial/source_checks.py`：检查本书依赖的 speedrun、模型常量、SFT/ChatCORE、RL 和本地链接等易漂移事实。

从本教程仓库根目录运行：

```bash
# 教程与固定源码的关键契约：无第三方依赖
python3 tutorial/source_checks.py

# 上游真实测试：需要先安装 nanochat 开发依赖
cd nanochat
uv sync --extra cpu --group dev   # 或 --extra gpu
# 必须以 `python -m pytest` 运行，nanochat/ 未被安装为包，该形式会把当前目录加入 sys.path
uv run python -m pytest
# macOS 上 RLIMIT_AS 不生效，请跳过那条内存沙箱测试：
# uv run python -m pytest --deselect tests/test_execution.py::test_memory_limit
```

注意两点：

- CUDA / Flash-Attention-3 / Hopper 特性的测试在 CPU 或 Mac 环境下会被 `skip`，这是预期行为；
- `tests/test_execution.py::test_memory_limit` 在 macOS 上必失败，因为 Darwin 内核忽略 `resource.RLIMIT_AS`；它是沙箱对平台能力的依赖，不是教程或模型问题。

两层检查不能互相替代：Source Checks 发现教程与源码的漂移，pytest 验证真实实现。模型质量仍需 BPB、CORE、ChatCORE 和固定协议下的实验。

---

## 版本、资料与引用说明

- **上游项目**：[karpathy/nanochat](https://github.com/karpathy/nanochat)
- **本书对应 revision**：`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`
- **判断优先级**：固定 revision 的源码与测试 > 上游 README / `dev/LOG.md`

本书独立编写，不隶属于 nanochat 上游作者或任何课程。引用实现细节或实验数字时，请同时注明源码 revision、运行配置和评估协议。

---

## 开始阅读

从 **[第 01 章：总览](./01-overview.md)** 开始，先建立全局地图；随后进入第一部分，亲手追踪一段原始文本如何成为模型的训练张量。
