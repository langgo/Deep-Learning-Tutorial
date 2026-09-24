# 第 08 章：Base Eval：BPB、CORE 与 sample

> **本章源码地图**：逐 token 的 BPB 在 `nanochat/nanochat/loss_eval.py::evaluate_bpb`；base model 的 CORE 渲染、续写区间定位与打分在 `nanochat/nanochat/core_eval.py`；统一入口是 `nanochat/scripts/base_eval.py`；样本由 `nanochat/nanochat/engine.py::Engine.generate_batch` 生成。
>
> **调用链**：
>
> ```text
> base_eval.py
>   ├─ evaluate_bpb → GPT.forward(x, y, loss_reduction='none')
>   ├─ evaluate_core → evaluate_task → evaluate_example → forward_model
>   └─ Engine.generate_batch（人工阅读 sample）
> ```
>
> **本章边界**：只评估 **base model** 的文本续写行为：BPB、CORE 与 sample。聊天提示、ARC/MMLU 的受限字母分类、GSM8K/HumanEval 的生成验证、ChatCORE 与 pass@k 属于[第 11 章：Chat Eval](./11-chat-evaluation.md)。

上一章解决了“怎样更新参数”。但训练日志上的 loss 下降，离“模型会回答问题”还差很远。一个模型可能很擅长预测新闻文本中常见的标点和词尾，却不会做算术；也可能在某份公开题库上分数很高，却只是在训练数据里见过题目。**评估不是一个数字，而是一组受协议约束的测量。**本章从 held-out BPB 走到 nanochat 的 base CORE 和生成样本，最后说明这些 base-model 指标不能告诉我们的边界。

## 学习目标

读完本章，你应能：

1. 解释为什么低验证 loss 是必要信号，却不等于问答、推理或编程能力；
2. 从 nats 手推到 bits，再推到 bits-per-byte（BPB），并说明 BPB 为什么比每 token loss 更适合比较不同 tokenizer；
3. 沿 `evaluate_bpb` 读懂特殊 token 与 `ignore_index=-1` 如何被排除，以及分布式下为何先汇总再相除；
4. 说明 CORE 如何用 token 续写来评估多选、schema 和语言建模任务，并解释 few-shot 样例如何进入 prompt；
5. 说明 sample 为什么只是快速的定性 smoke test，而不是能力分数；
6. 在报告 base 评估时主动交代 checkpoint、split、few-shot、截样和题库污染等协议差异。

## 前置概念

建议先完成[第 06 章：Base Train：数据到损失与训练循环](./06-base-train-loop.md)与[第 07 章：Base Train：优化器与规模法则](./07-base-train-optim-scaling.md)。本章仍使用 `(B,T,V)`：`B` 是 batch 行数，`T` 是 token 位置数，`V` 是词表大小。另引入：

| 符号 | 含义 | 代码对应 |
|---|---|---|
| \(p(y_t\mid x_{<t})\) | 模型给真实目标 token 的概率 | softmax 后真实类别概率 |
| \(\ell_t\) | 位置 `t` 的负对数概率，单位为 nat | `loss2d` 的元素 |
| \(N\) | 累计的有效负对数概率和 | `total_nats` |
| \(C\) | 目标 token 对应的原始字节数和 | `total_bytes` |
| \(a_i\) | 第 `i` 个任务的 accuracy | `results[label]` |
| \(r_i\) | 任务的随机答对率 | `random_baseline` / `baseline_acc` |

这里的“验证集”应与训练数据隔离；它回答的是**同一数据分布上，模型对未参与参数更新文本的预测能力**。基准题则试图探测某种任务行为。两者都重要，也不能互相替代。

---

## 1. 先建立正确预期：loss 很有用，但它不是能力总分

语言模型训练的经典目标是最大化真实文本的概率，等价于最小化每个目标 token 的交叉熵：

\[
\ell_t=-\ln p(y_t\mid x_{<t}).
\]

真实 token 预测概率越高，\(\ell_t\) 越小。验证集 loss 下降通常意味着模型没有只记住训练 batch，而是更会压缩同分布的新文本；它也很密集、便宜、稳定，适合训练中频繁观察。

但它有三个根本局限。

1. **平均掩盖能力结构。**自然文本里高频空格、标点、常见短语很多；在它们上面进步，足以降低平均 loss，却未必改善长链推理、严格格式或代码执行。
2. **目标与使用方式不同。**base model 训练的是“续写下一个 token”；用户关心的常是遵循聊天格式、选择正确选项、输出最终数字或让程序测试通过。后者还受 prompt、解码温度、工具协议影响。
3. **分布和知识来源不同。**held-out 预训练文本与评测题不一定同分布；反过来，题库若已经混入训练语料，题目分数也不等于新题泛化。

所以实践上要把证据分层：BPB 看通用语言建模质量；CORE 看 base model 在固定 ICL（in-context learning）协议下能否用上下文完成多种任务；Chat 评估看 SFT/RL 后面对聊天提示的可用行为；样本则是快速的定性 smoke test。任何一项单独漂亮，都不该被解释成“全面更聪明”。

| 指标 | 常见形式 | 归一化单位 | 跨 tokenizer 直接比较？ |
|---|---|---|---|
| mean token loss | `total_nats / token_count` | 每 token 的 nat | 不宜 |
| perplexity | `exp(mean token loss)` | 每 token 的等效分支数 | 不宜 |
| BPB | `total_nats / (ln(2) × total_bytes)` | 每原始有效字节的 bit | 更合适，但仍须固定数据与协议 |

## 2. 从 nats 到 bits，再到每 byte：BPB 的账怎么记？

### 2.1 nat 是什么

`F.cross_entropy` 使用自然对数，因此 `GPT.forward(..., loss_reduction='none')` 产出的逐位置损失是 nat。若正确 token 概率为 \(p\)，损失 \(-\ln p\)；例如 \(p=1/4\)，则

\[
-\ln(1/4)=\ln4\approx1.386\ \text{nats}.
\]

信息论也常用以 2 为底的 bit：\(-\log_2p\)。换底公式给出

\[
-\log_2p=\frac{-\ln p}{\ln2}.
\]

因此 \(\ln4/\ln2=2\) bits：四个等可能选项需要两次二元判断。这一步正是 `evaluate_bpb` 最终除以 `math.log(2)` 的原因。

### 2.2 为什么不能只看每 token loss

token 不是天然统一的文本单位。一个 tokenizer 可能把某段 UTF-8 文本切为 10 个 token，另一个切为 6 个；即使二者对同样原始字符串分配了相近概率，“每 token”平均后的数也会因切分粒度改变。词表大小、合并规则和特殊 token 都会干扰横向比较。

BPB 用 token 所代表的**原始字节长度**做分母：

\[
\operatorname{BPB}=\frac{N}{\ln2\times C}
=\frac{\sum_t\ell_t}{\ln2\times\sum_t\operatorname{bytes}(y_t)}.
\]

其中 \(N\) 是所有有效目标的 nat 损失和，\(C\) 是这些目标 decode 后的 byte 数和。它可以理解为：平均编码一个原始 byte，模型还需要多少 bit 的不确定性；越低越好。

**手推 nats → bits/byte。**假设两处有效 target 的损失分别是 \(0.693\) 和 \(1.386\) nats，对应 token 分别表示 1 和 2 bytes：

\[
N=0.693+1.386=2.079,\qquad C=1+2=3.
\]

先转 bits：\(2.079/\ln2\approx3\) bits；再除以 3 bytes，BPB \(\approx1.0\)。注意不能把两处的 BPB `(0.693/ln2/1)` 与 `(1.386/ln2/2)` 随便做无权平均；必须按 byte 数加权，也就是先累积 \(N,C\) 再相除。

### 2.3 `evaluate_bpb` 的真实路径

`nanochat/nanochat/loss_eval.py::evaluate_bpb(model, batches, steps, token_bytes)` 的输入 `token_bytes` 是形状 `(vocab_size,)` 的整型张量：下标是 token id，值是该 token 的 byte 数；不计入指标的 token 值为 0。它由 `nanochat/nanochat/tokenizer.py::get_token_bytes` 从 tokenizer 目录读入，而不是在评估循环临时猜测。

每一步的核心是：

```python
x, y = next(batch_iter)                         # 各 (B, T)
loss2d = model(x, y, loss_reduction='none')     # (B, T)，单位 nat
loss2d, y = loss2d.view(-1), y.view(-1)         # 各 (B*T,)
num_bytes2d = token_bytes[y]                    # (B*T,)
total_nats += (loss2d * (num_bytes2d > 0)).sum()
total_bytes += num_bytes2d.sum()
```

这是教学摘录；实际代码还处理 `y<0`。因为 `-1` 是 `F.cross_entropy` 的 ignore index，不能拿负下标访问 `token_bytes`；它先构造 `valid=y>=0`，将无效 id 临时替换为 0 用于安全索引，再把无效位置的 byte 数置零。这样特殊 token（byte 数为 0）和主动 mask 的 token 都不会贡献损失或字节。

多 GPU 时各 rank 看到不同 batch。函数对 `total_nats` 和 `total_bytes` 各做一次 `dist.all_reduce(..., SUM)`，**然后**才计算 BPB。这保持了全体有效 byte 的加权平均。若没有有效 byte，代码返回 `float('inf')`，避免除零。这里没有 `model.eval()`；调用者 `base_eval.py` 负责模型加载的 eval phase，而函数本身用 `@torch.no_grad()` 保证不构建反传图。

## 3. `base_eval.py`：一次 base model 检查不只跑 BPB

`scripts/base_eval.py` 的 `--eval` 是逗号分隔的 `core,bpb,sample`，默认正好是三者全跑。它先经 `compute_init` 建立设备/分布式上下文，`load_model("base", ..., phase="eval")` 读取 checkpoint 和 tokenizer，再从 checkpoint 的 `meta["model_config"]["sequence_len"]` 取序列长度。不要把脚本头部的示例命令或参数默认值当作任何机器上的质量承诺。

BPB 部分以 `device_batch_size * sequence_len * ddp_world_size` 得出一次全局 step 的 token 数。若 `split_tokens` 不能整除它，脚本向下调整到最近的倍数；随后分别对 `train` 与 `val` 创建 `tokenizing_distributed_data_loader_bos_bestfit(...)`，报告两个 BPB。训练 BPB 与验证 BPB 的差距是有价值的诊断线索，但差距小也不能排除题库污染或任务短板。

sample 部分只在 rank 0 上运行。它用一组硬编码短提示词，以 `Engine.generate_batch(..., max_tokens=16, temperature=0)` 生成条件样本，并从 BOS 开始以 `temperature=1.0` 生成 8 个无条件样本、每个最多 128 token。这些是人类肉眼检查，不进入 CORE/BPB 数学指标；更不该只凭几条顺眼样本宣称能力提升。

## 4. CORE：用 token 续写评估多类任务

### 4.1 经典基线：直接生成答案

最直观的多选评估是让模型生成 `A/B/C/D`，然后做字符串比较。但小模型可能输出解释、空格或格式错误；不同选项长度也会令生成停止规则干扰结果。CORE 对 base model 采用更接近训练目标的办法：把每个候选答案接到相同上下文后面，比较模型认为该**continuation**有多自然。

在 `core_eval.py::forward_model` 中，`input_ids` 是 `(B,T)`，模型输出 logits `(B,T,V)`。代码用 `torch.roll(input_ids, shifts=-1, dims=1)` 构造“下一 token” target，逐位置交叉熵后再 reshape 回 `(B,T)`；最后一列没有下一个真实目标，设为 `nan`。预测 `predictions=outputs.argmax(dim=-1)` 也一并返回。

例如某候选完整序列为 `[BOS, q1, q2, a1, a2]`，答案从位置 `si=3` 开始、末尾 `ei=5`。位置 2 的 logits 预测 `a1`，位置 3 的 logits 预测 `a2`，所以 continuation 的 loss 切片恰是 `losses[i, si-1:ei-1]`。少减这一格会把“预测答案第一个 token”的 loss 漏掉；多取最后一格又会使用无意义的 `nan`。

### 4.2 三类任务，三种“共同区域”定位方式

`evaluate_example` 先按 `task_meta['task_type']` 渲染文本、tokenize（都 prepend BOS），再定位真正打分的区间。它不是用字符长度切割，而是用 token 序列的共同前/后缀，避开 BPE 边界陷阱。

| 类型 | 渲染函数 / 任务结构 | 要比较的部分 | 源码如何定位 |
|---|---|---|---|
| `multiple_choice` | `render_prompts_mc`：相同 query、不同 choice | 每个候选答案 continuation | `find_common_length(..., 'left')` 找共同 token 前缀 |
| `schema` | `render_prompts_schema`：不同 context option、相同 continuation | 每个候选 context | `find_common_length(..., 'right')` 找共同 token 后缀 |
| `language_modeling` | `render_prompts_lm`：同一项的“无 continuation / 有 continuation”两版 | 固定 continuation | 断言前者是后者 token 前缀 |

多选或 schema 任务取各候选 continuation 的**平均** loss，`min(mean_losses)` 的候选即预测项；取平均避免更长字符串仅因累加项更多而天然吃亏。语言建模任务不是比较候选概率，而要求 continuation 中每一 token 的贪心预测都等于真实 token：`torch.all(predicted_tokens == actual_tokens)`。这比“平均 loss 较低”严格得多，读 CORE 分数时不能把三种 task type 当成完全同一难度或同一统计量。

若模型带 `max_seq_len` 且 prompt 太长，代码保留序列最后 `max_tokens` 个 token，并将 `start/end` 同步左移；若 continuation 被截掉，断言会失败。padding 时 `stack_sequences` 将不同长度序列右侧填 BOS；评分区间只到每条实际 `end_idx`，因而 padding 不应进入这些 loss 切片。

### 4.3 few-shot 不是训练，也不是随机不可复现

CORE 的 prompt 可包含 demonstrations。`render_prompts_mc` 会将每个 few-shot example 的 `query + delimiter + gold choice` 写在当前题目之前；schema 也写入其正确 `context_options[gold]` 与 continuation；语言建模则写 context 与 continuation。它们是模型**当前上下文中看到的例子**，不产生梯度、不更新权重，因此称 few-shot/ICL。

`evaluate_example` 对第 `idx` 题用局部随机数生成器 `random.Random(1234 + idx)`，从除当前题外的数据中抽取 `num_fewshot` 个例子。固定 seed 加题号意味着同一数据顺序和配置下例子可复现，又避免把待测题本身抽作示范。`base_eval.py::evaluate_core` 从 `core.yaml` 读取每个任务的 `num_fewshot[0]`、类型、数据 URI 和可选 delimiter（未写时为单个空格）；这些元数据来自下载到 `eval_bundle` 的 bundle，而不是硬编码在 `core_eval.py`。

### 4.4 多任务平均前，先扣掉猜题优势

`evaluate_core` 逐任务读取 JSONL，按固定 `random.Random(1337)` shuffle；`--max-per-task` 大于 0 时才截取前若干条。全量与截样不能混在同一排行榜解释：截样更快，却会增加方差，也可能改变任务构成。

每项 accuracy \(a_i\) 从 `evaluate_task` 得到。多 rank 时每张卡按 `rank, rank+world_size, ...` 的步长处理题目，再 all-reduce 一个长度为数据集的 `correct` 向量，最后取均值。随机答对率 \(r_i\) 从 `eval_meta_data.csv` 读取，注意那里是**百分数**，所以实际中心化为：

\[
c_i=\frac{a_i-0.01r_i}{1-0.01r_i},\qquad
\operatorname{CORE}=\frac{1}{M}\sum_{i=1}^{M}c_i.
\]

若模型恰好随机水平，\(c_i=0\)；若满分，\(c_i=1\)。这避免四选题的 25% accuracy 与例如不同候选数任务的“猜中难度”直接混为一谈。它仍是**对各任务等权**的汇总选择：任务大小、真实使用频率、风险都没有自动进入权重。脚本把各 task 的原始 accuracy、centered 值和最终 CORE 写入 `base_eval/base_model_<step>.csv`。

## 5. sample：让人读，不替人下结论

`base_eval.py --eval sample` 只在 rank 0 上创建 `Engine(model, tokenizer)`。它对几条硬编码短 prompt 以 `temperature=0` 生成最多 16 个 token，又从 BOS 开始以 `temperature=1.0` 生成 8 条、每条最多 128 token 的无条件样本。它们不参与 BPB 或 CORE 的数学计算。

sample 的价值是快速发现接口或训练健康问题：例如 tokenizer/checkpoint 不匹配、输出全是重复符号、明显无法停止，或训练刚开始时文本尚不连贯。它不能替代定量指标：少量好句子可能是巧合，少量坏句子也可能只是采样随机性。聊天任务的分类、生成验证、ChatCORE 和 pass@k 应在[第 11 章：Chat Eval](./11-chat-evaluation.md)按其自己的 prompt 与 verifier 讨论。

## 6. 最小形状实验：CORE 比较的是 continuation loss

下面是教学化简的评分区间例子，不加载模型。某候选完整 token 序列是 `[BOS, q1, q2, a1, a2]`，答案从 `si=3` 开始，到 `ei=5` 结束。模型在位置 `t` 的 logits 预测 `t+1` 的 token，因此 `a1/a2` 对应的逐位置 loss 是 `losses[si-1:ei-1]`：

```python
import torch

# 假设 forward_model 已给出每个输入位置预测下一个 token 的 NLL。
losses = torch.tensor([0.2, 0.4, 1.1, 0.7, float("nan")])
si, ei = 3, 5
continuation_loss = losses[si - 1:ei - 1]
assert continuation_loss.tolist() == [1.1, 0.7]
print(continuation_loss.mean().item())  # 候选答案的平均 continuation loss
```

CORE 对多选/schema 候选比较这一段的平均 loss，而不是让 base model 自由生成整段答案。最后一个 `nan` 没有下一个真实 token，故切片必须在它之前停止。不同长度候选取平均，也避免仅因 token 更多而累计损失更大。

---

## 常见误区

1. **“验证 loss/BPB 低，所以模型肯定会推理。”**它只说明在该 held-out 文本分布上预测概率更好；难题、长程规划和格式遵循仍需单独测。
2. **“perplexity 和 BPB 是一回事。”**perplexity 通常是 `exp(mean nats per token)`；BPB 是先转 bit 再按原始 byte 归一化。二者的分母不同。
3. **“换 tokenizer后直接比较平均 token loss。”**token 粒度变了；BPB 的设计正是减弱这一不可比性，但要求 `token_bytes` 与 tokenizer匹配。
4. **“特殊 token 的 loss 应计入 BPB。”**`evaluate_bpb` 对 byte 数为 0 的 token 不计分，`-1` mask 也被排除。
5. **“CORE 就是让模型生成正确答案。”**当前 base CORE 多数是比较候选 continuation 的平均 loss；LM 类型又要求逐 token greedy 全对。
6. **“few-shot 是微调。”**few-shot example 仅写进本题 prompt，不更新任何权重；它的数量、抽样 seed、delimiter 都是评估协议的一部分。
7. **“一两条 sample 好看就说明模型全面更强。”**sample 是定性 smoke test；应和 BPB、CORE、checkpoint 和固定评测协议一起看。
8. **“基准高分证明泛化。”**训练数据或互联网语料可能包含题目、近重复题或答案，即 benchmark contamination；还要考虑 prompt 泄漏、开发集调参和只报最好的随机 seed。聊天 benchmark 的额外边界见[第 11 章](./11-chat-evaluation.md)。

## 小结

评估首先从语言建模的逐 token 负对数概率开始。nanochat 将 nat 损失累加、除以 \(\ln2\) 变 bit，再除以有效 target 的原始字节数，得到 tokenizer 较可比的 BPB；特殊 token 和 ignore mask 不参与，分布式时必须先合并总 nats 与总 bytes。

CORE 把 base model 的多任务能力转成 continuation likelihood：多选比较不同答案，schema 比较不同上下文，LM 则检查 continuation 的逐 token 贪心预测；few-shot 示例是固定可复现的上下文条件。任务准确率先按随机基线中心化，再等权平均为 CORE。`base_eval.py` 的 sample 则是肉眼检查，不进入上述指标。

最后，数字永远嵌在协议里。报告 checkpoint、数据 split、任务/子集、样本数、few-shot、是否截样，以及污染风险和方差，才是“base 模型学得怎样”这一问题的诚实答案；聊天任务的温度、top-k、`num_samples` 与 verifier 口径留给[第 11 章](./11-chat-evaluation.md)。

## 练习

1. **单位换算。**若一个有效目标 token 的预测概率是 \(1/8\)，写出它的 nat loss 和 bit loss。若该 token 表示 3 bytes，它单独贡献多少 BPB？解释为什么不能只从这个单项推断整套评测的 BPB。
2. **手算中心化。**某四选任务 accuracy 为 0.40，随机基线为 25%。按 `evaluate_core` 的百分数写法计算 centered score；再验证它与 \((0.40-0.25)/(1-0.25)\) 相同。
3. **追踪区间。**给一个候选 token 序列 `[BOS, q1, q2, a1, a2]`，答案起止为 `si=3, ei=5`。指出 `forward_model` 的哪两个 loss 位置分别预测 `a1`、`a2`，为何切片为 `si-1:ei-1`。
4. **sample 审计。**列出 `base_eval.py` 的条件 sample 与无条件 sample 各自的 prompt、temperature、样本数和最大 token 数。为什么它们不应被合成一个准确率？
5. **污染审计。**若你发现某项 CORE 高分而 val BPB、相邻任务和人工新题都没有改善，列出三种可能解释（包含污染），并说明各自需要什么额外证据。
6. **边界辨认。**为 BPB、CORE、sample 各写一句“它能支持的结论”和一句“它不能支持的结论”；再到[第 11 章](./11-chat-evaluation.md)列出聊天评估额外需要记录的两个协议参数。

## 前后章导航

- ← 上一章：[第 07 章：Base Train：优化器与规模法则](./07-base-train-optim-scaling.md)。那里解释参数为何能稳定更新；本章检验这些更新有没有带来可泛化的改善。
- → 下一章：[第 09 章：监督微调 SFT：教会模型对话](./09-sft.md)。base checkpoint 如何通过对话协议与监督 mask 变成聊天模型，将在那里展开；生成与聊天任务评估随后分别在第 10、11 章讨论。
