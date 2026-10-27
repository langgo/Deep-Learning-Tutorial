# 第 08 章　基础模型评估：BPB、CORE 与生成样本

> **本章定位**　第二部分·基础模型的最后一章。把"**数据 → 训练 → 评估**"闭环收尾。
>
> **中心问题**　训练 loss 一直在降，但那不代表模型变强了。那至少用什么指标来回答"**这个 base 模型会什么、不会什么**"？
>
> **范围**　只讲 base 模型的评估协议：BPB、CORE、固定 prompt 采样。聊天场景下的任务评估（MMLU、GSM8K、HumanEval 这些）属于[第 11 章](./11-chat-evaluation.md)。
>
> **先决条件**　读过[第 06 章](./06-base-train-loop.md)、[第 07 章](./07-base-train-optim-scaling.md)。知道 loss 从哪来、怎么被累计。
>
> **本章产物**　**单位换算**：从 nats 到 bits 再到 bits-per-byte 的完整推导；**CORE 评分规则**：三类任务（多选 / schema / 语言建模）各自怎么判对；**sample 的边界**：为什么它只是 smoke test。
>
> **源码入口**　`nanochat/loss_eval.py::evaluate_bpb`、`nanochat/core_eval.py`、`scripts/base_eval.py`、`nanochat/engine.py::Engine.generate_batch`。

---

## 本章讲什么

[第 06 章](./06-base-train-loop.md)讲完"**怎么算 loss**"，[第 07 章](./07-base-train-optim-scaling.md)讲完"**怎么用 loss 更新参数**"。两章读完，你会以为"**loss 一直在降 = 模型在变好**"。

**但 loss 下降和"模型会答对问题"之间差得很远**。举两个常见的失误：

- 一个模型可能特别擅长预测"**句尾的句号 / 常见短语的下一个 token**"，于是训练 loss 一路降，但**完全不会算 1 + 1**；
- 一个模型可能在**某公开题库**上分数很高，不是因为它会做题，而是因为**那套题的题目在训练数据里被爬到过**（benchmark contamination，数据污染：评测题目泄露进了训练集，就像考试前试卷泄露，分数反映的是“背题”而不是能力）。

所以评估不是**算一个数字**，而是**用若干不同协议各看一面**——每个数字都带着自己的局限。本章就把 nanochat 的 base 评估分三条看：

- **BPB**：延续训练目标，看在**没见过的文本**上还能把字节压缩得多紧；
- **CORE**：用上下文学习（in-context learning）看 **base 模型能否用 prompt 完成多种任务**；
- **Sample**：固定 prompt 肉眼看，**最便宜的健康检查**。

**读完本章你应该能**：

1. 讲清为什么**低 loss 必要但不充分**；
2. 推导：**nat → bit → BPB**，并说清 BPB 为什么比平均 token loss 更公平；
3. 读懂 `evaluate_bpb` 怎么**先汇总再相除**、怎么排除特殊 token 和 `-1` mask；
4. 讲清 CORE 对**多选 / schema / 语言建模**三类任务各自怎么判对；
5. 讲清 few-shot example 为什么**不是训练**，而是 prompt 的一部分；
6. 讲清 CORE 为什么要**先扣掉随机猜题的优势**再平均；
7. 报告 base 评估时主动交代 **checkpoint / split / few-shot / 截样 / 污染风险** 这些协议差异。

> **本章结构**：§1 先讲为什么 loss 不够；§2 用一张表把 `base_eval.py` 的三件事定位清楚；§3–§5 再把 BPB、CORE、Sample 各自讲成一个完整的小篇。

---

## 预备词汇

本章继续用 `(B, T, V)`。另引入几个新符号：

| 符号 | 含义 | 代码对应 |
|---|---|---|
| `x_<t` | 位置 `t` 之前的所有 token，即前缀 `(x_0, …, x_{t-1})`（论文常用记法） | 因果 mask 保证位置 `t` 只能看到 `x_<t` |
| `p(y_t \| x_<t)` | 模型给**真实目标 token** 的条件概率（已知前缀 `x_<t`） | logits 做 softmax 后真实类别那一列 |
| `ℓ_t` | 位置 `t` 的负对数概率（单位 nat） | `loss2d` 的元素 |
| `total_nats` | 所有有效目标位置的 nat 损失总和（BPB 分子） | `total_nats` |
| `total_bytes` | 这些有效目标 token 对应的**原始字节数**总和（BPB 分母） | `total_bytes` |
| `a_i` | 任务 `i` 的 accuracy | `results[label]` |
| `r_i` | 任务 `i` 的"**纯猜能对的概率**"（随机基线） | `random_baseline` |

**"验证集" vs "基准题"**：

- **验证集**（val split）是**和训练数据同分布但不参与训练**的文本。它回答："**模型在同分布的新文本上预测概率怎样？**"
- **基准题**（例如 ARC、MMLU）是**特定任务**：选择题、问答、代码。它回答："**模型能不能完成某种任务？**"

两者都重要，**都不能替代对方**。

---

## 1. 为什么 loss 必要但不充分

### 1.1 loss 到底在说什么

训练的目标是让模型尽量给真实文本更高的概率。对每个位置，交叉熵只看一件事：**模型给真实下一个 token 的概率是多少**。

```
ℓ_t = -ln p(y_t | x_<t)
```

负对数把“概率”翻译成“惊讶程度”：

```text
p = 1.0   → ℓ = 0       完全确定，毫不意外
p = 0.5   → ℓ ≈ 0.69    猜对一半
p = 0.1   → ℓ ≈ 2.30    相当意外
p → 0     → ℓ → ∞      真实 token 被认为几乎不可能
```

所以 `ℓ_t` 可以读成：**看到正确答案时，模型有多意外**。整段文本的 loss 就是这些意外程度的平均（第 08 章后面还会把它换算成每字节的 bit 数）。

这里有一个关键细节：loss 评价的是**模型对整个词表的概率分布**，而不是“最终选对没选对”。举个例子，输入是 `The capital of France is`，真实下一词是 `Paris`：

```text
模型 A: p(Paris)=0.9   → ℓ ≈ 0.11
模型 B: p(Paris)=0.4   → ℓ ≈ 0.92
```

两个模型用 argmax 生成时可能都会输出 `Paris`，但 loss 不同——因为 A 更“自信”。这说明 **loss 是一个比“对/错”更细粒度、更平滑的信号**：哪怕模型还没能把概率推过 0.5，loss 下降也能反映它正在变好。这正是它适合当训练目标的原因。

所以：loss 下降意味着模型**在同分布文本上对真实 token 的预测概率更高**。这是一个**必要**的信号：一个连语言建模都做不好的模型，几乎不可能会推理。但它远不是**充分**的——下一节解释为什么。

### 1.2 为什么它不够：三个根本局限

**局限 1：平均掩盖能力结构**

自然文本包含大量相对容易、频繁出现的模式，例如空格、标点、常见短语和词尾。模型只在这些位置上进步，也可能让平均 loss 明显下降；而长链推理、严格格式输出、代码执行等稀少且困难的能力未必同步改善。具体占比取决于语料和 tokenizer，不能用一个固定百分比概括。

类比：一个考试的平均分提高 10 分，可能来自：
- 简单题人均多对 10 分（能力提升）；
- 简单题人均多对 15 分 + 困难题人均少对 5 分（能力结构改变）。

平均分一样，但模型的真实能力完全不同。

**局限 2：训练目标和用户关心的事不一样**

- 训练在学"**预测下一个 token**"；
- 用户关心的是"**选对答案 / 写对代码 / 遵循聊天格式**"。

能预测新闻报道的下一个 token，不等于会解数学题。

**局限 3：数据分布和知识来源**

- held-out 预训练文本可能和评测题**不同分布**（新闻 vs. 数学）；
- 反过来，**评测题的题目可能在训练数据里出现过**（benchmark contamination）——分数高只是记忆题，不是能力。

### 1.3 把证据分层

实际做法是**用多个指标互相印证**：

| 指标 | 看什么 | 适合什么比较 |
|---|---|---|
| **mean token loss** | 平均每个 token 的 nat | **同 tokenizer 内部**对比训练进度 |
| **perplexity** | `exp(mean token loss)` | 同 tokenizer，和 token loss 等价 |
| **BPB** | 每个**原始字节**需要多少 bit | **跨 tokenizer** 比较语言建模质量 |
| **CORE** | 固定协议下，base 模型能不能完成多种任务 | ICL（few-shot）能力 |
| **Sample** | 固定 prompt 的生成文本 | 定性 smoke test |

**任何一项单独好看都不能宣称"模型全面变强"**。

---

## 2. 一次 base 评估的全貌：base_eval.py 跑哪三件事

在深入每个指标前，先看它们怎样被一起跑起来。`scripts/base_eval.py` 是 base 评估的统一入口：

```bash
python -m scripts.base_eval --eval core,bpb,sample
```

`--eval` 是三件事的任意组合（逗号分隔），默认三件都跑。启动过程是：

1. `compute_init()` 建立设备和分布式上下文；
2. `load_model("base", phase="eval")` 读 base checkpoint 和 tokenizer；
3. 从 checkpoint 的 `meta["model_config"]["sequence_len"]` 读序列长度；
4. 跑 `--eval` 指定的项目。

这三件事回答的是三个不同层面的问题，互相补充，谁都不能替代谁：

| 协议 | 回答什么问题 | 产物 | 本章小节 |
|---|---|---|---|
| **BPB** | 在没见过的文本上，还能把字节压缩得多紧？ | 一个可跨 tokenizer 比较的数字 | §3 |
| **CORE** | base 模型能否用 prompt 完成多种任务？ | 一个中心化后的综合分数 | §4 |
| **Sample** | 模型当前“看起来”正常吗？ | 给人看的生成文本 | §5 |

接下来 §3–§5 依次展开每个协议：先讲它衡量什么、怎么算，再讲 `base_eval.py` 给它配了哪些具体设定。

---

## 3. BPB：把 loss 从 nat 换成 bits-per-byte

### 3.1 Nat 是什么

PyTorch 的 `F.cross_entropy` 用**自然对数**，所以 `loss2d` 的单位是 **nat**：

```
ℓ_t = -ln p(y_t | x_<t)
```

举例：如果真实 token 的概率 `p = 1/4`：

```
ℓ_t = -ln(1/4) = ln(4) ≈ 1.386 nats
```

### 3.2 从 nat 到 bit：换底

信息论更常用的单位是 **bit**（以 2 为底的对数）：

```
-log₂(p) = -ln(p) / ln(2)
```

所以 `1.386 nats = 1.386 / ln(2) ≈ 2 bits`。直觉：**四个等可能选项需要两次是/否判断才能定位**——完全吻合。

**这就是 `evaluate_bpb` 最后要除以 `math.log(2)` 的原因**：把 nats 换成 bits。

### 3.3 为什么不能只看"**平均 token loss**"

先讲结论：“每 token 平均 loss”把 **token 数当成了分母**，而 token 是个**会随 tokenizer 变化的单位**。分母不稳定，比较就不公平。

用一个例子看清楚。同一段 UTF-8 文本、同样的原始字节，两个 tokenizer 切法不同：

```text
Tokenizer A（细切）: 切成 100 个小 token
Tokenizer B（粗切）: 切成  25 个大 token
```

关键点是：**两个模型要编码的是同一段文本。** 一段文本本身含有多少“信息量”是固定的，和你怎么切无关。所以如果两个模型质量相当，它们给整段文本的**总 nats 也差不多**（假设都是 100 nats）。只是这 100 nats：

- 在 A 里被摊到 **100 个 token** 上；
- 在 B 里被摊到 **25 个 token** 上。

现在分别除以 token 数：

| 模型 | 总 nats（整段文本） | token 数 | 每 token 平均 loss |
|---|---|---|---|
| 模型 A（细切） | 100 | 100 | 100 / 100 = **1.0** |
| 模型 B（粗切） | 100 | 25 | 100 / 25 = **4.0** |

A 的“每 token 平均 loss”看起来低很多，但这纯粹是因为它把同样的信息量**分摄到更多 token 上**——每个 token 平均承担得少，并不是模型更强。两个模型对这段文本做的其实是**同一件事**。

这就是问题所在：只要换 tokenizer，mean token loss 的数字就会变，但它并没告诉你模型到底好不好。所以**跨 tokenizer 比质量时，需要一个不随切法变化的分母**——这就是下一节的“字节”。


### 3.4 BPB：用**字节数**做分母

**bits-per-byte** 的做法是用 token 对应的**原始字节数**做归一化：

```
         ∑_t ℓ_t                   分子 total_nats: 所有位置的 nat 总和
BPB = ─────────────────
       ln(2) × ∑_t bytes(y_t)    分母 total_bytes: 这些位置目标 token 的字节总和
```

直觉：**平均而言，模型要花多少 bit 才能编码一个原始字节的文本**？

- 字节是一个**跨 tokenizer 不变的单位**（UTF-8 规则下所有字节都一样）；
- 所以 BPB 让"**模型 A 和模型 B 在同样文本上的质量**"真正可比。

### 3.5 手算一次：从 nats 到 BPB

假设评估两个位置，分别得到：

- 位置 1：loss = `0.693 nats`，对应 token = **1 字节**；
- 位置 2：loss = `1.386 nats`，对应 token = **2 字节**。

**正确做法：先累加分子分母，再相除**。

```
total_nats  = 0.693 + 1.386 = 2.079 nats
total_bytes = 1 + 2 = 3 bytes

BPB = total_nats / (ln(2) × total_bytes)
    = 2.079 / (ln(2) × 3) ≈ 2.079 / 2.079 ≈ 1.0 bits/byte
```

**错误做法**：对每个位置算 BPB 再平均。

```
位置 1 单独 BPB = 0.693 / (ln(2) × 1) ≈ 1.0
位置 2 单独 BPB = 1.386 / (ln(2) × 2) ≈ 1.0
无权平均:        (1.0 + 1.0) / 2 = 1.0
```

**这次恰好得到同一个数**，但只是巧合——两个位置字节数本来就不同，无权平均没有按字节加权。换组数就对不上了：如果位置 1 是 10 bytes、位置 2 是 1 byte，无权平均就会偏得很明显。

**所以 `evaluate_bpb` 必须先累积 `total_nats` 和 `total_bytes`，最后再相除**。

### 3.6 `evaluate_bpb` 的真实流程

```python
# nanochat/loss_eval.py::evaluate_bpb 简化
total_nats  = 0.0
total_bytes = 0

for _ in range(steps):
    x, y = next(batch_iter)                           # 各 (B, T)
    loss2d = model(x, y, loss_reduction='none')       # (B, T) nats
    loss2d, y = loss2d.view(-1), y.view(-1)           # 展平

    # 1) 把 y < 0 (SFT mask=0 的位置) 临时替换成 0 以安全索引
    valid = y >= 0
    y_safe = y.where(valid, torch.zeros_like(y))
    num_bytes2d = token_bytes[y_safe]                 # (B*T,)
    num_bytes2d = num_bytes2d.where(valid, torch.zeros_like(num_bytes2d))

    # 2) 累积有效位置的 nats 和 bytes
    total_nats  += (loss2d * (num_bytes2d > 0)).sum()
    total_bytes += num_bytes2d.sum()

# 3) 多 rank：先 all-reduce，再相除
dist.all_reduce(total_nats,  op=dist.ReduceOp.SUM)
dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)

return total_nats / (math.log(2) * total_bytes)
```

**三件关键的事**：

**① 特殊 token 的字节数是 0**。`token_bytes` 对 `<|bos|>`、`<|user_start|>` 这些给 0。字节数为 0 的位置既不进分子（会被 `(num_bytes2d > 0)` mask 掉）也不进分母。所以 **BPB 不包含特殊 token 的 loss**。

**② `y < 0` 的位置被跳过**。SFT 阶段 `-1` 是 `cross_entropy` 的 `ignore_index`，不能当下标访问 `token_bytes`。源码先用 `valid = y >= 0` 把无效位置 mask 掉，再临时把 `-1` 替换成 `0` 做索引。

**③ 多 rank 下**：先对所有 rank 的 `total_nats` 和 `total_bytes` 做 `all_reduce(SUM)`，最后才相除。错误做法是让每个 rank 自己算 BPB 再平均——各 rank 的字节数可能不同，无权平均不正确。

### 3.7 base_eval.py 给 BPB 的具体设定

- 每一次全局 step 的 token 数 = `device_batch_size × sequence_len × ddp_world_size`；
- `split_tokens` 不能整除时向下调整到 `tokens_per_step` 的倍数；若配置得比一个全局 step 还小，会被调整为 0，最终 BPB 无有效字节而返回 `inf`，所以至少要覆盖一个 step；
- 对 `train` split 和 `val` split **各算一次 BPB**。

**为什么 train 和 val 都算**？因为**它们的差距**是关键诊断信号：

| 现象 | 诊断 |
|---|---|
| train BPB 低 + val BPB 高 | 过拟合 |
| train 和 val 都低 | 泛化不错，但要警惕 contamination |
| train 和 val 差距小 | 数据可能太少、模型没充分学到训练分布 |

但**差距小不代表一切都好**——还是要靠 CORE 和 Sample 补充证据。

### 3.8 BPB 不是万能的

- BPB **只衡量字节级压缩质量**——它看不到"**是否选对答案**"；
- **换数据分布** BPB 就会变——同一个模型在新闻文本上 BPB 低、在代码上 BPB 可能高，不代表模型变差；
- **污染**：如果 val split 和训练数据有重复，BPB 会假性低。

所以 BPB 要**和 CORE、Sample、chat 评估一起看**。

---

## 4. CORE：用 continuation loss 评估多种任务

> 本节反复出现的 **continuation**（续写）指：接在 prompt（上文）后面、交给模型“接着写”并用来打分的那段 token。在不同任务里它可能是候选答案、正确下文或被比较的目标。

### 4.1 为什么不直接生成答案

**最直观的做法**：给模型选择题 prompt，让它生成 `A`，然后字符串比较。

**问题**：

- Base 模型没经过 SFT，可能生成"**让我想想，答案是 B 因为...**"而不是单字母；
- 不同选项**字符串长度**可能不同，影响生成停止；
- 格式错误要靠正则兜底，不稳定。

**CORE 的做法**：**把每个候选答案接在相同上下文后面，比较模型给它们的 loss**。loss 最低的候选 = 模型最相信的答案。

这叫 **continuation likelihood**：底层仍使用 next-token 负对数似然，但评估协议还会限定 continuation 区间、按候选 token 数取平均，再在候选之间取最小值。它复用了预训练信号，不等于训练时对整段文本求平均 loss 的完整目标。

### 4.2 continuation 的 loss 怎么算

在钻进细节前，先把从代码输出到最终 CORE 分数的整条链看一遍（详细步骤分布在 §4.2–§4.6）：

```text
§4.2  过一遍模型，得到逐位置的 losses[b,t] 和 predictions[b,t]
        │
        │ 只取 continuation 区间（用 §4.2 的切片规则）
        ▼
§4.3  把一道题判成一个 0/1：correct
        │   多选 / schema：比哪个候选的平均 loss 最低
        │   语言建模：  逐 token greedy 是否全对
        ▼
§4.6  一个任务：accuracy a_i = 该任务所有题 correct 的均值
        │
        ▼
§4.5  中心化：c_i = (a_i - 0.01·r_i) / (1 - 0.01·r_i)   扣掉瞎猜优势
        │
        ▼
§4.5  最终：CORE = 所有任务 c_i 的等权平均
```

这一节先讲第一步：怎样算出 `losses` 和 `predictions`，以及怎样从中取出 continuation 的 loss。

`losses` 和 `predictions` 都是 `(B, T)`，逐位置含义不同，分别服务两种判对方式：

```text
losses[b,t]      = 在位置 t 预测“位置 t+1 的真实 token”的交叉熵
                   → 越小 = 模型越相信真实的下一个 token
                   → 多选 / schema 用它比“哪个候选 loss 最低”
predictions[b,t] = 在位置 t 概率最高的 token（argmax）
                   → “若让模型 greedy 生成，这一步会吐什么”
                   → 语言建模用它判“逐 token 是否全对”
```

要算“模型对续写多自信”，就是把整条序列过一遍模型，再把属于 continuation 的那几个位置的 loss 取出来。难点只有一个：**模型在位置 `t` 预测的是下一个 token（位置 `t+1`），所以“某个 token 的 loss”并不在它自己的位置上。**

#### 先看每个位置算出的 loss 对应谁

```python
# input_ids: 一批候选序列，每条形如 [<BOS>, q_1, q_2, a_1, a_2]
input_ids = ...                                              # (B, T) 候选序列

# 一次 forward：每个位置 t 输出对“下一个 token”的打分（未归一化的 logits）
logits = model(input_ids)                                    # (B, T, V)

# 因果语言模型的规则：位置 t 的目标是位置 t+1 的 token。
# torch.roll 把整条序列向左移一位，让 targets[t] == input_ids[t+1]。
# （最后一位被 roll 回开头，是无意义的垃圾值，稍后置 NaN 丢掉）
targets = torch.roll(input_ids, shifts=-1, dims=1)           # (B, T)

# 对每个位置单独算交叉熵（reduction='none' 表示不求和、不平均），
# 得到每个位置一个 loss；再 reshape 回 (B, T)。
# 注意：losses[t] 是“在位置 t 预测位置 t+1”的 loss，不是 token t 自己的 loss。
losses = F.cross_entropy(
    logits.view(-1, V), targets.view(-1), reduction='none'
).view(B, T)                                                 # (B, T)

# 最后一列没有真实的“下一个 token”（已是序列结尾），它的 target 是上面 roll
# 带过来的垃圾值，所以显式置 NaN，防止误用。
losses[:, -1] = float('nan')                                 # (B, T)

# argmax 取每个位置概率最高的 token，供语言建模任务的“逐 token 全对”判定用（§4.3）
predictions = logits.argmax(dim=-1)                          # (B, T)
```

关键是理解 `losses[t]` 到底是谁的 loss：**它是“在位置 `t` 预测位置 `t+1`”的 loss**，而不是 token `t` 自己的 loss。

#### 用一个具体例子对齐

候选序列是 `[<BOS>, q_1, q_2, a_1, a_2]`，prompt 占位置 0–2，answer（要评分的 continuation）是 `a_1, a_2`，在位置 3–4：

```text
位置 t:        0      1      2      3      4
token:        <BOS>  q_1    q_2    a_1    a_2
              └─── prompt ──┘  └─ answer ─┘
                                 si=3   ei=5   ← answer 区间 [si, ei)

losses[t] 预测谁:
  losses[2] = “看到 <BOS>,q_1,q_2 后预测 a_1” 的 loss
  losses[3] = “看到 ...,a_1 后预测 a_2”       的 loss
  losses[4] = NaN（a_2 后面没有下一个 token）
```

所以，“a_1 和 a_2 这段 continuation 的 loss”并不在位置 `3,4`，而是在 **预测它们的那两个位置 `2,3`** 上。换成区间就是：

```text
answer token 位在 [si, ei) = [3, 5)
对应的 loss 在   [si-1, ei-1) = [2, 4)
```

因此源码取的是 **`losses[si-1 : ei-1]`**，即 **`losses[2 : 4]`**，恰好是“预测 a_1”和“预测 a_2”两条 loss。把这两条平均，就是这个候选的 continuation loss。

#### 为什么容易切错

- 写成 `losses[si : ei]` = `losses[3:5]` → 拿到的是“预测 a_2”和“NaN”，**漏掉了 a_1、又把 NaN 带进来**；
- 写成 `losses[si-1 : ei]` = `losses[2:5]` → 多包了最后一列的 **NaN**。

两种都会让平均 loss 变成 NaN 或偏错，所以这个 `-1` 偏移是 CORE 实现的正确性关键（§6 有一个可跑的最小验证）。

### 4.3 三类任务，三种评分方式

`core.yaml` 里的任务按 **task_type** 分三类。先用一张表把它们放在一起对比，再逐个展开：

| 任务类型 | 多个候选之间什么变、什么固定 | 评分方式 | 选哪个 |
|---|---|---|---|
| 多选 multiple_choice | 上文相同，**continuation 不同** | 比各候选的平均 loss | loss 最低的候选 |
| Schema | **上文不同**，continuation 相同 | 比同一 continuation 在各上文下的平均 loss | 让它 loss 最低的上文 |
| 语言建模 language_modeling | 只有一个正确 continuation | 逐 token 是否 greedy 全对 | 全对算对，错一个就错 |

前两类是镜像：都是“比 continuation loss”，只是**把什么当成变量**不同。第三类是另一种评分规则：不比 loss 大小，而是要求逐 token 全对。

#### 多选（multiple_choice）

固定一个问题，几个不同答案做 continuation：

```text
query:   "法国的首都是哪里？"
choices: ["巴黎", "伦敦", "柏林", "罗马"]

拼成 4 条候选序列，上文都是“法国的首都是哪里？”：
  "...哪里？" + "巴黎"
  "...哪里？" + "伦敦"
  "...哪里？" + "柏林"
  "...哪里？" + "罗马"
```

每条序列按 §4.2 算出 continuation 的**平均 loss**，取 `min(mean_losses)` 的候选作为预测。

**为什么用平均而不是总 loss**：候选长度不一（"巴黎" 2 token，"柏林" 3 token）。用总 loss 的话，长候选多加了一项正 loss，天生吃亏；除以 token 数做平均后才公平。

**找评分区间**：因为四条候选的**前缀（prompt）相同**，源码用 `find_common_length(..., 'left')` 找到公共前缀长度，前缀之后各自不同的那段就是 continuation，分别评分。

#### Schema

schema 把多选反过来：**固定同一句 continuation，换不同上文**，问哪个上文更能引出这句话：

```text
contexts:     ["如果 X 相信 Y, 那 X 会...", "尽管 X 相信 Y, 那 X 不会..."]
continuation: "...救助 Y."

拼成 2 条候选，continuation 都是“救助 Y”：
  "如果 X 相信 Y, 那 X 会..." + "救助 Y"
  "尽管 X 相信 Y, 那 X 不会..." + "救助 Y"
```

谁让“救助 Y”这段 continuation 的平均 loss 更低，就选谁对应的上文。

**找评分区间**：这次候选的**后缀（continuation）相同**，所以用 `find_common_length(..., 'right')` 找公共后缀，把这段公共 continuation 设为 loss 区间——正好和多选题用 `'left'` 找前缀相反。

#### 语言建模（language_modeling）

给一段 context，后面只有一个**唯一正确**的 continuation，没有其他候选可比。所以评分规则换成更严的一种：要求 continuation 里**每一个 token 的 greedy 预测都等于真实 token**：

```python
torch.all(predicted_tokens == actual_tokens)
```

这不是比 loss 大小，是逐 token 严格全对。所以**语言建模任务的分数通常更低**——哪怕模型给正确 continuation 的概率最高，只要中间有一个 token 的 argmax 预测错了，整题就算错。

所以读 CORE 分数时，**不能把这三类任务当成同一难度**：语言建模天生比选择题苛刻。

### 4.4 Few-shot：在 prompt 里塞示范

CORE 的 prompt 可以包含 **demonstrations**（few-shot examples）：

```
<示范 1 的 query>  <delimiter>  <示范 1 的正确答案>
<示范 2 的 query>  <delimiter>  <示范 2 的正确答案>
...
<真正要测的 query>  <delimiter>  <候选 continuation>
```

**三件要点**：

**① 这不是训练**。few-shot example 只写进 **prompt** 里让模型"**看**"——它们**不产生梯度、不更新权重**。这叫 **in-context learning（ICL）**：模型用 attention 在 prompt 里就地看懂规律。

**② 示例可复现**。`evaluate_example` 用 `random.Random(1234 + idx)` 为每一题抽 few-shot 示例——`idx` 是题号，所以**同一数据顺序和配置下示例集完全可复现**，而且**不会抽到当前题自己**。

**③ 示例数由 `core.yaml` 决定**。每个任务都定义了自己的 `num_fewshot[0]`、delimiter、数据 URI。这些**元数据跟着任务来**，不是硬编码在 `core_eval.py`。

### 4.5 中心化：扣掉"**纯猜**"的优势

**问题**：不同任务的"**瞎猜能对的概率**"不同：

- 4 选题瞎猜：25%；
- 二元 schema 瞎猜：50%；
- 语言建模任务瞎猜：接近 0%。

**如果直接把 accuracy 加起来平均，4 选题的 25% 会给"**瞎猜模型**"看起来还不错的平均分**——不公平。

CORE 的解决办法：**先把每个任务的 accuracy 减去随机基线、再归一化到 [0, 1]**：

```
          a_i - 0.01 × r_i
c_i  =  ──────────────────       r_i 是百分比形式的随机基线（如 25）
          1 - 0.01 × r_i
```

- 模型**恰好瞎猜水平**：`c_i = 0`；
- 模型**满分**：`c_i = 1`；
- 模型**比瞎猜还差**：`c_i < 0`。

**例子**：4 选题 accuracy = 0.40，随机基线 25：

```
c_i = (0.40 - 0.25) / (1 - 0.25) = 0.15 / 0.75 = 0.20
```

然后对所有任务**等权平均**：

```
            1   M
CORE  =  ─── ∑  c_i
            M  i=1
```

**注意是等权**——不按任务数据量、不按真实使用频率加权。所以 CORE 高不等于"**对用户关心的任务都强**"——它是**对评估协议里选的那些任务等权**。

### 4.6 多卡时怎么算

多 rank 情况下，每张卡按 `rank, rank+world_size, 2*world_size+rank, ...` 的步长拿题目，每题各自算完 correct/wrong。最后用 **`all_reduce`** 聚合一个长度为 `len(dataset)` 的 `correct` 向量（0/1 的）——再取均值得到任务 accuracy。

**不要**让每张卡自己算 accuracy 再平均（不同 rank 看的题数可能不等）。

### 4.7 关于截样和复现

- `--max-per-task` 大于 0 时，每个任务**先按固定 seed `random.Random(1337)` shuffle**，然后截前若干条；
- **全量 CORE 和截样 CORE 不能混在同一张排行榜解释**——截样噪声大，分数可能偏高也可能偏低；
- 结果写入 `base_eval/base_model_<step>.csv`，含每个任务的原始 accuracy、centered 值、最终 CORE。

---

## 5. Sample：肉眼看的 smoke test

### 5.1 定位

Sample 是**最便宜、最快的健康检查**。它不进 BPB / CORE 的任何数学指标，只给**人类**看。

**便宜**：几秒钟就能看完所有样本。
**不可定量**：少量好句子可能是巧合，少量坏句子也可能只是随机采样。

### 5.2 base_eval.py 给 Sample 的具体设定

**只在 rank 0 上跑**，做两件事：

| 场景 | prompt | temperature | 条数 | max_tokens |
|---|---|---|---|---|
| 条件生成 | 几条硬编码短 prompt | **0**（贪心） | 一条 prompt 一条样本 | 16 |
| 无条件生成 | 只有 `<|bos|>` | **1.0** | 8 条 | 128 |

**温度 0 的用意**：**同一个 checkpoint 多次采样结果一致**，方便 checkpoint 之间对比。
**温度 1.0 的无条件生成**：看模型在**没有引导**时倾向生成什么——训练刚崩溃时（例如 NaN）就会表现为全是空字符、全是重复 token、全是乱码。

`Engine.generate_batch` 返回的序列包含 prompt，当前 `base_eval.py` 直接 decode 整条序列，因此条件样本输出里也会包含原 prompt。两类样本都不进入 BPB / CORE，只供人工检查。

### 5.3 Sample 能看出什么

| 现象 | 看出的问题 |
|---|---|
| 全是重复 token（`!!!!!!!!`） | 检查 loss/权重是否 NaN、采样设置和训练退化 |
| 大量空白或异常控制符 | 检查 tokenizer、特殊 token 概率和 checkpoint 健康度 |
| 看似流畅但语义飘忽 | 可能是容量/训练不足，也可能只是随机样本 |
| 到 `max_tokens` 仍不停 | 检查终止 token 学习、prompt 和生成上限；base 模型本来也未必按聊天协议停止 |
| 明显乱码 | 优先检查 tokenizer 与 checkpoint 是否匹配，再查权重损坏 |

### 5.4 它不能替代 BPB / CORE

**三种错误用法**：

- "**模型生成的那条很顺畅，所以模型变好了**"——不行，10 条里只挑一条看，连样本分布都没覆盖；
- "**样本答对了问题，所以模型会答题**"——不行，可能是 prompt 恰好触发了训练数据里的类似表达；
- "**我们在 X 版的 sample 里看到 Y 问题，所以 X 比 X-1 差**"——不行，温度 1.0 下随机性很大。

**正确用法**：**在出问题时用它快速定位**。日常评估还是看 BPB 和 CORE。


---

## 6. 一个最小 continuation loss 实验

不加载模型，手动验证 §4.2 的切片规则：

```python
import torch

# 假装 forward_model 已经给出了每个输入位置预测下一个 token 的 NLL
# 候选序列: [<BOS>, q_1, q_2, a_1, a_2]
losses = torch.tensor([0.2, 0.4, 1.1, 0.7, float("nan")])
#                      ↑    ↑    ↑    ↑    ↑
#                      预测  预测  预测  预测  (没有下一token)
#                      q_1   q_2  a_1   a_2

si, ei = 3, 5                          # answer 从位置 3 到位置 5（不含）
continuation_loss = losses[si - 1 : ei - 1]
assert continuation_loss.tolist() == [1.1, 0.7]
print("mean loss =", continuation_loss.mean().item())
```

**三件验证**：

- `losses[si-1]` 是**预测 answer 第一个 token** 的 loss；
- `losses[ei-1]` 是**预测 answer 最后一个 token** 的 loss；
- 不包含最后一列的 `nan`。

**练习**：如果 `[si, ei] = [3, 5]`、`[si-1, ei-1] = [2, 4]`，而你写成 `losses[si : ei]` = `losses[3 : 5]`，拿到的是 `[0.7, nan]`——漏掉了 `a_1` 的 loss 还把 `nan` 带进来，平均变 nan。所以**精确切片位置是 CORE 实现的核心正确性**。

---

## 常见误区

1. **"验证 loss 低 = 模型会推理。"** 不是。它只说明在该 held-out 文本分布上预测概率变好，长链推理 / 格式遵循 / 代码能力都要单独测。
2. **"perplexity 和 BPB 是一回事。"** 不是。perplexity = `exp(mean nats per token)`，分母是 token；BPB 的分母是**原始字节**。换 tokenizer 后 perplexity 不可比，BPB 可比。
3. **"换 tokenizer 后直接比 mean token loss。"** 不行。token 粒度变了。要比质量就用 BPB——而且 `token_bytes` 必须和当前 tokenizer 匹配。
4. **"特殊 token 的 loss 应该计入 BPB。"** 不计。`token_bytes` 对特殊 token 给 0 字节，`-1` ignore 位置也排除。
5. **"CORE 就是让模型生成答案字符串。"** 不是。CORE 的大部分任务是**比较候选 continuation 的平均 loss**；只有语言建模任务要求 greedy 全对。
6. **"few-shot 是在做 fine-tuning。"** 不是。few-shot 示例只进 prompt，**完全不更新权重**。它是 in-context learning。
7. **"CORE = 对所有任务 accuracy 平均。"** 不对。要**先按随机基线中心化**再等权平均，否则 4 选题的 25% 会被当成"不错"。
8. **"几条 sample 看着好就证明模型变强了。"** 不是。sample 是 smoke test，不是分数。
9. **"benchmark 高分证明模型泛化。"** 不是。还要考虑污染（训练数据里有题）、prompt 泄漏（dev set 调过参）、以及"只报最好的 seed"。

---

## 小结

- **loss 下降必要但不充分**。平均掩盖能力结构、训练目标和评估目标不一样、数据分布可能不匹配——所以要**用多个指标互相印证**。
- **BPB** 把 nat loss按字节归一化，使相同字节数据/协议下的跨-tokenizer比较更合理。关键三步：先累积 nats 和字节总和、排除特殊 token 与 ignore 位置、多 rank 下先 all-reduce 再相除。
- **CORE** 用 **continuation likelihood** 评估 base 模型：多选和 schema 比较候选的平均 loss，语言建模要求 greedy 全对。few-shot 示例只写进 prompt、不训练。
- **中心化**：先扣掉随机基线再等权平均，避免 4 选题的 25% 被误读成"**不错**"。
- **Sample** 是最便宜的 smoke test，用来发现崩溃式问题（NaN、死循环、tokenizer 不匹配），**不是分数**。
- **报告 base 评估**必须交代 **checkpoint、split、few-shot、是否截样、污染风险**——数字只有在协议明确时才有意义。

---

## 练习

1. **单位换算**。有效目标 token 预测概率是 `1/8`，写出它的 nat loss 和 bit loss。如果它对应 3 bytes，单独贡献多少 BPB？为什么不能从这单项推整个评测的 BPB？
2. **手算中心化**。某 4 选任务 accuracy = 0.40、随机基线 25。按 `evaluate_core` 的百分数写法算 `c_i`；再用 `(0.40 - 0.25) / (1 - 0.25)` 验证两种写法等价。
3. **切片精确位置**。给出候选 token 序列 `[<BOS>, q_1, q_2, a_1, a_2, a_3]`，`si = 3, ei = 6`。写出 continuation loss 的切片下标；指出各位置分别预测什么 token。
4. **Sample 协议题**。打开 `base_eval.py --eval sample` 的代码，列出条件 sample 和无条件 sample 各自的：prompt、temperature、样本数、max_tokens。**为什么它们不该被合成一个 accuracy？**
5. **污染审计**。某 CORE 分数高，但 val BPB 没变、相邻任务也没变、人工新出的类似题也没变。列出至少三种可能解释，并说明分别需要什么证据来排除。
6. **边界辨认**。给 BPB、CORE、Sample 各写一句话：
   - 它**能**支持什么结论；
   - 它**不能**支持什么结论。

---

## 前后章导航

- **上一章**：[第 07 章：基础模型训练：优化器、精度与规模法则](./07-base-train-optim-scaling.md)。讲完了"**怎样稳定更新参数**"。
- **本章**：讲清**稳定更新带来的改进要靠评估证明**——训练 loss 低不代表模型变好。
- **下一章**：[第 09 章：监督微调：教会模型对话](./09-sft.md)。把 base checkpoint 变成能聊天的模型，涉及对话渲染、监督 mask、SFT 的训练循环和 base 的差异。聊天场景下的任务评估协议留到[第 11 章](./11-chat-evaluation.md)。
