# 第 06 章：Base Train：数据到损失与训练循环

> **本章源码地图**：数据下载、Parquet 与分片见[第 02 章](./02-dataset.md)；本章从 `nanochat/nanochat/dataloader.py` 输出的 `(x, y)` 开始，沿 `nanochat/scripts/base_train.py` 的训练循环走到 `nanochat/nanochat/gpt.py::GPT.forward` 和 `nanochat/nanochat/loss_eval.py::evaluate_bpb`。传入 `targets` 时 `GPT.forward` 返回损失，未传入时返回 logits。

上一章的 Transformer 已能把一行 token 转成每个位置上的词表分数。但“它会预测下一个 token”还不是训练：我们还需要准备题目、定义错多少分，并把分数变成权重修改。本章沿着真实调用链走完这条路径：

```text
loader 的长度 T+1 token row → x/y（各 B,T） → GPT.forward
→ cross_entropy → backward → optimizer.step → zero_grad
```

这个 row 如何从 Parquet 文档、BOS 和 packing 得到，已在[第 02 章](./02-dataset.md)完整说明；这里仅使用它的训练接口。

## 学习目标

读完本章，你应该能够：

1. 说明 loader 输出为何是 GPU 上形状 `(B, T)` 的 `x`、`y`，以及二者为何只错开一格；
2. 手推带 BOS 的小 token 序列，并找出文档边界在右移目标中的位置；
3. 用直觉和公式解释交叉熵；
4. 读懂一次 `forward → backward → step → zero_grad`，以及梯度累积为何要除以累积次数；
5. 区分核心语言建模原理、nanochat 的数据布局、工程吞吐优化和后续的优化配方；
6. 说明验证、采样、checkpoint 在训练闭环中的职责。

## 前置概念

请先阅读[第 02 章：Dataset](./02-dataset.md)、[第 03 章：Tokenizer](./03-tokenizer.md)、[第 04 章：GPT attention](./04-gpt-attention.md)与[第 05 章：GPT MLP / 现代技巧](./05-gpt-mlp-tricks.md)。本章使用的记号如下。

| 符号 | 含义 | 在源码中的对应 |
|---|---|---|
| `B` | 单个设备上一份 micro-batch 的序列条数 | `args.device_batch_size` |
| `T` | 每条输入序列的 token 数 | `args.max_seq_len` |
| `V` | 真实词表大小 | `GPTConfig.vocab_size` |
| `x` | 输入 token id，`(B,T)` | loader 输出 |
| `y` | 下一个 token 的目标 id，`(B,T)` | loader 输出 |
| `logits` | 未归一化的词表分数，`(B,T,V)` | `GPT.forward` |

token id 是整数标签，不是有语义的数值向量；进入 `GPT` 的 embedding 后才变成浮点表示。

---

## 1. 预训练在做什么：把文本变成“续写题”

经典 decoder-only 语言模型的训练目标很简单。若一段文字已经编码为

\[
z_0,z_1,z_2,\ldots,z_N,
\]

就让模型从已有前缀预测紧随其后的 token：

\[
(z_0\rightarrow z_1),\quad(z_0,z_1\rightarrow z_2),\quad\ldots
\]

这叫**因果语言建模**。第 04 章介绍的因果 mask 保证位置 `t` 不能看答案 `z_(t+1)`；否则训练时泄题，生成时却没有答案。

真正高效的做法不是为每个前缀单独跑一次模型，而是取一行 `T+1` 个 token，左右各切掉一个：

```text
row: [z0, z1, z2, ..., zT]       长度 T+1
x:   [z0, z1, ..., z(T-1)]        长度 T
 y:  [z1, z2, ..., zT]            长度 T
```

这样 `x[t]` 位置的预测目标是 `y[t]`，每行一次 forward 就有 `T` 个监督信号。`dataloader.py` 的真实代码正是：

```python
# nanochat/nanochat/dataloader.py
cpu_inputs.copy_(row_buffer[:, :-1])
cpu_targets.copy_(row_buffer[:, 1:])
```

### 手推一个带文档边界的例子

假设 `BOS=0`，文档 A 编码为 `[11, 12]`，文档 B 编码为 `[21, 22, 23]`。加载器不是只在整个语料开头放一次 BOS，而是对**每一篇文档**调用：

```python
tokenizer.encode(doc_batch, prepend=bos_token, ...)
```

所以候选文档为：

```text
A: [0, 11, 12]
B: [0, 21, 22, 23]
```

令 `T=5`，一行容量是 `T+1=6`。放入 A 和 B 的前半段后：

```text
row = [0, 11, 12, 0, 21, 22]
x   = [0, 11, 12, 0, 21]
y   = [11, 12, 0, 21, 22]
```

| 位置 | 当前输入末尾 | 目标 | 含义 |
|---:|---|---:|---|
| 0 | `BOS` | 11 | 文档 A 从哪里开始？ |
| 1 | `BOS, 11` | 12 | 11 后是什么？ |
| 2 | `..., 12` | `BOS` | A 结束，新的文档开始 |
| 3 | `..., BOS` | 21 | 文档 B 从哪里开始？ |
| 4 | `..., 21` | 22 | 21 后是什么？ |

它仍是一条连续的计算序列，后面的 token 在因果 mask 下可以看到前面 token；但 BOS 给了模型清楚的文档开始标记。不要误解成“BOS 自动让模型完全看不到上一篇文档”。

---

## 2. loader 到 `x/y`：只回顾训练接口

完整的数据职责已经在[第 02 章：Dataset](./02-dataset.md)展开：下载和排序 Parquet shard、最后一个 shard 的 `val` 约定、按 row group 的 DDP 分工、每篇文档的 BOS、有限 buffer 的 best-fit、裁剪取舍，以及近似 resume state 都属于那里。本章不重复这些实现细节，只接住它的输出。

训练入口调用 `tokenizing_distributed_data_loader_with_state_bos_bestfit(...)` 后，已经得到 device 上的 `inputs, targets`。loader 先构造每行长度 `T+1` 的 token，再做：

```python
cpu_inputs.copy_(row_buffer[:, :-1])
cpu_targets.copy_(row_buffer[:, 1:])
```

因此无论上游用了何种 Parquet 读取或 packing 策略，训练接口的不变量都是：`x.shape == y.shape == (B,T)`，且 `y[b,t]` 是 `x[b,t]` 后紧邻的目标 token。以 `B=2,T=5` 为例，loader、模型和 loss 的形状是：

| 对象 | 形状 | 训练循环关心什么 |
|---|---:|---|
| 完整 row | `(2,6)` | 只为制造右移的一对序列 |
| `x`、`y` | 各 `(2,5)` | 分别是模型输入与正确类别 id |
| logits | `(2,5,V)` | 每个位置对整个词表的分数 |
| 默认 loss | `()` | 所有有效位置归约后的标量 |

CUDA 路径会使用 pinned CPU staging buffer 和 non-blocking copy；这是吞吐实现，不改变上述语义。若要修改 shard、DDP、best-fit 或裁剪行为，应回到第 02 章对照 `dataset.py`、`dataloader.py`，而不要只从这一章的训练循环推断数据行为。

---

## 3. logits 如何变成分数：交叉熵

在 `(b,t)` 位置，模型输出长度为 `V` 的 logits 向量 \(s\)。logit 是任意实数，不是概率。softmax 把它转为候选 token 的概率：

\[
p_i=\frac{e^{s_i}}{\sum_{j=1}^{V}e^{s_j}}.
\]

其中 \(i\) 是候选 token id，\(s_i\) 是它的 logit，\(p_i\) 是预测概率。若正确 token 是 \(y\)，该位置的损失为：

\[
\ell=-\log p_y.
\]

直觉很直接：给正确答案 `0.8` 概率，损失约 \(0.223\)；只给 `0.01`，损失约 \(4.605\)。它不是“猜对记 0、猜错记 1”，而是连续惩罚模型对正确答案缺乏信心。

通常 batch 平均损失为：

\[
L=\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T}-\log p_{b,t,y_{b,t}}.
\]

在真实 `nanochat/nanochat/gpt.py::GPT.forward` 中，模型先产出 `(B,T,V)` logits，再展平并调用：

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)), targets.view(-1),
    ignore_index=-1, reduction=loss_reduction,
)
```

因此不要先手写 softmax 再传给 `F.cross_entropy`：该函数期待 logits，内部会以更稳定的方式执行 log-softmax 和负对数似然。`ignore_index=-1` 表示目标为 `-1` 的位置不计入损失，后续 SFT 的回答掩码会使用它；本章的预训练 loader 没有 padding，正常输出 token id。

### 最小形状实验（教学化简代码）

下段不是仓库的 GPT 或 loader，只验证右移和逐 token 损失：

```python
import torch
import torch.nn.functional as F

row = torch.tensor([[0, 11, 12, 0, 21, 22],
                    [0, 31, 32, 33, 0, 41]])
x, y = row[:, :-1], row[:, 1:]
B, T = x.shape
V = 50
logits = torch.randn(B, T, V)  # 假装是模型输出
loss2d = F.cross_entropy(logits.view(B*T, V), y.reshape(B*T), reduction="none")

assert x.shape == y.shape == (2, 5)
assert logits.shape == (2, 5, 50)
assert loss2d.shape == (10,)
print(x[0].tolist(), "->", y[0].tolist(), loss2d.mean().item())
```

`reduction="none"` 保留每个位置的损失。`GPT.forward(..., loss_reduction='none')` 提供同样能力，验证 bpb 正是靠它取得逐位置 loss。

---

## 4. 一个参数更新：forward、backward、step、清梯度

先忽略各种现代优化器，一个训练步的经典骨架为：

```python
loss = model(x, y)       # forward：预测并得到标量损失
loss.backward()          # backward：计算并累积每个参数的梯度
optimizer.step()         # 按梯度更新参数
model.zero_grad()        # 清除旧梯度
```

梯度 \(\partial L/\partial\theta\) 描述参数 \(\theta\) 微小变化会怎样改变损失。优化器通常沿使损失下降的方向修改参数。一个很重要的 PyTorch 行为是：`.backward()` 默认向已有的 `param.grad` **累加**，不会自动覆盖。这既使梯度累积成为可能，也意味着忘记清梯度会造成 bug。

本仓库的调用链是：

```text
base_train.py: next(train_loader) 得到 x,y
  → model(x,y)
  → GPT.forward(idx=x, targets=y)
  → F.cross_entropy(...)
  → loss.backward()
```

`base_train.py` 用 `torch.compile(model, dynamic=False)` 得到训练模型；同时保留 `orig_model` 以供保存和输入形状会变化的评估/采样。这是执行效率选择，不改变反向传播的数学含义。

### 梯度累积：显存装不下大 batch 时怎么办？

`device_batch_size=B` 是单 rank 单次 forward/backward 的序列数，序列长度为 `T`。所有 rank 的单次 micro-step token 数是：

\[
\text{world\_tokens\_per\_fwdbwd}=B\times T\times\text{ddp\_world\_size}.
\]

脚本要求全局 `total_batch_size` 可以被它整除，并计算：

\[
\text{grad\_accum\_steps}=
\frac{\text{total\_batch\_size}}{B\times T\times\text{ddp\_world\_size}}.
\]

例如 `B=2,T=8,world_size=2,total_batch_size=64`，每个 micro-step 全局有 32 token，所以连续反传两次才更新参数一次。

真实训练循环的核心片段是：

```python
for micro_step in range(grad_accum_steps):
    loss = model(x, y)
    train_loss = loss.detach()
    loss = loss / grad_accum_steps
    loss.backward()
    x, y, dataloader_state_dict = next(train_loader)

optimizer.step()
model.zero_grad(set_to_none=True)
```

为什么必须除？每个 `loss` 已是 micro-batch 的平均，多个 `.backward()` 又把梯度相加。不除就把有效学习率放大了 `grad_accum_steps` 倍；先除再累加，才对应整个有效 batch 的平均梯度。

例如两个 micro-batch 对某参数给出的平均梯度分别是 `2` 和 `4`：正确的大 batch 平均梯度是 `(2+4)/2=3`；若直接连续 backward 而不除，累积成 `6`，更新量恰好放大 2 倍。这不是“多看数据所以理应更大”，而是把求平均误写成了求和。

`detach()` 取得脱离计算图的日志值。下一批数据在反传后就开始准备，是为让数据准备与 GPU 工作重叠的工程预取，不会混入当前已计算的梯度。

`optimizer.step()` 前，脚本还会设置各参数组的学习率、Muon 动量和权重衰减；这些优化器、学习率日程、FP8 与规模法则留给下一章。若 fp16 使用 `GradScaler`，代码也会先缩放/反缩放，并在 DDP 下同步异常梯度标志；这是数值安全机制，不是另一种学习目标。

最后 `model.zero_grad(set_to_none=True)` 在一次更新后清掉梯度，避免错误跨 step 累加。设为 `None` 而非填零也能少做一部分内存写入。

---

## 5. 不更新参数时，训练循环还在做什么？

### 验证：bpb 不是普通平均 loss

`base_train.py` 每隔 `args.eval_every` 步进入 `model.eval()`，新建验证 loader，并调用：

```python
val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
```

随后恢复 `model.train()`。当前 GPT 没有 dropout，模式切换未必明显改变输出，但这是正确习惯，也使未来加入不同训练/推理行为的层时仍正确。

`loss_eval.py::evaluate_bpb` 使用 **bits per byte**：

\[
\operatorname{bpb}=\frac{\text{total nats}}{\ln 2\times\text{total bytes}}.
\]

这里 nat 是自然对数下的交叉熵单位；除以 \(\ln2\) 转成 bit，再按目标 token 所代表的原始字节数归一化。若换一个 tokenizer，同一文本的 token 数会改变，普通“每 token loss”不再完全可比；bpb 较少受这个切分差异影响。

函数调用 `model(x,y,loss_reduction='none')` 取得 `(B,T)` loss，按 `token_bytes[y]` 得到目标 token 的字节数。字节数为 0 的特殊 token（如 BOS）不计入分子和分母；`y<0` 的 ignore 位置也被排除。DDP 时先用 `dist.all_reduce` 汇总所有 rank 的总 nats 与总 bytes，最后再相除。不要把“每 batch bpb 再平均”当成等价做法：各 batch 的有效字节数可能不同。

### 采样：快速但非严格的健康检查

每隔 `args.sample_every`，且仅在 `master_process`，训练脚本通过 `Engine(orig_model, tokenizer)` 对固定 prompt 调用 `generate_batch(..., max_tokens=16, temperature=0)`。`temperature=0` 是贪心选择最大 logit，方便稳定比较 checkpoint。采样不能证明泛化能力，却能很快看出乱码、重复、格式错误或训练崩坏。

脚本还会按 `args.core_metric_every` 调用 `scripts/base_eval.py::evaluate_core`。它是另一类任务评估；本章只需知道它与 val bpb、采样共同提供不同证据，而不是梯度更新的一部分。

### Checkpoint：保存权重之外的状态

在训练结束或满足 `args.save_every` 时，`save_checkpoint(...)` 写入 checkpoint 目录：

- `model_{step:06d}.pt`：仅 rank 0 保存的模型参数；
- `optim_{step:06d}_rank{rank}.pt`：每个 rank 保存自己的优化器状态；
- `meta_{step:06d}.json`：模型/用户配置、step、最近验证 bpb、batch 信息、loader 状态和循环状态。

优化器状态包含动量等历史，丢掉它后“继续训练”并不等于原轨迹的续接。meta 中的 `dataloader_state_dict` 含 `pq_idx`、`rg_idx`、`epoch`，让 `_document_batches` 近似跳过已读数据。checkpoint 是**恢复性工程**，不会在保存时额外让模型学会内容。

---

## 6. 分层理解：不要把所有代码都叫作“训练技巧”

| 层次 | 本章内容 | 含义 |
|---|---|---|
| 核心原理 | 右移目标、因果 mask、交叉熵、反传、沿梯度更新 | 大多数 decoder-only LM 的共同骨架 |
| nanochat 数据布局 | 每文档 BOS 与固定长度 row packing | 当前仓库的质量/利用率权衡；完整算法与裁剪代价见第 02 章 |
| 工程优化 | Parquet row group、DDP 分片、预分配、pinned memory、预取、compile、checkpoint | 尽量不改训练语义而改善吞吐、恢复性 |
| 实验性/优化配方 | 全局 batch、学习率调度、Muon/AdamW、FP8、规模法则 | 改变优化轨迹，下一章讨论 |
| 诊断 | val bpb、CORE、固定 prompt 采样 | 观察模型，不直接更新参数 |

## 常见误区

1. **“`x`、`y` 是两份无关文本。”** 它们来自同一行 `T+1` token，只差一个位置。
2. **“BOS 只加一次。”** 此 loader 对每篇文档 prepend BOS，文档交界处也会成为预测目标。
3. **“本章的 `x/y` 接口足以说明上游数据策略。”** 不对；BOS、row group 分片、best-fit 与裁剪的完整取舍在第 02 章。
4. **“低 loss 表示每个 token 都猜对。”** 交叉熵是概率质量的平均；简单高频 token 可能掩盖困难样本。
5. **“交叉熵的输入该是 softmax 概率。”** PyTorch 的 `F.cross_entropy` 要 logits。
6. **“梯度累积就是连续 backward。”** 还要除以 `grad_accum_steps`，并只在真正更新后清梯度。
7. **“应在每个 micro-step 后 zero_grad。”** 这样会抹掉已经累计的梯度。
8. **“验证 bpb 包含 BOS。”** `token_bytes` 为特殊 token 给出 0 字节，函数会排除它们。
9. **“样本文本通顺就等于模型好。”** 它只是 smoke test；仍须看验证与任务评估。

## 小结

预训练进入训练循环时，loader 已交出长度 `T+1` 的 token 行；右移后得到 `(B,T)` 的输入 `x` 和目标 `y`。上游的 Parquet、每文档 BOS、best-fit 与裁剪取舍属于[第 02 章](./02-dataset.md)；本章的关键不变量是每个 `y[b,t]` 都是 `x[b,t]` 右边紧邻的目标。

模型在每个位置输出 `(B,T,V)` logits，交叉熵将正确下一个 token 的低概率转成高损失。训练用若干已除以累积次数的 micro-batch 反传，累计出全局 batch 梯度，再 `optimizer.step()` 更新、`zero_grad()` 清场。验证 bpb、采样、CORE 和 checkpoint 分别回答“未见文本怎样”“输出看起来怎样”“任务表现怎样”“中断后如何继续”。

## 练习

1. 令 `BOS=0`，文档 A 为 `[4,5,6]`、B 为 `[8,9]`、`T=5`。写出 row、x、y，并标出预测 BOS 的位置。
2. 若 `B=4,T=128,ddp_world_size=8,total_batch_size=16384`，分别计算单 rank micro-batch token 数、全局 micro-step token 数和 `grad_accum_steps`。不能整除时，`base_train.py` 的哪条断言会阻止训练？
3. 运行本章最小实验。将某一位置正确类别的 logit 增加 10，再将错误类别增加 10，比较该位置 loss 的变化并解释。
4. 阅读 `loss_eval.py::evaluate_bpb`，解释为何它累计总 nats 和总 bytes 后才相除。
5. 画出 `shard_*.parquet → optimizer.step()` 数据流；在每条箭头上标注它是字符串、`list[int]`、`torch.long`、浮点 logits 还是标量 loss。
6. 回到[第 02 章](./02-dataset.md)，若比较连续 token 流切块和 BOS packing，哪些模型、tokenizer、有效 token budget、优化配方必须固定？除 loss 外还应报告哪些利用率、吞吐与验证数据？

## 前后章导航

- ← 上一章：[第 05 章：GPT MLP / 现代技巧：MLP、残差流与现代改进](./05-gpt-mlp-tricks.md)。那里说明 `GPT.forward` 如何由 `x` 产生 logits。
- → 下一章：[第 07 章：Base Train：优化器与规模法则](./07-base-train-optim-scaling.md)。我们将拆解 `model.setup_optimizer`、学习率/权重衰减日程、混合精度与分布式训练，回答“有了梯度后，怎样稳定高效地更新”。
