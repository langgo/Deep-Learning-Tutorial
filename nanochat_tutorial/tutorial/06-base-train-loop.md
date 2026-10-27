# 第 06 章　基础模型训练：从数据到损失与梯度

> **本章定位**　第二部分·基础模型。把前四章拼好的数据和模型连成一次完整的训练更新。
>
> **中心问题**　当 loader 交出 `(B, T)` 的 `x` 和 `y` 之后，到底是哪几行代码把它们变成交叉熵、梯度，然后变成一次 `optimizer.step()` 后更新完的参数？
>
> **范围**　**训练循环的主干**：数据接口、模型前向、交叉熵、反向传播、梯度累积、验证评估、checkpoint 保存。**优化器内部**（AdamW / Muon / FP8 / 学习率曲线）和多卡通信的细节留到[第 07 章](./07-base-train-optim-scaling.md)。
>
> **先决条件**　读过[第 02 章](./02-dataset.md)（数据）、[第 03 章](./03-tokenizer.md)（tokenizer / mask）、[第 04](./04-gpt-attention.md)–[05 章](./05-gpt-mlp-tricks.md)（模型）。
>
> **本章产物**　对"**一次训练步到底在做什么**"的完整认识——每一行代码都能对应到核心原理 / 数据布局 / 工程优化 / 实验性配方中的哪一层。
>
> **源码入口**　`scripts/base_train.py`、`nanochat/gpt.py::GPT.forward`、`nanochat/loss_eval.py::evaluate_bpb`。数据部分复用[第 02 章](./02-dataset.md)。

---

## 本章讲什么

读完前五章你已经有：

- 一条数据流水线：Parquet shard → `(B, T)` 整数张量 `x, y`；
- 一个能用的模型：`GPT.forward(x) → logits (B, T, V)`。

**但这还不是训练**。你有了"**题目**"（输入），也有了"**标准答案**"（右移目标），但还缺三件事：

1. **怎么算错多少分**（loss）；
2. **怎么把分数变成"该怎么改参数"**（反向传播）；
3. **怎么在有限显存下让"等效 batch 够大"**（梯度累积）。

本章把这三件事串成一条**完整的训练步**。核心调用链短到可以背下来：

```text
next(train_loader) → model(x, y) → loss.backward() → optimizer.step() → zero_grad()
```

但真实 `scripts/base_train.py` 在这几行上下还做了：梯度累积、验证评估、采样 smoke test、checkpoint 保存——它们都不是"**另一种训练**"，是**同一个训练循环里不同职责的岔路**。

**读完本章你应该能做到**：

1. 讲清"**loader 为什么交出 `(B, T)` 的 x 和 y，而不是 `(B, T+1)` 一整条**"；
2. 用一个小例子手推"**带 BOS 的 token 序列**" 里，文档边界落在 `y` 的哪些位置；
3. 用一句话（和一个公式）解释交叉熵；
4. 讲清一次 `forward → backward → step → zero_grad` 发生了什么，以及梯度累积**为什么必须除以累积次数**；
5. 把"**验证 bpb**"、"**采样 smoke test**"、"**保存 checkpoint**" 定位到训练循环的哪个岔路。

---

## 预备词汇

| 符号 | 含义 | nanochat 命名 |
|---|---|---|
| `B` | 单卡单次 forward 的 micro-batch 序列数 | `args.device_batch_size` |
| `T` | 每条序列的 token 数 | `args.max_seq_len` |
| `V` | 词表大小 | `GPTConfig.vocab_size` |
| `x` | 输入 token id，`(B, T)` | loader 输出 |
| `y` | 目标 token id，`(B, T)` | loader 输出 |
| `logits` | 未归一化词表分数，`(B, T, V)` | `GPT.forward` 输出 |
| `grad_accum_steps` | 梯度累积次数 | 由 `total_batch_size / (B×T×world_size)` 推导 |

**和前面章节的小衔接**：第 02 章讲过 loader **先构造 `(B, T+1)` 的大 row、再切出 `(B, T)` 的 x 和 y**；第 03 章讲过 SFT 的 mask=0 位置会被改成 `-1`，`cross_entropy(..., ignore_index=-1)` 自动跳过。这一章会把这两件事直接当作已知。

---

## 1. 预训练在学什么：把文本变成"续写题"

一句话：**根据前缀预测紧随其后的 token**。

形式化地：一段文本编码成 token 序列 `z_0, z_1, ..., z_N`，训练目标是学会：

```
给 z_0, z_1, ..., z_t  →  预测 z_{t+1}
```

这叫**因果语言建模**。因果 mask（[第 04 章](./04-gpt-attention.md)）保证训练时位置 `t` 看不到 `t+1`，否则等于偷看答案。

### 高效的做法：一次 forward 并行完成 `T` 道续写题

一行 `T+1` 个 token：

```text
row = [z_0, z_1, z_2, ..., z_T]
```

可以组成 `T` 道“根据前缀预测下一个 token”的题：

```text
题 0：看到 [z_0]                    → 预测 z_1
题 1：看到 [z_0, z_1]               → 预测 z_2
题 2：看到 [z_0, z_1, z_2]          → 预测 z_3
...
题 T-1：看到 [z_0, ..., z_{T-1}]    → 预测 z_T
```

#### 朴素做法：把 `T` 道题分开算

最直接的实现会对每个前缀分别调用模型：

```python
model([z_0])                         # 用最后一个位置预测 z_1
model([z_0, z_1])                    # 用最后一个位置预测 z_2
model([z_0, z_1, z_2])               # 用最后一个位置预测 z_3
...
model([z_0, ..., z_{T-1}])           # 用最后一个位置预测 z_T
```

这样需要 `T` 次 forward，而且前缀被反复计算：预测 `z_3` 时又算了一遍 `z_0,z_1`，预测 `z_4` 时还会再算一遍。训练数据很长时，这种重复代价无法接受。

#### 真实训练：把输入和目标错开一位

loader 把 `row` 切成两条长度为 `T` 的序列：

```text
row: [z_0, z_1, z_2, ..., z_{T-1}, z_T]    长度 T+1

x:   [z_0, z_1, z_2, ..., z_{T-1}]         长度 T，送进模型
y:   [z_1, z_2, z_3, ..., z_T    ]         长度 T，作为答案
       └──────── 每一项都比 x 左移一位 ────────┘
```

代码就是：

```python
# nanochat/dataloader.py
cpu_inputs.copy_(row_buffer[:, :-1])        # 去掉最后一个 token，得到 x
cpu_targets.copy_(row_buffer[:, 1:])        # 去掉第一个 token，得到 y
```

模型只执行一次：

```python
logits = model(x)                            # (B, T, V)
```

但这一次会同时产生 `T` 个位置的预测：

```text
logits[:, 0, :]：看到 z_0                  → 和目标 y[0] = z_1 比
logits[:, 1, :]：看到 z_0,z_1              → 和目标 y[1] = z_2 比
logits[:, 2, :]：看到 z_0,z_1,z_2          → 和目标 y[2] = z_3 比
...
logits[:, T-1, :]：看到 z_0,...,z_{T-1}    → 和目标 y[T-1] = z_T 比
```

然后对这 `T` 个位置一起计算交叉熵。因此：

```text
T+1 个原始 token
→ T 组“输入前缀 / 下一个 token”监督关系
→ 1 次 forward
→ T 个位置的 loss 并行计算
```

#### 答案都放进 `x` 了，模型不会偷看吗？

不会。虽然 `x` 一次包含 `z_0...z_{T-1}`，但因果 mask 规定位置 `t` 只能看 `0...t`：

```text
位置 0 只能看 z_0
位置 1 只能看 z_0,z_1
位置 2 只能看 z_0,z_1,z_2
...
```

所以 `logits[:, t, :]` 在预测 `z_{t+1}` 时，看不到 `z_{t+1}` 以及更后面的 token。**训练利用已知整段文本并行出题，因果 mask 负责把每道题的答案遮住。**

### 手推一个带文档边界的例子

[第 02 章](./02-dataset.md)讲过 loader **给每一篇文档 prepend 一个 BOS**。来看文档边界在 y 里的具体位置。

**设置**：

- `BOS = 0`
- 文档 A 编码为 `[11, 12]`，加 BOS 后 `[0, 11, 12]`，长度 3；
- 文档 B 编码为 `[21, 22, 23]`，加 BOS 后 `[0, 21, 22, 23]`，长度 4；
- `T = 5`，行容量 `T + 1 = 6`。

Best-fit packing 把 A 完整放入、再放 B 的前 3 个 token：

```text
row = [0, 11, 12, 0, 21, 22]
x   = [0, 11, 12, 0, 21]
y   = [11, 12,  0, 21, 22]
```

**逐位置看模型要学什么**：

| 位置 `t` | 位置 `t` 能看到的输入前缀 | 该位置的目标 `y[t]` | 要模型学的意思 |
|---:|---|---:|---|
| 0 | `[0]` | 11 | "**文档 A 开始了，接下来是 11**" |
| 1 | `[0, 11]` | 12 | "**11 之后是 12**" |
| 2 | `[0, 11, 12]` | **0** | "**A 这篇结束了，下一个该出现 BOS**" |
| 3 | `[0, 11, 12, 0]` | 21 | "**看到 BOS，下一篇开始，第一个 token 是 21**" |
| 4 | `[0, 11, 12, 0, 21]` | 22 | "**21 之后是 22**" |

**两件事值得注意**：

- **BOS 本身也成为预测目标**（位置 2）。模型会在训练流中的文档拼接边界学习预测 BOS。因为数据没有显式 EOS，这不等于模型获得了可靠的“语义完结判断”；它学到的是当前 packing 分布下，某些文档末尾之后会出现 BOS。
- **BOS 并不自动屏蔽上一篇**。位置 3 的 attention 在因果 mask 下仍能看到前面 `[0, 11, 12, 0]`——它从前文看到的内容包括前一篇文档。BOS 只是**标记一个"**自己的新开始**"**的 token。

---

## 2. 从 loader 到 `x, y`：只用它的训练接口

[第 02 章](./02-dataset.md)已经把数据链路完整讲完了：Parquet shard 的下载和排序、哪个 shard 当 val、按 row group 的 DDP 分工、每篇文档 prepend BOS、best-fit packing、裁剪、resume state 的近似恢复——这些都属于那一章。

**本章只用它的训练接口**。真实训练入口：

```python
# scripts/base_train.py
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len,
    split="train", device=device,
    resume_state_dict=dataloader_resume_state_dict,
)
x, y, dataloader_state_dict = next(train_loader)
```

调一次 `next(...)` 就得到：

- `x`：形状 `(B, T)`，已在 device 上，整数 token id；
- `y`：形状 `(B, T)`，已在 device 上，`y[b, t]` 是 `x[b, t]` 紧邻右边的目标 token；
- `dataloader_state_dict`：用于 checkpoint 时保存 loader 位置。

### 一个全章用的小例子

全章记号用这组数字（和[第 02 章](./02-dataset.md)例子对齐）：

```text
B = 2, T = 5, V = 50（假设词表大小）
```

**那么训练循环里出现的形状都是**：

| 对象 | 形状 | 这一步循环关心什么 |
|---|---|---|
| 原始 row（loader 内部） | `(2, 6)` | 只是中间产物，训练循环不直接看 |
| `x` | `(2, 5)` | 模型输入 |
| `y` | `(2, 5)` | 每个位置的正确答案 token id |
| `logits = model(x)` | `(2, 5, 50)` | 每个位置对词表的分数 |
| `loss = model(x, y)` | 标量 `()` | 所有有效位置归约后的损失 |

**CUDA 下**，loader 用 pinned CPU staging buffer + non-blocking copy 把 x、y 送到 device。这是**吞吐优化**——数据在 GPU 算模型时后台异步 copy，不改变训练接口。想改数据策略要回[第 02 章](./02-dataset.md)动 `dataset.py` / `dataloader.py`，别只从训练循环的两行推测。

---

## 3. 交叉熵：把 logits 变成损失

### 3.1 从 logits 到概率

模型在 `(b, t)` 位置输出一个长度 `V` 的 **logits** 向量 `s = (s_1, ..., s_V)`。logit 是任意实数，可以是 `-7.3`、`0.5`、`12.1`——**不是概率**。

**softmax** 把它变成概率分布：

```
          exp(s_i)
p_i = ─────────────────      p_i >= 0,  Σ p_i = 1
      Σⱼ exp(s_j)
```

**直觉**：`exp` 把 logits 变成非负数，分母再把它们归一化成概率。关键是：两个 token 的**概率比**只由它们的 logit 差决定：

```text
p_i / p_j = exp(s_i) / exp(s_j) = exp(s_i - s_j)
```

所以 logit 不是“多 2 就多两倍”：

| logit 差 `s_i - s_j` | 概率比 `p_i / p_j` |
|---:|---:|
| 1 | `e¹ ≈ 2.72` 倍 |
| 2 | `e² ≈ 7.39` 倍 |
| `ln(10) ≈ 2.30` | 正好 10 倍 |
| 5 | `e⁵ ≈ 148` 倍 |

例如只有两个候选 token，logits 分别是 `2` 和 `0`：

```text
softmax([2, 0]) ≈ [0.881, 0.119]
0.881 / 0.119 ≈ 7.39
```

所以更准确地说：**logit 相差约 2.3，两个 token 的概率就相差一个数量级**。如果词表里还有其他 token，它们会改变这两个概率的绝对值，但不会改变二者的概率比 `exp(s_i-s_j)`。

### 3.2 交叉熵 = 对正确答案的"负对数概率"

如果位置 `(b, t)` 的正确 token id 是 `y`，那这个位置的损失是：

```
ℓ = -log p_y
```

**直觉**：

| 模型给正确答案的概率 | loss |
|---:|---:|
| 1.0（数学极限） | 0.0 |
| 0.8 | 0.22 |
| 0.5 | 0.69 |
| 0.1 | 2.30 |
| 0.01 | 4.61 |
| 0.0（数学极限） | ∞ |

有限 logits 的 softmax 通常不会精确得到 0 或 1，表中两端是数学极限。这不是“猜对记 0 分、猜错记 1 分”，而是连续惩罚：正确 token 即使排名第一，只要概率仍低，loss 依然可观。这里“更自信”只是指提高目标 token 的 softmax 概率、拉开相对 logits，不表示模型具有主观信心。

### 3.3 batch 平均

通常 batch 内所有有效位置取平均：

```
         1    B   T
L  =  ────── ΣΣ  -log p_{b,t,y_{b,t}}
       B·T  b=1 t=1
```

### 3.4 PyTorch 的 `F.cross_entropy` 直接吃 logits

**不要先 softmax 再传给它**。`F.cross_entropy` **期望 logits**，内部用数值更稳定的 **log-softmax + 负对数似然**算损失。

真实调用（在 `GPT.forward` 里）：

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),      # (B*T, V)
    targets.view(-1),                       # (B*T,)
    ignore_index=-1,                        # SFT mask=0 的位置
    reduction=loss_reduction,               # 'mean' 或 'none'
)
```

**三个参数的作用**：

- 把 `(B, T, V)` 展平成 `(B*T, V)`，`(B, T)` 展平成 `(B*T,)`，交叉熵逐位置计算后归约；
- `ignore_index=-1` 让 target 为 `-1` 的位置**完全不计损失、也不进分母**（SFT 阶段用，预训练用不着——loader 不会产出 `-1`）；
- `reduction='mean'` 在训练循环里用（返回标量）、`'none'` 在[第 03 章](./03-tokenizer.md)讲 bpb 时用（返回 `(B, T)` 的逐位置 loss）。

### 3.5 最小形状实验

```python
import torch
import torch.nn.functional as F

row = torch.tensor([[0, 11, 12,  0, 21, 22],
                    [0, 31, 32, 33,  0, 41]])
x, y = row[:, :-1], row[:, 1:]
B, T = x.shape                    # (2, 5)
V = 50
logits = torch.randn(B, T, V)     # 假装是模型输出

# 用 'none' 看每个位置的 loss
loss2d = F.cross_entropy(
    logits.view(B * T, V), y.reshape(B * T), reduction="none"
).view(B, T)

print("x[0] =", x[0].tolist())
print("y[0] =", y[0].tolist())
print("loss2d.shape =", loss2d.shape)        # torch.Size([2, 5])
print("mean loss =", loss2d.mean().item())
```

关键：**每个 `(b, t)` 位置**都算了一个 loss；默认 `reduction='mean'` 就是对这些 loss 取平均。SFT 的 bpb 评估走的是 `'none'`，拿到 `(B, T)` 的逐位置 loss 后，用 `token_bytes` 做字节加权——那是[第 03 章](./03-tokenizer.md)讲过的事。

---

## 4. 一次参数更新：forward / backward / step / zero_grad

### 4.1 经典四行

```python
loss = model(x, y)       # ① forward: 预测 + 算 loss (标量)
loss.backward()          # ② backward: 把每个参数的 ∂L/∂θ 填进 .grad
optimizer.step()         # ③ step: 按 .grad 更新参数
model.zero_grad()        # ④ zero_grad: 把 .grad 清零
```

**每一行干什么**：

**① forward**：从 `x` 一路走到 logits，再算 cross_entropy。loss 是一个**标量**，但 PyTorch 内部记录了"**这个 loss 是通过哪些操作、从哪些参数算出来的**"——这就是**计算图**。

**② backward**：从 loss 出发，**沿计算图反向传播**，对每个参数 `θ` 算出 `∂L/∂θ`，填进 `θ.grad`。

**③ step**：优化器**读 `.grad`、算出该怎么改参数**，然后真的改参数：
```python
# 最简形式的 SGD
θ ← θ - learning_rate * θ.grad
```
真实的 Muon / AdamW 复杂得多（动量、二阶矩、权重衰减...）——留给[第 07 章](./07-base-train-optim-scaling.md)。

**④ zero_grad**：把所有参数的 `.grad` 清零。**不清会怎样？下面讲**。

### 4.2 关键细节：`.backward()` 是**累加**，不是覆盖

PyTorch 的 `.backward()` 默认把梯度**累加到 `.grad` 上**，而不是覆盖。这是**故意设计**，为了让梯度累积这件事能成立：

```python
loss1 = model(x1, y1); loss1.backward()   # θ.grad += ∂L₁/∂θ
loss2 = model(x2, y2); loss2.backward()   # θ.grad += ∂L₂/∂θ
optimizer.step()                           # 用累加后的梯度一次性更新
model.zero_grad()
```

**好处**：显存只能装一个小 batch 的 forward，但你可以通过多次 forward+backward 累积出一个大 batch 的等效梯度，再更新一次。

**代价**：忘了 `zero_grad` → 下一轮的梯度会和上一轮的加在一起，等效学习率翻倍，训练很快发散。所以 `zero_grad` 不是可有可无的清理，而是**正确性要求**。

### 4.3 梯度累积的必要除法

显存上一次只能装 `B` 条序列，但训练算法希望每次更新对应 `total_batch_size` 个 token——怎么办？做 `grad_accum_steps` 次 micro-batch forward+backward，再做一次 step。

```python
# scripts/base_train.py 的训练循环骨架
for micro_step in range(grad_accum_steps):
    loss = model(x, y)              # loss 已经是这个 micro-batch 的平均
    train_loss = loss.detach()      # 日志用，和梯度无关
    loss = loss / grad_accum_steps  # ← 关键：除以累积次数
    loss.backward()                  # 累加到 .grad
    x, y, dataloader_state_dict = next(train_loader)  # 预取下一批

optimizer.step()
model.zero_grad(set_to_none=True)
```

**为什么必须除以 `grad_accum_steps`**？因为：

- 每个 micro-batch 的 `loss` **已经是这个 micro-batch 的平均**；
- 多次 backward 把这些 micro-batch loss 的梯度**相加**；
- 想要的效果是**整个大 batch 的平均梯度**，所以要除。

**不除会怎样**？用数字验证：假设 `grad_accum_steps = 2`，两个 micro-batch 对某参数的梯度各是 `2` 和 `4`。

- **正确的大 batch 平均梯度**：`(2 + 4) / 2 = 3`；
- **不除直接累加**：`2 + 4 = 6`，正好**错了 2 倍**；
- **先除再累加**：`2/2 + 4/2 = 1 + 2 = 3`。✓

**这不是"**多看了数据所以梯度本来就应该更大**"**——那是在把"求平均"误写成了"求和"。除掉是把 micro-batch 的平均还原成"对整个大 batch 求总 loss"后再取平均。

### 4.4 几个和训练语义无关的工程细节

**`loss.detach()`**：得到一个不参与反传的张量供日志使用，避免把计算图继续挂在日志变量上。当前循环每个 micro-step 都覆盖 `train_loss`，所以最终记录的是**最后一个 micro-batch 的未缩放 loss**，不是所有累积 micro-batch loss 的平均；它适合做平滑趋势日志，不应误读成这次 optimizer step 的精确全局 batch loss。

**`next(train_loader)` 放在 `.backward()` 之后**：PyTorch 的 `.backward()` 会把 GPU 占满。数据预取放在它之后，意思是"**让下一批数据的准备和这一批的 backward 并行**"——数据准备在 CPU + 异步 copy，GPU 继续反传。CUDA 下的 pinned memory + non-blocking copy 把两件事真正并行。这是吞吐优化，不混入已算完的梯度。

**`torch.compile(model, dynamic=False)`**：训练用一个 compile 过的模型版本（更快）。但保存 checkpoint、评估、采样等**可能遇到不同输入形状**的地方用原始 `orig_model`。两者共享参数，compile 只是给 forward 加速，训练数学不受影响。

**fp16 的 `GradScaler`**：如果用 float16（而不是 bfloat16），梯度数值可能太小被下溢。`GradScaler` 先把 loss 放大、backward 完再缩回。这是数值安全机制，不改变训练目标。nanochat 默认用 bf16，所以这里一般不触发。

**多卡梯度同步**：nanochat 虽由 `torchrun` 启动多个 rank，但模型没有包进 PyTorch DDP。因此 `.backward()` 后每个 rank 仍持有自己的本地梯度；真正的同步在 `MuonAdamW.step()` 里完成：大参数用 `reduce_scatter(AVG)`，小参数用 `all_reduce(AVG)`，更新后再 `all_gather` 参数。详见[第 07 章](./07-base-train-optim-scaling.md)。

---

## 5. 训练循环的三条岔路

训练循环不是只做"forward + backward"。每隔一段步数，它会跳到三条**不更新参数**的岔路：

### 5.1 验证评估：用 bpb，不是平均 loss

**什么时候触发**：每 `args.eval_every` 步，`master_process` 调一次：

```python
model.eval()
val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
model.train()
```

`model.eval()` / `model.train()` 切换的是 **dropout / BatchNorm** 之类的训练/推理行为。当前 nanochat 没有 dropout，切不切看起来一样，但这仍是正确习惯——未来加了行为不同的层也不会出错。

**为什么是 bpb 不是普通 loss**？[第 03 章](./03-tokenizer.md)讲过——换 tokenizer 后"每 token loss"不可比，**bits per byte** 把 loss 归一化到字节：

```
                total nats
bpb  =  ────────────────────────────
          ln(2) × total bytes
```

实现上 `evaluate_bpb` 用 `loss_reduction='none'` 拿到逐位置 loss，再按 `token_bytes[y]` 汇总字节数，并排除 `y < 0` 的位置。多 rank 评估时**先 all-reduce 总 nats 和总字节数、最后再除**——如果每个 batch 或 rank 自己算 bpb 再无权平均就不对，因为字节数可能不同。

### 5.2 采样 smoke test：固定 prompt、贪心生成几个 token

**什么时候触发**：每 `args.sample_every` 步，master process 调：

```python
engine = Engine(orig_model, tokenizer)
engine.generate_batch(prompt, max_tokens=16, temperature=0)
```

`temperature=0` 是**贪心解码**（每次取 logit 最大的 token）——这样同一个 checkpoint 多次采样结果一致，方便对比。

**它测的不是"**模型聪不聪明**"**，是"**模型是不是在胡言乱语、死循环、格式崩了**"。扫一眼采样文本就能发现 base_train 出了 NaN、输出全是同一个 token、输出全是空格之类的问题——比 loss 曲线更直观。

### 5.3 Checkpoint：保存**权重 + 元数据 + loader 位置**

**什么时候触发**：训练结束，或每 `args.save_every` 步。写入三类文件：

| 文件 | 由谁保存 | 内容 |
|---|---|---|
| `model_000123.pt` | 只有 rank 0 | 模型参数（所有 rank 的权重都一样） |
| `optim_000123_rank0.pt`、`..._rank1.pt`... | 每个 rank 自己 | 优化器状态（动量、二阶矩等；按 rank 分片） |
| `meta_000123.json` | 只有 rank 0 | step、模型配置、最近 val bpb、batch 配置、**loader 位置**、循环状态 |

**三件事值得注意**：

- `meta_*.json` 里的 `dataloader_state_dict` 包含 `pq_idx, rg_idx, epoch`。恢复训练时 loader 会**近似**跳过已读数据（[第 02 章](./02-dataset.md) §6 讲过"approximate resume"）。
- 丢掉 optimizer state 后"**继续训练**"**不是**原轨迹的严格续接——动量没了。所以 checkpoint 一旦被不完整复制（例如忘了带 `optim_*`），你以为在续训，实际是从该 step 冷启。
- **保存只是存快照，不会让模型"顺带学到更多"**。

---

## 6. 分层：别把所有代码都叫作"训练技巧"

读到这里训练循环里又出现一堆名词（grad_accum、detach、pinned memory、compile、all-reduce...），用一张表归位——下章开始要大量讨论优化器时，这个表就是导航：

| 层次 | 本章出现的东西 | 性质 |
|---|---|---|
| **核心原理** | 右移目标、因果 mask、交叉熵、反传、沿梯度更新参数、梯度累积 | 几乎所有 decoder-only LM 都这么做 |
| **nanochat 数据布局** | 每文档 BOS、best-fit packing、固定 `(B, T+1)` row | 当前仓库的质量 / 利用率取舍（详 [02](./02-dataset.md)） |
| **工程优化** | Parquet 的按-rank数据分片、优化器内梯度通信、预分配 buffer、pinned memory、non-blocking copy、预取、`torch.compile`、`optim_*_rank*.pt` 分片 | 不改变目标函数，但影响吞吐、数值顺序与恢复性 |
| **实验性 / 优化配方** | 全局 `total_batch_size`、学习率调度、Muon / AdamW、FP8、规模法则 | 改变优化轨迹，详 [07](./07-base-train-optim-scaling.md) |
| **诊断** | val bpb、CORE、固定 prompt 采样 | 观察模型健康，不更新参数 |

读源码时每出现一行新东西，先问**它属于哪一层**。属于"工程优化"的东西关掉只会变慢，不会变错——所以第一遍可以忽略；属于"核心原理"的东西关掉就不是训练了。

---

## 常见误区

1. **"`x`、`y` 是两份独立文本。"** 不是。它们来自**同一行** `T+1` 个 token，只错开一格。
2. **"BOS 只在整个语料开头出现一次。"** 不是。loader 对**每一篇文档** prepend BOS，一行里可以有多个 BOS，而且 BOS 本身也会成为预测目标。
3. **"低 loss 等于每个 token 都猜对。"** 不等。交叉熵是**概率质量的平均**——高频 token 猜对可以把平均 loss 拉低，困难样本的错预测可能被淹没。所以要搭配 CORE / bpb / 采样一起看。
4. **"交叉熵的输入应该是 softmax 后的概率。"** PyTorch 的 `F.cross_entropy` 要 **logits**。先手写 softmax 再传进去数值不稳。
5. **"梯度累积就是连续 `.backward()`。"** 还要**除以 `grad_accum_steps`**，并且**只在真正 `step()` 之后 `zero_grad`**。
6. **"每个 micro-step 后应该 zero_grad。"** 会抹掉已经累积的梯度——等于不做累积。
7. **"val bpb 把 BOS 也算进去。"** 不会。`token_bytes` 给特殊 token 的字节数是 0，分子分母都排除。SFT mask=0 的位置（`y < 0`）也排除。
8. **"采样文本通顺 = 模型好。"** 它只是 smoke test。真正评估要看 val bpb、CORE、chat_eval。

---

## 小结

- **一个训练步的核心**是四件事：`forward → backward → step → zero_grad`。`.backward()` 是**累加**到 `.grad`，所以能支持梯度累积、也必须记得清零。
- **loader 的输出接口是不变量**：`x, y` 都是 `(B, T)`、`y` 是 `x` 右移一格。数据层面的所有事（Parquet、best-fit、BOS、resume）都在[第 02 章](./02-dataset.md)讲完，训练循环只用这个接口。
- **BOS 既是文档标记、也是边界处的预测目标**。模型学习训练流里的文档分隔模式，但 BOS 不能替代对语义完结能力的独立评估。
- **交叉熵是对正确答案的负对数概率**，是连续惩罚，输入是 **logits** 不是概率。
- **梯度累积必须除以 `grad_accum_steps`**，否则等效学习率被放大。用数字验证一遍最清楚：`(2 + 4) / 2 = 3`，不除变成 `6`。
- **训练循环还做三件不更新参数的事**：val bpb 评估（看泛化）、固定 prompt 采样（smoke test）、checkpoint 保存（支持恢复）。
- **分层**：核心原理 / 数据布局 / 工程优化 / 实验性配方 / 诊断——下章讲优化器时会继续用这个分类。

---

## 练习

1. **BOS 边界手推**。`BOS = 0`，文档 A = `[4, 5, 6]`、B = `[8, 9]`，`T = 5`。写出 row、x、y，并标出**哪个位置的目标是 BOS**。
2. **梯度累积数字题**。`B = 4, T = 128, world_size = 8, total_batch_size = 16384`。算：
   - 单 rank 单次 forward 的 token 数；
   - 一轮全局 micro-step 的 token 数；
   - `grad_accum_steps`。
   如果不能整除会怎样？`base_train.py` 里的哪条断言会阻止训练？
3. **交叉熵感觉题**。跑第 3.5 节的最小实验，然后：
   - 把 `logits[0, 2, y[0, 2]]` 加 10（提高正确类的 logit），对比 `loss2d[0, 2]` 的变化；
   - 换成把**错误类**的某个 logit 加 10，再看 `loss2d[0, 2]`。解释差异。
4. **bpb 的 all-reduce 顺序**。阅读 `loss_eval.py::evaluate_bpb`。为什么它要**先把所有 rank 的总 nats 和总字节数 all-reduce 完、最后再除**？如果换成"每个 batch 自己算 bpb、最后取平均"，结果为什么会不同？
5. **数据流图**。画一张从 `shard_*.parquet` 到 `optimizer.step()` 的数据流箭头图，在每条箭头上标注数据类型（字符串、`list[int]`、`torch.long`、浮点 logits、标量 loss）。
6. **做实验的注意事项**。如果你想对比"**连续 token 流**"和"**每文档 BOS packing**"哪个训练效果好，必须**固定**哪些变量（模型 / tokenizer / 有效 token budget / 优化器配方）？除了 loss，还要报告哪些指标（利用率、吞吐、val 评估）才算一次负责任的消融？

---

## 前后章导航

- **上一章**：[第 05 章：GPT 的 MLP、残差流与现代改进](./05-gpt-mlp-tricks.md)。讲清了 `GPT.forward` 怎么从 `x` 产生 logits。
- **本章**：logits + 右移目标 → 交叉熵 → 反传 → 一次参数更新。
- **下一章**：[第 07 章：基础模型训练：优化器、精度与规模法则](./07-base-train-optim-scaling.md)。拆 `model.setup_optimizer` 里 Muon / AdamW 的分工、学习率曲线、FP8 的用法，以及 `--depth` 怎么联动决定训练规模。
