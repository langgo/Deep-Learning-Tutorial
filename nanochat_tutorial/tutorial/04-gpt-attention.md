# 第 04 章　GPT 注意力：嵌入、因果注意力与 RoPE

> **本章定位**　第二部分·基础模型。打开模型第一层黑箱。
>
> **中心问题**　整数 token id 是怎么被模型读进去的？模型为什么能知道 token 的先后顺序？在"只能看前面"的约束下，模型是如何把过去的信息搬到当前位置的？
>
> **范围**　Embedding、因果多头自注意力、RoPE、GQA、QK norm、滑动窗口、FA3 / SDPA 后端。MLP、残差流、value embedding、smear、backout 留到[第 05 章](./05-gpt-mlp-tricks.md)。
>
> **先决条件**　读过[第 03 章](./03-tokenizer.md)：知道 `(B, T)` 是 token id 的矩形整数张量，`B = 一次并行的序列数`、`T = 每条序列的 token 位置数`。
>
> **本章产物**　把任意一行 `CausalSelfAttention.forward` 的代码追回标准三件事：`Q = xW_Q、K = xW_K、V = xW_V` 和因果 mask；能说出"**什么是核心机制**、**什么是本仓库的架构选择**、**什么是工程优化**"的分界线。
>
> **源码入口**　`nanochat/gpt.py`、`nanochat/flash_attention.py`、`tests/test_attention_fallback.py`；训练阶段的调用链起点是 `scripts/base_train.py::build_model_meta` 与 `model(x, y)`。

---

## 本章讲什么

[第 03 章](./03-tokenizer.md)的终点是一串 token id：`[17, 42, 9, 301]`。它们只是**词表里的行号**——**数值大小和数值距离都没有语义**。`17` 不比 `42` 更“大”（就像员工工号 17 和 42 不存在谁更重要）；`301` 和 `302` 数值虽然相邻，对应的 token 在含义上**也不比** `301` 和 `17` 更接近。模型从这堆行号里必须一步步重建出“**狗咬人**”和“**人咬狗**”的差别。

本章想一次性回答三个问题：

1. 这些整数是**怎么变成模型能计算的向量**的？
2. 一个 token 怎么**从前面所有历史 token 那里"拉"到信息**？
3. 既然模型里没有一张"位置表"，它**怎么知道哪个 token 在前、哪个在后**？

这三件事串起来，就是 `CausalSelfAttention` 这个类所做的全部。先给出全图：

```text
idx (B, T)              整数 token id        ← 第 03 章的终点
    │ wte 查表
x (B, T, C)             每个 token 一个 C 维向量
    │ ──┐
    │   │   每个 Block 内部重复做：
    │   │     • 用 Q、K、V 做因果注意力（本章）
    │   │     • 过一个 MLP（第 05 章）
    │   │     • 加残差回主干（第 05 章）
    │ ──┘
x (B, T, C)             每个位置的向量都带上了"看过的历史"
    │ lm_head
logits (B, T, V)        每个位置对"下一个 token 是什么"的打分
```

本章关注第二步到第三步里**注意力**这一半；MLP 和残差留给[下一章](./05-gpt-mlp-tricks.md)。

> **读法建议**：第一遍只读 §1–§4 和 §6（它们构成"**经典注意力 + 位置信息**"的最小闭环）。§5 的 GQA、§7 的 QK norm / 滑窗 / FA3 都是**选修**：架构取舍和工程后端，第二遍再回来。

---

## 预备词汇

本章会反复出现这几个名字。先一次说清，后文直接用：

| 符号 | 含义 | nanochat 对应 |
|---|---|---|
| **`B`** | batch size，一次并行多少条序列 | `B, T = idx.size()` |
| **`T`** | 本次 forward 送进模型的 token 数 | 训练 / prefill 时 = 整段序列长度；decode 时 = 1（历史存在 KV cache 里） |
| **`C`** | 每个 token 的向量宽度（模型主干宽度） | `config.n_embd` |
| **`H`** | query 头的数量 | `config.n_head` |
| **`H_kv`** | key / value 头的数量 | `config.n_kv_head` |
| **`D`** | 一个头内部的向量宽度，`D = C / H` | `self.head_dim` |
| **`V`** | 词表大小 | `config.vocab_size` |

另外三个高中数学工具，等会儿会反复用到：

- **线性层**：`y = x @ W`。只改最后一维的宽度，其他维不动。
- **点积**：两个向量方向越一致、点积越大。它可以为负（夹角 > 90°）或 0（正交）。
- **softmax**：把一排任意分数 `[s_1, s_2, ..., s_n]` 变成一组非负、加起来等于 1 的权重。分数越大，权重越大。

### 一组贯穿全章的小数字

为了避免每节都重新造例子，整章用这组参数：

```text
B    = 2       两条序列并行
T    = 5       每条序列 5 个 token
C    = 12      每个 token 一个 12 维向量
H    = 3       3 个 query 头
H_kv = 3       3 个 kv 头（本章基础配置：n_kv_head = n_head）
D    = 4       每个头内部 4 维（= C / H）
```

这是一个微型配置。真实训练里 `C` 可能是 768、1024，`H` 可能是 12、16。本章凡出现"输入形状 `(2, 5, 12)`"，都是指这组数字。

> **一个常见误读**：`gpt.py::GPTConfig` 的 dataclass 默认值（`n_layer=12`、`n_head=6`、`n_embd=768` ...）**不是真实训练的参数**。真正跑训练时，`scripts/base_train.py::build_model_meta` 会根据 `--depth`、`--head-dim`、`--aspect-ratio` 现算出来，并显式传 `n_kv_head=num_heads` —— 也就是**基础训练就是普通多头注意力**，不启用 GQA。

---

## 1. GPT 到底在学什么：预测下一个 token

把这一节当成"训练目标的 FAQ"。先一张图：

```text
真实文本 token 序列:   [t0, t1, t2, t3, t4]

训练时右移一格:
    输入 idx:          [t0, t1, t2, t3]
    目标 y:            [t1, t2, t3, t4]
```

**位置 0 要从 `t0` 猜 `t1`；位置 2 要从 `t0, t1, t2` 猜 `t3`**。四个位置可以**同时并行**算损失——这正是训练效率的关键。但代价是：**位置 `i` 绝对不能看到位置 `i+1` 及以后**。否则就等于偷看了答案；等到真正生成文本时，未来 token 根本还不存在，模型得不出合理的结果。

这个"不能偷看"的约束就是本章要反复出现的 **causal（因果）**。

### 走一遍真实入口

```python
# gpt.py::GPT.forward
def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
    ...
```

两种调用模式：

- **只有 `idx`**（没传 `targets`）→ 做前向，返回 `logits`，形状 `(B, T, V)`；
- **同时传 `idx` 和 `targets`** → 用 `F.cross_entropy` 算损失，返回标量。

损失的那一支还有一个小细节：`ignore_index=-1`。[第 03 章](./03-tokenizer.md)里讲 SFT mask 时就见过——mask=0 的目标会被改成 `-1`，这里就是它被真正忽略的地方。

### 最后一步：logits 不是概率

模型主干的输出是 `(B, T, C)`。最后几行代码：

```python
logits  = self.lm_head(x)                   # (B, T, V_padded)
logits  = logits[..., :self.config.vocab_size]   # 砍掉 padding，变成 (B, T, V)
logits  = logits.float()
softcap = 15
logits  = softcap * torch.tanh(logits / softcap)  # 把 logits 压进 ±15 范围
```

两件事值得专门指出：

**① 词表 padding 只是矩阵效率**。`V_padded` 是把 `V` 向上凑到某个对齐数（64 或 128 的倍数），GPU 矩阵乘更快。砍回 `V` 后 padding 的那几列就不参与采样了。

**② 用 `softcap = 15` 挤压 logits**。`tanh(x)` 的值永远在 `(-1, 1)` 之间，所以 `15 * tanh(x/15)` 永远在 `(-15, 15)` 之间。这样做让极端大的 logits（例如 50、100）平滑地回落到可控范围，避免 softmax 下某个 token 的概率失控地占到几乎 1——训练会更稳。这是**稳定性的工程选择**，不是语言建模的理论必要。

**logit 不是概率**：它可以为负、整行加起来也不要求等于 1。只有对某一个位置 `logits[b, t, :]` 做 softmax，才得到词表上"下一个 token 是什么"的概率分布。模型并不是"**一次输出整句话**"——它反复做同一件事：**取最后一个位置的 logits → softmax → 采一个 token → 把它接到序列末尾 → 再来一次**。

---

## 2. Embedding：把"行号"变成可计算的向量

单纯一个 `t = 17` 没法做乘法——`17 * 2.3` 对语言建模来说毫无意义。所以第一步永远是**查表**。

```python
# gpt.py::GPT.__init__
"wte": nn.Embedding(padded_vocab_size, config.n_embd)

# gpt.py::GPT.forward
x = self.transformer.wte(idx)      # (B, T) → (B, T, C)
x = x.to(COMPUTE_DTYPE)
x = norm(x)
```

`wte` 可以看成一张形状 `(V_padded, C)` 的参数表（**weights of token embedding**）。`idx` 里每个整数指定 `wte` 里的一行；一次查表完成后，`(B, T)` 的整数张量就变成了 `(B, T, C)` 的向量张量。

这**不是**"把整数乘成浮点数"。它是**查行号**：第 17 行是什么、第 42 行是什么，全部由训练过程中反向传播调整。**语义不在 id 的数值里，语义在表的那些行向量里**。

### 两个常被忽略的细节

**① 没有位置表**。经典 Transformer 会在 `wte(idx)` 之后再加一个"位置 embedding"：

```python
# 经典做法（nanochat 不是这样！）
x = token_embed[idx] + position_embed[0..T]
```

nanochat 文件开头明确写着 **"rotary embeddings (and no positional embeddings)"**。它没有可学习的位置表，而是在后面每一层的注意力里**旋转 Q 和 K**。为什么要这样？[§6](#6-rope不用位置表也让注意力知道顺序) 专门讲。

**② 输入 `wte` 和输出 `lm_head` 是两套独立参数**。有些 GPT 实现会**权重绑定**（weight tying）：`lm_head.weight = wte.weight`。nanochat **没有**这么做。前者负责"读 token"，后者负责"向词表写分数"，它们训练时各自调整——这叫 **untied weights**。读源码别从别处套这个假设。

### `norm(x)` 又是什么

`norm` 是无参数的 RMS 归一化（`F.rms_norm(x, (x.size(-1),))`）：对每个 token 的 `C` 维向量，统一除以它的 RMS 幅度，让它们的"长度"进入一个可控范围。它在整个 Transformer 里反复出现，这里先当"把每行向量的尺度整理一下"看待；下一章讲 RMSNorm 的具体角色。

---

## 3. 单头注意力：具体位置具体看

讲完 embedding，每个 token 都有一个向量了。但这些向量彼此**毫无关系**——"苹果"在"吃苹果"和"苹果公司"里是**同一行表**。所以第二步要做的是：**让每个位置从前面的位置拉信息进来，让它变成"带上下文的苹果"**。

### 三个线性投影：Q、K、V

**先把多头、batch、mask 全部放一边**，只看一条序列：

```text
X ∈ ℝ^(T × C)     T 行，每行是一个 token 的 C 维向量
```

做三个独立的线性投影，分别得到：

```text
Q = X @ W_Q      "问题" (query)       形状 (T, D)
K = X @ W_K      "标签" (key)         形状 (T, D)
V = X @ W_V      "内容" (value)       形状 (T, D)
```

> 这里先用 `D` 表示单头的宽度（后面接到多头时会自然用上）。本节暂时把 `D = C` 就行。

直观上三者的角色是：

- **Query `q_i`**：第 `i` 个位置**现在想找什么样的信息**；
- **Key `k_j`**：第 `j` 个位置**自我介绍——我是什么样的信息**；
- **Value `v_j`**：第 `j` 个位置**如果被选中，实际要搬运什么内容**。

Key 和 Value 为什么要分开？因为"我值得被看"和"我实际携带什么"是两件可以**单独训练**的事。类比图书馆：Key 是书脊上的分类号，Query 是你查找的关键词，Value 是书里的内容。

### 走一个具体位置：`i = 3`

用贯穿全章的小数字。我们看第二条序列的第 3 个 token（`b=1, i=3`）要怎么收集信息。假设用的是单头（`H=1`、`D=C=12`）：

**第一步：算分数**。拿 `q_3`（一个 12 维向量）去和 `k_0, k_1, k_2, k_3, k_4` 五个 key 各做一次点积：

```
scores =
  q_3 · k_0 = 2.1
  q_3 · k_1 = 0.3
  q_3 · k_2 = 4.7    ← 最大
  q_3 · k_3 = 1.9
  q_3 · k_4 = 3.5
```

**第二步：缩放**。除以 `√D = √12 ≈ 3.46`。为什么要这么做？`D` 越大，两个随机向量点积的典型幅度就越大；不缩放的话 softmax 会变得极尖、几乎只有一个 1 其余全 0，梯度就消失了。

**第三步：加因果 mask**。位置 3 **只能看到 `j ≤ 3`**，所以要把 `j = 4` 的分数改成 `-∞`：

```
scores (缩放+mask) =
   0.61,  0.09,  1.36,  0.55,  -∞
```

**第四步：softmax**。得到 5 个权重，和为 1：

```
weights =  [0.22, 0.13, 0.46, 0.19, 0.00]
                              ← 位置 4 权重为 0（因为 e^-∞ = 0）
```

**第五步：加权求和**。拿这 5 个权重去对 5 个 value **做加权平均**：

```
y_3 = 0.22 * v_0 + 0.13 * v_1 + 0.46 * v_2 + 0.19 * v_3 + 0.00 * v_4
```

这个 `y_3`（一个 12 维向量）就是**位置 3 看过前面所有允许看的位置之后、带上了上下文的新向量**。它主要吸收了 `v_2` 的内容（权重 0.46），也吸收了 `v_0` 和 `v_3` 的一部分。

### 整条序列并行：一个矩阵乘法

把上面"位置 3"的流程**同时对所有 5 个位置做**，用矩阵乘法一次算完，就是经典的注意力公式：

```
S = Q @ K.T / √D           形状 (T, T)    分数矩阵
A = softmax(S + M)          形状 (T, T)    权重矩阵（每行和为 1）
Y = A @ V                   形状 (T, D)    每个位置的输出
```

`S[i, j]` 就是位置 `i` 的 query 和位置 `j` 的 key 的点积分数。`M` 是因果 mask —— 一个把右上三角填成 `-∞`、左下三角和对角线填成 0 的矩阵：

```
         key j = 0   1   2   3   4
query  0:   0   -∞  -∞  -∞  -∞
query  1:   0    0  -∞  -∞  -∞
query  2:   0    0   0  -∞  -∞
query  3:   0    0   0   0  -∞
query  4:   0    0   0   0   0
```

于是 `A` 的"可见图"就是：

```
query 0:  █ · · · ·            每行权重加起来 = 1
query 1:  █ █ · · ·            看到的位置用加权平均
query 2:  █ █ █ · ·            看不到的权重为 0
query 3:  █ █ █ █ ·
query 4:  █ █ █ █ █
```

这就是 **causal self-attention** 的全部核心机制。它**不是**"每个位置只看上一个 token"——完整模式下**位置 `i` 可以同时看到 `0 ... i` 全部历史**；它只是**禁止看未来**。

---

## 4. 多头：一组 token 同时用多种眼光看

单头有一套 `W_Q, W_K, W_V`。每个 token 只能用**一种匹配方式**检索上下文。但实际语言里，"该找谁"是多维度的——有的时候要找句法上的主语，有的时候要找语义上相关的名词，有的时候要找最近的标点。

**多头**的做法很直接：**把 `C` 维向量切成 `H` 份，每份 `D = C/H` 维，每份独立做一次上面讲的注意力**。这样每个 token 相当于**同时用 H 种不同的匹配空间检索上下文**，最后再把 H 个结果拼回来。

**注意**：多头**不是**把文本切成 `H` 段分给不同头。**每个头看到的还是完整的 T 个位置**，只是每个头内部的 Q/K/V 都只有 `D` 维。

### 看 `CausalSelfAttention.__init__`

```python
self.head_dim = self.n_embd // self.n_head
assert self.n_embd    %  self.n_head    == 0
assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

self.c_q    = Linear(self.n_embd, self.n_head    * self.head_dim, bias=False)
self.c_k    = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
self.c_v    = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
self.c_proj = Linear(self.n_embd, self.n_embd,                     bias=False)
```

对我们的小数字来说：

- `head_dim = C / H = 12 / 3 = 4`；
- `c_q`、`c_k`、`c_v` 都是 `12 → 12` 的 Linear（因为 `H = H_kv = 3`，`H × D = 12`）；
- `c_proj` 是 `12 → 12` 的 Linear，用来把 H 个头的结果合回主干。

### Forward 的形状操作

```python
B, T, C = x.size()                                   # (2, 5, 12)
q = self.c_q(x).view(B, T, self.n_head,    self.head_dim)   # (2, 5, 3, 4)
k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)   # (2, 5, 3, 4)
v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)   # (2, 5, 3, 4)
```

`.view(...)` **不复制数据**，只是**换一种解释内存的方式**。原来是 `(2, 5, 12)` 一条长 12 的向量，现在重新解释成 `(2, 5, 3, 4)` —— 3 个头，每头 4 维。

计算完注意力后，输出仍是 `(B, T, H, D)`。合头的代码极其直接：

```python
y = y.contiguous().view(B, T, -1)    # (2, 5, 3, 4) → (2, 5, 12)
y = self.c_proj(y)                    # (2, 5, 12) → (2, 5, 12)
```

`.contiguous()` **不能随便去掉**：某些注意力后端或转置操作让底层内存不再是"按最后一维连续排布"，而 `.view(...)` 要求内存按某种布局连续才能原地解释。少了 `.contiguous()` 会报运行时错误。

### 一个具体的形状练习

你可以在脑子里（或者 Python 里）走一遍：

```python
import torch

B, T, C, H = 2, 5, 12, 3
D = C // H

x = torch.randn(B, T, C)                             # (2, 5, 12)
Wq = torch.randn(C, C)
q = (x @ Wq).view(B, T, H, D)                        # (2, 5, 3, 4)

# 真实 nanochat 保持 (B, T, H, D) 布局，因为 FA3 要求它。
# 下面只是为了用矩阵乘法手动算一次，临时把 H 换到 T 前面：
qh = q.transpose(1, 2)                               # (B, H, T, D) = (2, 3, 5, 4)
scores = qh @ qh.transpose(-2, -1) / D**0.5          # (2, 3, 5, 5)
```

最后那个 `(2, 3, 5, 5)` 的张量，含义是：**2 条序列、3 个头、每个头内部 5×5 的"query 位置 × key 位置"分数矩阵**。每个头都对所有 5 个位置做一次完整的注意力。

---

## 5. GQA：很多 query，共享几套 K/V

多头的默认配置是 `H_kv = H`：query 头和 kv 头一样多。但 nanochat 的注意力还支持 **Grouped-Query Attention（GQA）**，允许 `H_kv < H`，而且必须整除。

### 为什么要有 GQA

训练时无所谓——Q、K、V 都是算一次就扔的。但**推理时**不一样：

- 每生成一个新 token，前面所有历史 token 的 K、V 都要保存下来，这叫 **KV cache**；
- 每层都保存一份；
- 显存占用和带宽都和 `H_kv` 成正比。

举个数：假设 `C = 1024`、`H = 16`、32 层、上下文 8192、`float16`：

- 普通多头 (`H_kv = 16`)：每层 KV cache = `2 × 8192 × 16 × 64 × 2 B` ≈ 32 MiB，32 层 ≈ **1 GiB**；
- GQA 把 `H_kv` 减到 4：`H_kv` 砍到 1/4，KV cache 也砍到 1/4，变成 **256 MiB**。

在这个例子里，**KV cache 本身**和读取 KV 的理论字节数都缩小到 1/4。端到端推理吞吐能提升多少，还取决于模型权重读取、batch、上下文长度、kernel 和硬件；不能直接从“KV 少 4 倍”推出“整体快 4 倍”。代价是多个 query head 共用较少的 K/V 表示，独立表达能力受到约束。

GQA 已被多种现代模型采用，但不是所有模型或所有规格的统一标配。

### 具体怎么分组

以 `H = 16, H_kv = 4` 为例（两者必须整除）：

```
group_size = H / H_kv = 4         ← 每组有 4 个 Q head

Q head:    0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
           └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
KV head:         0               1               2               3
```

- Q 头数 **不变**：模型仍然输出 16 个 `(T, D)` 的 attention 结果，拼回 `(T, 16×D) = (T, C)`；
- K/V **投影矩阵只有 4 份**：`c_k`、`c_v` 输出维度是 `H_kv × D`，不是 `H × D`（所以 K/V 的权重和 KV cache 都小 4 倍）；
- **计算上**：Q head 0、1、2、3 都和 KV head 0 的 K、V 算注意力——所以说“**共享**”而不是“**重复用 4 次**”：常规 kernel 在 FA3 / SDPA 内部把 KV head 按对应关系广播到 16 份，**不在 Python 层写 `repeat_interleave`**。

对应到 nanochat 的代码：

```python
# gpt.py::CausalSelfAttention.forward (简化)
q = self.c_q(x).view(B, T, self.n_head,    self.head_dim)   # (B, T, 16, D)
k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)   # (B, T,  4, D)
v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)   # (B, T,  4, D)
# FA3 路径：把 Q head 数与 KV head 数这两个维度直接交给 kernel
# SDPA 路径：显式传 enable_gqa=True 让 PyTorch 自己广播
```

**一句话**：**Q head 全额保留、KV head 按组共用**——是“多对一”共享，不是“给 Q 调用 KV 四次”。

### 本仓库的现状

GQA 的**代码能力准备好了**，但**基础训练不启用**：

```python
# scripts/base_train.py::build_model_meta
# ... 构造 model_config 时传入 n_kv_head=num_heads
```

也就是说，基础训练默认跑的是**普通多头**。GQA 的代码路径由专门的测试覆盖：`tests/test_attention_fallback.py::TestFA3VsSDPA.test_gqa` 用 `n_heads=8, n_kv_heads=2` 验证两条后端都能正确处理 GQA。

读到这个"存在但未启用"的状态不要意外——它是**预留的能力**，等到推理阶段真的需要节省 KV cache 时就可以打开。本章先知道有这件事就行。

---

## 6. RoPE：不用位置表，也让注意力知道顺序

### 为什么必须有位置信息

重看一下 §3 的注意力公式：

```
S = Q @ K.T / √D
A = softmax(S + M)
Y = A @ V
```

如果没有任何位置编码，Q/K/V 投影本身只依赖 token 内容；对同一组**已可见**的历史 token，attention 没有显式坐标告诉它“这是前一个”“那是十步以前”。纯 self-attention 对位置排列具有等变性。

不过 decoder 的 causal mask 已经提供了一部分顺序结构：位置 `t` 只能看 `0..t`，因此不同位置的可见前缀不同。RoPE 补上的，是已可见 token 之间更具体的相对位移信息，使模型能区分“相邻”和“相隔很远”，而不只知道“都在过去”。

### 两条路线

**经典做法：绝对位置 embedding**。给第 `p` 个位置加一个固定或可学习的位置向量：

```
x_p = token_embed(t_p) + position_embed[p]
```

缺点：

- 位置向量占参数预算；
- 长度超过训练过的最大 `T` 就不好外推。

**nanochat 用的做法：RoPE（Rotary Positional Embedding，旋转位置编码）**。它不在 `x` 上加任何位置向量，而是在**每层注意力里**对 **Q 和 K**（不是 V）做一个和位置有关的**旋转**。

> **一句话直觉**：RoPE 让 "**位置 i 的 Q 乘以位置 j 的 K**" 的结果里，自动带上 "**i - j 的相对位置信息**"。因此 attention 的分数就天然地知道两个 token 的相对距离。

### 怎么算这些“旋转”

#### 先把直觉建起来：把每对通道当成一个“时钟盘”

RoPE 把一个头的 `D` 个通道配成 `D/2` 对。每对通道被看成一个二维平面上的向量：

> **术语**：**通道（channel）** 就是“向量的一个分量”。一个 token 在模型主干里是长度 `C = n_embd` 的向量，一共 `C` 个通道；**分成 `H` 个头之后，每个头看到的是 `D = C/H` 个通道**。本节讲 RoPE 对“**一个头内部的 `D` 个通道**”做旋转，不同头各自独立。比如 `D = 4` 时某个头的 Q 向量形如 `[0.3, -1.1, 0.7, 0.2]`，四个数就是四个通道。
>
> **源码细节**：为了画图直观，可以先把“配对”想成 `(0,1)、(2,3)` 这种相邻配对；但 nanochat 源码实际是 `x1 = x[..., :D/2]`、`x2 = x[..., D/2:]`，也就是**前半和后半配对**。两种配对方式数学作用相同，都会把两个数放进一个 2D 平面旋转；区别只影响 checkpoint 的排列约定。

```
没 RoPE：                      RoPE 后（位置 t 处）：

   x₂ ↑                            x₂ ↑
      | • (x₁, x₂)                   |      • (y₁, y₂)
      |/                             |     /
      └─────→ x₁                    └─────→ x₁
                                     ↑
                               旋转了 θ = t × freq[c]
```

每对通道都转，但 **每对转的速度不同**：通道 0-1 转得最快（高频）、通道 D-2, D-1 转得最慢（低频）。总共 `D/2` 条不同速度的“时钟指针”——就像一个多尺度的位置钟。

> 🧡 **交互式可视化**：[./assets/rope-visualization.html](./assets/rope-visualization.html)。本地双击打开（或 `python3 -m http.server` 后浏览器访问）。包含三个面板：① 单对通道在 2D 上的旋转；② `D/2` 对通道的频率分布；③ 旋转后 `Q(i)·K(j)` 如何自动带上相对距离。

#### 再看代码

```python
# gpt.py::GPT._precompute_rotary_embeddings
channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
inv_freq      = 1.0 / (base ** (channel_range / head_dim))     # base = 100000
t             = torch.arange(seq_len, dtype=torch.float32, device=device)
freqs         = torch.outer(t, inv_freq)                        # (seq_len, D/2)
cos, sin      = freqs.cos(), freqs.sin()                        # (seq_len, D/2)
cos, sin      = cos[None, :, None, :], sin[None, :, None, :]    # (1, seq_len, 1, D/2)
```

逐行对图看：

- **`channel_range = [0, 2, 4, ..., D-2]`**：`D/2` 对通道的编号。
- **`inv_freq = 1 / 100000^(c/D)`**：这里 `c` 取 `0, 2, ..., D-2`。`c=0` 时 `inv_freq=1`（最快）；最后一项是 `100000^{-(D-2)/D}`，接近但通常大于 `1/100000`。
- **`freqs = outer(t, inv_freq)`**：**位置 × 角速度 = 角度**。产物是 `(seq_len, D/2)` 的角度表。
- **`cos[None, :, None, :]`**：形状改成 `(1, seq_len, 1, D/2)`，之后广播到 `(B, T, H, D/2)` 一次性作用到所有头。

然后 `apply_rotary_emb` 把每对通道按 2×2 旋转矩阵计算：

```
y_1 =  x_1 * cos + x_2 * sin
y_2 = -x_1 * sin + x_2 * cos
```

如果这两行公式卡住，先看这个小推导： [./assets/rotation-matrix-derivation.html](./assets/rotation-matrix-derivation.html)。它从 `(x₁, x₂)` 的极坐标开始，推到标准旋转矩阵，再对比 nanochat 源码里的 `-θ` 写法。

调用本身非常简洁：

```python
cos, sin = cos_sin
q = apply_rotary_emb(q, cos, sin)
k = apply_rotary_emb(k, cos, sin)
# 注意：没有 V ！
```

**为什么只转 Q 和 K，不转 V？** RoPE 的设计目标是让位置进入 `Q·K` 的匹配分数；V 仍承载被权重汇总的内容。这里是架构约定，不意味着“V 从原则上绝不能编码位置”——其他位置编码方案可以采用不同设计。

#### 为什么这样做 “Q·K” 就自动带上了相对距离

先只看**一对通道**。假设：

- 位置 `i` 的 query 原始向量是 `q = (q₁, q₂)`；
- 位置 `j` 的 key 原始向量是 `k = (k₁, k₂)`；
- 这对通道的频率是 `freq`；
- RoPE 给位置 `i` 的 Q 旋转 `θ_i = i × freq`；
- RoPE 给位置 `j` 的 K 旋转 `θ_j = j × freq`。

按 nanochat 的旋转公式，旋转后是：

```
Q_i = ( q₁ cosθ_i + q₂ sinθ_i,
       -q₁ sinθ_i + q₂ cosθ_i )

K_j = ( k₁ cosθ_j + k₂ sinθ_j,
       -k₁ sinθ_j + k₂ cosθ_j )
```

attention 分数里要算的是内积 `Q_i · K_j`。展开后会得到：

```
Q_i · K_j
= (q₁k₁ + q₂k₂) · cos(θ_i - θ_j)
  + (q₂k₁ - q₁k₂) · sin(θ_i - θ_j)
```

这行公式不用背，只看它的结构：

| 部分 | 含义 |
|---|---|
| `q₁k₁ + q₂k₂` | 原来 q 和 k 的普通相似度，也就是不带位置时的点积 |
| `q₂k₁ - q₁k₂` | q 和 k 在二维平面里的有向面积项 |
| `cos(θ_i - θ_j)`、`sin(θ_i - θ_j)` | **位置差**带来的调制因子 |

关键在最后一列：

```
θ_i - θ_j = i × freq - j × freq = (i - j) × freq
```

所以 RoPE 后的 `Q_i · K_j` 不只取决于 q、k 的内容分量，还会通过 `sin/cos((i-j)×freq)` 被相对位置调制。这里不是“整个点积乘一个距离因子”，而是内容相关的两项分别乘 cos 和 sin。换句话说：

> **模型在比较 query 和 key 时，分数函数能够区分不同的相对位移 `i-j`。**

一个最简单的特例更直观。假设 `q = k = (1, 0)`，也就是内容完全一样，那么：

```
Q_i · K_j = cos((i - j) × freq)
```

- `i = j` 时，相对距离 0，分数是 `cos(0) = 1`；
- `i` 和 `j` 差 1 个位置，分数是 `cos(freq)`；
- 差 2 个位置，分数是 `cos(2 × freq)`；
- 差多少，分数就按那个距离变化。

这就是 RoPE 的关键性质：代码分别用绝对位置角度 `θ_i`、`θ_j` 旋转 Q/K，但一做点积，位置部分只以差值 `i-j` 出现。它提供的是**相对位移信号**，不是预先写死的“距离越近分数越高”规则；真实 attention 是否偏爱近邻，要由 q/k 内容和训练共同决定。

> 在交互式可视化的图 3 里，拖动 “Q 的位置 i” 滑块，会看到同一相对位移对应同样的 RoPE 调制形状；曲线可能振荡，并不保证随距离单调下降。

### KV cache 时的位置偏移

训练阶段 Q 和 K 用的位置是 `[0, 1, 2, ..., T-1]`，很直接。但推理时如果已经把前面 100 个 token 的 K、V 存进了 cache，现在刚算的那**一个**新 token 的位置应该是 **100**，不是 **0**。

源码精确处理了这件事：

```python
T0 = 0 if kv_cache is None else kv_cache.get_pos()
cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
```

`kv_cache.get_pos()` 返回当前已缓存的长度，RoPE 直接从那个位置继续取旋转角度。KV cache 的完整机制留给[第 10 章](./10-inference-engine.md)细讲，这里只要记住："**位置是 cache 内部累积出来的，不是每次都从 0 开始**"。

### 预分配的 RoPE 表上限

`GPT.__init__` 把 cos / sin 的长度预分配为 `config.sequence_len * 10`，当作 non-persistent buffer 注册。`GPT.forward` 会断言当前 `T` 不超过这张表。这**不是"无限上下文"的承诺**，而是**预先多算一些以供复用**——真正的训练窗口仍由 `sequence_len` 决定。

---

## 7. 架构选择与工程优化：和核心机制分开看

前面六节讲的是**注意力的核心机制**——Q/K/V、softmax、因果 mask、多头、RoPE。读到这里，你已经理解了一个标准 decoder-only Transformer 的注意力层在做什么。

本节讲的三件事都是**可以从核心机制中拆出来、独立评估的东西**。你完全可以关掉它们、注意力还是正常工作；它们的作用是**训练更稳、推理更快、兼容更多硬件**。

### 7.1 QK norm 和 1.2：先把分数尺度稳住

前面说过，attention 的分数来自：

```text
score = Q @ K.T / √D
```

这里最怕的是 **Q 或 K 的长度突然变大**。因为点积会跟着变大，softmax 就会变得特别尖：某个位置接近 1，其他位置接近 0。太尖以后梯度容易不稳；反过来，如果 Q/K 太小，softmax 又会摊得太平，模型分不清重点。

nanochat 在 RoPE 之后马上做两步：

```python
q, k = norm(q), norm(k)
q = q * 1.2
k = k * 1.2
```

#### 第一步：`norm(q), norm(k)`

`norm` 是 RMSNorm。放在 Q/K 上时，可以把每个位置、每个头里的向量长度拉回一个稳定范围。直觉上就是：

```text
原来：有些 Q/K 很长，有些很短 → 点积分数尺度乱飘
现在：Q/K 先归一化             → 点积分数尺度更可控
```

注意：这里不是在改“看谁”的逻辑，只是在改**分数的尺度**。Q/K 方向仍然保留，长度被压稳。

#### 第二步：`* 1.2`

归一化后，分数可能偏保守。于是源码又把 Q 和 K 各乘 `1.2`：

```text
(Q * 1.2) @ (K * 1.2).T = 1.44 × (Q @ K.T)
```

也就是说，attention logits 整体放大约 `1.44` 倍，softmax 会稍微更尖一点，更愿意集中到少数几个位置。

#### 怎么看待它

| 代码 | 作用 | 属于什么 |
|---|---|---|
| `Q @ K.T / √D` + causal mask + softmax + `@ V` | 注意力的数学核心 | 必需 |
| `norm(q), norm(k)` | 稳定 Q/K 的尺度 | 架构选择 |
| `q = q * 1.2; k = k * 1.2` | 调整 softmax 尖锐程度 | 实验配方 |

所以不要把 `1.2` 当成理论常数。它是 nanochat 当前规模和训练配方下的经验值，换模型可能要重调。

### 7.2 滑动窗口：不是“不看长上下文”，而是“多数层省着看”

先看 full attention。假设 `T = 8`，位置 7 在因果注意力里能看所有历史：

```text
位置 7 能看：0 1 2 3 4 5 6 7
```

整张可见矩阵是一个下三角：

```text
      key 位置 →
query↓   0 1 2 3 4 5 6 7
0        ✓
1        ✓ ✓
2        ✓ ✓ ✓
3        ✓ ✓ ✓ ✓
4        ✓ ✓ ✓ ✓ ✓
5        ✓ ✓ ✓ ✓ ✓ ✓
6        ✓ ✓ ✓ ✓ ✓ ✓ ✓
7        ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓
```

这叫 **full attention**。它表达力强，但矩阵大小是 `(T, T)`，计算量随 `T²` 增长。`T = 32K` 时，代价就很高。

#### 滑动窗口做了什么

如果窗口 `left = 4`，每个位置最多看左边 4 个历史位置，再加当前位置自己。位置 7 就变成：

```text
位置 7 能看：3 4 5 6 7
```

可见矩阵只保留靠近对角线的一条带子：

```text
      key 位置 →
query↓   0 1 2 3 4 5 6 7
0        ✓
1        ✓ ✓
2        ✓ ✓ ✓
3        ✓ ✓ ✓ ✓
4        ✓ ✓ ✓ ✓ ✓
5          ✓ ✓ ✓ ✓ ✓
6            ✓ ✓ ✓ ✓ ✓
7              ✓ ✓ ✓ ✓ ✓
```

计算量从大约 `O(T²)` 变成 `O(T × (left+1))`；做渐近分析时通常简写成 `O(T × window)`。窗口远小于 `T` 时，会省很多。

#### nanochat 的 `(left, right)`

`gpt.py::GPT._compute_window_sizes` 给每一层生成一个 `(left, right)`：

```python
(long_window, 0)
(short_window, 0)
```

- `left`：当前位置能看左边多少个 token；
- `right`：当前位置能看右边多少个 token；
- decoder-only 模型不能看未来，所以 `right = 0`。

源码里：

```python
long_window = config.sequence_len
short_window = -(-long_window // 4 // 128) * 128
```

直觉上可以把 `short_window` 理解成：**大约是 `sequence_len / 4`，再对齐到 128 的倍数**。默认 `sequence_len = 2048` 时：

```text
short_window = 512
long_window  = 2048
```

所以：

- `L` = long：基本等于 full attention；
- `S` = short：只看最近 512 个左右。

> 源码注释里有一句 “2048 -> 768”，看起来像旧逻辑遗留；按当前代码实际算出来是 `512`。读源码时以表达式为准。

#### `window_pattern = "SSSL"`

默认 pattern 是：

```text
S S S L
```

如果模型有 12 层，就循环铺开：

```text
层号:    0    1    2    3    4    5    6    7    8    9    10   11
窗口:    512  512  512  全   512  512  512  全   512  512  512  全
```

并且源码最后还会强制：

```python
window_sizes[-1] = (long_window, 0)
```

也就是**最后一层一定看全上下文**。

这个设计的直觉是：

```text
大多数层：只做局部加工，便宜
少数层：做全局混合，贵但必要
最后层：看全局后再给 logits 做最终判断
```

所以滑动窗口不是说模型整体“不支持长上下文”。更准确地说：**多数层只看近处，少数层负责远距离信息交换**。

### 7.3 FA3 和 SDPA：同一套注意力，两种底层实现

到这里为止，我们一直在讲“数学上的注意力”：

```text
Q/K/V → causal mask → softmax → 加权 V
```

但在真实硬件上，怎么算这件事有不同实现。nanochat 用 `nanochat/flash_attention.py` 做了一个统一入口，让上层代码不用关心底层走哪条路。

#### 路径 A：FA3

FA3 = Flash Attention 3。它的目标是：**别真的把 `(B, H, T, T)` 的分数矩阵完整写出来**。

朴素实现像这样：

```text
先算 scores = Q @ K.T       # 巨大的 T×T 矩阵
再做 softmax(scores)
再乘 V
```

Flash Attention 的思路是把这些步骤融合到一个高效 kernel 里，边算边做 softmax、边累积输出，尽量少写中间大矩阵。收益是：

- 显存少；
- 带宽压力小；
- H100 / Hopper 这类 GPU 上速度很快。

nanochat 启动时会尝试加载 FA3：

```python
_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None
USE_FA3 = _resolve_use_fa3()
```

FA3 不可用、硬件不支持、dtype 不合适时，就不走它。

#### 路径 B：SDPA fallback

SDPA = PyTorch 的 `F.scaled_dot_product_attention`。它不是另一种模型算法，而是 PyTorch 提供的通用 attention API。

优点是兼容性好：CUDA、CPU、MPS 都能走。你在 Mac/MPS 上跑教程时，靠的就是这条 fallback 路径。

但 SDPA 的张量布局和 FA3 不一样。FA3 接口用：

```text
(B, T, H, D)
```

SDPA 需要：

```text
(B, H, T, D)
```

所以 fallback 里会转一下：

```python
q = q.transpose(1, 2)
k = k.transpose(1, 2)
v = v.transpose(1, 2)
y = F.scaled_dot_product_attention(...)
return y.transpose(1, 2)
```

#### fallback 为什么还要自己造 mask

最简单的 full causal attention 可以直接交给 SDPA：

```python
F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

但 nanochat 还支持：

- 滑动窗口；
- KV cache；
- prefill / decode 时 `Tq` 和 `Tk` 不一样；
- GQA (`H != H_kv`)。

这些组合起来时，简单的 `is_causal=True` 不够表达全部规则。fallback 里就会显式造一个 bool mask：

```python
mask = col_idx <= row_idx
mask = mask & ((row_idx - col_idx) <= window)
```

第一行表示“不能看未来”；第二行表示“不能看超过滑动窗口的远处历史”。GQA 情况则传：

```python
enable_gqa=True
```

#### 怎么看待 FA3 / SDPA

| 路径 | 作用 | 价值 |
|---|---|---|
| FA3 | 高性能 fused kernel | 快、省显存，适合支持的 GPU |
| SDPA | PyTorch 通用实现 | 兼容 CPU / MPS / 普通 CUDA |

两条路径目标是**同一个注意力结果**，但浮点运算顺序不同，所以不能要求逐 bit 一致。测试里用容差比较（`atol = rtol = 1e-2`），确认语义上足够接近。

一句话：**FA3 / SDPA 是执行后端，不是两种模型能力**。换后端只应该影响速度、显存和微小浮点误差，不应该改变模型定义。

---

## 把全章串成一段代码

把前七节合在一起，一个 `CausalSelfAttention.forward` 的简化骨架就是：

```python
def forward(self, x, cos_sin, kv_cache=None, ...):
    B, T, C = x.size()

    # 1. 三个线性投影 → Q, K, V
    q = self.c_q(x).view(B, T, self.n_head,    self.head_dim)   # (B, T, H,    D)
    k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)   # (B, T, H_kv, D)
    v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

    # 2. RoPE 旋转 Q 和 K，V 不动
    cos, sin = cos_sin
    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)

    # 3. 架构选择：QK norm + 1.2 缩放
    q, k = norm(q), norm(k)
    q = q * 1.2
    k = k * 1.2

    # 4. 真正的注意力（底层是 FA3 或 SDPA）
    #    自带 causal mask、滑窗、以及 KV cache 读写
    y = attention(q, k, v, causal=True, window_size=self.window_size, kv_cache=...)

    # 5. 合头回主干
    y = y.contiguous().view(B, T, -1)
    y = self.c_proj(y)
    return y
```

把这段代码对照回前面七节，每一行都能定位到某个概念。

---

## 常见误区

1. **"token id 本身就是语义数值。"** 错。id 只是 `wte` 表里的行号；语义来自表的那些**向量**和训练。
2. **"logits 就是概率。"** 错。logits 可以为负，整行加起来不为 1，softmax 后才是概率分布。
3. **"causal mask = 只能看上一个 token。"** 错。它只是**禁止看未来**；完整 attention 下位置 `i` 可以看到 `0 ... i` 全部历史。滑窗注意力下只能看最近的 `window + 1` 个 key。
4. **"多头 = 把句子切成 H 段，每个头负责一段。"** 错。**每个头都看完整的 T 个位置**，只是把每个 token 的 `C` 维向量切成了 `H` 份、每份独立做一次注意力。
5. **"GQA 会让输出头数变成 `H_kv`。"** 错。Q 和输出头数仍然是 `H`。GQA 共享的是 **K/V 头**。
6. **"RoPE = 把位置 embedding 加到 x。"** 那是经典绝对位置编码的描述。nanochat 的 `wte` 后**没有位置表**，RoPE **在每层注意力里旋转 Q 和 K**，而且**不旋转 V**。
7. **"FA3 不可用 attention 就跑不了。"** 错。SDPA fallback 支持 CPU / MPS / 非 Hopper CUDA，`test_attention_fallback.py` 专门有一组 `TestSDPAOnly` 覆盖这条路径。
8. **"`gpt.py` 里所有代码都是 Transformer 必须的定义。"** 错。核心是 **因果 QK-softmax-V**；QK norm、`* 1.2`、滑窗、FA3 都是本仓库选择的**稳定性 / 效率 / 兼容性**改进，不是数学必要。

---

## 小结

- **GPT 的训练目标** = 右移一格的 next-token prediction。`GPT.forward(idx, targets)` 一次性对所有 `(B, T)` 个位置算损失；`idx` 不带 target 时只返回 `logits`。
- **Embedding** = 查 `wte` 表。`(B, T)` 整数张量变成 `(B, T, C)` 向量。nanochat 的 `wte` 和 `lm_head` 是独立参数，且**不用可学习的位置表**。
- **注意力的核心** = `Q = xW_Q, K = xW_K, V = xW_V`，`softmax(Q K.T / √D + 因果 mask) @ V`。因果 mask 保证训练时位置 `i` 看不到 `i+1`。
- **多头** = 把 C 切成 H 份，每份独立算注意力，再合头回主干。每个头看完整序列，只是在**不同的匹配空间**里。
- **RoPE** = 不加位置表，在每层注意力里按位置**旋转 Q 和 K**（不转 V）。多种频率提供粗细不同的位置刻度；KV cache 时位置要从 `kv_cache.get_pos()` 继续偏移。
- **架构选择**（QK norm + `* 1.2`、滑动窗口 `SSSL`）和**后端优化**（FA3 vs SDPA）是独立的两类决策，别把它们和核心机制混为一谈。GQA 的能力已经写好，但当前基础训练**不启用**（`n_kv_head = num_heads`）。
- 下一章把 `CausalSelfAttention` 的输出接回 `Block.forward`：MLP 为什么要"先扩张再压回"、残差为什么必要、`norm` 到底该放在哪，以及 value embedding、smear、backout 这些 nanochat 额外加进来的东西。

---

## 练习

1. **形状追踪**。设 `B = 4, T = 16, C = 64, H = 4, H_kv = 2`。写出下列每一步的形状：`idx` → `wte(idx)` → `q` → `k` → `v` → attention 输出 → 合头后 → `c_proj` 后 → `lm_head` 后。注明 `D` 是多少。
2. **因果 mask 小动手**。用 `torch.randn(5, 5)` 当 `scores`，把严格上三角填成 `-inf`，`softmax(dim=-1)`。打印权重矩阵，检查每行加起来是否接近 1、以及右上三角是否全为 0。
3. **具体位置走一遍**。沿着 §3 的"位置 `i = 3`"例子，自己造一组 `q_3, k_0..k_4, v_0..v_4`（可以用 `torch.randn`，每个向量 4 维），手动算出缩放分数、因果 mask 后分数、softmax 权重、`y_3`。验证权重和 ≈ 1、位置 4 权重 = 0。
4. **RoPE 推理**。`apply_rotary_emb` 把最后一维分成等大的两半。如果 `D` 是奇数会发生什么？回去读 `GPTConfig` 和 `CausalSelfAttention.__init__`，说清当前代码靠哪些整除关系**间接**保证 `D` 为偶数——它有没有**单独**显式断言这一点？
5. **滑窗数字题**。默认 `sequence_len = 2048`，`window_pattern = "SSSL"`。写出每层的 `(left, right)` 窗口数值。层数设 8，再问：**哪一层能看到最远距离？** 距离具体是多少？
6. **机制分类**。把下面这些项分成"**核心机制 / 架构或训练选择 / 工程后端优化**"三类，并解释理由（允许一项同时属于两类）：因果 mask、RoPE、GQA、QK norm、`q = q * 1.2`、滑动窗口、`softcap = 15`、FA3、SDPA fallback、`.contiguous()`。

---

## 前后章导航

- **上一章**：[第 03 章：分词器：从字节到 Token](./03-tokenizer.md)。解释了本章 `idx` 里的整数是如何从原始文本中产生的，以及 tokenizer 的词表与模型嵌入表为何必须成套匹配。
- **本章**：`idx(B, T)` → `wte` → RoPE + 因果多头注意力 → `(B, T, C)` 带上下文的向量 → `lm_head` → `(B, T, V)` 下一 token 打分。
- **下一章**：[第 05 章：GPT 的 MLP、残差流与现代改进](./05-gpt-mlp-tricks.md)。从 `Block.forward` 继续：把 attention 的输出接到 MLP、理解残差连接的角色、以及 value embedding / smear / backout 这些 nanochat 独有的改动。
