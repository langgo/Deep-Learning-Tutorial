# 第 05 章　GPT 的 MLP、残差流与现代改进

> **本章定位**　第二部分·基础模型。承接[第 04 章](./04-gpt-attention.md)的注意力，补齐一个 `Block` 的另一半——MLP、残差流、以及 nanochat 独有的那堆小改动。
>
> **中心问题**　`Block.forward` 里除了 attention，还做了什么？`GPT.forward` 从 `wte(idx)` 到 `lm_head` 之间，这一堆 `resid_lambdas`、`x0_lambdas`、smear、backout、value embedding、softcap 分别在什么位置、各自解决什么问题？
>
> **范围**　注意力之后的全部层次。训练目标、交叉熵、反向传播、优化器都留给[第 06 章](./06-base-train-loop.md)。
>
> **先决条件**　读过[第 04 章](./04-gpt-attention.md)。知道 attention 让一个位置从前面的位置拉信息、但只拉**内容**，还没对内容做"加工"。
>
> **本章产物**　一张贯穿全章的形状表：从 `idx(B,T)` 一路跟到 `logits(B,T,V)`；一张三层分类法：把 nanochat 的每个可学习小标量分到"**核心原理 / 架构选择 / 实验性调参**"的哪一层。
>
> **源码入口**　`nanochat/gpt.py`（`Block`、`MLP`、`GPT.forward`、`GPT.init_weights`）、`nanochat/engine.py`（看 `KVCache` 怎么配合 smear）。

---

## 本章讲什么

读完第 04 章之后，一个常见的困惑是：attention 既然能"从前面拉信息"，为什么还需要 MLP？

一个很有用的类比：attention 像**图书馆检索**——它决定"**从哪些位置拿什么**"。但拿回来的只是一堆 value 向量的加权和，相当于一个带权摘要。**模型还需要把这个摘要加工成有用的内部状态**——比如"**这个位置看了前文，判断主语是人；下一个 token 更可能是动词**"。这一步"**加工**"由 MLP 做。

换个说法：**attention 和 MLP 一起才构成一次完整的"思考步"**——横向看前文，纵向加工内容。少了哪一半 Transformer 都不工作。本章就把这"另一半"讲清楚，并顺带把 nanochat 挂在主干上的那堆实验性改动一一归位。

**读完本章你应该能做到**：

1. 顺着 `MLP.forward` → `Block.forward` → `GPT.forward`，写出每一步的张量形状；
2. 讲清 attention 和 MLP 的**分工**，以及残差连接为什么会形成一条"残差流"；
3. 分清**经典 pre-norm 基线**和 nanochat 的具体选择（RMSNorm 无参数、relu²、untied head）；
4. 在源码里一眼找到 **smear、value embedding、resid_lambdas / x0_lambdas、backout、softcap**，并知道它们属于**实验性增量**而非 Transformer 必要机制；
5. 理解为什么 nanochat 的初始化**让每个 Block 刚开始接近恒等映射**，而不是放大随机噪声。

---

## 预备词汇和贯穿例子

接着第 04 章的记号用：

| 符号 | 含义 | 本章例子 |
|---|---|---|
| `B` | batch 中的序列数 | 2 |
| `T` | 本次输入的 token 数 | 5 |
| `C` | 模型宽度 / 嵌入宽度 | 12 |
| `H` | query 头数 | 3 |
| `H_kv` | kv 头数（GQA 时 ≤ H） | 1（本章特意用 GQA 配置演示 value embedding） |
| `D` | 每个头内部宽度 `C/H` | 4 |
| `V` | 词表大小 | 100 |

**本章会反复用 "`每个位置`" 这个说法**：它指 batch 和 time 坐标都固定下来的那一个点 `(b, t)`，这个点上有 `C` 个通道组成一个向量。整个 Block 的工作就是**在残差流上，把这个 C 维向量反复编辑**。

### 一个最简基线：先把 nanochat 的所有花活都删掉

把 nanochat 独有的东西全部删掉之后，一个最经典的 **pre-norm decoder-only Transformer** 就是这 5 行：

```python
# 教学化简伪代码：先把这个结构记牢
x = token_embedding(idx)                       # (B, T, C)
for block in blocks:
    x = x + attention(norm(x))                 # 子层 1：跨 token 拉信息
    x = x + mlp(norm(x))                       # 子层 2：逐位置加工
logits = lm_head(norm(x))                      # (B, T, V)
```

**这 5 行就是本章的主干**。张量全程保持 `(B, T, C)`，一条"残差流"从头流到尾，每层只往上**加增量**。本章后面讲的所有现代改进（smear、value embedding、backout、softcap...）都是在这条主干上**往某个位置加小修改**；你随时可以回到这 5 行作为对照。

---

## 1. 一个 Block 做两种工作

真实的 `nanochat/gpt.py::Block.forward` 骨架：

```python
# 仓库真实代码（简化掉 ve / cos_sin / window_size / kv_cache 的参数展示）
x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
x = x + self.mlp(norm(x))
return x
```

这就是基线主干的两行。它有三个值得特别指出的性质：

**① Pre-norm，不是 post-norm**。写法是 `x + sublayer(norm(x))`，不是 `norm(x + sublayer(x))`。两者**不是简单括号移位**：

- **Pre-norm** 在残差分支上保留显式的恒等项，反向传播时至少有一条直接梯度通路，通常更容易训练深层网络；但总梯度仍包含子层 Jacobian，不是“完全无阻碍”。
- **Post-norm** 把 norm 放在残差相加之后，深层训练通常更依赖初始化、学习率和额外稳定技巧；不能简单概括为 norm 每层把梯度“压缩”。

现代 decoder-only LLM 常采用 pre-norm，但并非所有 Transformer 都如此。

**② Attention 和 MLP 分工**：

- **Attention**：**跨 token 搬运信息**。位置 `t` 从 `0..t` 的 value 向量里取加权和。
- **MLP**：**在一个位置内加工信息**。它**完全不跨 token**——位置 `t` 的 MLP 只处理位置 `t` 自己的 `C` 维向量，看不到任何别的位置。

这就解释了为什么**MLP 的参数在所有位置共享**：每个位置做的是**同一种加工**。类比：图书馆检索（attention）告诉你**该读哪本书**，然后每个位置独立做**同样的笔记整理动作**（MLP）。

**③ “残差流”的概念**：`x` 从输入一路传到最后，每个子层都把输出作为增量加回主干。可以把它想成一本不断修订的草稿：旧状态有直接通路保留，但后续增量可以抵消、旋转或重编码早期信息，因此不能理解成“每层内容都能从最终向量中无损拆回”。这个意象在后面讲 backout、`x0_lambdas` 时还会用到。

---

## 2. MLP：扩张、relu²、压回

### 2.1 看代码：核心就是三行

```python
class MLP(nn.Module):
    def __init__(self, config):
        self.c_fc   = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x
```

去掉 batch 和位置坐标后，一个位置的向量 `z` 走过 MLP 的操作就是：

```
MLP(z) = W2 · [relu(W1 · z)]²        W1: C → 4C,  W2: 4C → C
```

形状上就是**先把 C 维扩张到 4C，做一次非线性，再压回 C**。两个 Linear 都没有 bias。

### 2.2 为什么要"**先扩张再压回**"

直觉上要做的是：**非线性之前先给模型足够多的"中间特征"**。如果 `C=12`，直接在 12 个通道上做 relu 之后再线性，表达力太受限。扩张到 48 之后相当于"**我同时监测 48 种不同的组合特征**"，哪种被 relu 激活取决于当前输入；最后压回 12 维送回残差流。

**如果没有非线性会怎样？** 两个线性层连乘还是线性变换：

```
W2 · (W1 · z) = (W2 · W1) · z
```

对这个 MLP 子层而言，它会和一个单层线性变换等价，扩张到 `4C` 不再带来额外的分段非线性表达力。整个 Transformer 仍含 softmax attention、RMSNorm 等非线性，不能说整网完全退化成线性模型；但 MLP 失去了它最主要的非线性特征加工能力。

### 2.3 为什么是 `relu²` 而不是 ReLU / GELU

| 名字 | 公式 | 哪里用 |
|---|---|---|
| ReLU | `max(0, x)` | 经典 MLP、CNN |
| GELU | `x · Φ(x)`（高斯 CDF） | GPT-2、GPT-3、BERT 等老 Transformer |
| SwiGLU | `(xW₁) ⊙ silu(xW₂)` | Llama、Mistral、Qwen 等现代 Transformer |
| **relu²** | `relu(x)²` | **nanochat 用的** |

`relu²` 对正值是**二次放大**，比普通 ReLU 更能把"确实激活"的特征放大、把"刚过零线"的特征压下去，总体更锐利；相比朴素 SwiGLU，它**少一条门控投影**。如果两者都用 `4C` hidden，relu² 的 MLP 参数量约是 `8C²`，SwiGLU 约是 `12C²`；实际大模型常会把 SwiGLU 的 hidden 宽度调小来对齐参数量。

**但不要把它神化**。它是 nanochat 的当前**架构选择**，不是什么理论定律；换成 GELU 或 SwiGLU 一样能跑，只是训练动力学略有不同。读源码时把它当成"**作者在这个规模、这份数据上试出来的调参**"更安全。

### 2.4 一个最小形状实验

```python
import torch
import torch.nn.functional as F

B, T, C = 2, 4, 8
x  = torch.randn(B, T, C)              # (2, 4, 8)
w1 = torch.randn(4 * C, C)             # Linear(C, 4C) 的 weight
w2 = torch.randn(C, 4 * C)             # Linear(4C, C) 的 weight

h   = F.linear(x, w1)                  # (2, 4, 32)       扩张到 4C
h   = F.relu(h).square()               # (2, 4, 32)       逐元素，形状不变
out = F.linear(h, w2)                  # (2, 4, 8)        压回 C
assert out.shape == x.shape            # 回到残差流宽度
```

关键观察：**`F.linear` 只改最后一维**，`(B, T)` 两维完全不动——这是"**MLP 不跨 token**"在张量操作层面上的直接表现。

---

## 3. RMSNorm 和残差流：怎么"稳定地反复编辑"

### 3.1 norm 到底在做什么

```python
# gpt.py 顶部
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```

对一个位置的 `C` 维向量 `z = (z_1, ..., z_C)`，RMSNorm 做的是**除以均方根**：

```
RMS(z) = sqrt((1/C) · Σ z_j² + ε)
RMSNorm(z) = z / RMS(z)
```

直觉上就是"**把向量的长度拉回标准范围**"——不管这个 `z` 当前的能量是 100 还是 0.01，归一化之后它的 RMS 都接近 1。

**为什么要做这件事**？两个原因：

- Attention 和 MLP 都是矩阵乘法，**输入向量的尺度直接决定输出的尺度**。如果不归一化，`x` 一层层累加会漂到很大的数值，训练不稳定。
- 多层残差网络的表示尺度可能随层数漂移，进而改变 attention / MLP 的输入分布。Pre-norm 在每个子层入口提供受控尺度，但不会把残差流本身永久“重置”，也不能单独保证数值稳定。

**和 LayerNorm 的两个区别**：

- LayerNorm 要**先减均值**，RMSNorm 不减——略省一点计算；
- 本仓库这里的 `norm(x)` **没传入可学习的 `weight` 和 `bias`**，所以是**完全无参数**的 RMSNorm。

> 不要把别处 Transformer（例如 Llama）的 `RMSNorm` 想成有可学习 scale。那种是另一种实现。nanochat 这里就是无参数的。读源码时每次看到 `norm(...)` 都是**同一个函数**。

### 3.2 残差连接不等于"跳过这一层"

读源码的新手容易把 `x = x + sublayer(x)` 读成"**跳过这一层**"。不是——它是：

> **保留 `x`，再加上一个增量**。

这个"加法"不是把旧信息覆盖掉，而是把子层输出当作**增量**叠到原来的 `x` 上。它的关键作用是：

- **浅层信息不被覆盖**：第 1 层学到的特征在第 10 层仍然看得到；
- **梯度有一条直接通路**：对 `x + f(x)`，Jacobian 含恒等项 `I`，因此梯度不必只穿过 `f`；但总 Jacobian 是 `I + J_f`，仍可能受子层、深度和数值尺度影响。

残差、pre-norm 和合适初始化共同改善深网优化，但都不是单独的稳定性保证。

### 3.3 nanochat 加的第一组实验性标量：`resid_lambdas` 和 `x0_lambdas`

看 `GPT.forward` 里 Block 循环的真实写法：

```python
x0 = x
for i, block in enumerate(self.transformer.h):
    x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0   # ← nanochat 加的
    x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
```

两组参数：

- **`resid_lambdas[i]`**：缩放当前残差流 `x` 的系数；
- **`x0_lambdas[i]`**：**把最初那条归一化后的 embedding `x0` 再混回这一层**。

两者都是形状 `(n_layer,)` 的**可学习标量**（不是向量）。初始化在 `init_weights()` 里：

- `resid_lambdas`：从第 0 层约 `1.15` 线性递减到最后一层约 `1.05`；
- `x0_lambdas`：从第 0 层约 `0.20` 线性递减到最后一层约 `0.05`。

**直觉解释**：这是给模型加一条"**捷径**"——每层除了接收上一层的残差流，还能**稍微**看看最初的 token embedding。初始化上让浅层的"捷径"更强、深层的"捷径"更弱——换句话说：**浅层更多地直接看到原始输入，深层则更依赖前面层已经加工过的结果**。

**是不是必要的**？不是。它是实验性增强，相当于给模型额外的残差流旋钮。删掉后可恢复更标准的 pre-norm 结构，但“是否收敛、需要多少步、最终质量如何”都取决于其余初始化和训练配方，必须实测，不能预先断言只会慢一点。

---

## 4. 完整 `GPT.forward`：从 token id 到 logits

现在把前三节的内容对回真实 forward。我按**出现顺序**走一遍 `GPT.forward`，每一节落到一个具体代码片段。

### 4.1 Embedding：token id → `(B, T, C)` 向量

```python
# GPT.__init__
"wte":     nn.Embedding(padded_vocab_size, config.n_embd)
self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
```

**两个工程细节**：

**① 词表 padding**：`padded_vocab_size` 把真实 `vocab_size`（例如 32768）**向上凑到 64 的倍数**（可配）。GPU 的矩阵乘法在对齐边界（64 / 128 的倍数）上吞吐最高——padding 是**纯粹的硬件对齐优化**。padding 出来的那些行在最后会被切掉，**不会**被采样为真实 token。

**② Untied head**：`wte` 和 `lm_head` 是**独立模块**，权重不共享。有些 LLM 做**权重绑定**（weight tying），把两张表的权重设成同一组参数以省参数量。**nanochat 没做**——输入表负责"读 token"，输出 head 负责"给候选 token 打分"，它们各自训练。参数多一些，但自由度更高。详见[第 03 章](./03-tokenizer.md)。

### 4.2 Embedding 之后：先 norm，再 smear

先看真实代码的入口：

```python
x = self.transformer.wte(idx)          # (B, T, C) 整数 → 向量
x = x.to(COMPUTE_DTYPE)                # bf16 / fp32，看 common.py
x = norm(x)                            # 每个 token 的 C 维向量先 RMS 归一化
# ↓ 这里开始是 smear
```

#### smear 想解决什么

标准 GPT 里，位置 `t` 想知道上一个 token `t-1` 的信息，通常要等到第一层 attention：

```text
x_t  → attention → 从 x_{t-1} 拉信息
```

nanochat 额外加了一条更便宜的短路：**在进入第一个 Block 之前，先把上一个位置的 embedding 少量混进当前位置**。

```text
原本：  x_{t-1}      x_t
          │          │
          └── attention 才能交互

smear： x_{t-1} ──少量混入──▶ x_t
          │                  │
          └── 后面再正常走 attention
```

所以 smear 可以理解成一个很轻量的 **bigram 通路**：它让模型在最底层就直接看到“上一个 token 是什么”，而不用等 attention 自己学出这件事。

#### 训练 / prefill 路径：整段序列一次进来

训练时 `x` 的形状是 `(B, T, C)`，整段序列都在。源码用切片直接拿相邻位置：

```python
gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
    self.smear_gate(x[:, 1:, :24])
)
x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
```

逐项看：

| 表达式 | 形状 | 含义 |
|---|---|---|
| `x[:, 1:, :24]` | `(B, T-1, 24)` | 取当前位置 `1..T-1` 的前 24 个通道，用来决定“要不要混上一个 token” |
| `smear_gate(...)` | `(B, T-1, 1)` | 每个位置输出一个门控标量 |
| `sigmoid(...)` | `(B, T-1, 1)` | 把门控压到 `(0, 1)` |
| `smear_lambda` | `(1,)` | 全局可学习强度 |
| `x[:, :-1]` | `(B, T-1, C)` | 上一个位置的 embedding，即 `x_0..x_{T-2}` |
| `gate * x[:, :-1]` | `(B, T-1, C)` | 要混入当前位置的那一小份上文信息 |

写成位置级公式就是：

```text
x'_0 = x_0                                      # 第一个位置没有上一个 token，不动
x'_t = x_t + λ_smear · σ(g(x_t[:24])) · x_{t-1}  # t >= 1
```

注意：门控 `g(...)` 看的是**当前位置** `x_t` 的前 24 个通道，不是看 `x_{t-1}`。意思是：由当前位置决定“我需要从上一个 token 借多少”。

#### 一个小数字例子

假设某个位置的门控是 `0.3`，训练中 `λ_smear` 学成了 `0.5`：

```text
实际混入比例 = 0.5 × 0.3 = 0.15

x'_t = x_t + 0.15 × x_{t-1}
```

也就是说，这不是把上一个 token 复制过来，而是**轻轻加一点**。后面真正的上下文建模仍然靠 attention。

#### decode 路径：一次只进来 1 个 token

推理逐 token 生成时，本次 forward 的 `T = 1`：

```text
x.shape = (B, 1, C)
```

这时没有 `x[:, :-1]`，因为本次输入里根本没有“上一个位置”。但 smear 又需要 `x_{t-1}`。怎么办？源码让 `KVCache` 额外保存一份 `prev_embedding`：

```python
x_pre_smear = kv_cache.prev_embedding
kv_cache.prev_embedding = x[:, -1:, :]

if T > 1:
    ...  # prefill，和训练一样用 x[:, :-1]
elif x_pre_smear is not None:
    gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
        self.smear_gate(x[:, :, :24])
    )
    x = x + gate * x_pre_smear
```

流程是：

```text
第 t-1 步 decode 结束：cache.prev_embedding = x_{t-1}
第 t 步 decode 开始：  读出 x_{t-1}，混进当前 x_t
                     再把 cache.prev_embedding 更新为 x_t，留给下一步
```

这也是为什么[第 10 章](./10-inference-engine.md)说 nanochat 的 KV cache 不只缓存 K/V，还缓存 `prev_embedding`：smear 需要它。

#### smear 的边界

- 它**不跨很远**：只看 `t-1`，不是长程记忆；
- 它**不替代 attention**：后面每层 attention 仍然照常看历史；
- 它**默认关闭**：`smear_lambda` 在 `init_weights()` 里初始化为 0；
- 它**是实验性机制**：训练如果发现有用，`smear_lambda` 会自己长大；如果没用，它可以保持接近 0。

所以最安全的理解是：**smear 是 embedding 后、Block 前的一条可学习 bigram shortcut**。它给模型一个便宜的“看一眼上一个 token”的机会，但不是 Transformer 的必要组成部分。

### 4.3 Block 循环：先看 value embedding 这个额外的小表

```python
for i, block in enumerate(self.transformer.h):
    x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0

    # 查一张"按层单独维护"的额外 embedding 表（不是每层都有）
    ve_table = self.value_embeds[str(i)] if str(i) in self.value_embeds else None
    ve = ve_table(idx).to(x.dtype) if ve_table is not None else None

    x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
```

**value embedding** 是 nanochat 加的另一个实验：**不是**每层都有，由 `has_ve(layer_idx, n_layer)` 决定——它选"**和最后一层同奇偶性的那些层**"，保证最后层有，前面层交错。

表的宽度是 `kv_dim = n_kv_head × head_dim`，所以 `ve` 查出来形状是 `(B, T, kv_dim)`。进 attention 后 reshape 成 `(B, T, H_kv, D)`，然后**加到普通 V 上**：

```
V ← V + ( 3 · σ(ve_gate(x[:12])) ) ⊙ VE
```

- `ve_gate` 只读 `x` 的前 12 个通道，输出每个 kv 头一个门；
- `σ` 后乘 3，让门的范围变成 `(0, 3)`——比普通 sigmoid 范围大，允许更强的注入；
- ⊙ 是逐元素乘。

**直觉**：这给 attention 的 V 加了一条由 token id 直接决定的额外内容，绕过“token id → 残差流 → V”的完整路径。普通 V 来自当前上下文化表示；value embedding 更像直接从 token id 查一张层专属卡片，再由当前 `x` 决定注入强度。

**和 GQA 的交互**：ve 用 `H_kv` 而不是 `H`。所以在 GQA 配置下，同组 query head 共享同一个 ve 增量。本章开头的例子特意选了 `H_kv = 1`，就是为了让这个共享关系在形状表里看得见。

**证据边界**：`dev/LOG.md` 记录了 value embedding 在本项目多组速度/规模实验中的明显收益，但这仍是 nanochat 当前规模、数据和配方下的项目内证据，不代表所有 LLM 都应采用同样实现。

### 4.4 滑动窗口和 attention 后端

Block 内部真正做 attention 的还是[第 04 章](./04-gpt-attention.md)的那套 Q/K/V/softmax 流程，但**每层带着一个 `window_size`**：

```python
self.window_sizes = self._compute_window_sizes(config)
```

默认 `window_pattern = "SSSL"`：三层**短窗**（short，约 `sequence_len/4`，再按源码表达式对齐到 128）加一层**长窗**（long = `sequence_len`），循环铺开，最后一层**强制 L**。默认 `sequence_len=2048` 时短窗是 **512**。

**读源码的分界线**（第 04 章讲过，这里只回顾）：

- 滑窗**改变了可见图**（短层只能看 512 个历史）；
- FA3 / SDPA fallback **不改变可见图**，只是两条执行路径。

这两件事别混。详见[第 04 章 §7.2 和 §7.3](./04-gpt-attention.md#7-架构选择与工程优化和核心机制分开看)。

### 4.5 Backout：最后输出前，减掉一部分“中段草稿”

先看它在代码里到底做了什么：

```python
backout_layer = n_layer // 2
x_backout = None

for i, block in enumerate(self.transformer.h):
    ...
    x = block(...)
    if i == backout_layer:
        x_backout = x               # 记住中间那层结束后的残差流

# 所有 Block 都跑完之后
x = x - self.backout_lambda.to(x.dtype) * x_backout
x = norm(x)
```

如果 `n_layer = 12`，那 `backout_layer = 6`。也就是说：

```text
第 0 层 → 第 1 层 → ... → 第 6 层      → ... → 第 11 层 → 输出前
                           │                         │
                           └── 存一份 x_backout       └── 减掉 λ × x_backout
```

写成一句话：**先在网络中间拍一张快照，最后输出前从当前残差流里减掉这张快照的一部分**。

公式是：

```text
x_final ← x_final - λ_backout · x_middle
```

其中：

- `x_middle = x_backout`：中间层的残差流快照；
- `λ_backout`：一个可学习标量，初始化为 `0.2`；
- 最后再 `norm(x_final)`，送进 `lm_head`。

#### 为什么叫 backout

你可以把残差流想成一本草稿：

```text
embedding 草稿 → 浅层改一遍 → 中层改一遍 → 深层改一遍 → 输出
```

中间层的草稿里，可能还带着很多“低层 / 表面”的信息：比如原始 token 形状、局部拼写、词面特征。深层继续加工后，理论上应该把真正对预测有用的高级特征留下来。但残差连接会把许多早期信息一路带到最后。

一种可能的解释是：最终输出前减去部分中段表示，可以抵消某些沿残差流保留下来的中层成分，让输出更突出“后半段网络相对中段新增了什么”。但这只是帮助理解的假设；当前源码只标明在 halfway point 缓存表示，没有给出“remove low-level features”的理论或实验结论。

因此不要把 `x_middle` 直接等同于“低层噪声”。它包含当时已经形成的全部表示，其中既有表面特征，也可能有有用的上下文信息。

#### 一个极简数字例子

假设只有一维，方便看：

```text
x_middle = 10
x_final  = 18
λ_backout = 0.2

backout 后：
x_final' = 18 - 0.2 × 10 = 16
```

它不是把中间层完全删掉，只是减掉一部分。训练后的 `λ_backout` 能说明优化器选择了多大的全局扣除系数，但不能单独证明被减去的成分是“无用信息”；判断收益仍需消融和评估。

#### 它和普通残差连接是什么关系

普通残差连接是：

```text
x ← x + 增量
```

一路把旧信息保留下来。这对训练很有帮助，但也可能让最后的 residual stream 里混着很多早期信息。backout 是一个反向的小动作：

```text
前面一直在加、一直在保留；最后减掉一部分中段快照。
```

所以它不是 Transformer 的必要机制，而是 nanochat 的实验性残差流线性组合；“清理”只是直觉比喻，不是已验证的语义分解。

#### 边界

- 它**不改变 attention 的可见范围**；
- 它**不影响中间层继续计算**，因为扣除发生在所有 Block 之后；
- 它只影响最终送进 `lm_head` 的表示；
- 它是**实验性机制**，不是主流 Transformer 标配；
- 是否有用要靠消融：把 `backout_lambda` 固定为 0，比较 BPB / CORE / chat eval。

### 4.6 输出 head 和 softcap

```python
x       = norm(x)
logits  = self.lm_head(x)                              # (B, T, V_padded)
logits  = logits[..., :self.config.vocab_size]         # 砍掉 padding → (B, T, V)
logits  = logits.float()
softcap = 15
logits  = softcap * torch.tanh(logits / softcap)       # 把 logits 挤进 (-15, 15)
```

**softcap** 的作用[第 04 章 §1](./04-gpt-attention.md#1-gpt-到底在学什么预测下一个-token) 已经讲过——`15 · tanh(l/15)` 把 logits 平滑压到 `(-15, 15)`：

- 小 logit（`|l| < 5`）几乎不动；
- 大 logit（`|l| > 50`）被平滑压近 ±15；
- 不是硬裁剪，一阶可导。

**作用**：限制极端 logits 的有效幅度，减少 softmax 过度饱和和异常大分数带来的数值风险。它不保证训练一定不会卡住；而且 `tanh` 在饱和区自身导数也会变小，所以这是经验性的稳定化取舍。

**同样，这是训练稳定性的实验性选择**，不是 Transformer 定义。常量 `15` 本身也是作者在这个规模上**试出来的**。

### 4.7 两种调用模式

```python
if targets is not None:
    # 训练：返回标量 loss
    loss = F.cross_entropy(
        logits.view(-1, V), targets.view(-1),
        ignore_index=-1, reduction=loss_reduction,
    )
    return loss
return logits   # 推理
```

**两种模式**：

- **训练**：传 `targets`，内部计算 `F.cross_entropy`，返回标量；`ignore_index=-1` 跳过 SFT mask=0 的位置（见[第 03 章](./03-tokenizer.md)、[第 09 章](./09-sft.md)）。
- **推理**：不传 `targets`，直接返回 logits；`Engine.generate` 只取 `logits[:, -1, :]` 做采样（见[第 10 章](./10-inference-engine.md)）。

---

## 5. 三层分类法：核心 / 架构 / 实验

读到这里你应该发现：`gpt.py` 里确实挂了**很多东西**。为了后面读其他源码时不乱，用一张表把它们分到三个篮子里：

### 第一层：核心原理（删掉就不是 Transformer 了）

| 组件 | 作用 |
|---|---|
| Token embedding | 整数 → 可学习向量 |
| 因果 self-attention | 跨 token 搬运信息，受因果 mask 约束 |
| MLP | 位置内非线性加工 |
| 残差 + 归一化 | 让多层堆叠训练稳定 |
| LM head | 产生 `(B, T, V)` 的 logits |

**把这些留下，其他全删，模型仍然能做 next-token prediction**。这是"**最小理解单元**"，建议第一遍只读这些。

### 第二层：工程优化（不改语义，但影响速度和兼容性）

| 组件 | 优化什么 |
|---|---|
| Padded vocabulary | GPU 矩阵乘对齐，不改模型行为 |
| 自定义 `Linear`（dtype 转换） | 混合精度存储和计算的分工 |
| FA3 和 SDPA 后端 | 不同硬件下的 attention 实现 |
| KV cache | 自回归生成的中间状态复用 |

**关键认识**：这些东西**不是"模型学会了什么"**。KV cache 不是"模型有额外记忆"——它只是把前面已经算过的 K/V 缓存起来避免重复算。读源码时如果把它们误读成模型机制，会一直困惑。

### 第三层：架构选择 / 实验性调参（可选，可被消融）

| 组件 | 性质 |
|---|---|
| Pre-norm vs post-norm | 架构选择；nanochat 采用 pre-norm，其他架构可能不同 |
| relu² | 架构选择（相对 SwiGLU / GELU 的替代） |
| 无参数 RMSNorm | 架构选择 |
| **QK norm + `*1.2`** | 第 04 章讲过的训练稳定性调参 |
| **滑动窗口 "SSSL"** | 架构选择，改变可见图 |
| **value embedding** | 实验性增量，只在部分层启用 |
| **`resid_lambdas` / `x0_lambdas`** | 实验性残差流调节 |
| **smear** | 实验性 bigram 捷径，初始为 0 |
| **backout** | 实验性残余移除 |
| **softcap = 15** | 实验性数值稳定化 |
| 具体初始化尺度（`0.8`、`0.001`、`0.4`、`0.2`...） | 实验性调参 |

**实验的性质**：每一项都可能**只在这个规模、这份数据、这个优化器配方下**有效。不要外推成"所有 GPT 都该加 smear"。读[附录 C](./performance-research.md)里讲的消融方法论——"**一次只开一个，固定其他**"——如果你想亲自验证哪些真的有用。

---

## 6. 初始化：让深层 Block 刚开始接近"恒等"

为什么要专门讲初始化？因为**残差流 + 合适的初始化**才是深网能训练起来的秘密。

### 为什么用 `meta` device 构造

`GPT.__init__` 的注释特别提醒：训练启动时模型在 PyTorch 的 **meta device** 上构造——那里**只有形状，没有真实数据**。真正写入数值的地方是 `GPT.init_weights()`。调用链是：

```text
base_train.py::build_model_meta()        # 在 meta device 上构造模型骨架
    → model.to_empty(device=device)      # 分配真实显存，但内容未定义
    → model.init_weights()                # 这里才写入初始化数值
```

好处是：**大模型的构造完全避免 CPU 上的无用分配**，省显存也省时间。

### 关键初始化策略

对照 `init_weights()` 看这几件事：

- **`wte.weight`**：正态初始化，`std = 0.8`——给输入 embedding 一个较大的初始能量；
- **`lm_head.weight`**：正态初始化，`std = 0.001`——**非常小**；
- **`c_q / c_k / c_v`**：uniform，标准差约 `1/√C`；
- **`c_proj.weight`（attention 和 MLP 的输出投影）**：**全零！**
- **`mlp.c_fc.weight`**：uniform 尺度 × `0.4`；
- **`value_embeds`、`ve_gate`**：和 Q/K/V 同尺度 / 小正值；
- **`resid_lambdas` / `x0_lambdas` / `smear_lambda` / `backout_lambda`**：按 §3.3 / §4.2 / §4.5 讲的曲线和初值设置。

**最关键的那条**：`c_proj.weight = 0` 让 **attention 和 MLP 的子层输出一开始全是 0**。那么 Block 内部的两次残差都是 `x + 0 ≈ x`——**每个 Block 一开始接近恒等映射**。严格说，`GPT.forward` 在每个 Block 前还会做 `resid_lambdas[i] * x + x0_lambdas[i] * x0`，所以整网不是把 `wte(idx)` 逐值原封不动地传到底；但由于最后还有 `norm(x)`，初始化阶段的主方向仍主要来自初始 embedding，子层不会一上来注入随机噪声。

**为什么这很重要**：如果初始化时每个子层都向残差流注入较大的随机增量，多层叠加会让初始优化更困难。让输出投影从 0 开始，使 attention/MLP 分支最初不改写主干，再逐渐学出增量，是当前配方的一项稳定化设计；它不是单独的“根本保证”，仍依赖 norm、残差系数、学习率和优化器。

**具体数字（`0.8`、`0.001`、`0.4`、`std=1/√C`...）是当前仓库配方，别外推。** 零初始化残差分支输出在一些现代深网中很常见，但不是所有 Transformer 的统一做法。

### 自定义 `Linear` 的额外职责

`nanochat/gpt.py::Linear` 的 `forward` 会把权重临时转到输入 dtype 之后再做 `F.linear`：

```python
# 简化
def forward(self, x):
    return F.linear(x, self.weight.to(x.dtype))
```

这是**混合精度训练**的工程细节：**权重按 fp32 存储**（给优化器用高精度），**计算按 bf16 进行**（给矩阵乘吞吐）。读源码时看到 `.to(x.dtype)` 就是这个意思——不改变矩阵运算的含义。

---

## 7. 一张形状表把全章串起来

用贯穿全章的小配置 `B=2, T=5, C=12, H=3, H_kv=1, V=100`（于是 `D=4`、`kv_dim=4`）：

| 位置 | 形状 | 含义 |
|---|---|---|
| `idx` | `(2, 5)` | 整数 token id（第 03 章的终点） |
| `wte(idx)` → norm → smear 后的 `x` | `(2, 5, 12)` | 残差流 |
| 某 Block 内部的 `q` | `(2, 5, 3, 4)` | 3 个 query head |
| 该 Block 的 `k`, `v` | `(2, 5, 1, 4)` | 1 个 KV head，GQA 共享 |
| 该 Block 的 `ve`（若该层启用） | `(2, 5, 4)` → `(2, 5, 1, 4)` | 额外 value embedding |
| Attention 合头 + c_proj 后 | `(2, 5, 12)` | 加回残差流 |
| MLP `c_fc` 后 | `(2, 5, 48)` | 扩张到 `4C` |
| MLP `c_proj` 后 | `(2, 5, 12)` | 压回 `C` |
| 最终 norm 后 | `(2, 5, 12)` | 残差流末端 |
| `lm_head` + 裁剪 + softcap 后的 logits | `(2, 5, 100)` | 每个位置对词表的分数 |

**带 KV cache 的 decode**：本次 `T` 可以是 1，RoPE 从 `kv_cache.get_pos()` 偏移位置、attention 从 cache 读历史 K/V、Block 结束才推进 cache 位置。详见[第 10 章](./10-inference-engine.md)。

---

## 常见误区

1. **"MLP 不跨 token，所以不重要。"** 错。attention 搬信息，MLP 加工信息——缺一不可。
2. **"残差 = 跳过这一层。"** 错。它是**保留 `x` 并加增量**；训练后的增量可以很大，恒等路径只是保证梯度和浅层信息传得过去。
3. **"RMSNorm 一定有可学习 scale。"** nanochat 的 `norm(...)` **没传 weight/bias**，是完全无参数的。
4. **"value embedding 就是 `wte` 的别名。"** 错。两者都查 token id，但 `wte` 输出残差流宽度 `C`、每层都读；`value_embeds` 输出 `kv_dim`、只在部分层读、而且 **只加到 V 上**。
5. **"滑窗等于模型失忆。"** 短层确实看不到远处，但**最终层强制 L** 和跨层残差共同保留了长程通路；它是**可见图的局部修改**，不是全局失忆。
6. **"softcap / 滑窗 / FA3 都是同一种优化。"** 它们**解决的不是同一类问题**：softcap 改 logits 的数值变换、滑窗改可见图、FA3 vs SDPA 改执行后端。读源码时要分开讨论。
7. **"所有写在 `gpt.py` 里的技巧都是 Transformer 定义。"** 错。核心只有 §5 第一层那五项；其他都是可被消融的架构选择或实验性调参。
8. **"词表 padding 的那些行可能被采样成 token。"** 不会。`lm_head` 算完就立刻切回真实 `V`，padding 的 logits 根本不进 softmax。

---

## 小结

- **一个 Block 的主干**是两次 pre-norm 残差：`x = x + attn(norm(x))`、`x = x + mlp(norm(x))`。attention 跨 token 搬信息，MLP 逐位置加工。
- **MLP 的套路是"扩张 → 非线性 → 压回"**：`C → 4C → C`，非线性用 `relu²`（nanochat 的选择，不是定律）。
- **残差流**是一本被反复编辑的草稿，每层只加增量。**无参数 RMSNorm** 在每个子层前把能量重置。
- **nanochat 挂在主干上的实验性机制**：embedding 后的 smear、Block 循环内的 `resid_lambdas` / `x0_lambdas` / value embedding、全部 Block 之后的 backout、`lm_head` 之后的 softcap。它们原则上可分别做消融；当前公开日志对 value embedding 有较强项目内证据，对其他项应以实际对照实验为准。smear 的全局系数初始化为 0。
- **初始化的核心思想**：`c_proj.weight = 0` 让每个 Block 的 attention/MLP 输出分支从零增量开始，再逐层学会改写残差流；它与 norm、残差系数、学习率和优化器共同影响稳定性。
- **三层分类表**是本章最该带走的工具：下次读源码看到任何小技巧，先问"**它属于核心 / 工程优化 / 实验性调参**哪一层"。

下一章开始，我们从模型内部跳出来，看**训练循环**：`loss = model(x, y)` 之后的 `loss.backward()`、`optimizer.step()` 做了什么，以及 DDP、梯度累积、学习率调度怎么配合。

---

## 练习

1. **形状追踪**。`B = 4, T = 16, C = 64, H = 4, H_kv = 2`。写出：`x` → 某 Block 内 `q` → `k/v` → 有 ve 时的 `ve`（经 attention reshape 后）→ MLP 中间层 → 最终 logits 的形状。
2. **RMSNorm 手动实现**。写一个无参数 RMSNorm，对随机 `(2, 3, 8)` 的输入验证：每个 `(b, t)` 向量归一化后 RMS 接近 1。再写一个"**先减均值再做 RMS**"的 LayerNorm 版本，用同一输入对比两者输出。
3. **滑窗数字题**。`sequence_len = 2048`、`n_layer = 8`、`window_pattern = "SL"`。写出 8 层各自的 `(left, right)`。为什么**最后一层**必定是 L？短窗为什么等于 **512**（算一遍）？
4. **最小基线复原**。从 nanochat 的 `GPT.forward` 里**删掉**：`smear`、`resid_lambdas`、`x0_lambdas`、`value_embeds`、`backout_lambda`、`softcap`。写出剩下最小经典 pre-norm GPT 的 10 行伪代码。
5. **KV cache + smear 的协作**。读 `engine.py::KVCache.prefill` 和 `Engine.generate`。为什么复制 KV cache 做并行采样时**必须同时复制 `prev_embedding`**？如果忘了会出什么现象（在 smear 这个机制上）？

---

## 前后章导航

- **上一章**：[第 04 章：GPT 注意力：嵌入、因果注意力与 RoPE](./04-gpt-attention.md)。讲清了 Q/K/V、因果 mask、RoPE、GQA 和 FA3/SDPA 后端。
- **本章**：补齐 Block 的另一半——MLP、RMSNorm、残差流、以及 nanochat 挂在主干上的那堆实验性小机制。
- **下一章**：[第 06 章：基础模型训练：从数据到损失与梯度](./06-base-train-loop.md)。本章得到的 logits 怎么和右移目标算交叉熵、怎么反向传播、怎么把 8 张 H100 用起来——完整训练循环展开。
