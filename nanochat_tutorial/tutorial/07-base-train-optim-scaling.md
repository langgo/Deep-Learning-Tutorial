# 第 07 章　基础模型训练：优化器、精度与规模法则

> **本章定位**　第二部分·基础模型。承接[第 06 章](./06-base-train-loop.md)的反向传播，打开紧接着的 `optimizer.step()`，解释梯度怎样真正变成参数更新。
>
> **中心问题**　为什么不能总是直接执行 `θ -= lr * g`？AdamW、Muon、学习率调度、低精度和多卡通信分别解决什么问题？模型变大后，训练 batch、学习率和 token 预算又该怎样联动？
>
> **范围**　优化器主线（SGD、momentum、AdamW、Muon）、参数分组、学习率 / momentum / weight decay 调度、bf16 / fp16 / FP8、多卡通信和规模法则。loss 与梯度的产生过程在[第 06 章](./06-base-train-loop.md)，训练结果的评估在[第 08 章](./08-base-evaluation.md)。
>
> **先决条件**　读过[第 06 章](./06-base-train-loop.md)，知道 `.backward()` 会把梯度写入参数的 `.grad`。
>
> **本章产物**　一条“**梯度 → AdamW 或 Muon → 参数更新**”的主干，以及围绕它展开的三条工程分支：精度、多卡和训练规模。
>
> **源码入口**　`scripts/base_train.py`、`nanochat/gpt.py::GPT.setup_optimizer`、`nanochat/optim.py`、`nanochat/common.py`、`nanochat/fp8.py`。

---

## 本章讲什么

第 06 章结束时，训练循环的核心是：

```python
loss = model(x, y)
loss.backward()       # 为每个参数得到梯度 p.grad
optimizer.step()      # ← 本章要打开它
model.zero_grad()
```

现在我们已经知道“参数应该往哪个方向改”，但还不知道三个问题：

1. **怎样更新才稳定**：当前 minibatch 的梯度有噪声，不同参数的梯度尺度也不同；
2. **怎样更新才高效**：模型可能使用低精度计算，也可能分布在多张 GPU 上；
3. **怎样把训练配方扩展到不同规模**：模型变深、变宽后，batch、学习率和训练 token 数不能完全不变。

`optimizer.step()` 看起来只有一行，nanochat 实际会沿下面这条路径工作：

```text
梯度 g
  │
  ├─ 1. 按参数职责选择 AdamW 或 Muon
  ├─ 2. 读取当前 step 的 lr / momentum / weight decay
  ├─ 3. 多卡时聚合并分片梯度
  ├─ 4. 根据梯度和优化器状态计算更新量 Δθ
  └─ 5. 写回参数，并让所有 rank 重新得到一致参数
```

本章先讲这条**参数更新主干**，再展开三条分支：

- **精度分支**：bf16、fp16、GradScaler 和 FP8 决定用什么精度存储与计算；
- **多卡分支**：reduce-scatter 和 all-gather 决定梯度与优化器状态怎样跨卡协作；
- **规模分支**：`--depth` 通过参数量联动 token 预算、batch、学习率和训练步数。

**读完本章你应该能**：

1. 讲清 `SGD → momentum → AdamW` 每一步解决了什么问题；
2. 解释 Muon 中 `G_t → M_t → UVᵀ` 的含义，并说明被处理的是更新矩阵而不是参数矩阵；
3. 说明 nanochat 为什么让 embedding、输出层和标量走 AdamW，让 Transformer Block 内矩阵走 Muon；
4. 读懂 lr、Muon momentum 和 Muon weight decay 的调度；
5. 区分参数存储精度、矩阵乘精度和优化器内部计算精度；
6. 解释 nanochat 为什么不是 DDP，以及它的 ZeRO-2 式分片发生在哪里；
7. 沿 `--depth → P → D → B_tok → lr / WD → steps` 讲清训练规模如何联动。

> **读法建议**：第一遍读 §1–§3 和小结，先抓住“梯度怎样变成更新”；第二遍再读 §4–§6 的低精度、多卡与规模法则。

---

## 预备词汇

| 符号 | 含义 |
|---|---|
| `θ_t` | 完成前 `t` 次更新后的参数，也是第 `t` 次更新的输入；可以是标量、向量或矩阵 |
| `g_t` | 在参数 `θ_t` 上，由当前 minibatch 计算出的梯度 |
| `m_t` | 梯度的一阶历史统计，即 momentum |
| `v_t` | 梯度平方的二阶历史统计 |
| `η_t` | 当前学习率（lr） |
| `λ_t` | 当前权重衰减系数（weight decay） |
| `B_tok` | **每步吃多少数据**：一次 `optimizer.step()` 累计处理的全局 token 数 |
| `D` | **整场训练吃多少数据**：从开始到结束处理的 token 次数；重复读到的 token 也重复计数，不等于数据集中的唯一 token 数 |
| `P` | **模型有多大**：规模法则按特定口径统计的参数个数；当前口径只计 Transformer 矩阵和 `lm_head`，不是模型总参数量，详见 §6.2 |

“batch size”必须同时说明单位。第 06 章形状 `(B, T)` 中的 `B` 是**单次 forward 的序列条数**；本章规模公式中的 `B_tok` 是**一次参数更新累计的全局 token 数**。nanochat 源码把后者命名为 `total_batch_size`。两者关系见 §6.5。

本章的 **step** 始终指一次 `optimizer.step()`，不是梯度累积中的 micro-step。

---

## 1. 核心主线：从 SGD 到 AdamW 和 Muon

### 1.1 SGD：沿当前梯度走一步

最基本的更新规则是：

```text
θ_{t+1} = θ_t - η_t · g_t
```

例如：

```text
θ_t = 10
g_t = 2
η_t = 0.1

θ_{t+1} = 10 - 0.1 × 2 = 9.8
```

这个规则只回答了两个问题：

- **往哪走**：梯度的反方向；
- **走多远**：学习率乘以梯度大小。

它没有利用历史信息，也没有区分不同坐标的典型梯度尺度。因此真实训练中会遇到三类问题：

| 问题 | 表现 | 后续机制 |
|---|---|---|
| minibatch 梯度有噪声 | 相邻 step 的方向来回抖动 | momentum |
| 不同坐标的梯度尺度不同 | 同一个 lr 对一些坐标太大、另一些太小 | Adam |
| 参数范数缺少约束 | 权重可能在平坦或尺度冗余方向持续漂移 | AdamW 的 weight decay |

二维权重矩阵还有第四个问题：整张矩阵的更新可能被少数奇异方向支配。Muon 专门处理这一层矩阵几何，见 §2.3。

### 1.2 Momentum：不要只相信当前 batch

当前 minibatch 的梯度可以写成：

```text
g_t = 总体趋势 + 当前 batch 的采样噪声
```

momentum 不直接使用 `g_t`，而是维护历史梯度的指数滑动平均：

```text
m_t = μ · m_{t-1} + (1 - μ) · g_t
θ_{t+1} = θ_t - η_t · m_t
```

其中 `μ` 接近 1 时，历史占比更大。

看一个数值例子。假设上一时刻的平均方向是 `m_{t-1}=0.4`，当前梯度突然变成 `g_t=1.0`，`μ=0.9`：

```text
m_t = 0.9 × 0.4 + 0.1 × 1.0 = 0.46
```

更新使用 `0.46`，而不是立刻完全跟随 `1.0`。如果某个方向连续多步一致，它会保留下来；如果噪声一会儿为正、一会儿为负，就会在平均中部分抵消。

> 有些资料把公式写成 `m_t = μm_{t-1} + g_t`，省略 `(1-μ)`。两种写法只是缩放约定不同，学习率要相应调整。本章采用 nanochat `lerp_` 对应的指数滑动平均写法。

### 1.3 Adam：给每个坐标一把自己的“步长尺子”

momentum 只平滑方向，还没有解决坐标尺度不同的问题。Adam 同时维护：

```text
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²
```

忽略偏差校正时，更新方向近似为：

```text
AdamDirection = m_t / (√v_t + ε)
```

这里：

- `m_t` 回答“最近总体往哪走”；
- `√v_t` 回答“这个坐标最近的梯度通常有多大”；
- 两者相除后，大梯度坐标的步子被压小，小梯度坐标的步子被相对放大。

一个刻意简化的例子：若两个坐标长期梯度分别约为 `10` 和 `0.1`，那么二阶统计约为 `100` 和 `0.01`：

```text
坐标 A：10  / √100  ≈ 1
坐标 B：0.1 / √0.01 ≈ 1
```

原本相差 100 倍的梯度，经尺度归一化后可以得到相近量级的更新。

#### 为什么 Adam 要做偏差校正

Adam 把 `m_0` 和 `v_0` 初始化为 0。训练刚开始时，滑动平均里真正观测到的梯度还很少，其余权重实际上仍落在初始值 0 上，因此 `m_t` 和 `v_t` 会被系统性地拉向 0。这是**零初始化造成的偏差**，不是说当前梯度本身有偏。

以一阶统计为例，把递推式展开：

```text
m_t = (1 - β₁) · (g_t + β₁g_{t-1} + ... + β₁^{t-1}g_1)
```

这些已观测梯度的权重之和不是 1，而是：

```text
(1 - β₁) · (1 + β₁ + ... + β₁^{t-1}) = 1 - β₁^t
```

所以训练初期的 `m_t` 只拥有完整滑动平均的 `1-β₁^t` 倍。除以这个权重和，就把它恢复到正常尺度：

```text
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
```

例如梯度始终为 `g=2`、`β₁=0.9`：

```text
m_1 = 0.9 × 0 + 0.1 × 2 = 0.2
m̂_1 = 0.2 / (1 - 0.9) = 2
```

`0.2` 偏小不是因为真实趋势只有 `0.2`，而是第一步只积累了 `10%` 的权重。二阶统计同理：如果 `β₂=0.999`，第一步只有 `0.1%` 的梯度平方进入 `v_1`，因此尤其需要校正。

随着 `t` 增大，`β₁^t` 和 `β₂^t` 都趋近 0，校正因子趋近 1，偏差校正便几乎不再产生影响。严格地说，“无偏”依赖梯度统计近似平稳等假设；在实际 Adam 中，它主要用于消除训练初期明确的零初始化缩小效应。

最终更新为：

```text
θ_{t+1} = θ_t - η_t · m̂_t / (√v̂_t + ε)
```

### 1.4 AdamW：把 weight decay 从自适应梯度中分离

如果在 loss 中加入 L2 正则 `(λ/2)‖θ‖²`，梯度会多出 `λθ`：

```text
g_t ← g_t + λθ_t
```

在普通 SGD 中，这和每步按比例缩小参数可以得到相同的单步形式：`θ_{t+1} = θ_t - η(g_t + λθ_t) = (1 - ηλ)θ_t - ηg_t`。也就是说，L2 项 `λθ_t` 在普通 SGD 里恰好等价于先把参数乘以 `1 - ηλ`，再执行梯度更新。但在 Adam 中，`λθ_t` 也会进入 `m_t / √v_t` 的自适应缩放，不再等于“所有坐标都按同一比例收缩”。

AdamW 将两件事明确拆开：

```text
① θ_t ← (1 - η_tλ_t) · θ_t      # 独立缩小参数
② θ_t ← θ_t - η_t · AdamDirection
```

继续用 `θ=10, η=0.1, λ=0.01`：

```text
先 decay：10 × (1 - 0.1 × 0.01) = 9.99
若 AdamDirection 简化为 2：9.99 - 0.1 × 2 = 9.79
```

nanochat 的融合实现正是这个顺序：

```python
# nanochat/optim.py::adamw_step_fused
p32.mul_(1 - lr_t * wd_t)
# 更新一阶/二阶统计、偏差校正，再应用 Adam 更新
```

到这里可以先记住一条主线：

```text
SGD        只看当前梯度
  + momentum  利用历史，减少方向抖动
  + Adam      按坐标归一化典型梯度尺度
  + AdamW     再把参数收缩从自适应梯度中分离
```

Muon 不是简单替代这条链。它面向二维矩阵，在 momentum 之后额外处理更新矩阵的几何结构。

---

## 2. nanochat 的真实选择：AdamW 和 Muon 分工

### 2.1 先看结论：一套模型，两条更新路线

`GPT.setup_optimizer()` 没有让所有参数共用一种优化器：

```text
模型参数
  ├─ embedding、输出 head、控制标量 ──→ AdamW
  └─ Transformer Block 内的矩阵参数 ─→ Muon
```

具体分组如下：

| 路线 | 参数 | 为什么这样分 |
|---|---|---|
| **AdamW** | `wte`、`value_embeds`、`lm_head`、`resid_lambdas`、`x0_lambdas`、smear/backout 参数 | embedding、最终输出层和标量更适合标准自适应优化器 |
| **Muon** | `transformer.h` 内 attention、MLP 等参数 | 这些参数主要是参与 token 变换的二维矩阵，可利用矩阵更新的几何结构 |

这不是“二维就用 Muon，一维就用 AdamW”的通用规则：

- `wte` 和 `lm_head` 也是二维张量，但明确走 AdamW；
- `transformer.h` 内的参数按模块归属进入 Muon 组；
- 这是当前模型和优化器实现的工程决策，不是由 `p.ndim` 自动推出的定理。

### 2.2 AdamW 路线：不同职责使用不同基础学习率

`base_train.py` 先按全局 batch 缩放 CLI 学习率，再传给 `GPT.setup_optimizer()`。这里的 CLI 学习率是以参考全局 batch `B_tok,ref = 524,288 tokens` 调好的基础值；实际全局 batch 变大时，梯度通常更稳定，因此代码按经验规则 `lr ∝ √B_tok` 放大学习率，变小时则相应缩小。这个平方根缩放是工程配方，并非数学定律；源码也注明，将它用于 Muon 仍是尚未充分验证的假设。

随后，`GPT.setup_optimizer()` 又对大型 AdamW 参数应用宽度缩放：

```text
s_batch = √(B_tok / B_tok,ref)
s_width = (model_dim / 768)^(-1/2)
```

因此常规训练入口下，各组的初始 lr 为：

| 参数组 | 初始 lr（尚未乘训练中的全局 lr multiplier） | betas | weight decay |
|---|---|---|---:|
| `lm_head` | `unembedding_lr × s_batch × s_width` | `(0.8, 0.96)` | 0.01 |
| `wte` | `embedding_lr × s_batch × s_width` | `(0.8, 0.995)` | 0.001 |
| `value_embeds` | `embedding_lr × s_batch × s_width × 0.5` | `(0.8, 0.995)` | 0.01 |
| `resid_lambdas` | `scalar_lr × s_batch × 0.01` | `(0.8, 0.95)` | 0.05 |
| `x0_lambdas` | `scalar_lr × s_batch` | `(0.96, 0.95)` | 0 |
| smear/backout 参数 | 固定 `0.2` | `(0.8, 0.95)` | 0 |

这张表最重要的阅读方法不是记住每个数字，而是分清三层：

```text
CLI 基础 lr
  × batch 缩放
  × 参数组内部缩放（例如 width、0.5、0.01）
  × 当前 step 的 lr multiplier
  = 当前真正使用的 lr
```

例如 CLI 的 `--embedding-lr=0.3` 不是 `wte` 最终使用的学习率。只盯着任意一个默认值都会看错。

### 2.3 Muon 路线：处理更新矩阵，而不是正交化权重

先只看一张二维参数矩阵 `W_t`。反向传播得到同形状的梯度矩阵 `G_t`，momentum 再把当前梯度与历史梯度平滑为候选更新矩阵 `M_t`：

```text
M_t = μ · M_{t-1} + (1 - μ) · G_t
```

三者的角色不同：

```text
W_t：模型当前保存的参数矩阵
G_t：当前 minibatch 计算出的梯度矩阵
M_t：融合历史梯度后，这一步准备采用的更新矩阵
```

普通 momentum 会直接执行近似如下的更新：

```text
W_{t+1} = W_t - η_t · M_t
```

Muon 认为，对二维矩阵来说，只平滑 `M_t` 还不够：`M_t` 可能被少数矩阵变化模式支配。因此它在真正更新 `W_t` 前，多做一步矩阵正交化：

```text
当前梯度 G_t
  → momentum / Nesterov 混合得到 M_t
  → 将 M_t 各个非零奇异方向的强度压到相近量级
  → 得到处理后的更新矩阵 O_t
  → W_{t+1} = W_t - η_t · O_t
```

理想情况下，若：

```text
M_t = U · Σ · Vᵀ
```

Muon 希望使用它的极分解因子：

```text
O_t = U · Vᵀ
```

也就是保留 `M_t` 的主要矩阵变化模式，但去掉原奇异值之间悬殊的尺度差异。

需要特别区分：

- 被处理的是**本次更新矩阵 `M_t`**，不是参数矩阵 `W_t`；
- Muon 不会强迫训练后的权重始终保持正交；
- “压平奇异值有利于优化”是 Muon 的设计假设和经验结果，不是 SVD 本身给出的定理。

#### 2.3.1 补充：SVD 中的“方向”和“强度”是什么

设 `M` 是一张 `m × n` 矩阵，秩为 `r`。使用紧致形式的奇异值分解（SVD）：

```text
M = U · Σ · Vᵀ
```

其中：

```text
U：m × r，每一列是单位向量 u_i
Σ：r × r，对角线上是非负奇异值 σ_i
V：n × r，每一列是单位向量 v_i
```

把矩阵乘法展开，可以得到更直观的形式：

```text
M = σ₁u₁v₁ᵀ + σ₂u₂v₂ᵀ + ... + σᵣuᵣvᵣᵀ
```

这里准确的说法不是“`U` 是方向、`V` 也是方向”，而是：

- 每一对 `u_i`、`v_i` 共同定义一个**矩阵方向** `u_i v_iᵀ`；
- `σ_i` 表示原矩阵沿这个方向的**强度**。

##### 矩阵方向为什么是一张矩阵

对普通向量，可以写成“大小 × 单位方向”。矩阵也类似，只是它的一个方向本身是一张标准化矩阵。

例如：

```text
u = [1, 0]ᵀ，v = [1, 0]ᵀ

uvᵀ = [[1, 0],
       [0, 0]]
```

这个矩阵方向表示“只改变左上角元素”。再看一个不与单个元素对齐的方向：

```text
u = (1/√2)[1,  1]ᵀ
v = (1/√2)[1, -1]ᵀ

uvᵀ = [[ 0.5, -0.5],
       [ 0.5, -0.5]]
```

它表示四个元素按固定比例协同变化：第一列一起增加，第二列一起减小。因此这里的“方向”是**整张矩阵中各元素协同变化的一种模式**。

##### `U` 和 `V` 各自有什么含义

把矩阵看成线性变换 `y=Mx`。单个 SVD 分量作用于输入 `x` 时：

```text
σ_i u_i v_iᵀx = σ_i u_i · (v_iᵀx)
```

可以分成三步：

```text
1. v_iᵀx：取出输入在 v_i 方向上的分量
2. × σ_i：将这个分量放大或缩小
3. × u_i：把结果放到输出的 u_i 方向上
```

所以 `v_i` 是输入空间中的方向，`u_i` 是与之配对的输出空间方向；`v_i → u_i` 这一对映射共同构成一个矩阵变化模式。

##### 为什么 `σ_i` 是强度

`u_i` 和 `v_i` 都是单位向量，因此 `u_i v_iᵀ` 已经是标准化的方向；前面的 `σ_i` 只控制这个方向在 `M` 中占多大分量。并且有：

```text
M v_i = σ_i u_i
```

也就是说，把单位输入 `v_i` 送入 `M`，输出方向是 `u_i`，输出长度是 `σ_i`。所以 `σ_i` 可以直接理解为该方向的放大倍数或更新强度。

##### 为什么示例里有多个方向

方向数量不是固定为 3。`m × n` 矩阵最多有 `min(m,n)` 个奇异方向，其中非零奇异值的数量等于矩阵的秩：

```text
2 × 2 矩阵      最多 2 个方向
3 × 5 矩阵      最多 3 个方向
768 × 3072 矩阵 最多 768 个方向
```

例如：

```text
M = [[100, 0],
     [  0, 2]]
  = 100u₁v₁ᵀ + 2u₂v₂ᵀ
```

它有两个矩阵方向，强度分别为 `100` 和 `2`。普通 momentum 保留这个 50 倍的差距；理想化的 Muon 将非零奇异值改为 1：

```text
M = UΣVᵀ  ──→  UVᵀ = u₁v₁ᵀ + u₂v₂ᵀ
```

对 `(768, 3072)` 这类长方形矩阵，`UVᵀ` 是**半正交矩阵**，不是方阵意义上的正交矩阵。

### 2.4 Polar Express：不用完整 SVD，近似得到 `UVᵀ`

§2.3 描述的是理想目标：做 SVD、丢掉 `Σ` 的尺度差异，再得到 `UVᵀ`。但训练的每一步都有许多大矩阵，逐张执行完整 SVD 代价很高；而 Muon 真正需要的是 `UVᵀ`，并不需要单独保存 `U`、`Σ`、`V`。

nanochat 因此使用 Polar Express 多项式迭代。可以把它直观理解成一个反复执行的“奇异值整形器”：不显式求出 `U`、`Σ`、`V`，而是直接用矩阵乘法把各个非零奇异值推到 1 附近。

#### 第一步：把矩阵缩放到合适范围

先忽略 MuonEq 等附加处理。Polar Express 的输入是 momentum/Nesterov 处理后的更新矩阵 `M`，源码先按 Frobenius 范数归一化：

```text
X₀ = M / (1.01‖M‖F + ε)
```

其中 `‖M‖F` 是矩阵所有元素平方和的平方根。归一化只调整整体尺度，不改变各奇异值之间的比例；`1.01` 和 `ε` 用来留出数值安全余量。

#### 第二步：执行五轮矩阵多项式

对于宽矩阵 `X ∈ R^{m×n}`（`m ≤ n`），每一轮使用一组系数 `(a,b,c)`：

```text
A = XXᵀ
B = bA + cA²
X ← aX + BX
```

展开后就是：

```text
X ← aX + b(XXᵀ)X + c(XXᵀ)²X
```

高矩阵 `m > n` 则从右侧计算：

```text
A = XᵀX
B = bA + cA²
X ← aX + XB
```

也就是 `X ← aX + bX(XᵀX) + cX(XᵀX)²`。两种形式目标相同，只是选择尺寸更小的中间方阵。以 `(768, 3072)` 的宽矩阵为例：

```text
X @ Xᵀ  → (768, 768)
Xᵀ @ X  → (3072, 3072)
```

所以计算 `X @ Xᵀ` 更省计算和显存。

#### 为什么这个多项式能改变奇异值

若当前矩阵为 `X=UΣVᵀ`，对于宽矩阵有：

```text
XXᵀ       = UΣ²Uᵀ
(XXᵀ)X    = UΣ³Vᵀ
(XXᵀ)²X   = UΣ⁵Vᵀ
```

因此一轮更新等价于：

```text
X_new = U(aΣ + bΣ³ + cΣ⁵)Vᵀ
```

也就是说，在这段核心迭代中，`U`、`V` 对应的方向保持不变，每个奇异值 `σ` 独立经过同一个标量函数：

```text
σ_new = aσ + bσ³ + cσ⁵
```

这就是“不做 SVD，却能整形奇异值”的关键。nanochat 连续使用五组不同系数：

| 轮次 | `a` | `b` | `c` |
|---:|---:|---:|---:|
| 1 | 8.1566 | -22.4833 | 15.8788 |
| 2 | 4.0429 | -2.8089 | 0.5000 |
| 3 | 3.8917 | -2.7725 | 0.5061 |
| 4 | 3.2858 | -2.3681 | 0.4645 |
| 5 | 2.3465 | -1.7098 | 0.4232 |

第 `k` 轮把每个奇异值送入 `f_k(σ)=a_kσ+b_kσ³+c_kσ⁵`。这些系数针对五轮组合后的效果设计，中间轮次可能振荡或暂时远离 1，并不要求每一轮都单调变好。

源码的核心计算可以写成：

```python
# X 已按 Frobenius 范数归一化
if X.size(-2) <= X.size(-1):               # 宽矩阵
    for a, b, c in polar_express_coeffs[:5]:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
else:                                       # 高矩阵
    for a, b, c in polar_express_coeffs[:5]:
        A = X.mT @ X
        B = b * A + c * (A @ A)
        X = a * X + X @ B
```

#### 一个具体例子：`M = diag(4, 1)`

取候选更新矩阵：

```text
M = [[4, 0],
     [0, 1]]
```

它的两个奇异方向强度为 4 和 1，相差 4 倍；因为 `U=V=I`，理想目标 `UVᵀ` 就是单位矩阵。

先归一化：

```text
‖M‖F = √(4² + 1²) = √17 ≈ 4.1231
X₀ = M / (1.01 × 4.1231)
   ≈ diag(0.9605, 0.2401)
```

接着将两个奇异值依次送入五轮多项式，结果约为：

| 阶段 | 第一个奇异值 | 第二个奇异值 | 强度比 |
|---|---:|---:|---:|
| 归一化后 | 0.9605 | 0.2401 | 4.000 |
| 第 1 轮 | 0.8929 | 1.6600 | 1.859 |
| 第 2 轮 | 1.8941 | 0.1651 | 11.471 |
| 第 3 轮 | 0.8690 | 0.6302 | 1.379 |
| 第 4 轮 | 1.5315 | 1.5242 | 1.005 |
| 第 5 轮 | 1.0178 | 1.0039 | 1.014 |

这个例子展示了两点：

1. 中间过程不是单调逼近 1——第 2 轮的差距反而暂时增大；
2. 五轮组合后，原本相差 4 倍的奇异值变为约 `1.0178` 和 `1.0039`，最终矩阵已经接近 `UVᵀ=I`。

因此当前输出不是数学上精确的 `UVᵀ`，更准确地说接近：

```text
U · Σ' · Vᵀ
```

其中 `Σ'` 的非零奇异值已被压到相近范围，但不保证严格全部等于 1。源码还叠加了几项实验性配方：

| 处理 | 作用 |
|---|---|
| MuonEq row equilibration | 先均衡行范数，改善进入迭代时的条件数 |
| Muon+ renormalization | 将 Frobenius 范数拉回半正交矩阵应有的量级 |
| NorMuon 式方差缩减 | 缓解正交化后不同神经元更新尺度仍不均匀的问题 |
| cautious weight decay | 只在 `g * W >= 0` 的位置施加衰减 |
| 形状补偿 | 内部 lr 乘 `max(1, m/n)^(1/2)` |

第一次阅读时，可以先跳过这张表，只记住：

```text
2.3 讲“想得到什么”：近似 UVᵀ
2.4 讲“实际怎么算”：用 5 轮 GPU 友好的矩阵乘近似它
```

### 2.5 Stack：批量计算，不会混合不同层

模型中通常有许多同形状矩阵。假设 12 层各有一张 `(768, 3072)` 的 MLP 矩阵，逐张调用 Muon 会反复启动 GPU kernel。nanochat 会先按形状分组，再把这 12 张矩阵堆成一个 batch：

```text
单张矩阵：       (768, 3072)
12 张堆叠后： (12, 768, 3072)
```

这和把多张图片组成 image batch 类似：新增的第一维只表示“有 12 张矩阵”，不会让不同层的元素互相混合。GPU 仍对每张矩阵独立执行 §2.4 的迭代，只是一次批量提交，减少 Python 循环和 kernel 启动开销。

只有形状相同的矩阵才能直接堆叠：

```text
(768, 3072) + (768, 3072)  → 可以 stack
(768, 3072) + (768, 768)   → 不能直接 stack，必须分到不同组
```

完整张量形状为：

```text
stacked_grads           (12, 768, 3072)  # 12 张当前梯度
stacked_params          (12, 768, 3072)  # 12 张参数
momentum_buffer         (12, 768, 3072)  # 12 张历史动量
second_momentum_buffer  (12,   1, 3072)  # 每张宽矩阵按列保存二阶统计
```

最后一个 buffer 属于 NorMuon 的附加方差缩减，不是理解基本 Muon 的前提。对宽矩阵 `(768, 3072)`，它将每列的统计量沿 768 行汇总，因此每张矩阵只保存 `(1, 3072)`，而不是再保存一张完整的 `(768, 3072)` 二阶状态。

三节可以合起来记成：

```text
2.3：Muon 要做什么      —— 改造更新矩阵的奇异值
2.4：怎样高效做到       —— 用 Polar Express 近似 SVD 目标
2.5：怎样处理许多矩阵   —— 同形状矩阵组成 batch 批量计算
```

### 2.6 把一次 nanochat 更新串起来

现在可以完整描述一个 step：

```text
.backward() 产生每个参数的梯度
  │
  ├─ AdamW 组
  │    梯度 → 一阶/二阶统计 → 解耦 decay → 参数更新
  │
  └─ Muon 组
       同形状梯度堆叠 → momentum → 近似正交化
       → 方差缩减 / cautious decay → 参数更新

两条路线都使用当前 step 的 lr；只有 Muon 组还会被训练循环动态改写 momentum 和 weight decay。
```

---

## 3. 超参数调度：更新规则不变，但力度随训练阶段变化

这里的“调度”是指：随着训练 step 推进，按照预先设定的规则改变学习率、momentum、weight decay 等超参数。优化器回答“怎样根据梯度更新”，超参数调度回答“当前这一阶段应该更新多猛”。每个 step 前，`base_train.py` 会执行：

```python
lrm = get_lr_multiplier(step)
muon_momentum = get_muon_momentum(step)
muon_weight_decay = get_weight_decay(step)

for group in optimizer.param_groups:
    group["lr"] = group["initial_lr"] * lrm
    if group["kind"] == "muon":
        group["momentum"] = muon_momentum
        group["weight_decay"] = muon_weight_decay
```

### 3.1 学习率：warmup → plateau → warmdown

```text
lr multiplier
1.0 ┤      ┌───────────────┐
    │     ╱                 ╲
    │    ╱                   ╲
0.05┤                         └
    └──────────────────────────── step
       warmup      plateau    warmdown
```

默认参数是：

- `--warmup-steps=40`；
- `--warmdown-ratio=0.65`；
- `--final-lr-frac=0.05`。

三个阶段分别解决：

| 阶段 | 做法 | 直觉 |
|---|---|---|
| warmup | 从 `1/warmup` 线性升到 1 | 刚初始化时状态不稳定，先小步启动 |
| plateau | multiplier 保持 1 | 用完整学习率快速学习 |
| warmdown | 线性降到 `final_lr_frac` | 接近预算末尾时减小步长，避免持续大幅震荡 |

例如某参数组的 `initial_lr=0.02`，当前 `lrm=0.25`，真正使用的是：

```text
lr = 0.02 × 0.25 = 0.005
```

一个容易忽略的边界：若总训练只有 1000 步，默认 warmdown 就占 `650` 步，warmup 再占 `40` 步，完整学习率平台只剩约 `310` 步。长训练配方不应原样照搬到很短的实验。

### 3.2 Muon momentum：先增强历史，再在末段减弱惯性

默认调度是：

```text
前 400 步：0.85 → 0.97
中段：     保持 0.97
warmdown： 0.97 → 0.90
```

训练初期先用较低 momentum，避免尚不稳定的早期梯度过度累积；中段提高历史平滑；末段降低到 `0.90`，让更新更快响应当前局部方向。

源码先判断 `step < 400`，再判断是否进入 warmdown。因此对极短训练，前 400 步的 momentum warmup 会优先于末段分支。

### 3.3 Muon weight decay：从初始值余弦降到零

Muon 的 weight decay 使用：

```text
λ_t = λ_0 · (1 + cos(πt/N)) / 2
```

其中 `N` 是总 step 数。它从 `λ_0` 平滑降到 0。

注意两条路线不同：

- **Muon 组**：训练循环每步改写 weight decay；
- **AdamW 组**：表 §2.2 中的 weight decay 系数保持不变，但 lr warmdown 会让实际每步收缩量 `η_tλ` 一起变小。

---

## 4. 精度分支：存什么精度、用什么精度算

低精度部分最容易混淆，是因为“精度”至少涉及三个位置：

```text
参数长期怎样存
    ↓
forward / backward 的矩阵乘怎样算
    ↓
optimizer.step() 内部怎样算并写回
```

### 4.1 nanochat 的常见 bf16 路径

| 位置 | 常见做法 |
|---|---|
| Transformer `Linear` 参数 | 长期保存为 fp32 |
| `Linear` 矩阵乘 | forward 时将权重临时转成输入 dtype，通常是 bf16 |
| `wte` / `value_embeds` | bf16 模式下直接以 bf16 保存，节省显存 |
| AdamW 更新 | 将参数、梯度和状态临时转 fp32 计算，再写回原 dtype |
| Muon 更新 | 状态跟随参数 dtype；主要矩阵乘可用 bf16，范数等敏感计算显式用 fp32 |

`Linear.forward` 很直接：

```python
def forward(self, x):
    return F.linear(x, self.weight.to(dtype=x.dtype))
```

因此“矩阵乘用 bf16”不等于“所有参数和优化器状态都长期存成 bf16”。反过来，“AdamW 内部转 fp32”也不等于存在一份永久的 fp32 embedding 副本：bf16 参数在更新结束后仍写回 bf16。

以 bf16 embedding 为例：

```text
bf16 参数 / 梯度 / Adam 状态
          │
          ▼ 临时转 fp32
    decay + moment + update
          │
          ▼ 写回原 dtype
bf16 参数 / Adam 状态
```

### 4.2 bf16 和 fp16：关键差异是指数范围

| 格式 | 指数位 | 尾数位 | 主要特点 |
|---|---:|---:|---|
| fp32 | 8 | 23 | 范围大、精度高、成本高 |
| bf16 | 8 | 7 | 范围接近 fp32，尾数较短 |
| fp16 | 5 | 10 | 尾数比 bf16 长，但范围明显更窄 |

fp16 的小梯度更容易下溢为 0，大值也更容易溢出。nanochat 只有在显式使用 fp16 时创建 GradScaler：

```python
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
```

GradScaler 的流程是：

```text
loss × S → backward 得到放大的梯度
         → step 前除以 S
         → 若发现 inf / NaN，则跳过这次更新并调整 S
```

它保护的是 fp16 backward 的数值范围，不负责改变 AdamW 或 Muon 的数学，也不会补回低精度格式丢失的尾数。

多卡时，任意一个 rank 发现 inf / NaN，所有 rank 都必须一起跳步，否则参数会分叉。nanochat 对 `found_inf` 做 `all_reduce(MAX)` 来保证一致决定。

### 4.3 FP8：只替换部分 Linear 的矩阵乘

`--fp8` 不会把整套模型参数永久改存为 FP8。它会把满足形状要求的 `nn.Linear` 替换成共享原参数的 `Float8Linear`，让三次主要 GEMM 使用 FP8：

```text
forward:   input @ weight.T
backward:  grad_output @ weight       → grad_input
           grad_output.T @ input      → grad_weight
```

当前实现：

- 要求输入、输出维度都能被 16 整除，且较小维度至少为 128；
- input / weight 使用 `float8_e4m3fn`；
- grad output 使用指数范围更大的 `float8_e5m2`；
- 每个 tensor 使用一个 scale，即 tensorwise scaling；
- 评估时临时换回普通 `Linear`，继续共享同一份 weight / bias。

源码边界也要说清：CLI 虽然列出 `rowwise`，当前 `Float8LinearConfig` 实际只接受 `tensorwise`，传 `rowwise` 会报错。

---

## 5. 多卡分支：通信发生在优化器里

### 5.1 它使用 torchrun，但没有把模型交给 DDP 管理

多卡启动方式是：

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train
```

`torchrun` 负责启动多个 Python 进程，并为每个进程提供 `RANK`、`LOCAL_RANK`、`WORLD_SIZE` 等分布式信息；它本身不等于 DDP。标准 PyTorch DDP 通常还会显式执行：

```python
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
)
```

这一步就是“用 DDP 包装模型”：在原始模型外套一层 `DistributedDataParallel` 对象，由它在 backward 过程中通过 hook 自动聚合各卡梯度。nanochat 没有执行这一步，模型仍是普通的 `GPT` 对象。

因此，多卡时每个进程都会独立执行 forward 和 backward；`.backward()` 刚结束时，每张卡拿到的仍是根据本卡数据计算的**本地梯度**。随后，`MuonAdamW.step()` 自己调用 all-reduce 或 reduce-scatter 聚合梯度，完成本地更新，再用 all-gather 恢复各卡一致的完整参数：

```text
标准 DDP：
本地 backward → DDP 在 backward 中自动 all-reduce → 各卡完整更新

nanochat：
本地 backward → MuonAdamW.step() 自己通信
               → 各卡更新负责的切片或矩阵
               → all-gather 完整参数
```

所以，“没有把模型交给 DDP 管理”不等于没有多卡通信，也不等于各卡最终参数不同；它只表示梯度同步和参数同步由自定义优化器负责，而不是由 `DDP(model)` 自动负责。

### 5.2 AdamW 大参数：reduce-scatter → 本地更新 → all-gather

多卡时，大参数的路径是：

```text
每个 rank 的完整本地梯度
          │
          ▼ reduce_scatter(AVG)
每个 rank 只得到自己负责的梯度切片
          │
          ▼
只更新对应参数切片和 Adam 状态
          │
          ▼ all_gather
每个 rank 重新得到一致的完整参数
```

这样每个 rank 只保存大参数中自己那一片的 `exp_avg` 和 `exp_avg_sq`。小于 1024 个元素的参数没有必要分片，直接 all-reduce 梯度并在每张卡完整更新。

大 AdamW 参数还有一个实现要求：第 0 维必须能被 `world_size` 整除，否则无法按当前方式 reduce-scatter，源码会触发 `assert`。

### 5.3 Muon 参数：先按形状堆叠，再按矩阵数量分片

Muon 已经把同形状矩阵堆成 `(K, m, n)`。多卡时：

```text
K 张同形状矩阵
  → 必要时 padding，使 K 能均分给所有 rank
  → reduce_scatter，每个 rank 负责其中若干张矩阵
  → 本地执行 Muon 更新
  → all_gather，恢复完整 stack
  → 复制回各个 Parameter
```

momentum 和二阶状态也只为本 rank 负责的矩阵保存。

### 5.4 为什么称为“ZeRO-2 风格”，而不是标准 ZeRO-2 实现

**ZeRO（Zero Redundancy Optimizer）要解决的问题，是多卡训练中的重复存储。** 如果每张卡都保存完整参数、完整梯度和完整优化器状态，那么这些数据会在所有 rank 上重复，占用大量显存。ZeRO 按阶段逐步消除这些副本：

| Stage | 每个 rank 分片保存什么 | 每个 rank 仍完整保存什么 |
|---|---|---|
| ZeRO-1 | 优化器状态 | 参数、梯度 |
| ZeRO-2 | 优化器状态、梯度 | 参数 |
| ZeRO-3 | 优化器状态、梯度、参数 | 三者都不再长期完整保存 |

nanochat 和 ZeRO-2 的核心思路相同。以一个大型 AdamW 参数为例：

```text
各 rank 完成本地 backward，得到本地完整梯度
  → reduce-scatter 对梯度求平均并切片
  → 每个 rank 只保留并更新自己负责的梯度、参数切片
  → exp_avg / exp_avg_sq 也只为该切片保存
  → all-gather 更新后的参数切片
  → 每个 rank 重新得到完整参数
```

因此，在**参数更新阶段**：

- 梯度按 rank 分片；
- AdamW 状态也按相同参数切片分片；
- 参数在更新后重新 all-gather，所以每个 rank 最终仍保存完整参数。

这正好符合 ZeRO-2 的主要特征：**分片梯度和优化器状态，但不长期分片参数**。

不过，nanochat 不是 DeepSpeed ZeRO-2 等标准框架实现的直接调用或逐步复刻，所以更准确地称为“ZeRO-2 风格”：

1. **完整梯度曾短暂存在。** `.backward()` 刚结束时，每个 rank 仍持有由本地数据产生的完整梯度；进入 `optimizer.step()` 后才执行 reduce-scatter。它没有在 backward 过程中就让完整梯度始终不落地。
2. **小参数不分片。** 少于 1024 个元素的 AdamW 参数直接 all-reduce 梯度，每个 rank 都保存并更新完整参数与状态，因为为这些小张量分片得不偿失。
3. **Muon 使用自己的分片单位。** 它先把同形状矩阵堆成 `(K,m,n)`，再按矩阵数量把 stack 分给各 rank，而不是统一把所有参数展平后按同一种规则切片。
4. **通信由自定义优化器管理。** 这里没有接入一个完整的 ZeRO 运行时；reduce-scatter、局部更新和 all-gather 都写在 `MuonAdamW.step()` 中。

所以“ZeRO-2 风格”表达的是：

> nanochat 采用了 ZeRO-2 的核心去冗余原则，但通信时机、分片单位和小参数策略是项目自己的实现。

它显然不是 ZeRO-3，因为更新完成并 all-gather 后，每个 rank 仍持有一份完整模型参数。

整个 `step()` 又分成三个阶段：先异步启动 reduce，再等待各组梯度并计算更新、启动 gather，最后等待 gather 并写回参数。这样可以让一部分通信和计算重叠。单卡时所有通信分支自然跳过。

---

## 6. 规模法则：把模型大小换算成一套训练计划

前几节都在解释**一次 `optimizer.step()` 内部发生什么**。训练真正开始前，还有另一类问题要回答：模型应该做多大、总共训练多少 token、一次更新使用多少 token，以及模型规模变化后 lr 和 weight decay 是否也要变化。

这里的“规模法则”不是严格的物理定律，而是根据一组实验拟合或选定的经验关系。nanochat 用这些关系把许多配置串起来，让不同深度的模型可以从同一套参考配方外推。为了避免把定义和经验公式混在一起，先看默认路径的全貌：

```text
1. --depth 决定默认的层数、宽度和 head 数
                    ↓
2. 构造模型并统计用于规模法则的参数量 P
                    ↓
3. 按经验比例 R=12，得到目标训练量 D_target = R × P
                    ↓
4. 若未指定全局 batch，则由 D_target 估算 B_tok
                    ↓
5. 根据 B_tok 和 D_target 缩放 lr 与 Muon weight decay
                    ↓
6. 由 D_target ÷ B_tok 得到默认训练步数 N
                    ↓
7. 再把 B_tok 拆到各张卡和梯度累积 micro-step 上
```

这条链描述的是**没有显式覆盖训练预算时的常规路径**。用户传入 `--num-iterations`、`--target-flops` 或 `--total-batch-size` 后，相应环节会被覆盖，见 §6.4。

### 6.1 第一步：由 `--depth` 得到模型形状

默认配置不是“只增加层数、其他都不变”。`build_model_meta(depth)` 还会根据 depth 推导模型宽度和 attention head 数：

```text
model_dim = round_up(depth × aspect_ratio, head_dim)
num_heads = model_dim / head_dim
```

`round_up` 表示向上取整到 `head_dim` 的整数倍。例如：

```text
depth = 24
aspect_ratio = 64
head_dim = 128

model_dim = round_up(24 × 64, 128) = 1536
num_heads = 1536 / 128 = 12
```

因此默认情况下，`--depth` 增大通常意味着模型同时**更深、更宽、head 更多**，参数量也随之增加。`--aspect-ratio` 和 `--head-dim` 都可以显式覆盖，所以这只是仓库的默认模型族，而不是 Transformer 必须遵守的形状关系。

### 6.2 第二步：构造模型后，按统一口径统计 `P`

形状确定后，代码先在 meta device 上构造模型。meta tensor 只有形状，没有真实参数数据，因而可以低成本地统计参数量和估算 FLOPs。

`GPT.num_scaling_params()` 会分别报告：

- 输入 embedding `wte`；
- `value_embeds`；
- 输出投影 `lm_head`；
- Transformer 矩阵；
- 控制标量；
- 总参数量。

但当前规模公式中的 `P` 采用：

```text
P = transformer_matrices + lm_head
```

也就是说，`P` **不是模型总参数量**：它不包含输入 embedding、value embedding 和控制标量。源码选择这个口径，是因为它在当前模型族的 scaling 实验中拟合得更干净。

参数统计口径会直接影响 `D/P`。因此看到“每个参数训练多少 token”时，必须先确认 `P` 包含哪些参数；口径不同，比例数字不能直接横向比较。

### 6.3 第三步：由 `P` 得到默认 token 预算 `D_target`

默认参数是：

```text
R = target_param_data_ratio = 12
D_target = R × P
```

`R=12` 的含义是：按上述 `P` 口径，每个 scaling parameter 计划训练约 12 个 token。例如：

```text
P = 50,000,000
R = 12
D_target = 12 × 50,000,000 = 600,000,000 tokens
```

这里的 `12` 是当前仓库根据自身模型族、数据和评估目标采用的工程默认，不是普适最优值。其他论文即使写着 `20 token/parameter`，也可能采用不同的参数口径、数据和训练目标，不能只比较两个数字。

此时得到的只是**目标 token 预算**，还没有决定每次更新吃多少 token，也没有决定最终训练多少 step。

### 6.4 第四步：选择全局 batch，并确定训练步数

#### 先选择一次更新覆盖多少 token

`B_tok` 是一次 `optimizer.step()` 累计处理的全局 token 数。如果用户显式传入 `--total-batch-size`，代码直接使用它；否则以 depth 12 模型为参考，按经验关系估算：

```text
B_tok / B_tok,ref = (D_target / D_ref)^0.383
B_tok,ref = 524,288 tokens
```

估算结果还会取到最近的 2 的幂，方便硬件高效执行。直觉上，训练规模和数据预算变大时，可以使用更大的全局 batch；但指数 `0.383` 是实验配方，不是 batch size 的定义。

#### 再确定训练多少 step

训练脚本按下面的优先级选择总步数 `N`：

1. 若传入 `--num-iterations`，直接使用指定步数；
2. 否则若传入 `--target-flops`，用总 FLOPs 预算除以每 step 的估算 FLOPs；
3. 否则使用默认的 token/parameter 路径：

```text
N = floor(D_target / B_tok)
```

继续前面的例子，若：

```text
D_target = 600,000,000 tokens
B_tok = 524,288 tokens / step
```

则：

```text
N = floor(600,000,000 / 524,288)
  = 1,144 steps

D_actual = N × B_tok
         = 599,785,472 tokens
```

step 必须是整数，所以实际训练量 `D_actual` 通常与目标值 `D_target` 有少量差异。

这三种训练预算不会同时生效：**显式 step 数优先于 FLOPs 预算，FLOPs 预算又优先于 token/parameter 比例。** 不过 `target_param_data_ratio` 默认仍为 12；如果实验采用固定 step 或固定 FLOPs，最好同时明确哪些自动 batch / weight-decay 推导仍依赖这个默认目标 token 数。

### 6.5 第五步：按 batch 和训练规模缩放 lr 与 Muon weight decay

参考模型是 depth 12，参考全局 batch 为：

```text
B_tok,ref = 524,288 tokens
```

当实际 `B_tok` 改变时，所有 CLI 基础学习率先按平方根规则缩放：

```text
η = η_ref · √(B_tok / B_tok,ref)
```

例如 batch 变成参考值的 4 倍，基础 lr 就乘 `√4=2`。直觉是更大的 batch 通常给出噪声更小的平均梯度，可以尝试更大的步长；这仍是经验规则，而非数学保证。源码也明确指出，将同一规则用于 Muon 尚未经过充分研究。

Muon 的初始 weight decay 还按以下配方缩放：

```text
λ = λ_ref · √(B_tok / B_tok,ref) · (D_ref / D_target)
```

随后训练过程中，§3.3 的余弦调度还会让 Muon weight decay 从这个初始值逐渐降到 0。AdamW 各参数组则使用 §2.2 中自己的固定 decay 系数。

这里有一个实现边界：当前代码在最终的固定 FLOPs horizon 确定**之前**，就根据 `target_param_data_ratio × P` 推导自动 batch 和 Muon weight decay。`runs/scaling_laws.sh` 会把 ratio 设为 `-1`；此时分子、分母的负号在比例中抵消，程序仍能得到正值，但它实际更接近按 `P/P_ref` 外推，而不是按最终 FLOPs 路径产生的真实 token 数外推。做严谨的固定 FLOPs 实验时，需要注意这不是完全一致的推导顺序。

### 6.6 第六步：把全局 `B_tok` 拆到各张卡

最后还要确保目标全局 batch 能由当前硬件配置实现：

```text
B_tok = device_batch_size
      × sequence_length
      × world_size
      × grad_accum_steps
```

其中：

- `device_batch_size`：每张卡一次 forward 处理的序列条数；
- `sequence_length`：每条序列的 token 数；
- `world_size`：参与训练的进程 / GPU 数；
- `grad_accum_steps`：累积多少个 micro-step 后才执行一次 `optimizer.step()`。

例如：

```text
device_batch_size = 16
sequence_length = 2048
world_size = 8
B_tok = 524,288
```

一次跨 8 卡的 micro-step 处理：

```text
16 × 2048 × 8 = 262,144 tokens
```

因此需要：

```text
grad_accum_steps = 524,288 / 262,144 = 2
```

也就是说，每张卡连续做两次 forward/backward、累积梯度，然后所有 rank 才共同执行一次 `optimizer.step()`。降低 `device_batch_size` 可以缓解单卡 OOM，只要相应增加 `grad_accum_steps`，仍可保持同一个全局 `B_tok`。

### 6.7 把整个默认过程串起来

可以用一句话概括：

> `--depth` 先决定模型形状；模型形状决定参数量 `P`；默认配方用 `P` 得到 token 预算 `D_target`，再选择全局 batch `B_tok`、缩放 lr 和 Muon weight decay，并把 `D_target` 换算成总步数 `N`；最后由 GPU 数量、单卡 batch 和梯度累积共同实现这个 `B_tok`。

各个量在默认路径中的职责如下：

| 量 | 回答的问题 | 主要来源 |
|---|---|---|
| `P` | 模型按 scaling 口径有多大？ | 模型形状和参数分组 |
| `D_target` | 计划总共训练多少 token？ | 默认 `12 × P` |
| `B_tok` | 每次参数更新累计多少 token？ | 用户指定或按 `D_target^0.383` 估算 |
| `N` | 一共更新多少次？ | 显式 step、FLOPs 预算，或 `D_target / B_tok` |
| `η`、`λ` | 每一步更新多猛、衰减多强？ | 参考值再按 batch / token 预算缩放 |

这些关系的价值，是让一套参考配方能随模型规模自动外推；它们的风险也在这里：除了形状和计数关系外，大部分指数、比例和参考值都是经验结果，换模型、换数据或换优化器后都需要重新验证。

---

## 7. 怎样读这一章对应的源码

不要从 `optim.py` 第一行直接扎进融合 kernel。按数据流阅读更容易：

1. **`scripts/base_train.py` 的 CLI**：有哪些基础超参数；
2. **规模推导区**：怎样得到 `B_tok`、lr scale、weight decay 和 `num_iterations`；
3. **`GPT.setup_optimizer()`**：每个参数进入 AdamW 还是 Muon，组内超参数是什么；
4. **训练循环的 scheduler 段**：当前 step 怎样覆盖 lr、Muon momentum 和 decay；
5. **`MuonAdamW.step()`**：多卡 reduce、局部更新、gather 的顺序；
6. **`adamw_step_fused()` / `muon_step_fused()`**：最后再看单个更新 kernel 的数学和数值细节。

阅读时也要区分三种性质：

| 层次 | 本章例子 | 改掉后意味着什么 |
|---|---|---|
| 核心原理 | Adam 的二阶统计、Muon 处理矩阵更新 | 换了优化方法 |
| 工程实现 | 融合 kernel、堆叠、异步通信、FP8 GEMM | 主要影响速度和显存 |
| 实验配方 | 参数组 lr、momentum 调度、ratio=12、Muon 附加处理 | 需要用消融实验验证 |

项目做过但未保留的优化实验，以及 MFU / MBU 和 scaling-law 实验方法，放在[性能研究附录](./performance-research.md)中，不再打断本章主线。

---

## 常见误区

1. **“Momentum 的 `M` 是另一个神秘参数。”** 不是。它就是历史梯度滑动平均在二维参数上的矩阵形式。
2. **“AdamW 就是 Adam 加 L2 loss。”** 不是。AdamW 的参数收缩不经过 Adam 的自适应分母。
3. **“Muon 把模型权重正交化。”** 不是。它处理的是本次更新矩阵，参数本身不会被强制保持正交。
4. **“所有二维参数都走 Muon。”** 不是。embedding 和 `lm_head` 也是二维参数，但当前明确走 AdamW。
5. **“CLI 中的 lr 就是最终 lr。”** 不是。还会经过 batch、宽度、参数组和当前 step multiplier 的缩放。
6. **“bf16 计算意味着所有参数和状态都是 bf16。”** 不是。计算 dtype、参数存储 dtype 和优化器内部 dtype 要分开看。
7. **“bf16 也必须开 GradScaler。”** 当前代码只对 fp16 创建 GradScaler；bf16 的指数范围和 fp32 相同。
8. **“用了 torchrun 就是 DDP。”** 不是。这里由优化器自己同步梯度，没有包装 `DDP(model)`。
9. **“ZeRO-2 风格意味着每张卡没有完整参数。”** 不是。更新后的参数会 all-gather，每张卡最终仍持有完整参数。
10. **“12 token/parameter 是普适最优值。”** 不是。它依赖参数口径、模型族、数据、优化器和评估目标。

---

## 小结：抓住一条主干和三条分支

**主干只有一句话**：

> `.backward()` 给出梯度；AdamW 用一阶/二阶统计和解耦衰减更新 embedding、输出层和标量，Muon 用 momentum 加矩阵更新正交化处理 Transformer 内矩阵；超参数调度再决定当前 step 的更新力度。

围绕它有三条工程分支：

- **精度分支**：bf16 / fp16 / FP8 决定矩阵乘和更新以什么精度执行；
- **多卡分支**：优化器内部用 reduce-scatter 和 all-gather 分片状态、同步参数；
- **规模分支**：`--depth` 先决定模型形状和参数量，再联动 token 预算、batch、lr、weight decay 和 step 数。

如果只记流程，可以记成：

```text
训练前：depth → P → D → B_tok → lr / WD → N
每一步：gradient → schedule → AdamW 或 Muon → distributed sync → new parameters
计算中：bf16 / fp16 / FP8 在不改变训练目标的前提下换取速度和显存
```

---

## 练习

1. **手算 AdamW**：给定 `θ=10`、`η=0.1`、`λ=0.01`、简化后的 `AdamDirection=2`，算出更新后参数，并指出 decay 和梯度更新各贡献多少。
2. **解释 Muon 的 M**：用自己的话解释 `G_t → M_t → UVᵀ` 三者分别是什么，以及为什么被近似正交化的不是参数 `θ`。
3. **追踪真实 lr**：选择 `wte`，沿 `base_train.py → GPT.setup_optimizer → scheduler` 写出它从 CLI 基础 lr 到当前 step lr 的完整乘法链。
4. **计算梯度累积**：全局 batch 为 `1,048,576` tokens，8 张卡，每卡 `device_batch_size=8`、序列长度 2048，需要多少个 micro-step？
5. **区分三种 dtype**：以 bf16 embedding 为例，分别写出参数存储、forward 计算、AdamW 内部更新使用的 dtype。
6. **检查短训练调度**：总步数为 200、warmup 为 40、warmdown ratio 为 0.65 时，完整学习率平台有多少步？这个配方可能有什么问题？

---

## 前后章导航

- **上一章**：[第 06 章：基础模型训练——从数据到损失与梯度](./06-base-train-loop.md)。讲到 `.backward()` 填充梯度为止。
- **本章**：解释梯度怎样经优化器、超参数调度、低精度和多卡通信变成参数更新。
- **下一章**：[第 08 章：基础模型评估——BPB、CORE 与生成样本](./08-base-evaluation.md)。检验这些训练选择最终是否真的让模型变好。
