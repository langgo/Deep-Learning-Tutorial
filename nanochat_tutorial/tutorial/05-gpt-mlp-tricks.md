# 第 05 章：GPT MLP / 现代技巧：MLP、残差流与现代改进

> **本章源码地图**：核心实现是 `nanochat/nanochat/gpt.py`；注意力后端在 `nanochat/nanochat/flash_attention.py`；训练从 `nanochat/scripts/base_train.py` 构造 `GPTConfig`、`GPT` 并以 `loss = model(x, y)` 调用；推理的 `nanochat/nanochat/engine.py::Engine.generate` 则将 `KVCache` 传入 `model.forward`。本章接上第 04 章的注意力，走完从 token id 到 logits 的完整路径。

## 学习目标

读完本章，你应该能够：

1. 沿着 `MLP.forward`、`Block.forward` 和 `GPT.forward` 追踪数据，写出关键张量形状；
2. 说明 attention 与 MLP 的分工，理解残差连接为什么形成“残差流”；
3. 分清经典 decoder-only Transformer 的基线，与 nanochat 的 RMSNorm、pre-norm、relu²、untied head 等具体选择；
4. 在源码中定位滑动窗口、value embedding、`resid_lambdas`/`x0_lambdas`、smear、backout、softcap，并知道它们属于需要验证的实验性增量；
5. 理解初始化为什么让深层网络在训练刚开始时接近恒等映射，而不是把随机噪声层层放大。

## 前置概念

请先读完[第 04 章：GPT attention](./04-gpt-attention.md)。本章继续使用以下符号：

| 符号 | 含义 | 源码名称 |
|---|---|---|
| `B` | batch 中的序列数 | `B` |
| `T` | 本次输入的 token 数 | `T` |
| `C` | 模型/嵌入宽度 | `config.n_embd` |
| `H` | query head 数 | `config.n_head` |
| `H_kv` | key/value head 数 | `config.n_kv_head` |
| `D` | 每个头的宽度，`C/H` | `head_dim` |
| `V` | 原始词表大小 | `config.vocab_size` |

输入 `idx` 是整数 token id，形状 `(B,T)`；进入 embedding 后才成为浮点表示 `(B,T,C)`。本章会频繁说“每个位置”：它指固定的 batch 与时间坐标 `(b,t)`，其 `C` 个通道组成一个向量。

先把最小经典 pre-norm GPT 骨架放在眼前；本章后半的现代技巧都是在这条主干上增量加入：

```python
# 教学化简伪代码
x = token_embedding(idx)
for block in blocks:
    x = x + attention(norm(x))
    x = x + mlp(norm(x))
logits = lm_head(norm(x))
```

第一遍只要看懂这五行和张量始终保持 `(B,T,C)`，再认识 nanochat 在哪些位置增加机制即可。

---

## 1. 先立基线：一个 Block 做两种不同的工作

第 04 章的 attention 让一个位置从此前 token 中**取信息**。但它主要是按权重混合 value 向量；模型还需要在当前位置把取回来的信息进行非线性组合。这个职责由 MLP（也常称 FFN）承担。

一个经典的 decoder-only Transformer 层可以写成：

\[
u=x+\operatorname{Attention}(\operatorname{Norm}(x))
\]
\[
y=u+\operatorname{MLP}(\operatorname{Norm}(u))
\]

其中 \(x,u,y\in\mathbb{R}^{B\times T\times C}\)。`Norm` 整理每个 token 向量的尺度；attention 沿序列维度通信；MLP 对每一个位置独立计算；两处加号将新结果写回同一条残差流。

这正是 `nanochat/nanochat/gpt.py::Block.forward` 的主干：

```python
# 仓库真实代码
x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
x = x + self.mlp(norm(x))
return x
```

它采用 **pre-norm**：子层的输入先 norm，子层输出再加到原始 `x`。作为对照，早期常见的 post-norm 近似写作 `norm(x + sublayer(x))`。两者不是简单移动括号；pre-norm 保留了贯穿加法的直接路径，深层网络训练通常更稳定。

这里最重要的分工是：**attention 横向搬运上下文信息，MLP 纵向加工一个位置的通道特征**。MLP 的参数会在所有位置共享，但本次计算中位置 3 的 MLP 不会直接读取位置 2 的数值；跨 token 的交互仍要依靠 attention、smear 或跨层已经写入残差流的内容。

## 2. MLP：扩张、激活、压回

源码的完整 `MLP` 很小：

```python
class MLP(nn.Module):
    def __init__(self, config):
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x
```

省略 batch、位置坐标后，令隐藏宽度为 `4C`，它是：

\[
\operatorname{MLP}(z)=W_2\,[\operatorname{ReLU}(W_1z)]^2.
\]

`W1` 把 `C` 个通道扩为 `4C` 个中间特征，逐元素 relu² 做非线性筛选，`W2` 再压回 `C`，以便和残差相加。两个线性层均为 `bias=False`。如果没有激活函数，两个线性层连乘仍只是一个线性变换；非线性才使“先找出若干特征，再组合特征”成为可能。

### 最小形状实验

下面是**教学化简代码**，只验证线性代数和形状，不替代仓库的 `Linear` 或训练配方：

```python
import torch
import torch.nn.functional as F

B, T, C = 2, 4, 8
x = torch.randn(B, T, C)       # (2, 4, 8)
w1 = torch.randn(4 * C, C)     # Linear(C, 4C) 的 weight
w2 = torch.randn(C, 4 * C)     # Linear(4C, C) 的 weight

h = F.linear(x, w1)            # (2, 4, 32)
h = F.relu(h).square()         # (2, 4, 32)，逐元素
out = F.linear(h, w2)          # (2, 4, 8)
assert out.shape == x.shape
```

`F.linear` 只变换最后一维，故 `(B,T)` 保持不变。`4C` 是这个实现的选择；其他 Transformer 可能用 GELU、SwiGLU 或不同扩张比例。**relu² 是 nanochat 的当前架构选择，不是 Transformer 的定义，也不能脱离完整训练配方断言它必然更好。**

## 3. RMSNorm 与残差流：稳定地反复编辑表示

`gpt.py` 顶部的归一化函数是：

```python
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```

对一个位置的向量 \(z=(z_1,\ldots,z_C)\)，RMSNorm 的直觉公式为：

\[
\operatorname{RMS}(z)=\sqrt{\frac1C\sum_{j=1}^{C}z_j^2+\epsilon},\qquad
\operatorname{RMSNorm}(z)=\frac{z}{\operatorname{RMS}(z)}.
\]

\(C\) 是通道数，\(\epsilon\) 防止除零，由 PyTorch 实现处理。和 LayerNorm 相比，它不先减均值；更关键的是，当前调用没有传入可学习的 weight 或 bias，因此这里是**无可学习参数的 RMSNorm**。不要把其他模型的 norm 参数想象到本仓库 checkpoint 中。

残差连接 `x + update` 不是“跳过该层”，而是让每层在一块持续存在的工作内存上写增量。这样浅层信息不会被强制覆盖，梯度也存在一条直接的恒等通路。RMSNorm 让每次准备写入 attention/MLP 前的向量长度更可控，二者共同服务于深层训练稳定性。

nanochat 还增加了两组逐层标量：

```python
x0 = x
for i, block in enumerate(self.transformer.h):
    x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
    ...
    x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
```

- `resid_lambdas[i]` 缩放当前残差流；
- `x0_lambdas[i]` 将初始、已 norm 的 embedding `x0` 再混回第 i 层；
- 二者都是形状 `(n_layer,)` 的可学习标量，而非每个通道一组参数。

`init_weights()` 将前者从早层约 `1.15` 线性降到深层约 `1.05`，后者从约 `0.20` 降到约 `0.05`；单层时通过 `max(n_layer - 1, 1)` 避免除零。这是对残差流的**实验性增强**：它给模型额外的可学习捷径，不应误当作经典 Block 的必要公式。

注意同一 `norm` 还用于 Q、K（QK-norm），但此时最后一维是 `D` 而不是 `C`；第 04 章已解释其对 attention 分数尺度的作用。

## 4. 完整 `GPT.forward`：从 id 到下一个 token 的 logits

### 4.1 embedding、词表 padding 与 untied head

在 `GPT.__init__` 中，核心模块是：

```python
"wte": nn.Embedding(padded_vocab_size, config.n_embd)
"h": nn.ModuleList([Block(config, layer_idx) ...])
self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
```

`wte(idx)` 将 `(B,T)` 的 id 查表为 `(B,T,C)`。构造器会把词表大小补到 `pad_vocab_size_to` 的倍数（默认 `64`），注释说明这是 DDP 与 tensor core 对齐的工程优化；输出 logits 随后会切回 `config.vocab_size`，填充出的行不能成为正常预测类别。

有些语言模型将输入 embedding 与输出分类器权重共享，称 weight tying。这里的 `wte` 和 `lm_head` 是独立模块，称 **untied head**：输入表负责读取 token，输出 head 负责给候选 token 打分。这增加了参数与自由度，但不是放之四海的必选项。

### 4.2 embedding 后的 norm 与 smear

真实 forward 先执行：

```python
x = self.transformer.wte(idx)
x = x.to(COMPUTE_DTYPE)
x = norm(x)
```

然后是 **smear**。训练或无 cache 时，对位置 `t>=1`，源码等价于：

\[
x_t\leftarrow x_t+\lambda_{smear}\,\sigma(g(x_t[:24]))\,x_{t-1}.
\]

其中 `g` 是 `smear_gate = Linear(24, 1, bias=False)`，只读取当前位置前 24 个通道；`sigmoid` 生成门；`smear_lambda` 是一个可学习标量。它把**前一个位置的归一化 embedding**混入当前 token，提供一条廉价的、类似 bigram 的局部通路，并不替代 attention。

带 `kv_cache` 的单 token decode 时，本次 `x` 的长度是 1，无法用 `x[:, :-1]` 找到上一个 token。因此 `GPT.forward` 从 `kv_cache.prev_embedding` 取前一次保存的 embedding，再将当前 embedding 写回该字段。`engine.py::KVCache` 初始化、`reset()` 和 `prefill()` 都维护这个状态。这是保证训练/prefill 与逐 token 解码语义尽量一致的工程细节。

`smear_lambda` 在 `init_weights()` 中初始化为 0，故这个分支刚开始完全关闭；这也说明它是值得单独做消融实验的技巧，而不是基础结构。

### 4.3 Block 循环与 value embedding

循环中每层按键名决定是否查一张额外的表：

```python
ve_table = self.value_embeds[str(i)] if str(i) in self.value_embeds else None
ve = ve_table(idx).to(x.dtype) if ve_table is not None else None
```

`has_ve(layer_idx, n_layer)` 选择与最后一层同奇偶性的交错层，保证最后层包含；因此不是每层都有 `ve`。表的宽度为：

```python
head_dim = n_embd // n_head
kv_dim = n_kv_head * head_dim
```

所以 `ve` 初始形状 `(B,T,kv_dim)`。进入 `CausalSelfAttention.forward` 后，它 reshape 成 `(B,T,H_kv,D)`，并加入普通 value：

\[
V\leftarrow V+\bigl(3\sigma(\operatorname{ve\_gate}(x[:12]))\bigr)\odot VE.
\]

`ve_gate_channels=12`，`ve_gate` 为每个 KV head 产生一个门，故 gate 形状 `(B,T,H_kv)`，范围在 `(0,3)`。这是一种 ResFormer 风格的 **value embedding 实验**：这个名字来自相关研究脉络，不理解论文名称不影响本段；只需知道它额外由 token id 提供内容给 V，不改变 Q/K、RoPE 或因果掩码。它使用 `H_kv` 而非 `H`，因此在 GQA 中同组 query heads 共享这一份 value 增量。

### 4.4 滑动窗口：有选择地少看远处

`GPTConfig.window_pattern` 默认是 `"SSSL"`。`GPT._compute_window_sizes()` 循环铺开该字符串：`L` 使用 `(sequence_len, 0)`，`S` 使用 `ceil(ceil(sequence_len / 4) / 128) × 128` 的 `(short_window, 0)`，且最后一层强制 `L`。例如默认 `sequence_len=2048` 时，短窗口是 **512**，它本来已是 128 的倍数；不要采信源码注释中与实现矛盾的 `2048 -> 768` 示例。

元组左值表示向左可看的范围，右值为 0 保持因果性。短窗口层减少远距离 query-key 比较和 decode 时读取的 KV cache；长窗口层仍建立完整上下文路径。因此它是**架构折中兼性能优化**，而不是纯粹换一个更快 kernel：短层的可见连接确实改变了。

调用链是 `GPT.forward` 取 `self.window_sizes[i]` → `Block.forward` → `CausalSelfAttention.forward` → `flash_attn.flash_attn_func` 或 `flash_attn_with_kvcache`。后端优先用 FA3，不可用时 `flash_attention.py` 转为 PyTorch SDPA 并显式处理滑窗 mask。`base_train.py` 也明确警告：未使用 FA3 且窗口模式不是 `L` 时，SDPA 的训练利用率可能很差。故“滑窗一定更快”不是正确结论，必须连同硬件和后端测试。

### 4.5 backout、最终 norm、head 与 softcap

层循环会缓存：

```python
backout_layer = n_layer // 2
if i == backout_layer:
    x_backout = x
```

所有 Block 后执行：

```python
x = x - self.backout_lambda.to(x.dtype) * x_backout
x = norm(x)
```

这叫 **backout**：在最终 norm 和输出头前减掉中间层表示。`backout_lambda` 是标量，初始化为 `0.2`。源码注释的动机是移除低层特征；它是可学习的实验性假设。请以 `i == n_layer // 2` 这行精确定义为准，不要将“中层”脑补成其他层号。

接着：

```python
logits = self.lm_head(x)                       # (B,T,V_pad)
logits = logits[..., :self.config.vocab_size] # (B,T,V)
logits = logits.float()
softcap = 15
logits = softcap * torch.tanh(logits / softcap)
```

logits 是未归一化分数，不是概率。`softcap=15` 是当前 `forward` 的局部常量，公式 \(s\tanh(l/s)\) 在零附近近似保持 `l`，大数时平滑趋近于 \([-s,s]\)，不同于硬裁剪。这是数值/训练行为的**实验性稳定化技巧**，不是 softmax 或 Transformer 的定义。

给了 `targets` 时，模型把 `(B,T,V)` 展平为 `(B*T,V)`，以 `F.cross_entropy(..., ignore_index=-1, reduction=loss_reduction)` 返回损失；未给 targets 则返回 logits。训练脚本的真实调用是 `nanochat/scripts/base_train.py` 中的 `loss = model(x, y)`。推理的 `Engine.generate` 先用 prompt 做一次 prefill，再反复用形状 `(B,1)` 的新 token 调 `model.forward(ids, kv_cache=...)`，每次取 `[:, -1, :]` 采样。

## 5. 三层分类法：别把“能跑得快”当成“模型定义”

读到这里，代码中同时出现了许多技巧。一个有用的学习方法是把它们放进三层篮子；这能防止改模型时一次性删改太多而无法解释结果。

**第一层：核心原理。** token embedding 将离散 id 变成向量；因果 attention 让位置只能从允许的历史取信息；MLP 做逐位置的非线性变换；残差与 norm 让许多层可以堆叠；最后的线性 head 给词表产生 logits。这些构成了能做 next-token prediction 的 decoder Transformer 骨架。即使把本章的实验支路都删除，仍应保留这个最小模型作为理解与调试的基线。

**第二层：工程优化。** padded vocabulary 的矩阵对齐、自定义 `Linear` 的 dtype 转换、FA3 和 SDPA 的后端选择、KV cache 都主要解决相同数学计算怎样更省显存或更快完成。它们有时会影响浮点舍入、支持的硬件和吞吐，却不应被解释成模型“学会语言”的新机制。尤其 KV cache 并非让模型拥有额外记忆：它缓存的正是此前已经算出的 K/V，避免每生成一个 token 都从头计算 prompt。

**第三层：架构或训练实验。** 滑窗会改可见连接；relu²、untied head、无参数 RMSNorm、QK-norm 是明确的架构选择；value embedding、resid/x0 lambda、smear、backout、softcap 则是这版源码中更具试验色彩的增量。它们可能有效，也可能只在特定模型规模、数据、优化器和硬件配方下有效。正确的实验次序是：先让经典基线的损失与生成行为正常，再一次只启用一项，固定训练 token 数、随机种子和评估方法，记录质量、训练吞吐、decode 吞吐和显存；不能只凭一次 loss 波动就下结论。

这一区分也解释了为什么源码既有 `flash_attention.py` 的 fallback 测试，也把很多技巧写成很短的可学习分支：前者优先保证实现语义，后者提供可被消融的模型假设。对初学者而言，先能在纸上删掉第三层并写出剩余 forward，比记住每个技巧名称更重要。

## 6. 初始化：先让每个 Block 接近恒等

`GPT.__init__` 的注释特别提醒：训练构造会在 PyTorch `meta` device 上运行 `__init__`，那里只有形状、没有真实数据。因此写入真实数值的地方是 `GPT.init_weights()`。实际调用链为：`base_train.py::build_model_meta()` 在 meta 上构造 → `model.to_empty(device=device)` 分配存储 → `model.init_weights()` 初始化。

其中值得直接对照源码的策略是：

- `wte.weight` 正态初始化，`std=0.8`；独立的 `lm_head.weight` 正态初始化，`std=0.001`；
- attention 的 `c_q/c_k/c_v` 用标准差约为 \(1/\sqrt C\) 的 uniform；
- `c_proj.weight` 和 `mlp.c_proj.weight` 都初始化为全零，故 attention/MLP 的最终写回起初为零，Block 起初近似恒等映射；
- `mlp.c_fc.weight` 的 uniform 尺度额外乘 `0.4`；value embedding 与 Q/K/V 同尺度；相关 gate 小正值初始化；
- 上文的 residual 标量、smear 和 backout 也在这里明确设置。

残差与合适尺度是深网可训练的核心问题；但 `0.8`、`0.001`、`0.4`、标量曲线等精确数字是本仓库当前训练配方，属于应通过完整实验验证的部分。不要省略 `init_weights()`，也不要把它们当作所有模型的默认值。

自定义 `Linear.forward` 还有一项工程职责：它把权重转换为输入 dtype 后调用 `F.linear`，而主权重可保留 fp32 供优化器使用；embedding 在满足条件时会转为 `COMPUTE_DTYPE`。这改变的是混合精度的存储/计算安排，不改变这里的矩阵运算含义。

## 7. 把形状一次串完

设教学配置为 `B=2, T=5, C=12, H=3, H_kv=1, V=100`，则 `D=4`、`kv_dim=4`：

| 位置 | 形状 | 含义 |
|---|---:|---|
| `idx` | `(2,5)` | 整数 token id |
| `wte(idx)`、norm、smear 后的 `x` | `(2,5,12)` | 残差流 |
| 某层的 `q` | `(2,5,3,4)` | 3 个 query heads |
| `k`、`v` | `(2,5,1,4)` | 1 个 KV head，供 GQA 共享 |
| 该层的 `ve`（若存在） | `(2,5,4)` → `(2,5,1,4)` | 额外 value embedding |
| attention 投影后、Block 输出 | `(2,5,12)` | 可加回残差流 |
| `MLP.c_fc` 后 | `(2,5,48)` | `4C` 中间宽度 |
| MLP 投影后 | `(2,5,12)` | 回到模型宽度 |
| `lm_head` 与裁剪后 logits | `(2,5,100)` | 每位置对词表的分数 |

带 KV cache 的 decode 中，本次 `T` 可以是 1；RoPE 通过 `T0 = kv_cache.get_pos()` 取正确的 `cos/sin` 片段，attention 从 cache 中读取历史 K/V，最后一层处理后才推进 cache 位置。cache 如何节省自回归生成的重复计算将在第 10 章展开。

## 常见误区

1. **“MLP 不跨 token，所以可有可无。”** attention 搬运信息，MLP 进行位置内非线性计算；两者分工互补。
2. **“残差等于跳过层。”** 它是保留旧表示并加上更新；训练后更新可以很大，恒等路径主要帮助优化。
3. **“RMSNorm 一定有可学习 scale。”** 本仓库的 `norm(x)` 未传 weight/bias，没有这种参数。
4. **“value embedding 就是 `wte`。”** 两者都查 id，但 `wte` 输出 `C` 维初始残差；`value_embeds` 只在部分层输出 `H_kv*D` 并门控加入 V。
5. **“滑窗意味着模型只能记住短窗口。”** 短层不能直接看远处，但强制的最终长层及跨层残差仍可传递长程信息；代价是可见性结构和性能取舍，不是绝对失忆。
6. **“FA3、滑窗、softcap 都是同一种优化。”** FA3/SDPA 是尽量保持语义的后端实现优化；滑窗改变 attention 连接；softcap 改变 logits 的数值变换。它们应分开测量。
7. **“所有源码技巧都是 Transformer 标配。”** `resid/x0 lambda`、smear、value embedding、backout、softcap 都是当前仓库的实验性增量，应从经典基线逐一做消融评估。
8. **“词表 padding 会让额外 token 被采样。”** `lm_head` 先算 padding 是工程对齐，随后立即裁为真实 `vocab_size`。

## 小结

一个 nanochat `Block` 的不变主干是两次 pre-norm 残差更新：attention 在 token 之间取信息，relu² MLP 以 `(B,T,C)→(B,T,4C)→(B,T,C)` 在每个位置加工信息。无参数 RMSNorm、残差流和初始化共同帮助深层训练稳定。

完整 `GPT.forward` 将 embedding、smear、逐层 residual/x0 混合、可选 value embedding、Block、backout、最终 norm、untied `lm_head`、词表裁剪和 logit softcap 串起来。应将核心原理（残差、attention、MLP）与工程优化（padding、dtype、FA3/SDPA）以及实验性速度赛/训练技巧（滑窗、value embedding、lambda、smear、backout、softcap）明确区分。

## 练习

1. 对 `B=4,T=16,C=64,H=4,H_kv=2`，写出 `x`、`q`、`k/v`、有 value embedding 时的 `ve`、MLP 中间层和 logits 的形状。
2. 在独立脚本中实现无参数 RMSNorm，输入随机 `(2,3,8)`，检查每个 `(b,t)` 向量的均方根接近 1；再与“先减均值”的 LayerNorm 比较输出差异。
3. 手算 `sequence_len=2048,n_layer=8,window_pattern="SL"` 的八个窗口。解释为什么最后一个必为长窗口，以及短窗口为何是 512。
4. 从纸上删除 `resid_lambdas`、`x0_lambdas`、smear、value embedding、backout、softcap，写出最小经典 pre-norm GPT 的伪代码。
5. 阅读 `engine.py::KVCache.prefill` 和 `Engine.generate`。说明为何复制 KV cache 时还必须复制 `prev_embedding`，否则哪一项实验性机制会在多样本 decode 中失去上下文。

## 前后章导航

- ← 上一章：[第 04 章：GPT attention：嵌入、注意力与 RoPE](./04-gpt-attention.md)。回顾 Q/K/V、因果掩码、RoPE、QK-norm、GQA 与 attention 后端。
- → 下一章：[第 06 章：Base Train：数据到损失与训练循环](./06-base-train-loop.md)。本章最后得到的 logits 怎样与右移 targets 计算交叉熵、反向传播并更新参数，将在那里展开。
