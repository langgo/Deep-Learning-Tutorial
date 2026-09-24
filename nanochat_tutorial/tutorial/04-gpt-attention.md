# 第 04 章：GPT attention：嵌入、注意力与 RoPE

> **本章位置**：[`03｜Tokenizer：从字节到 Token`](./03-tokenizer.md) → **04｜GPT attention：嵌入、注意力与 RoPE** → [`05｜GPT MLP / 现代技巧：MLP、残差流与现代改进`](./05-gpt-mlp-tricks.md)
>
> **本章源码地图**：`nanochat/nanochat/gpt.py`、`nanochat/nanochat/flash_attention.py`、`nanochat/tests/test_attention_fallback.py`。调用链为：`nanochat/scripts/base_train.py::build_model_meta` 创建 `GPTConfig` 和 `GPT`，训练循环调用 `model(x, y)`；推理时 `nanochat/nanochat/engine.py::Engine.generate` 创建 `KVCache`，调用 `GPT.forward(ids, kv_cache=...)`。检查点加载路径在 `nanochat/nanochat/checkpoint_manager.py::build_model`。

第 03 章的终点是一串 token id，例如 `[17, 42, 9, 301]`。它们只是词表中的地址：17 不比 42 大，也没有“更接近某个词”的含义。本章要沿着真实的 `GPT.forward` 追踪它们：整数如何查表变成向量，向量如何通过因果自注意力读取前文，以及没有位置嵌入表时，RoPE 如何告诉模型 token 的顺序。

先给出全景主线。对输入 `idx`，模型做的是：

```text
idx (B, T) 整数 token id
  → wte 查表
x (B, T, C) token 向量
  → 多层 Block 中的因果注意力
x (B, T, C) 含上下文的向量
  → lm_head
logits (B, T, V) 每个位置对下一 token 的打分
```

本章只聚焦嵌入与注意力；注意力后的 MLP、残差流以及更多 nanochat 增量，留给下一章。

> **第一遍主线路线**：先读 §1–4，掌握 embedding、单头/多头因果注意力和 `B/T/C/H/D` 形状；再读 §6 的 RoPE 直觉。§5 的 GQA、§7 的 QK norm/滑窗/FA3-SDPA 后端属于架构取舍与工程实现，可第二遍再读。

---

## 学习目标

读完后，你应能：

1. 解释 GPT 为什么训练为 **next-token prediction**，并区分 token id、embedding、logit、概率和交叉熵；
2. 在 `GPT.forward` 与 `CausalSelfAttention.forward` 中追踪 `B/T/C/H/D` 张量形状；
3. 从单头注意力理解 Q、K、V、softmax 与因果掩码，再扩展到多头；
4. 说明 `apply_rotary_emb`、`_precompute_rotary_embeddings` 怎样实现 RoPE，以及缓存解码为何需要位置偏移；
5. 分清经典注意力原理、架构取舍（QK norm、GQA、滑窗）和实现优化（FA3/SDPA fallback）。

## 前置概念：形状比名词更重要

假定你已读过第 03 章，并知道 `(B, T)` 是由 token id 组成的矩形张量。以下符号贯穿本章：

| 符号 | 含义 | nanochat 对应符号 |
|---|---|---|
| `B` | batch size，一次并行的序列数 | `B, T = idx.size()` |
| `T` | 当前序列长度 | prompt 阶段常大于 1；逐 token 解码可为 1 |
| `C` | 模型宽度 / embedding 宽度 | `config.n_embd` |
| `H` | query head 数 | `config.n_head` |
| `H_kv` | key/value head 数 | `config.n_kv_head` |
| `D` | 单个 head 宽度，`C / H` | `self.head_dim` |
| `V` | 实际词表大小 | `config.vocab_size` |

`gpt.py::GPTConfig` 的类默认值是 `sequence_len=2048`、`vocab_size=32768`、`n_layer=12`、`n_head=6`、`n_kv_head=6`、`n_embd=768`。这只是 dataclass 的默认配置，不应当把它当作所有训练运行的参数。实际基础训练中，`base_train.py::build_model_meta` 根据 `depth`、`aspect_ratio` 与 `head_dim` 算出宽度和头数，并设置 `n_kv_head=num_heads`，也就是基础训练配置使用普通多头注意力。

还只需记住三件数学工具：线性层改变最后一维；点积衡量两个向量在某个学习到的空间里是否匹配；softmax 把一排任意分数变成非负、和为 1 的权重。

---

## 1. 语言模型的唯一基本作业：预测下一个 token

给一段 token 序列：

```text
[t0, t1, t2, t3, t4]
```

训练时常将它右移一格，形成：

```text
输入 idx: [t0, t1, t2, t3]
目标 y:   [t1, t2, t3, t4]
```

位置 0 只能根据 `t0` 猜 `t1`；位置 2 只能根据 `t0,t1,t2` 猜 `t3`。这样一条序列的所有位置可并行训练，却必须遵守生成时同样的信息条件：未来 token 尚不存在，不能偷看答案。这就是 decoder-only GPT 的“因果”约束。

真实入口是 `gpt.py::GPT.forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean')`：

- `idx` 形状是 `(B, T)`；
- 没有 `targets` 时，返回 `logits`，形状 `(B, T, V)`；
- 有 `targets` 时，调用 `F.cross_entropy` 并返回损失；目标为 `-1` 的位置由 `ignore_index=-1` 忽略。

最后的输出头是：

```python
logits = self.lm_head(x)
logits = logits[..., :self.config.vocab_size]
logits = logits.float()
softcap = 15
logits = softcap * torch.tanh(logits / softcap)
```

`lm_head` 先产生 padded vocabulary 的分数，再切到真实 `V`。padding 是矩阵计算效率优化，不是分词器新增的可生成 token。随后代码转为 `float32`，并以源码常量 `softcap=15` 平滑压缩 logit 幅度。

**logit 不是概率。** 它可以为负，全部 logit 也不要求和为 1。对某位置 `logits[b, t, :]` 做 softmax，才得到词表上“下一个 token”的概率分布；交叉熵会鼓励正确目标 token 有更高概率。模型并不是一次输出完整句子，而是反复完成这个分类任务：取最后位置分布，采一个 id，追加到输入，再预测下一次。

---

## 2. Embedding：从“词表地址”到可计算的向量

神经网络不能从 token id 的数值大小直接理解语言。`301` 与 `302` 恰好相邻，并不表示其语义相邻。第一步是查一张可学习表：

```python
# gpt.py::GPT.__init__
"wte": nn.Embedding(padded_vocab_size, config.n_embd)

# gpt.py::GPT.forward
x = self.transformer.wte(idx)
x = x.to(COMPUTE_DTYPE)
x = norm(x)
```

`wte` 可看成形状 `(V_pad, C)` 的参数表，`idx` 的每个 id 选择其中一行。因此：

\[
\operatorname{Embedding}: (B,T) \longrightarrow (B,T,C).
\]

这是一张查表，不是把整数乘成浮点数。训练会同时调整每行向量和后续网络，使常在相似上下文中起作用的 token 有机会形成可用的表示。

经典 Transformer 常把 token embedding 和**绝对位置 embedding**相加。nanochat 没有位置表：文件开头明确写着 “rotary embeddings (and no positional embeddings)”。它会先把内容向量送入各层，稍后只在注意力的 Q、K 上注入位置。

还有一个容易漏掉的设计：输入 `wte` 与输出 `lm_head` 是两套不同参数，即 *untied weights*。前者负责“读入 token”，后者负责“向词表写出分数”；这里没有把两张表的权重绑定。不要因为有些 GPT 实现会 weight tying，就把它套到本仓库。

`norm(x)` 是 `F.rms_norm(x, (x.size(-1),))`，没有额外可学习缩放参数。此处先把它当作整理每个 token 的通道尺度；它在完整 Transformer 层中的角色会在下一章展开。

---

## 3. 经典基线：单头自注意力究竟算了什么？

只有 embedding 时，一个 token 的向量与上下文无关。“苹果”在“吃苹果”和“苹果公司”中永远是同一行表。自注意力让每个位置提出问题，并从此前位置按需取信息。

先忽略 batch 和多头。令一条序列输入为：

\[
X \in \mathbb{R}^{T\times C}.
\]

三个独立线性投影得到：

\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
\]

其中 `W_Q/W_K/W_V` 是可学习权重，若一个头宽度为 `D`，则 `Q,K,V∈R^(T×D)`。第 `i` 行的含义可以这样把握：

- **Query** `q_i`：位置 i 当前想找哪类信息；
- **Key** `k_j`：位置 j 能以什么特征被匹配；
- **Value** `v_j`：位置 j 一旦被选中，实际带走的内容。

注意力的核心公式是：

\[
S=\frac{QK^\mathsf{T}}{\sqrt D},\qquad
A=\operatorname{softmax}(S+M),\qquad
Y=AV.
\]

符号逐一解释：`S∈R^(T×T)` 的 `S_ij` 是位置 i 的 query 与位置 j 的 key 的点积分数；`√D` 缩放点积，避免 D 变大时数值典型幅度增大、softmax 过尖；`M` 是掩码；`A` 是每一行和为 1 的注意力权重；最终 `Y∈R^(T×D)` 是对所有 value 的加权和。

所以注意力不是“在词表查相似词”。它比较的是**当前层、当前上下文**产生的 Q/K；相同 token 在不同句子中已有不同表示，因而可指向不同位置。Q/K 决定“看哪里”，V 决定“拿什么”。

### 因果掩码：并行训练时不许泄题

next-token 预测要求位置 i 只访问 `j≤i`。对于 `j>i`，令 `M_ij=-∞`，softmax 后对应权重为 0。长度为 4 时可见性是：

```text
query 位置 0： ✓ · · ·
query 位置 1： ✓ ✓ · ·
query 位置 2： ✓ ✓ ✓ ·
query 位置 3： ✓ ✓ ✓ ✓
```

这叫 causal mask。它不是“只看前一个 token”：完整注意力下，位置 i 可以看全部历史；它只禁止未来。训练可一次算完整个 `(T,T)` 矩阵，生成又不会获得训练时不该有的信息。

---

## 4. 多头：一组 token 同时用多种检索视角

单头只拥有一套 Q/K/V 投影。多头注意力将 `C` 个通道拆为 `H` 个头，每个头宽度 `D=C/H`；每个 token 同时用多种低维匹配方式检索上下文，最后再合并。它不是把文本切成 H 份，每个头都能看整段可见上下文。

`gpt.py::CausalSelfAttention.__init__` 直接体现了这些约束：

```python
self.head_dim = self.n_embd // self.n_head
assert self.n_embd % self.n_head == 0
assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
```

在 `CausalSelfAttention.forward` 中，投影后只是在最后一维重新解释形状：

```python
B, T, C = x.size()
q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
```

### 一个最小形状实验

设 `B=2, T=5, C=12, H=3, H_kv=3`，则 `D=4`。以下是**教学化简代码**，不替代仓库的 `Linear` 或注意力后端：

```python
import torch

B, T, C, H = 2, 5, 12, 3
D = C // H
x = torch.randn(B, T, C)
Wq = torch.randn(C, C)
q = (x @ Wq).view(B, T, H, D)
print(q.shape)                    # torch.Size([2, 5, 3, 4])

# 为了按数学公式做矩阵乘法，临时把 H 移到 T 前面
qh = q.transpose(1, 2)            # (B, H, T, D)
scores = qh @ qh.transpose(-2, -1) / D**0.5
print(scores.shape)               # torch.Size([2, 3, 5, 5])
```

最后两个 `T,T` 维代表“每个 query 位置对每个 key 位置”的分数。真实 nanochat 保持 `(B,T,H,D)`，因为 `gpt.py` 注释说明这是 FA3 的原生布局，避免转置；只有 SDPA fallback 会转换为 `(B,H,T,D)`，算完再转回。

注意力后输出仍是 `(B,T,H,D)`，代码通过：

```python
y = y.contiguous().view(B, T, -1)
y = self.c_proj(y)
```

将 H 个头拼回 `(B,T,C)`，再投影回残差流宽度。`.contiguous()` 不能随意删：某些后端或转置操作可能让内存布局不连续，而 `view` 需要可按该布局解释。

---

## 5. GQA：保留很多 query，少存一些 K/V

经典 MHA（multi-head attention）中 `H_kv=H`：Q、K、V 均有 H 个头。nanochat 支持 GQA（Grouped-Query Attention），即 `H_kv≤H` 且 `H` 可被 `H_kv` 整除。多个 query head 共用一个 K/V head；query 的头数和输出头数仍是 H。

为什么值得这么做？推理时，历史每个 token 的 K、V 要被逐层保存进 KV cache。缓存量与 `H_kv` 成正比。减少 K/V 头可以减少缓存占用和读取带宽，通常对逐 token 解码很重要；代价是不同 query 头失去一部分各自专属 K/V 的表达自由度。

这是**架构上的效率—表达力取舍**，不是注意力公式的必要部分。也不要误读为“当前基础模型肯定启用了 GQA”：如前所述，`base_train.py::build_model_meta` 传入 `n_kv_head=num_heads`。实现和配置预留了能力，测试 `tests/test_attention_fallback.py::TestFA3VsSDPA.test_gqa` 则专门以 `n_heads=8, n_kv_heads=2` 验证后端可处理 GQA。

---

## 6. RoPE：不加位置表，也让注意力知道顺序

若没有位置信息，attention 对输入行的排列没有顺序概念。交换两个 token 的行，计算只是相应交换，无法区分“狗咬人”和“人咬狗”。

经典基线是给第 p 个位置加一条位置向量：

\[
x_p=\operatorname{token\_embed}(t_p)+e_p.
\]

nanochat 采用的 RoPE（Rotary Positional Embedding）不同：token embedding 本身不加位置向量；每层注意力里，对 Q 和 K 的二维通道对按位置旋转。

`GPT._precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, ...)` 的核心是：

```python
channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
inv_freq = 1.0 / (base ** (channel_range / head_dim))
t = torch.arange(seq_len, dtype=torch.float32, device=device)
freqs = torch.outer(t, inv_freq)
cos, sin = freqs.cos(), freqs.sin()
cos, sin = cos[None, :, None, :], sin[None, :, None, :]
```

`base` 的默认值确为 `100000`。不同二维通道对拥有不同频率；不同位置拥有不同旋转角。最终 `cos/sin` 形状为 `(1, seq_len, 1, D/2)`，能自动广播到 `(B,T,H,D/2)`。

`apply_rotary_emb(x, cos, sin)` 接收 `(B,T,H,D)`，把最后一维分为两半：

\[
y_1=x_1\cos\theta+x_2\sin\theta,\qquad
y_2=-x_1\sin\theta+x_2\cos\theta.
\]

然后拼回 `(B,T,H,D)`。源码注释指出此处旋转方向与某些教材写法相反；只要 Q、K 始终使用同一约定，注意力依赖的相对旋转关系不变。

真实调用非常短：

```python
cos, sin = cos_sin
q = apply_rotary_emb(q, cos, sin)
k = apply_rotary_emb(k, cos, sin)
```

只转 Q、K，不转 V。位置应当影响“当前 query 与哪个 key 匹配”，而 V 是匹配成功后搬运的内容。RoPE 的关键直觉是：位置 i 的 Q 与位置 j 的 K 相乘时，结果能表达两者的**相对位置**；多种频率提供粗细不同的位置刻度。

初始化时，`GPT.__init__` 将表长度设为 `config.sequence_len * 10`，以非持久 buffer 注册 `cos/sin`；`GPT.forward` 会断言 `T` 不超过这张表。这里不是无限上下文承诺，而是当前实现预先多算的缓存上限。

### KV cache 时为什么位置必须偏移？

推理会先把 prompt 的 K/V 写入缓存，再一次输入一个新 token。新 token 的 RoPE 位置不能又从 0 开始。源码精确地做了：

```python
T0 = 0 if kv_cache is None else kv_cache.get_pos()
cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
```

`engine.py::KVCache.get_pos()` 从 `cache_seqlens` 取当前位置；最后一个 attention layer 完成后，`CausalSelfAttention.forward` 调用 `kv_cache.advance(T)`。这是位置与缓存同步的关键。KV cache 的内存与生成流程会在第 10 章专门讨论。

---

## 7. nanochat 的注意力实现：原理、取舍与优化要分开

### 7.1 QK norm 与 1.2：稳定性/实验性训练选择

RoPE 后，源码执行：

```python
q, k = norm(q), norm(k)
q = q * 1.2
k = k * 1.2
```

这里 `norm` 仍是无参数 RMSNorm，但作用在每个 head 的最后 `D` 维。QK norm 的直觉是限制 Q/K 长度的漂移，从而让点积分数和 softmax 更稳定。Q、K 各乘 1.2，使点积总幅度约乘 1.44，代码注释称其意图是更尖锐的 attention；这是当前源码的实验性常数，不应当作一般结论。

因此，Q/K/V、mask、softmax 是**核心原理**；QK norm 是稳定训练的架构选择；这里的 `1.2` 应视为当前仓库的实验性调节，不是普适常数，更不该从它反推出一条理论定律。

### 7.2 滑动窗口：改变可见连接以节省计算

`GPT._compute_window_sizes` 根据 `window_pattern` 为每层生成 `(left, right)`；`L` 是 `config.sequence_len` 的长窗口，`S` 是 `ceil(ceil(sequence_len / 4) / 128) × 128`，即先取约四分之一上下文、再向上对齐到 128。默认 pattern 是 `"SSSL"`，并且最后一层无条件改为 `L`。例如默认 `sequence_len=2048` 时，短窗为 **512**；源码注释里的 `2048 -> 768` 与实现不符，应以代码为准。右侧恒为 0，仍是因果的；按后端的 left-window 约定，当前位置还可见自身，所以最多涉及 `left + 1` 个 key。

短窗口层只看最近若干历史 token，减少远距离的比较；最终全窗口层保留全上下文路径。这既是计算优化，也确实改变了某些层的可见图，所以不能和“换一个更快 kernel”混为一谈。

### 7.3 FA3 与 SDPA：同一个注意力语义的两个执行后端

训练路径没有 KV cache：

```python
y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
```

推理路径从 `kv_cache.get_layer_cache(self.layer_idx)` 取每层缓存，调用 `flash_attn.flash_attn_with_kvcache(...)`，由后端原地写入新 K/V。

`nanochat/nanochat/flash_attention.py` 包装为统一接口。`_load_flash_attention_3()` 在 CUDA 可用时尝试加载 Flash Attention 3；`_resolve_use_fa3()` 再依据可用性和 `COMPUTE_DTYPE` 决定是否使用。否则走 PyTorch 的 `F.scaled_dot_product_attention`（SDPA）。fallback 会：

1. 将 `(B,T,H,D)` 转成 SDPA 使用的 `(B,H,T,D)`；
2. 在 Q 头与 K 头数量不同的 GQA 情况设置 `enable_gqa=True`；
3. 处理因果、滑窗，以及 cache/chunk 造成 `Tq` 与 `Tk` 不一致时所需的显式 mask；
4. 将结果转回 `(B,T,H,D)`。

FA3 与 SDPA 的目标是同一类因果注意力，不是两种模型算法。FA3 的价值主要在融合计算、降低中间内存读写和适配硬件后的速度；SDPA fallback 让 CPU、MPS 或不兼容 CUDA 环境仍可运行。浮点计算顺序不同，结果不必逐 bit 相同。

这正是测试文件表达的契约：

- `TestFA3VsSDPA` 仅在 `HAS_FA3` 时比较两者，使用 CUDA 和 `bfloat16`，覆盖普通因果、全上下文、滑窗、GQA、prefill、单 token cache，以及反向梯度；`assert_close` 默认 `atol=rtol=1e-2`；
- `TestSDPAOnly` 强制 SDPA，检查输出形状、没有 NaN、梯度存在，并用 `KVCache` 检查 prefill 后和再解码一个 token 后的位置推进；
- `test_kvcache_single_token_sliding_window` 特别防止单 token 解码时 fallback 忽略窗口、错误读取全部缓存的回归。

测试证明的是两条实现路径在允许浮点误差下语义接近、且可反传；它不声称任何硬件上的两次运行完全相同，也不应被误读为性能基准。

---

## 常见误区

1. **“token id 本身就是语义数值。”** 错。它只是 embedding 表的行号；语义来自查到的参数向量和训练。
2. **“logits 就是概率。”** 错。logits 是未归一化分数，softmax 后才是概率。
3. **“因果注意力只能看上一个 token。”** 错。它可看全部历史，或滑窗允许的历史，只是不看未来。
4. **“多头把句子分给不同头。”** 错。每个头都处理所有位置，只是各自有不同投影与匹配空间。
5. **“GQA 会使输出只剩 `H_kv` 个头。”** 错。Q 和输出仍按 H 个头组织；共享的是 K/V。
6. **“RoPE 等于把位置 embedding 加到 x。”** 那是经典绝对位置编码的描述；nanochat 的 `wte` 后没有位置表，RoPE 在每层旋转 Q/K。
7. **“FA3 不可用，attention 就不能工作。”** 错。该仓库提供 SDPA fallback，并有 CPU 可运行的 SDPA-only 测试。
8. **“所有写在 `gpt.py` 的技巧都是 Transformer 定义。”** 错。核心是因果 QK-softmax-V；GQA、QK norm、滑窗、FA3 解决的是不同的稳定性、容量或工程效率问题。

---

## 小结

1. GPT 的训练目标是右移后的 next-token prediction。`GPT.forward` 输入 `idx(B,T)`，无 target 时输出 `logits(B,T,V)`，有 target 时返回交叉熵损失。
2. `transformer.wte` 将 token 地址查成 `(B,T,C)` 向量；nanochat 的输入 embedding 与 `lm_head` 未绑定，且不使用可学习绝对位置表。
3. 注意力以 `QK^T/√D` 产生相关性分数，经因果 mask 和 softmax 得到权重，再加权汇总 V。因果 mask 保证并行训练不泄漏未来。
4. 多头将 C 拆为 H 个 D 维检索视角。GQA 可令 `H_kv<H`，以共享 K/V 换取更小的推理缓存。
5. RoPE 在 Q/K 上按位置旋转二维通道对，令匹配能够利用相对位置；使用 KV cache 时，`T0=kv_cache.get_pos()` 保证新 token 取正确的位置角度。
6. QK norm、1.2 缩放、滑窗属于本实现的训练或架构取舍；FA3/SDPA 是尽力保持同一注意力语义的后端优化与兼容方案。

下一章会把 `CausalSelfAttention` 的输出接回 `Block.forward`：为什么要残差相加、为什么 MLP 要先扩张再压回、RMSNorm 放在哪，以及 value embedding、smear、backout 等增量在完整 `GPT.forward` 的什么位置。

---

## 练习

1. **形状追踪**：设 `B=4,T=16,C=64,H=4,H_kv=2`。写出 `idx`、`wte(idx)`、`q`、`k/v`、attention 输出、合头后输出和 logits 的形状；计算 D。
2. **因果 mask 小实验**：在本章形状实验中构造 `(T,T)` mask，把严格上三角填为 `-inf`，对随机 scores 做 softmax。验证每行权重和接近 1，未来位置权重为 0。不要训练模型。
3. **读 fallback**：阅读 `flash_attention.py::_sdpa_attention`。当 cache 解码使 `Tq==1, Tk>1` 时，为什么不能简单使用“同长度下三角”的直觉？说明源码为何在该分支使用 `is_causal=False`，又为何还要按 `window+1` 裁 K/V。
4. **RoPE 推理**：`apply_rotary_emb` 要把最后一维分成等大的两半。若 D 为奇数会怎样？回到 `GPTConfig` 和 `CausalSelfAttention.__init__`，说明当前代码靠哪些整除关系约束 D；再思考它们是否单独显式断言 D 为偶数。
5. **测试阅读**：在不运行昂贵训练的前提下，阅读 `test_kvcache_single_token_sliding_window`。用自己的话说明它构造了什么缓存状态、期待避免什么 bug，以及为什么窗口为 8 时实际包含的是最近 `window+1` 个 key。
6. **边界分类**：将以下项分类为“核心原理 / 架构或训练取舍 / 工程后端优化”：因果掩码、RoPE、GQA、QK norm、`q=k=...*1.2`、SDPA fallback、FA3。允许某项有双重影响，但必须解释理由。

## 前后章导航

- ← [第 03 章：Tokenizer：从字节到 Token](./03-tokenizer.md)：解释了本章 `idx` 中的整数如何从文本而来，以及 tokenizer 与模型词表为何必须匹配。
- **本章**：`idx(B,T)` 经 embedding、因果多头注意力与 RoPE，成为带上下文的 `(B,T,C)` 表示，并最终可投影为下一 token 的 `(B,T,V)` 打分。
- → [第 05 章：GPT MLP / 现代技巧：MLP、残差流与现代改进](./05-gpt-mlp-tricks.md)：从 `Block.forward` 出发，补齐 attention 之后的 MLP、残差流和完整模型路径。
