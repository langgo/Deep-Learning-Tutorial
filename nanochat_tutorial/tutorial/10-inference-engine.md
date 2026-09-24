# 第 10 章：Inference：KV 缓存与高效生成

> **源码地图**：调度器在 `nanochat/nanochat/engine.py`，模型读写缓存的位置在 `nanochat/nanochat/gpt.py`，注意力内核/回退在 `nanochat/nanochat/flash_attention.py`，命令行流式聊天在 `nanochat/scripts/chat_cli.py`，性能测量在 `nanochat/scripts/infer_bench.py`，行为测试在 `nanochat/tests/test_engine.py`。  
> **调用链**：`chat_cli.py → Engine.generate → GPT.forward(..., kv_cache=...) → CausalSelfAttention.forward → flash_attn_with_kvcache`。不需要流式显示时，`Engine.generate_batch` 消费前者的 generator；`chat_eval.py` 和 `chat_rl.py` 都走这条批量路径。

训练时，模型一次看到一段 `(B,T)` token，并行计算每个位置的下一个 token 损失。推理时却必须先生成第一个 token，才能生成第二个。这个自回归依赖无法消除；但我们可以避免在每一步把已经读过的 prompt 再算一遍。本章的主角 KV 缓存正是为此而来。

## 学习目标

完成本章后，你应能：

1. 指出朴素生成的重复计算，并解释 prefill 与 decode 的分工；
2. 说清 `KVCache` 的真实维度、位置计数和 GQA 对缓存的影响；
3. 读懂单行 prefill 后复制到多样本 decode 的过程；
4. 区分 greedy、temperature、top-k，以及 generator 如何流式返回 token；
5. 解释 `RowState`、强制 token 和计算器状态机；
6. 用显存带宽、TTFT、TPOT、MFU/MBU 理解推理性能；
7. 明确 `eval` 的有限防护不是强安全沙箱。

## 前置概念

建议先读[第 04 章：GPT attention](./04-gpt-attention.md)、[第 05 章：GPT MLP / 现代技巧](./05-gpt-mlp-tricks.md)和[第 09 章：监督微调 SFT](./09-sft.md)。本章在其后解释真正的生成与 rollout。记号如下。

| 符号 | 含义 | 源码对应 |
|---|---|---|
| `B` | 并行生成的行数 | `num_samples` / `batch_size` |
| `P` | prompt token 数 | `len(tokens)` |
| `L` | Transformer 层数 | `config.n_layer` |
| `H_q`, `H_kv` | Q 头数、KV 头数 | `n_head`, `n_kv_head` |
| `D` | 头维度，`n_embd / n_head` | `self.head_dim` |
| `V` | 真实词表大小 | `config.vocab_size` |

token id 是整数；KV cache 存的是各层注意力投影产生的浮点 key/value，不是 token id，也不是最终 logits。

---

## 1. 经典基线：正确但越来越慢的朴素生成

给定前缀 \(x_{1:t}\)，模型输出末位置 logits \(z_t\in\mathbb R^V\)，并定义：

\[
p(x_{t+1}\mid x_{1:t})=\operatorname{softmax}(z_t).
\]

其中 `softmax` 把词表中 `V` 个未归一化分数变为概率。选一个 token，附到前缀尾部，再预测下一个，这就是自回归生成。

`nanochat/nanochat/gpt.py::GPT.generate` 保留了最清楚的基线（以下是教学摘录）：

```python
ids = torch.tensor([tokens])       # (1, P)
for _ in range(max_tokens):
    logits = self.forward(ids)     # (1, 当前长度, V)
    next_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # greedy: (1, 1)
    ids = torch.cat((ids, next_ids), dim=1)
    yield next_ids.item()
```

它没有错，却会反复计算旧前缀。生成 `N` 个 token 时，送入网络的位置数是：

\[
P+(P+1)+\cdots+(P+N-1)=NP+\frac{N(N-1)}2.
\]

第 20 个新 token 到来时，前 19 个新 token 和整个 prompt 又经过 embedding、所有 Transformer 层、MLP。这里要分清两种“重复”：历史 token 的 MLP、Q/K/V 投影乃至 attention 输出都被重复算；同时，新末位置的 attention 还会再次读取历史。KV cache 只消除前一种重复，后一种读取是让新 token 使用上下文所必需的。

例如 `P=4`、只生成 `N=3`：朴素基线依次前向 `(1,4)`、`(1,5)`、`(1,6)`，共处理 15 个输入位置；缓存版本是一次 `(1,4)` prefill，之后每步 `(1,1)` decode，共处理 7 个新送入网络的位置。两者最终给出的第三个 token 应相同（在同一权重、同一采样规则下），但这不代表二者运行时间必然按 `15/7` 成比例：attention、矩阵乘和内存读写的占比会随形状变化。

`engine.py` 的 `__main__` 用这个慢基线与 `Engine.generate` 在温度零下比较 token 序列，目的正是验证加速不改变结果。它是一个很好的工程习惯：先以简单、显然正确的实现作为 oracle，再写复杂的缓存路径；不要只因“更快”就假定位置和状态没有错。

## 2. KV 缓存：保存不会被未来改变的部分

一层注意力先投影：

\[
Q=xW_Q,\qquad K=xW_K,\qquad V=xW_V.
\]

当前位置的输出可写作：

\[
\operatorname{softmax}(Q_tK_{\leq t}^{\mathsf T}/\sqrt D)V_{\leq t}.
\]

因果 mask 保证旧位置看不到未来。因此某个旧 token 的 K、V 一旦算好，未来 token 不会改变它们。新 token 只需算自己的 Q/K/V：用新 Q 查询历史 K/V，然后把自己的 K/V 追加进去。**KV cache 复用的正是历史 K/V。**

它不让模型“跳过 forward”：每个新 token 仍须走全部层、MLP、lm head，仍须注意历史；它只免除了历史 token 的重复投影和重复前向。缓存位置错误会使结果错误，不只是变慢。

### 真实维度与位置

`engine.py::KVCache.__init__` 预分配：

```text
k_cache, v_cache: (L, B, T_max, H_kv, D)
cache_seqlens:    (B,) ，设备上的 int32
```

注意本仓库面向 Flash Attention 3：单层 cache 视图是 `(B,T_max,H_kv,D)`，不是许多旧实现的 `(B,H,T,D)`。`tests/test_engine.py::test_kv_cache_basic` 明确检查这一点。

设 `L=2, B=3, T_max=12, H_q=4, H_kv=2, n_embd=32`，则 `D=8`，总缓存形状为 `(2,3,12,2,8)`。已 prefill 5 个 token 后，`cache_seqlens=[5,5,5]`。下一步 decode 输入 `(3,1)`；本层 Q 是 `(3,1,4,8)`，K/V 是 `(3,1,2,8)`，新 K/V 写在位置 5，当前 Q 对长度 6 的缓存注意力。

GQA 让 `H_kv` 小于等于 `H_q`。`gpt.py::CausalSelfAttention.forward` 中 Q reshape 为 `(B,T,H_q,D)`，K/V 为 `(B,T,H_kv,D)`；多个 Q 头共享 KV 头。因此 KV 显存与 `H_kv` 成正比，GQA 是架构层面的推理减负。注意 `D` 仍由 query 头数计算，即 `n_embd // n_head`；KV 头变少并不把每个 head 的通道变宽。

### 谁写 cache，谁推进位置？

真正的写入发生在 `nanochat/nanochat/gpt.py::CausalSelfAttention.forward`。推理分支先通过 `kv_cache.get_layer_cache(self.layer_idx)` 取得这一层的 `(B,T_max,H_kv,D)` 视图，再把新的 `k`、`v` 和当前 `cache_seqlens` 传给 `flash_attn.flash_attn_with_kvcache(...)`。FA3 会原地更新缓存；`nanochat/nanochat/flash_attention.py::flash_attn_with_kvcache` 的 SDPA fallback 也显式执行相同语义的切片赋值。`KVCache` 本身不是每层都 append 一次的 Python list，它是预先分配的一整块张量。

`cache_seqlens` 是 `(B,)` 而不是单个 Python 整数，符合 FA3 的接口；但 `KVCache.get_pos()` 和 fallback 当前都假设各行处在相同位置，并读取第 0 行。`Engine.generate` 的所有样本每轮都走一次 forward，即使某行已结束也继续送入 token，所以这个“同一步长”假设得以保持。最后一层才调用 `kv_cache.advance(T)`：若每层都加 `T`，第 2 层会错误地从 prompt 之后的位置开始写，破坏层间对齐。

## 3. 两个阶段：prefill 与 decode

### Prefill：并行读 prompt

`Engine.generate` 先给 batch=1 建 `kv_cache_prefill`：

```python
ids = torch.tensor([tokens], dtype=torch.long, device=device)  # (1, P)
logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
logits = logits[:, -1, :].expand(num_samples, -1)              # (B, V)
```

这次新输入长度为 `P`。`GPT.forward` 以 `T0 = kv_cache.get_pos()` 切 RoPE：

```python
cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
```

空缓存的 `T0=0`，所以 prompt 使用位置 `0...P-1`。每层调用 `flash_attn_with_kvcache` 原地写 K/V；只有最后层 `kv_cache.advance(T)`。这是必要的：同一批 token 的所有层必须从同一旧位置读写，不能让第 0 层先推进位置。

prompt 内矩阵乘可以并行，prefill 通常更接近**计算受限**。`infer_bench.py::bench_generate` 把第一次 `next(generator)` 的时间记为 TTFT（首 token 时间）；它包含 prefill、缓存复制和第一次采样。

### Decode：每轮只送一个 token

随后每轮是：

```python
ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
# (B, 1)
logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]  # (B, V)
```

此时 `T=1`，缓存位置使 RoPE 正确落在新绝对位置。更具体地说，`GPT.forward` 取 `T0 = kv_cache.get_pos()`，再切 `self.cos[:, T0:T0+T]`、`self.sin[:, T0:T0+T]`。因而 prompt 的 `T0=0` 使用位置 `0...P-1`；它完成后位置推进到 `P`；第一枚生成 token 使用位置 `P`。若忘记这个 offset，decode 虽不一定报错，却会反复把每个新 token 当作位置 0，RoPE 的相对位置信息就不再与训练时一致。

`GPT.forward` 还维护 `KVCache.prev_embedding`：模型的 smear 机制需要上一 token 的归一化 embedding。整段训练可以直接用切片取前一位置；单 token decode 没有它，所以缓存额外保存该状态，保证 prefill 与 decode 边界的计算一致。它在 prefill 时保存最后一个 `(B,1,C)` embedding，在下一次 `T=1` decode 时参与当前 embedding 的混合。KV cache 因而是一个广义的“增量状态”：在这个模型里，除了 K/V 还含有这项额外状态。

decode 每步的时间称 TPOT。它算得少，却需要反复读取模型权重和不断变长的 KV，因此小 batch、长上下文下常常是**显存带宽受限**，而非算力受限。

## 4. 一次 prefill，复制给多个候选

同一 prompt 要生成 `B` 个候选时，nanochat 不做 B 次 prompt 前向，而是：

1. batch=1 prefill 共享前缀；
2. 建 batch=B 的 decode cache；
3. `KVCache.prefill(other)` 复制已写 K/V、位置和 smear 状态；
4. 将末 logits 展开为 `(B,V)`，每行独立抽第一个 token；
5. B 行从此独立 decode。

`prefill` 要求目标 cache 为空，检查层数、KV 头数、头维和容量；只复制到 `:other_pos`。`prev_embedding` 若存在，会从 `(1,1,C)` 扩展并 `clone` 为每行独立状态。`test_kv_cache_prefill` 用写入 K=1、V=2 的小缓存检查复制和位置。

这节省共享 prompt 的**计算**，并不节省 B 份缓存的**显存**。还有一个容量细节：prefill cache 的 `seq_len=len(tokens)`，因为它只装 prompt；decode cache 的长度是 `len(tokens)+max_tokens`（若未指定 `max_tokens`，则使用模型 `sequence_len`）。这是为了给后续位置预留槽位；`Engine.generate` 不会自动裁剪超出预分配长度的生成。

`test_multi_sample_first_token_diversity` 还守护一个易错点：展开 logits 后须逐行采样，不能抽一个 token 再广播；均匀 logits 下 16 行全相同几乎必然是实现错误。独立采样不意味着“答案彼此完全独立”：它们共享同一个随机数生成器和同一个模型/前缀，只是 `torch.multinomial` 在 `(B,V)` 的每一行各做一次选择。固定 `seed` 是复现实验的工具，不是保证不同硬件和不同 PyTorch 内核严格逐位一致的承诺。

## 5. 从 logits 选 token：greedy、温度、top-k

`engine.py::sample_next_token(logits, rng, temperature, top_k)` 接收 `(B,V)` 并返回 `(B,1)`。

- **经典 greedy**：`temperature == 0.0` 时直接 `argmax`，即 \(\arg\max_i z_i\)。它确定、适合基准；`test_temperature_zero_determinism` 验证 seed 不影响它。
- **temperature 采样**：对 \(\tau>0\)，\(p_i=\exp(z_i/\tau)/\sum_j\exp(z_j/\tau)\)。低温使分布尖锐，高温提高随机性。源码用同设备的 `torch.Generator` 和 `seed`，相同 seed 可复现（`test_seed_reproducibility`）。温度改变抽样，不会修改模型知识。
- **top-k**：若 `top_k=k>0`，只保留最高 `min(k,V)` 个 logits，在其中 softmax 与采样，再映回原 id；`None` 或非正数则使用全词表。它不是 top-p，本文件没有 nucleus sampling。

## 6. 流式聊天与 RowState 工具状态机

`Engine.generate` 是 generator，每轮 `yield token_column, token_masks`：前者是 B 行 token，后者标示该 token 是模型采样的 `1` 还是引擎强制注入的 `0`。所以 `chat_cli.py` 能立即 `decode([token])` 并 `print(..., flush=True)`，不必等整段回答。

CLI 维护 `conversation_tokens`，拼入 `<|user_start|>...<|user_end|><|assistant_start|>`。采到 `<|assistant_end|>` 或 `<|bos|>`，对应 `RowState.completed=True`；所有行完成或达到 `max_tokens` 后停止。CLI 的参数默认温度 `0.6`、top-k `50`，不要误写成 `Engine.generate` 默认值（后者为 `1.0` 与 `None`）。若 CLI 因 256 token 上限结束，会补 `<|assistant_end|>` 以保持下一轮格式完整。

每行的 `RowState` 保存 `current_tokens`、待注入的 `forced_tokens` 队列、`in_python_block`、表达式 token 和完成标记。模型采到 `<|python_start|>` 后，引擎收集 token；遇 `<|python_end|>`，解码表达式、调用 `use_calculator`，再把 `<|output_start|> 结果 <|output_end|>` 排进 `forced_tokens`。下一轮循环优先从 `deque.popleft()` 取这些 token，而不是使用刚采样的 token；不过所有 token——包括强制输出——仍会被喂回 `GPT.forward`，以便模型能以工具结果为条件继续生成解释。

强制 token 不来自模型抽样，故 `token_masks` 为 0；采样 token 为 1。`generate_batch` 一面累积最终 token 序列，一面累积这些 mask，并且不把终止的 `<|assistant_end|>` 或 BOS 放入结果。这正是后续 RL 能分辨“哪些位置是模型动作、哪些是环境工具返回”的接口。训练端 `tokenizer.py::render_conversation` 对 `python_output` 同样不监督：它应由工具给出，而非让模型背诵。

状态机只是协议执行器，不会理解 Python 语义。没有闭合的 `<|python_end|>` 不会执行；计算失败或被过滤时 `use_calculator` 返回 `None`，引擎不会插入 output token；一旦采到 assistant end 或 BOS，行被标记 completed。虽然 completed 行仍保持 batch 对齐而继续进行 forward，它们的后续 token 不会由 `generate_batch` 写入结果。这个选择以少量无效计算换取简单的固定形状批处理。

### `eval` 不是强沙箱

`use_calculator` 会移除数字逗号；数学分支只容许数字和基本运算符、拒绝 `**`；字符串分支只允许 `.count()` 相关字符并拒绝 `__`、`import`、`open` 等模式。`eval_with_timeout` 用空 `__builtins__` 和 3 秒 `SIGALRM` 超时执行。

这只是有限防护，**不是强安全沙箱**。黑名单、字符检查、语言对象模型、资源耗尽和平台差异都可能留下风险，`SIGALRM` 也不是跨平台隔离。绝不可把不可信文本当生产环境代码执行；应采用不含 `eval` 的 AST 白名单解析器，并在权限最小化的独立进程/容器中执行。

## 7. 性能账本：省计算，也消耗显存与带宽

`GPT.kv_bytes_per_token()` 的每行缓存成本是：

\[
\text{bytes/token}=L\times2\times H_{kv}\times D\times s,
\]

其中 2 是 K 和 V，`s` 是 `COMPUTE_DTYPE.itemsize`。上面的玩具模型若为 float32，结果是 `2×2×2×8×4=256` 字节/token。真实模型的层数和上下文远大得多，batch 增大时 KV 很快成为 VRAM 主角。

`GPT.kv_read_bytes(context_len)` 还估计每个 decode step 读取的历史 K/V；滑动窗口层只读约 `min(context_len, left_window)` 的历史（实际 attention 的 left-window 边界还包含当前位置），全上下文层读更长历史。GQA 与滑动窗口是**核心架构选择**；预分配缓存、FA3 原地更新以及 CPU/MPS/不兼容 CUDA 时的 SDPA 回退是**工程实现优化**；warmup、batch sweep、中位数 TPOT 与 roofline 指标则是**实验性速度测量技巧**，不能被说成模型能力提升。

`infer_bench.py` 在 batch=1 prefill 上报告 MFU（实际 FLOPs/峰值 FLOPs），在 decode 上以权重字节加 batch 倍 KV 读取字节计算 MBU（实际带宽/峰值带宽）。小 batch 时每步几乎都要重读全部权重；提高 batch 往往提高总 token/s，直到算力或 VRAM 饱和，但会改变延迟。这里 `tok/s` 是整个 batch 的总产出，不能与单请求的 TPOT 混为一谈。

脚本的测量边界也值得准确理解。`bench_generate` 第一次 `next(generator)` 计作 TTFT，故它不只是“模型算 prompt”的纯 prefill 时间，还包括 cache 分配/复制和第一次抽样；随后每次 `next` 才被记录为 decode step。它先 warmup，再用 decode step 时间的中位数作为 TPOT，减少内核首次编译、分配器和偶发抖动的影响。这些是有用的实验方法，不是推理算法本身。

该脚本明确要求 CUDA 和单 GPU，且把 `prompt + decode` 限制在 `sequence_len` 内。它会报告 `weight_bytes(model)`、`model.kv_bytes_per_token()` 和 `model.kv_read_bytes(context_mid)`；不要在没有实际运行测量时虚构某块 GPU 的 MBU、MFU 或 tokens/s。架构估算提供上界和解释，真实性能仍依赖 GPU、dtype、attention 实现、batch 和上下文长度。

## 常见误区

1. **“KV cache 后不再跑模型。”**错；新 token 仍经过全部层，只是不重算历史 K/V。
2. **“这里缓存是 `(B,H,T,D)`。”**错；本仓库整体为 `(L,B,T,H_kv,D)`，层视图为 `(B,T,H_kv,D)`。
3. **“复制 prefill 不增加显存。”**错；它省共享前缀计算，B 行仍各占一份 K/V。
4. **“温度零等于除以零。”**错；代码专门走 `argmax`。
5. **“top-k 等于 top-p。”**错；前者固定候选数量，本引擎未实现后者。
6. **“流式一定降低总耗时。”**它主要降低看到首段文字的等待感。
7. **“清空 builtins 后 eval 就安全。”**错；这不是不可信代码的隔离机制。

## 小结

自回归依赖使 decode 必须逐 token 进行；KV cache 利用因果性，把生成改为一次 prefill 加多次单 token decode。nanochat 用 `(L,B,T_max,H_kv,D)` 缓存、`cache_seqlens` 和 RoPE 偏移保证位置正确，并额外保存 smear 的前一 embedding。`Engine.generate` 还负责复制共享前缀、逐行采样、流式交付、结束控制和工具强制输出。

推理优化不是只看 FLOPs：decode 往往被权重和 KV 的显存带宽限制。GQA、窗口、缓存大小和 batch 共同决定延迟、吞吐与可承载上下文。

## 练习

1. 令 `P=4,N=3`，写出朴素生成每轮输入长度；再写出 KV 版 prefill/decode 的输入长度和缓存位置，比较总处理位置数。
2. 设 `n_layer=12,n_embd=768,n_head=6,n_kv_head=2,B=8,T_max=512`，求 `D`、完整 K/V 缓存形状，并写出每 token KV 字节公式。
3. 阅读 `test_seed_reproducibility`、`test_temperature_zero_determinism`、`test_multi_sample_first_token_diversity`，分别说出它们锁定的不变量。
4. 可选最小实验：复用测试中的 `MockModel`、`ByteTokenizer`，打印 `KVCache` 形状和 `advance(1)` 前后位置，再把 batch=1 cache 复制到 batch=4；不要把均匀 logits 的输出当语言质量。
5. 设计一个只支持 `+ - * /` 和括号的计算器：列出允许的 AST 节点、拒绝的节点，以及为什么仍应放在隔离进程中。

## 前后章导航

- ← [第 09 章：监督微调 SFT：教会模型对话](./09-sft.md)：SFT 定义聊天 special token、监督 mask 和模型应学习的动作边界。
- → [第 11 章：Chat Eval](./11-chat-evaluation.md)：这里的 `Engine.generate_batch` 如何成为分类/生成任务评估中的真实生成路径，将在那里展开。
