# 第 10 章　推理引擎：KV 缓存与高效生成

> **本章定位**　第三部分·从基础模型到聊天模型。第 09 章得到了 SFT checkpoint，本章解释它怎样真正生成回答。
>
> **中心问题**　回答必须逐 token 生成，但已经读过的 prompt 为什么不必每次重算？缓存保存什么、下一 token 怎样读取它？工具返回的结果又怎样加入生成过程？
>
> **范围**　朴素生成与 KV cache、prefill / decode 两阶段、位置与 RoPE、多样本生成、greedy / temperature / top-k、工具调用和推理性能测量。聊天评分在[第 11 章](./11-chat-evaluation.md)，RL 采样在[第 12 章](./12-rl.md)。
>
> **先决条件**　读过[第 04 章](./04-gpt-attention.md)的 Q/K/V 和 RoPE、[第 05 章](./05-gpt-mlp-tricks.md)的 smear，以及[第 09 章](./09-sft.md)的对话与工具协议。
>
> **本章产物**　一条“**prompt → prefill → 首 token → 单 token decode → 回答结束**”的主线；能解释缓存更新、多样本复制、采样和工具输出分别在哪一步发生。
>
> **源码入口**　`nanochat/engine.py`、`nanochat/gpt.py::GPT.forward`、`nanochat/flash_attention.py`、`scripts/chat_cli.py`、`scripts/infer_bench.py`、`tests/test_engine.py`。

---

## 本章讲什么

训练时，示范答案已经知道，模型可以在一次 forward 中并行预测各个位置。推理时，后面应该写什么还不知道，必须先得到一个 token，才能把它放进前缀、继续预测下一 token：

```text
prompt → token 1 → token 2 → token 3 → ...
```

本章讨论的是 nanochat 当前的普通自回归生成：**输出之间的先后依赖仍在，但旧前缀的重复计算可以省掉**。

例如 prompt 有 500 个 token，想生成 100 个新 token：

```text
朴素：输入长度依次为 500、501、502、...、599
缓存：先输入 500 个 prompt token，然后每轮只输入一个新 token
```

KV cache 的作用，就是保留每层已经计算好的历史 K/V，让新 token 仍能读取上文，却不用重新计算旧 token 的层输出。

完整调用链是：

```text
chat_cli：把聊天历史和用户问题编码为 prompt token ids
  → Engine.generate：prefill、采样、decode、处理工具
  → GPT.forward(..., kv_cache=...)：计算新输入的位置和表示
  → 各层 attention：写入新 K/V，读取历史 K/V
  → 返回 logits，选出下一个 token
  → 返回给调用者，并在继续生成时送回模型
```

**读完本章你应该能**：

1. 解释 KV cache 节省了什么，为什么新 token 仍要经过全部 Transformer 层；
2. 用一个小例子追踪 prefill、首 token 和 decode 的顺序；
3. 读懂缓存形状、缓存位置和 RoPE 为什么必须一致；
4. 区分一次 prompt 的多样本生成与不同 prompt 的请求 batch；
5. 用 logits 的具体数字解释 temperature 和 top-k；
6. 区分模型生成的 token、工具插入的 token 与终止 token；
7. 分别用首 token 延迟、decode 延迟、吞吐和显存评价推理。

> **读法建议**：§1–§3 讲生成主线和缓存；§4–§6 展开多样本、采样、工具；§7 是工具执行的安全边界；§8 再把计算流程换算成延迟、带宽与显存账。

---

## 预备词汇

| 名称 | 含义 | 代码对应 |
|---|---|---|
| `prompt` | 本次生成已经提供的全部上文，聊天时包括历史消息和角色标记 | `tokens: list[int]` |
| `prompt_len` | prompt 的 token 数，不是模型参数量 | `len(tokens)` |
| `num_new_tokens` | 推导中计划输出的新增 token 列数；包括工具插入和终止标记，不等于正文长度 | 有限的 `max_tokens` 上限 |
| `B` | 同一个 prompt 并行生成的样本数 | `num_samples` |
| `T` | 本次 forward 实际输入的 token 数；prefill 为 prompt_len，decode 通常为 1 | `idx.shape[1]` |
| `L` | Transformer 层数 | `config.n_layer` |
| `H_q` / `H_kv` | query 头数 / key-value 头数 | `n_head` / `n_kv_head` |
| `head_dim` | 每个 attention 头的向量长度 | `n_embd // n_head` |
| `C` | 模型隐藏宽度 | `n_embd` |
| `V` | 词表大小 | logits 最后一维 |
| `T_max` | 本次缓存预分配的容量，不是当前已经写入的长度 | `KVCache.max_seq_len` |
| `cache_seqlens` | 每行已经处理并写入缓存的 token 数 | `(B,)` int32 张量 |

**KV cache 存的是每层 attention 使用的 K/V 浮点张量，不是 token id，也不是 logits。** 历史 token id 还用于组织聊天文本，历史 K/V 用于计算 attention，两者职责不同。后文的 `q₀...q₃` 是 prompt token 的简写；attention 公式里的 `q_t` 才是 query 向量，不要混为一谈。

---

## 1. 从 prompt 到下一个 token：先把生成过程看清楚

先把模型看成一个函数：输入一串已经知道的 token，模型会为每个位置给出“下一个 token 可能是什么”的分数。生成文本时，我们取出最后一个位置的分数，选出一个新 token，把它接到输入末尾，然后重复这个过程。

### 1.1 一次 forward 并不会直接返回一句话

假设“法国的首都是”被切成 4 个 token。这里的切分只为说明过程，不代表 tokenizer 的真实切分：

```text
[法国] [的] [首都] [是]
```

送入模型后，一次 forward 返回的不是字符串“巴黎”，而是一张 logits 分数表：

```text
输入 ids shape： (1, 4)
输出 logits：    (1, 4, V)
```

`V` 是词表大小。4 个输入位置各有一行、每行有 `V` 个分数：

```text
位置 0 的 logits：根据“法国”预测后一个 token
位置 1 的 logits：根据“法国 的”预测后一个 token
位置 2 的 logits：根据“法国 的 首都”预测后一个 token
位置 3 的 logits：根据“法国 的 首都 是”预测后一个 token
```

我们现在要续写整个 prompt，所以只关心最后一行：

```python
next_token_logits = logits[:, -1, :]  # shape=(1, V)
```

再从这 `V` 个分数中选择一个 token。可以直接选最高分，也可以按概率采样；假设这次选中“巴黎”。然后把它接到末尾：

```text
[法国] [的] [首都] [是] [巴黎]
```

要继续生成，就用这条更长的前缀再预测一次。于是文本生成本质上是一个很小的循环：

```text
对当前前缀做 forward
  → 取最后位置的 logits
  → 选择一个新 token
  → 把新 token 接回前缀
  → 重复
```

`GPT.generate` 正是这个朴素版本。下面先只看“选最高分”的 greedy 分支：

```python
ids = torch.tensor([tokens], dtype=torch.long, device=model.get_device())
for _ in range(max_tokens):
    logits = model(ids)                          # 计算当前完整前缀
    next_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    ids = torch.cat((ids, next_ids), dim=1)      # 新 token 接回前缀
    yield next_ids.item()
```

这段代码慢，但非常重要：后面的高效引擎只是在避免重复计算，生成逻辑并没有改变。

### 1.2 聊天里的 prompt 到底包含什么

在命令行里，用户看见的可能只有一句：

```text
法国的首都是哪里？
```

但模型不会自动知道“这是用户的问题，现在轮到助手回答”。调用模型前，`chat_cli.py` 会把角色标记和聊天历史一起编码成 `conversation_tokens`。

第一轮聊天的 prompt 类似：

```text
BOS
+ user_start + 用户问题 1 + user_end
+ assistant_start
```

模型生成回答 1 后，回答及其 `assistant_end` 也会追加到 `conversation_tokens`。因此第二轮的 prompt 变成：

```text
BOS
+ user_start + 用户问题 1 + user_end
+ assistant_start + 助手回答 1 + assistant_end
+ user_start + 用户问题 2 + user_end
+ assistant_start
```

所以这里的 **prompt** 不是“用户刚输入的那句话”，而是：

> 在开始生成本轮回答之前，模型能够看到的完整 token 前缀。

末尾的 `assistant_start` 也很重要。它相当于告诉模型：“前面的用户消息结束了，接下来应按 assistant 的身份续写。”程序提供这个标记，模型生成它后面的回答正文。

> **其他入口怎样构造 prompt**：评估和 RL 使用 `render_for_completion(conversation)`。它会拿掉最后一条作为标准答案的 assistant 消息，再渲染上文并追加 `assistant_start`。这样模型看到的是题目和已有历史，不会提前看到参考答案。

### 1.3 为什么旧前缀可以复用，新 token 仍要经过模型

得到“巴黎”后，为了预测再下一个 token，朴素的 `GPT.generate` 会重新输入全部 5 个 token：

```text
[法国] [的] [首都] [是] [巴黎]
```

问题是：前 4 个位置刚刚已经计算过。继续生成时，前缀越来越长，这些旧位置也会一遍遍重算。

#### 旧位置为什么不需要重算

模型使用因果 attention。位置 0 只能看位置 0，位置 1 只能看位置 0～1，以此类推。后来把“巴黎”追加到位置 4，不会改变位置 0～3 当时能看到的内容：

```text
追加前：旧位置看不到未来的“巴黎”
追加后：旧位置仍然看不到“巴黎”
```

因此，只要模型权重、旧 token 和位置不变，旧位置先前算出的结果就可以复用。

#### 新位置需要从旧位置读取什么

现在只看 Transformer 的某一层。令：

- `t`：新 token“巴黎”所在的位置；
- `j`：它可以查看的某个历史位置。

这一层会为新位置计算 Query、Key 和 Value，分别简写为 `q_t、k_t、v_t`。可以先用“查资料”理解它们：

```text
q_t：新位置提出的查询
k_j：位置 j 的检索标签，用来判断它与查询是否相关
v_j：位置 j 真正提供的内容
```

新位置用自己的 `q_t` 与各位置的 `k_j` 比较，再对对应的 `v_j` 加权求和：

```text
相关性分数：scores_j = q_t · k_j / √head_dim
权重：      weights = softmax(scores)
输出：      output_t = Σ_j weights_j · v_j
```

因此，计算新位置时需要历史位置的 K/V，却不需要历史 Q。历史 Q 已经完成了“旧位置当时查询什么”的工作；现在发起新查询的是 `q_t`。

这就是 **KV cache**：把每层旧位置的 K/V 保存下来，下一轮直接读取。

```text
旧 token：不再经过这一层，读取已经保存的 K/V
新 token：计算自己的 Q/K/V，并把新 K/V 追加到缓存
```

#### 有 cache，为什么新 token 还要 forward

因为缓存里只有历史计算结果，不包含新 token 的结果。“巴黎”仍要从头经过模型：

```text
embedding
→ 第 0 层：计算新 Q/K/V，读取第 0 层的历史 K/V
→ 第 1 层：计算新 Q/K/V，读取第 1 层的历史 K/V
→ ...
→ 最后一层和 lm_head
→ 得到预测再下一个 token 的 logits
```

每一层处理新 token 时，都依赖上一层刚产生的新表示，所以每一层都必须执行，也都需要保存自己那一层的 K/V。

一句话总结：**KV cache 省掉的是旧 token 的重复 forward；新 token 仍要完整经过模型。**

### 1.4 KV cache 到底少算了多少

继续用一个长度为 4 的 prompt，并生成 3 个新 token。

朴素版本每次重算完整前缀：

| 要得到的输出 | 本轮送入模型的内容 | 本轮位置数 |
|---|---|---:|
| 新 token 1 | 4 个 prompt token | 4 |
| 新 token 2 | prompt + 新 token 1 | 5 |
| 新 token 3 | prompt + 新 token 1、2 | 6 |
| **合计** |  | **15** |

理想的缓存版本先处理一次 prompt，首个新 token 直接来自 prompt 最后位置的 logits；之后每轮只送回刚生成的一个 token：

| 要得到的输出 | 本轮送入模型的内容 | 本轮位置数 |
|---|---|---:|
| 新 token 1 | 4 个 prompt token，即 prefill | 4 |
| 新 token 2 | 只送入新 token 1 | 1 |
| 新 token 3 | 只送入新 token 2 | 1 |
| **合计** |  | **6** |

这就是后文两个术语的直觉：

- **prefill**：一次处理已知的完整 prompt，并填好历史 cache；
- **decode**：之后一次只处理一个刚生成的 token。

若 prompt 长度为 `prompt_len`，计划生成 `num_new_tokens` 个 token，朴素版本处理的位置总数是：

```text
prompt_len × num_new_tokens
+ num_new_tokens × (num_new_tokens - 1) / 2
```

例如 prompt 有 500 个 token、输出 100 个 token：

```text
朴素版本：500 + 501 + ... + 599 = 54,950 个输入位置
理想缓存：500 + 99 = 599 个输入位置
```

缓存版本为什么只加 99？因为处理 500-token prompt 的那次 forward 已经给出了第 1 个新 token；为了得到后面 99 个，才需要再做 99 次单 token decode。

不过，**位置数减少到约 1/92，不代表墙钟时间一定快 92 倍**。decode 虽然只输入一个新 token，但 attention 仍要读取历史 K/V；prefill 和 decode 使用的 kernel、并行度与内存访问模式也不同。这里只是在数避免了多少重复输入位置，不是在直接预测速度。

> 本节比较的是理想缓存流程。当前 `Engine.generate` 作为 Python generator 还有一次不产生新输出的尾部 forward；等 §3.5 讲完 `yield` 的暂停位置后，再看这个实现细节。

---

## 2. KV cache：把每层算过的 K/V 按位置存下来

§1 已经说明了为什么要缓存：旧 token 的 K/V 不会因为末尾多了一个 token 而改变。现在来看这些 K/V 具体存在哪里，以及“下一次写入位置”怎样推进。

继续使用长度为 4 的 prompt。为了让形状容易展开，先假设模型有 2 层、每次只生成 1 条回答。

### 2.1 从一个 token 的 K/V，搭出完整缓存

先只看**一层、一个 token**。

这一层会为该 token 计算 K 和 V。若模型有 `H_kv` 个 KV 头，每个头的宽度是 `head_dim`，那么一个 token 的形状是：

```text
K：(H_kv, head_dim)
V：(H_kv, head_dim)
```

接下来逐步增加维度：

```text
一个 token：              (H_kv, head_dim)
一条样本的所有 token：    (T_max, H_kv, head_dim)
一个 batch 的所有样本：   (B, T_max, H_kv, head_dim)
模型的所有层：            (L, B, T_max, H_kv, head_dim)
```

因此 `KVCache` 分别预分配两个五维张量：

```text
k_cache：(L, B, T_max, H_kv, head_dim)
v_cache：(L, B, T_max, H_kv, head_dim)
```

可以把它想成：**每一层都有两本按 token 位置排列的账本，一本记 K，一本记 V。** 不同层不能共用同一本账，因为同一个 token 经过不同层后会产生不同的 K/V。

先用“不共享 KV 头”的简单配置代入：

```text
层数 L=2
batch 大小 B=1
最多容纳 T_max=12 个位置
query 头数 H_q=2
KV 头数 H_kv=2
隐藏宽度 C=16
head_dim = C / H_q = 8

k_cache.shape = v_cache.shape = (2, 1, 12, 2, 8)
```

先只读 shape：模型有 2 层、1 条样本、每层最多存 12 个位置；每个位置有 2 个 KV 头，每个头保存 8 个数。

> **GQA 分支**：缓存最后两维取决于 `H_kv`，不是 `H_q`。如果改成 `H_q=4、H_kv=2、C=32`，`head_dim` 仍是 8，cache shape 也仍以 2 个 KV 头计为 `(2,1,12,2,8)`。这表示每 2 个 Q 头共享一组历史 K/V。当前 base 配方默认 `H_q=H_kv`，实际值以 checkpoint 为准。

### 2.2 `T_max` 是容量，`cache_seqlens` 才是当前长度

预分配 12 个位置，只表示“这块内存最多能放 12 个 token”，不表示 12 个位置都已经有效。还需要一个游标记录已经写了多少，这就是 `cache_seqlens`。

刚创建 cache 时：

```text
位置：              0  1  2  3  4  5  ... 11
是否有效：          ·  ·  ·  ·  ·  ·      ·
cache_seqlens：    [0]
下一次写入位置：    0
```

长度为 4 的 prompt 完成 prefill 后：

```text
位置：              0  1  2  3  4  5  ... 11
是否有效：          ✓  ✓  ✓  ✓  ·  ·      ·
cache_seqlens：    [4]
下一次写入位置：    4
```

注意 `cache_seqlens=[4]` 表示“前 4 个位置有效”，所以有效索引是 `0、1、2、3`，下一个空位才是 4。

现在假设模型已经选出了新 token“巴黎”。下一次 forward 只输入这个 token：

```text
开始时 cache_seqlens=[4]
  → 把“巴黎”在当前层的新 K/V 写到位置 4
  → 它的新 Q 可以读取位置 0...4 的 K/V，包括自己的 K/V
  → 所有层处理完成后，cache_seqlens 变成 [5]
```

于是位置 4 也变成有效历史，下一次写入位置是 5。

所以这里先记住一个关系：

```text
0 <= cache_seqlens <= T_max
已写范围是 [0, cache_seqlens)
下一次从 cache_seqlens 开始写
```

> **实现边界**：滑动窗口层可能只读取最近一段有效 K/V，但当前实现仍按完整 `T_max` 预分配空间，不会用环形缓存回收旧槽位。`Engine` 在多样本生成时还会把 prefill cache 复制到更大的 decode cache，这留到 §4 再展开。

### 2.3 token 的位置跨层不变，游标只能在最后一层推进

一个 token 会依次经过所有 Transformer 层。假设 prompt 已占据位置 `0...3`，现在处理位置 4 的“巴黎”：

```text
第 0 层：把第 0 层算出的 K/V 写入“第 0 层账本”的位置 4
第 1 层：把第 1 层算出的 K/V 写入“第 1 层账本”的位置 4
```

层号从 0 变成 1，但 token 在序列中的位置仍然是 4。因此，第 0 层写完以后不能马上把公共游标推进到 5；否则第 1 层会误以为自己正在处理下一个 token，把同一个“巴黎”写到位置 5。

源码让每层先取得自己的缓存切片，但所有层共享本轮开始时的 `cache_seqlens`：

```python
k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
y = flash_attn.flash_attn_with_kvcache(
    q, k_cache, v_cache,
    k=k, v=v,
    cache_seqlens=kv_cache.cache_seqlens,
    causal=True,
    window_size=window_size,
)
if self.layer_idx == kv_cache.n_layers - 1:
    kv_cache.advance(T)  # 最后一层完成后，才统一推进 T 个位置
```

两层模型处理一个 decode token 的时序是：

```text
开始：cache_seqlens=[4]
第 0 层写位置 4：cache_seqlens 仍为 [4]
第 1 层写位置 4：写完后 advance(1)
结束：cache_seqlens=[5]
```

prefill 也是同样逻辑，只是本轮 `T=4`：每层都写位置 `0...3`，最后一层完成后一次执行 `advance(4)`。

> **源码边界**：cache 保存的是该层真正交给 attention 的 K/V。K 已经过 RoPE、QK norm 和缩放，V 也可能混入 value embedding。FA3 在 kernel 内原地写入；SDPA fallback 在 Python 中显式写入，但对外表现相同。

### 2.4 为什么代码只读取第 0 行的 cache 位置

`cache_seqlens` 的 shape 是 `(B,)`，理论上可以分别记录 B 行的已写长度。但当前代码取位置时只读取第 0 行：

```python
def get_pos(self):
    return self.cache_seqlens[0].item()
```

这段代码成立，依赖一个重要前提：

> 当前 Engine 要求 batch 中所有行始终拥有相同的 cache 长度。

这个前提是怎样维持的？当前 `B` 行来自同一个 prompt 的 B 条候选回答。假设 prompt 长度为 4、`B=3`：

```text
prefill 后：cache_seqlens = [4, 4, 4]
```

之后每次 decode 都给三行各输入一个 token：

```text
输入 shape=(3,1)
第 0 行写位置 4
第 1 行写位置 4
第 2 行写位置 4
完成后统一 advance(1)

结果：cache_seqlens = [5, 5, 5]
```

下一轮仍是三行各输入一个 token，因此变成 `[6,6,6]`。既然所有值始终相同，读取第 0 行就能得到全体共同的写入位置。

这也解释了当前实现的限制：它不能让某一行独自停在长度 5，而其他行继续增长到 6、7。某行生成结束标记后，`generate_batch` 只是不再把它后续产生的 token 加入最终答案；底层仍保留这行并同步 forward，直到所有行结束。

所以本节的结论不是“所有 KV cache 天生都要求等长”，而是：

> **nanochat 用“所有行同步前进”简化了位置管理，因此可以只读第 0 行；代价是提前结束的行会产生无效计算，也不能承载长度不同、动态加入或退出的独立请求。**

§4 会解释为什么当前场景恰好是同一个 prompt 的 B 条候选回答，以及它们怎样共享一次 prefill。

---

## 3. 一次回答怎样生成：Prefill → 首 token → Decode

§1 讲了原理，§2 讲了 cache 的结构。现在只追踪一次真实生成：模型何时读 prompt、何时选出首 token、何时把新 token 写回 cache。

沿用 4-token prompt，并把它们记为 `p₀、p₁、p₂、p₃`。这里改用字母 `p`，避免与 attention 的 Query 符号 `q` 混淆。

### 3.1 Prefill：一次处理完整 prompt

Engine 首先建立空 cache，把整个 prompt 一次送入模型：

```text
输入：[p₀, p₁, p₂, p₃]，shape=(1,4)
  → 每层计算四个位置的表示
  → 每层把四个位置的 K/V 写入 cache
  → cache 已写长度从 0 变为 4
  → 取位置 3 的 logits，选择首个回答 token a₀
```

这一步叫 **prefill**：用已经知道的 prompt 填充 cache。

prompt 的四个位置可以在一次 forward 中并行计算；因果 mask 仍确保 `p₀` 看不到后面的 token、`p₁` 只能看见 `p₀～p₁`，以此类推。

最容易混淆的一点是：**首 token `a₀` 已经由 prefill 的最后位置 logits 选出，不需要额外做一次 decode。** 但此时 cache 中只有 `p₀～p₃`，还没有 `a₀`，因为 `a₀` 尚未作为模型输入。

### 3.2 Decode：输入刚生成的 token，预测再下一个

要继续生成，Engine 把刚选出的 `a₀` 送回模型：

```text
输入：a₀，shape=(1,1)，它在完整序列中的位置是 4
  → 每层计算 a₀ 的新 Q/K/V
  → 把 a₀ 的 K/V 写到 cache 位置 4
  → a₀ 的 Q 读取 prompt 和自己的 K/V
  → cache 已写长度从 4 变为 5
  → logits 用来选择 a₁
```

接下来重复同一过程：

```text
prefill [p₀,p₁,p₂,p₃] → 选择 a₀
输入 a₀                → 选择 a₁
输入 a₁                → 选择 a₂
输入 a₂                → 选择 a₃
```

这就是 **decode**：每轮只输入刚刚确定的一个 token，利用历史 cache 预测再下一个 token。

因此，“decode 生成 `a₁`”的完整含义是：**decode 输入 `a₀`，得到 logits，再从中选出 `a₁`。** 输出 token 总是比本轮输入向后一步。

### 3.3 单 token 输入仍要使用它在完整序列中的位置

decode 张量只有一个 token，shape 是 `(1,1)`；但 `a₀` 在完整序列中位于位置 4，不是位置 0。

`GPT.forward` 从 cache 读取本轮起点 `T0`，据此选择 RoPE 参数：

```python
T0 = kv_cache.get_pos()
cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
```

| forward | 本轮输入 | `T0` | RoPE 位置 |
|---|---|---:|---|
| prefill | `p₀、p₁、p₂、p₃` | 0 | 0、1、2、3 |
| decode 1 | `a₀` | 4 | 4 |
| decode 2 | `a₁` | 5 | 5 |

所以要区分两个概念：

```text
本轮张量中的局部位置：a₀ 是唯一一个 token，局部下标为 0
完整序列中的绝对位置：a₀ 接在四个 prompt token 后，位置为 4
```

如果每次 decode 都错误地使用 RoPE 位置 0，新 Q/K 与历史 K 的位置关系就会错乱。

### 3.4 nanochat 还要保存 smear 的上一 token embedding

KV cache 解决的是 attention 历史，但 nanochat 的 smear 还依赖前一个 token 的归一化 embedding。

prefill 一次收到整段 prompt，`p₃` 的前一个 token `p₂` 就在同一张量中。decode 只收到 `a₀`，它看不到同一张量里的 `p₃`，所以需要额外保存：

```text
prefill 完成：prev_embedding 保存 p₃ 的 embedding
输入 a₀：     读取 p₃ 与 a₀ 做 smear，再把 prev_embedding 更新为 a₀
输入 a₁：     读取 a₀ 与 a₁ 做 smear，再更新为 a₁
```

这揭示了一个通用原则：**增量推理必须保存模型中所有跨 token 的状态，不只是 attention 的 K/V。** `KVCache.prev_embedding` 是当前模型特有的附加状态。

### 3.5 `yield` 的暂停位置决定 token 何时写入 cache

`Engine.generate` 是 Python generator。调用 `Engine.generate(...)` 只得到 generator 对象；第一次 `next(generator)` 才开始执行。遇到 `yield` 后，函数暂停，下一次 `next` 才从暂停处继续。

对于 4-token prompt、`max_tokens=3`：

| 调用 | 恢复后送入模型 | 返回给调用者 | 暂停时 cache 包含 |
|---|---|---|---|
| `next` 1 | prompt `p₀～p₃` | `a₀` | `p₀～p₃` |
| `next` 2 | `a₀` | `a₁` | `p₀～p₃, a₀` |
| `next` 3 | `a₁` | `a₂` | `p₀～p₃, a₀, a₁` |

关键规律是：

> token 先被选出并 yield，下一次恢复 generator 时，才作为输入写入 cache。

因此 yield `a₂` 的瞬间，cache 中还没有 `a₂`。如果后面还要预测 `a₃`，才需要输入 `a₂`。

当前代码在调用者继续请求第 4 次 `next` 时，会先输入 `a₂`，然后循环回顶部发现已经达到 `max_tokens=3`，最终抛出 `StopIteration`。这就是 §1.4 提到的不产生新输出的尾部 forward。若 `generate_batch` 看到所有行都已返回终止 token，会主动停止消费 generator，不需要再回写最后的终止列。

> **运行边界**：`max_tokens` 统计 generator yield 的新增 token 列，工具插入和终止 token 也占预算。Engine 不会自动滚动或截断超长历史；调用者必须保证 prompt 与生成上限适配模型上下文、RoPE 和 cache 容量。

> **模式边界**：checkpoint 加载时的 `model.eval()` 控制模型训练/评估模式；`torch.inference_mode()` 关闭梯度记录。二者职责不同，都与 §7 中计算表达式使用的 Python `eval(...)` 无关。

### 3.6 一份 cache 只服务于当前这次回答

多轮聊天中需要区分两种状态：

```text
conversation_tokens：跨轮保存历史 token id
KV cache：            当前一次 Engine.generate 内的临时计算结果
```

第一轮：

```text
完整 prompt → 新建 cache → prefill → decode 回答 1
回答结束 → generate 结束 → 这份 cache 不再保留
回答 1 的 token id → 追加到 conversation_tokens
```

第二轮：

```text
conversation_tokens + 用户问题 2 + assistant_start
→ 再次调用 Engine.generate
→ 新建 cache
→ 对包含第一轮问答的完整 prompt 重新 prefill
```

所以 cache 避免的是**生成同一个回答期间**的重复计算。下一轮能够看见旧对话，是因为历史 token id 被重新送入模型，不是因为上一轮的 K/V 仍在。

### 3.7 可选自检：缓存路径应与完整前缀重算一致

缓存不仅要快，还要保证新位置的结果与完整前缀重算一致。下面在 `nanochat/` 的 Python 环境中运行；先用 `NANOCHAT_DTYPE=float32 python` 启动解释器，固定为 fp32 CPU 自检。不用 checkpoint，构造一个随机初始化的小 GPT，比较两条路径的最后位置 logits。

```python
import torch
from nanochat.common import COMPUTE_DTYPE
from nanochat.gpt import GPT, GPTConfig
from nanochat.engine import KVCache

torch.manual_seed(42)
config = GPTConfig(sequence_len=16, vocab_size=64, n_layer=2,
                   n_head=4, n_kv_head=2, n_embd=32, window_pattern="L")
with torch.device("meta"):
    model = GPT(config)
model.to_empty(device="cpu")
model.init_weights()
# 默认 attention 输出投影是零；自检改为非零，让错误的 K/V 真正影响 logits。
for block in model.transformer.h:
    torch.nn.init.normal_(block.attn.c_proj.weight, std=0.02)
model.eval()

cache = KVCache(batch_size=1, num_heads=2, seq_len=16, head_dim=8,
                num_layers=2, device="cpu", dtype=COMPUTE_DTYPE)
ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
with torch.inference_mode():
    full_prompt = model(ids[:, :4])[:, -1, :]
    cached_prompt = model(ids[:, :4], kv_cache=cache)[:, -1, :]
    torch.testing.assert_close(cached_prompt, full_prompt, atol=1e-5, rtol=1e-5)
    assert cache.get_pos() == 4          # prefill 四个 token，下一位置为 4
    for pos in (4, 5):
        full = model(ids[:, :pos+1])[:, -1, :]
        incremental = model(ids[:, pos:pos+1], kv_cache=cache)[:, -1, :]
        torch.testing.assert_close(incremental, full, atol=1e-5, rtol=1e-5)
        assert cache.get_pos() == pos + 1
```

这个 fp32 CPU 小例子同时走过 attention、RoPE 和 smear 的缓存分支，比较的是**同一固定前缀的 logits**，不是靠两个随机采样结果碰巧相同判断正确。低精度 / 不同后端需调整误差容差，真实 checkpoint 和长上下文仍需另外测试。

---

## 4. 同一个问题生成多条回答：Prompt 只算一次

有时我们希望模型对同一道题给出多条候选答案。例如 RL 要比较同题的不同 rollout，评估也可能从多个候选中选择最佳答案。

### 4.1 `num_samples` 表示一个 prompt 的多条回答

当前接口是：

```python
Engine.generate(tokens, num_samples=B)
```

`tokens` 只有一份，因此它表达的是：

```text
同一个 prompt ─┬→ 回答 0
                ├→ 回答 1
                └→ 回答 2
```

它不是同时传入 B 个不同问题。正因为 prompt 相同，prompt 的 forward 才有机会只做一次。

### 4.2 先用一行 prefill，再复制成 B 行

继续使用 4-token prompt `[p₀,p₁,p₂,p₃]`，同时生成 3 条回答。

第一步只用 batch=1 做 prefill：

```text
[p₀,p₁,p₂,p₃]
  → prompt forward 一次
  → 得到 prompt cache
  → 得到预测首 token 的 logits，shape=(1,V)
```

第二步扩展成三行：

```text
prompt cache ─┬→ 第 0 行 cache
              ├→ 第 1 行 cache
              └→ 第 2 行 cache

同一份首 token logits ─┬→ 为第 0 行采样 a₀
                        ├→ 为第 1 行采样 b₀
                        └→ 为第 2 行采样 c₀
```

第三步，三行开始各自 decode：

```text
本轮输入：[a₀, b₀, c₀]，shape=(3,1)
第 0 行写入 a₀ 的 K/V
第 1 行写入 b₀ 的 K/V
第 2 行写入 c₀ 的 K/V
```

从这里开始，三条回答可能沿不同 token 继续生成。

### 4.3 Logits 是展开读取，K/V 是实际复制

源码中有两种看似相同、实际不同的操作：

```python
logits = logits[:, -1, :].expand(num_samples, -1)
kv_cache_decode.prefill(kv_cache_prefill)
```

- logits 的 `expand` 创建共享底层存储的视图。这里只需读取相同分数并为每行分别采样，不需要修改 logits；
- K/V 必须复制到 B 个独立 cache 行，因为后续每行要写入不同回答，不能共享同一块可变存储。

`KVCache.prefill(other)` 这个名字容易让人误以为它会执行 prompt forward。实际上它只做“把 `other` 中已经算好的有效 K/V 复制过来”，同时复制 smear 所需的 `prev_embedding`。

### 4.4 节省的是 prompt 计算，不是 cache 显存

对 B 条回答，优化前后可以这样比较：

```text
朴素做法：prompt forward B 次，保存 B 份 cache
当前做法：prompt forward 1 次，再复制成 B 份 cache
```

所以省掉的是 `B-1` 次重复的 prompt 计算。decode 阶段仍需 B 个独立 cache 行，显存大致随 B 线性增加；创建、清零和复制 cache 本身也有开销。

### 4.5 分别采样不等于保证不同

采样器对 `(B,V)` 的每一行分别选择 token，不会先选一个 token 再广播到所有行。但“分别选择”不保证结果一定不同：

- `temperature=0` 时，每行都取同一个 argmax，同 prompt 的回答通常相同；
- `temperature>0` 时，每行独立抽取，可能不同，也可能碰巧相同。

每次 `generate` 调用会用 seed 初始化一个随机数生成器。模型、prompt、B 和采样参数都相同时，同 seed 应复现同一批输出；改变 B 会改变随机数的消费顺序，不能期待某一行仍与原来的单样本结果完全一致。

---

## 5. 从 logits 选出一个 token

模型只负责给词表中每个 token 打分；究竟选哪个 token，由采样策略决定。下面始终使用同一组小例子：

```text
词表 token id：  0    1    2
logits：          0    2    1
```

注意：logit `2` 是 token id 1 的分数，不是要返回 token id 2。

### 5.1 Softmax 把分数变成概率

logits 可以为任意实数，尚不是概率。softmax 将它们变为总和为 1 的分布：

```text
softmax([0,2,1]) ≈ [0.090, 0.665, 0.245]
```

顺序始终对应原词表 id `0、1、2`。id 1 概率最高，但随机采样仍可能选到 id 0 或 id 2。

### 5.2 `temperature=0`：不采样，直接取最高分

当前实现把温度 0 单独解释为 greedy：

```python
if temperature == 0.0:
    return logits.argmax(dim=-1, keepdim=True)
```

例子中最高 logit 位于索引 1，所以返回 token id 1。这条分支不做 `logits/0`，也不使用随机数，因此 seed 不影响结果。

严格地说，若两个最高分几乎相等，不同硬件或数值精度的微小差异仍可能改变 argmax；这里的“确定”指同一数值结果下不再随机抽样。

### 5.3 `temperature>0`：控制概率有多集中

温度 `τ` 的计算是：

```text
probs = softmax(logits / τ)
```

| 温度 | id 0、1、2 的概率约值 | 直觉 |
|---:|---|---|
| 0.5 | `[0.016,0.867,0.117]` | 分布更尖，最高分更容易被选中 |
| 1.0 | `[0.090,0.665,0.245]` | 原始 softmax 分布 |
| 2.0 | `[0.186,0.506,0.307]` | 分布更平，低分 token 机会增加 |

低温度减少随机性，高温度增加随机性。它只改变本次选择分布，不会改变模型参数；“更随机”也不等于“质量更高”或“更有创意”。

### 5.4 Top-k：只在最高的 k 个候选中抽样

假设 `top_k=2`。先找出最高的两个 logit：

```text
保留的分数 vals：       [2, 1]
它们的原词表 id idx：   [1, 2]
```

id 0 被排除。再只对 `[2,1]` 应用 temperature 和 softmax：

```text
τ=1 时，候选内部概率约为 [0.731, 0.269]
```

源码对应：

```python
vals, idx = torch.topk(logits, k, dim=-1)
probs = F.softmax(vals / temperature, dim=-1)
choice = torch.multinomial(probs, 1, generator=rng)
next_ids = idx.gather(1, choice)
```

`choice` 是“小候选表中的第几个”，不是原词表 id。例如 `choice=0` 要通过 `idx[0]` 映射回 token id 1，这就是最后一步 `gather` 的作用。

因此 temperature 和 top-k 不是二选一：top-k 先筛候选，temperature 再决定候选内的概率。`top_k=None` 或非正数表示不筛选；超过词表大小时会截到 `V`。当前项目没有实现 top-p。

### 5.5 最终参数由调用者决定

| 入口 | temperature | top_k | `max_tokens` |
|---|---:|---|---:|
| `Engine.generate` 函数默认值 | 1.0 | None | None |
| `chat_cli.py` 默认传入值 | 0.6 | 50 | 256 |

调用 `chat_cli.py` 时，真正生效的是第二行。温度为 0 时采样函数直接进入 greedy 分支，top-k 不再执行。阅读生成行为时应沿调用链看实参，而不是只看底层函数签名。

---

## 6. 工具调用：模型提出请求，Engine 插入结果

普通生成中，每个新 token 都由模型选择。工具调用多了一种来源：模型负责写出“我要计算什么”，Engine 执行工具后把结果插入 token 流，模型再基于结果继续回答。

### 6.1 先看一遍完整的 `2+2`

假设模型生成下面的工具请求：

```text
python_start  2+2  python_end
```

Engine 看见完整请求后执行计算器，得到 `4`，并插入：

```text
output_start  4  output_end
```

随后模型继续生成正文。整条序列是：

```text
模型：python_start → 2+2 → python_end
环境：output_start → 4 → output_end
模型：答案是 4。→ assistant_end
```

责任边界很清楚：

| token 区间 | 谁决定内容 | 来源 mask |
|---|---|---:|
| `python_start ... python_end` | 模型 | 1 |
| `output_start ... output_end` | Engine / 工具环境 | 0 |
| 工具后的回答正文 | 模型 | 1 |

这里的 mask 表示“谁产生了 token”，不是 attention mask。环境插入的 `4` 仍然会送入模型，后续回答当然可以看见它；但 RL 不应把 `4` 当成模型自主选择的动作计算策略梯度。

### 6.2 `RowState` 记录每条回答进行到哪里

多样本生成时，每一行都可能处于不同状态：某行正在写普通回答，另一行可能正在收集工具表达式。`RowState` 为每行保存控制信息：

| 字段 | 作用 |
|---|---|
| `current_tokens` | prompt 加上已经选出的续写 token |
| `in_python_block` | 是否已经生成 `python_start`、正在收集表达式 |
| `python_expr_tokens` | 当前工具表达式的 token id |
| `forced_tokens` | Engine 接下来必须插入的工具结果队列 |
| `completed` | 是否已经遇到 `assistant_end` 或 BOS |

这些是“生成流程走到哪一步”的控制状态。它们不保存 Transformer 各层的历史表示，不能替代 KV cache。

### 6.3 工具结果也要逐 token 送回模型

工具返回 `4` 后，Engine 不会一次性把整个结果偷偷塞进 cache。它仍遵循 §3 的生成节奏：先 yield 一个 token，下一次恢复 generator 时再把它送入模型。

```text
yield python_end
  → 执行工具，把 [output_start, 4, output_end] 放入 forced_tokens

下一次：输入 python_end → yield output_start
再下一次：输入 output_start → yield 4
再下一次：输入 4 → yield output_end
再下一次：输入 output_end → 模型基于完整工具结果继续生成
```

因此“调用者已经收到 token `4`”和“`4` 已经写入 KV cache”仍是两个时刻。只有下一次 forward 处理 `4` 后，它才成为后续预测可读取的历史。

当前实现即使 `forced_tokens` 非空，也会先从 logits 采样一个 token，再用队列首项替换它。被替换的采样不会进入上下文，但会消耗一次随机数；这是实现开销，不改变最终 token 流。

### 6.4 流式接口与批量收集接口

`Engine.generate` 每轮 yield 两个长度为 B 的 Python 列表：

```text
token_column：本轮每行输出的 token id
token_masks： 对应来源，1=模型采样，0=Engine 强制插入
```

`chat_cli.py` 消费这个流并逐 token 打印。

`generate_batch` 则把同一个流收集成完整结果：

```python
results, masks = engine.generate_batch(tokens, num_samples=4, max_tokens=128)
```

无工具的单行例子：

```text
prompt：        [p₀,p₁,p₂,p₃]
流式输出：      a₀, a₁, assistant_end
results[0]：   [p₀,p₁,p₂,p₃,a₀,a₁]
masks[0]：     [ 0, 0, 0, 0, 1, 1]
```

`results` 包含原 prompt，prompt mask 记为 0；`generate_batch` 识别终止 token，但不把它加入结果。若只需要新增回答，取 `results[i][len(tokens):]`。

### 6.5 失败和结束分别意味着什么

工具请求只有闭合后才执行：

- 没有先生成 `python_start` 的 `python_end` 不触发工具；
- 始终没有 `python_end` 的表达式不会执行；
- 表达式不符合过滤规则、求值失败或超时，工具返回 `None`，Engine 不插入输出区间。

生成也有两种不同结束原因：

- 模型生成 `assistant_end` 或 BOS：该行标记为完成；
- 达到 `max_tokens`：只是预算耗尽，回答可能截在正文甚至工具区间中间。

工具插入 token 和终止 token 都占 `max_tokens`。`chat_cli.py` 在预算耗尽且末尾不是 `assistant_end` 时，会补一个结束标记再存入历史；这只是在修补聊天记录边界，不代表模型主动生成了正确结束，也不会修复一个未闭合的工具协议。

> **显示边界**：一个 BPE token 不一定对应一个完整字符。当前 CLI 对单个 token 调用 `decode([token])` 并立刻打印，跨 token 的 UTF-8 字节片段可能显示为替换字符。严谨的流式 UI 需要缓冲不完整字节。

---

## 7. 工具安全：能过滤一些输入，不等于建立了沙箱

先给结论：当前计算器适合教学和受控实验，**不应直接用于执行不可信用户提供的任意表达式**。

### 7.1 当前代码做了哪些限制

`use_calculator` 先删除数字中的逗号，再只允许两类表达式：

1. 数学表达式：字符只能来自数字、`*+-/.()` 和空格，并额外拒绝 `**`；
2. 字符串计数：字符受限，必须包含 `.count(`，并拒绝 `__`、`import`、`open` 等关键词。

例如：

```text
2+2                 → 允许进入求值
"banana".count("a") → 允许进入求值
print(2+2)          → 不符合过滤规则
```

通过过滤后，表达式在清空 builtins 的环境中求值：

```python
eval(formula, {"__builtins__": {}}, {})
```

同时用 `SIGALRM` 设置默认 3 秒超时。过滤、执行或超时失败都会返回 `None`，Engine 因而不插入工具结果。

### 7.2 为什么这些措施仍不是安全隔离

可以把当前措施看成“门口检查”，不能把它当成“把代码关进另一间房”：

- 字符白名单和关键词黑名单没有完整理解 Python 语法，容易漏掉未预想的组合；
- 清空 `__builtins__` 限制了名称访问，但没有隔离 CPU、内存和当前进程；
- `eval` 与模型推理运行在同一个 Python 进程，异常资源消耗会直接影响 Engine；
- `SIGALRM` 依赖 POSIX 信号和主线程语义，并不适用于所有平台或并发模型；
- 修改进程级信号处理器本身也不适合作为通用多请求执行器。

所以“清空 builtins + 3 秒超时”只能降低部分风险，不能给出沙箱级安全保证。

### 7.3 面向真实用户时应该增加什么

如果工具只需算术，最小可靠方案是：解析受限语法，只实现允许的数字、运算符和函数，而不是把字符串交给 Python `eval`。

若必须运行更一般的代码，则至少需要：

```text
严格语法白名单与输入长度限制
+ 执行时间、内存、输出大小和数值大小限制
+ 与模型服务隔离的进程或受控运行环境
+ 明确的工具权限和审计
```

AST 白名单可以限制“允许写什么”，但不能单独限制无限计算或内存消耗；语法限制和资源隔离解决的是不同问题。

---

## 8. 推理性能：用户等多久，机器做多少，显存占多少

推理性能不能只报一个“tokens/s”。至少要分三本账：

```text
延迟：一个用户多久看到首 token，之后多久看到下一个
吞吐：GPU 每秒为整个 batch 生成多少 token
显存：模型权重、KV cache 和临时张量占多少空间
```

### 8.1 先算 KV cache 的显存

一条样本的一个 token，需要在所有层保存 K 和 V：

```text
每 token 的 cache 字节数
= L × 2 × H_kv × head_dim × dtype_bytes
      ↑ K 和 V 两份
```

B 条样本、每条预留 `T_max` 个位置时：

```text
cache_bytes = B × T_max × 每 token 的 cache 字节数
```

例如：

```text
L=12, H_kv=2, head_dim=8, dtype=float32（4 bytes）
每 token = 12 × 2 × 2 × 8 × 4 = 1,536 bytes

B=8, T_max=4096
总 cache = 8 × 4096 × 1536
           = 50,331,648 bytes
           = 48 MiB
```

因此 B 或 `T_max` 翻倍，cache 预分配空间也近似翻倍。bf16/fp16 每元素 2 bytes，相同 shape 下约为 fp32 的一半。当前 Engine 的 cache dtype 取 `COMPUTE_DTYPE`，不一定与所有模型参数的存储 dtype 相同。

这还不是完整峰值显存：还要加模型权重、logits、临时张量，以及 prefill cache 向 decode cache 复制时短暂同时存在的空间。GQA 通过减少 `H_kv` 降低 cache；滑动窗口虽然减少读取范围，但当前实现不会因此缩小预分配容量。

### 8.2 Prefill 和 decode 是两种不同的工作负载

| 阶段 | 输入形态 | 主要工作 | 常见瓶颈倾向 |
|---|---|---|---|
| prefill | 一次输入整段 prompt | 大批位置的矩阵计算和 attention | 较容易利用 GPU 算力 |
| decode | 每行每轮输入一个 token | 模型投影 + 读取历史 K/V | 小 batch 时常受带宽和启动开销影响 |

prefill 的 token 多，矩阵乘规模较大；decode 虽然每轮只有一个新 token，却要反复读取模型权重和越来越长的 K/V。二者不应混成一个平均速度。

这是常见倾向，不是硬规则。batch、上下文长度、模型结构、dtype、量化、attention 后端和硬件都会改变瓶颈。

### 8.3 三个最先要看的指标

| 指标 | 回答的问题 |
|---|---|
| TTFT（Time To First Token） | 用户提交 prompt 后，多久看到第一个 token？ |
| TPOT（Time Per Output Token） | 首 token 之后，平均/典型多久出现下一个 token？ |
| 总 tok/s | 整个 batch 每秒一共生成多少 token？ |

假设 B=8，每个 decode step 用 10 ms，并且每行都产生一个有效 token：

```text
单行：每 10 ms 得到 1 token → 100 tok/s
全 batch：每 10 ms 得到 8 token → 800 tok/s
```

所以 800 tok/s 是 GPU 总吞吐，不表示单个用户以 800 tok/s 阅读回答。增大 batch 常能提高总吞吐，却可能让单请求延迟不变甚至变差。

`infer_bench.py` 的 TTFT 从第一次 `next(generator)` 前计时，到首列 token 产生并完成 CUDA 同步。它包含 prompt prefill、cache 创建/复制和首次采样，但不包含 checkpoint 加载、分词、网络排队或 UI 显示，因此不是完整线上端到端延迟。

### 8.4 MFU 与 MBU 帮助判断在等算力还是等数据

两个利用率估计回答不同问题：

```text
MFU = 估算 FLOPs / (耗时 × GPU 理论峰值 FLOPs/s)
MBU = 估算字节流量 / (耗时 × GPU 理论峰值 bytes/s)
```

- MFU（Model FLOPs Utilization）看计算单元利用程度；
- MBU（Model Bandwidth Utilization）看显存带宽利用程度。

例如小 batch decode 的 MFU 很低，可能不是实现“什么都没做”，而是计算单元在等待权重和 K/V 从显存搬入；这时应同时看 MBU。

当前 benchmark 的 decode 估算为：

```text
每步字节数 ≈ weight_bytes + B × kv_read_bytes(context_len)
每步 FLOPs ≈ B × estimate_decode_flops(context_len)
```

这些是根据架构计算的账本，不是 profiler 或硬件计数器直接测得的真实流量。比如 embedding 是按行查询，不能据此断言每个参数每步都从显存完整读取。MFU/MBU 适合比较配置和发现瓶颈线索，最终归因仍需 profiler。

### 8.5 `infer_bench.py` 实际怎样测

在 `nanochat/` 目录运行：

```bash
python -m scripts.infer_bench -i sft -g d12 \
    --prompt-tokens 512 --decode-tokens 128 --batch-sizes 1,8,32
```

脚本当前只支持单张 CUDA GPU，不应使用 `torchrun`。执行流程是：

```text
加载 checkpoint
→ 构造指定长度的自然语言 prompt
→ 若 prompt + decode 超出 sequence_len，则缩短 prompt
→ warmup
→ 对每个 batch size 测量一次正式生成
```

每次正式生成中：

1. 第一次 `next` 的耗时记为 TTFT；
2. 后续每次成功 yield 的耗时记为 decode step time；
3. step time 的中位数记为 TPOT；
4. `B × 成功 decode step 数 / decode 总耗时` 记为 tok/s；
5. 同时报告 peak allocated VRAM，并估算 MFU/MBU；
6. 最后一行输出包含相同数据的 JSON，供脚本读取。

这里有两个计数边界：

- 首 token 已计入 TTFT，不进入 decode tok/s；generator 最后的无输出尾部 forward 也不进入“成功 yield”耗时；
- 若某行提前终止或包含工具强制 token，`B × step 数` 不一定等于有效自然语言正文 token 数。

一份可比较的性能报告至少要写清：checkpoint、GPU、dtype、attention 后端、prompt 长度、生成上限、batch、temperature、实际 decode steps、TTFT、TPOT、tok/s 和峰值显存。缺少这些上下文，单独一个 tok/s 几乎无法解释。

---

## 常见误区

1. **“首 token 来自第一次 decode。”** 它来自 prefill 最后位置的 logits；第一次 decode 输入的是首个已经生成的 token。
2. **“缓存后不再执行模型。”** 新 token 仍走全部层，只是不重算历史位置。
3. **“cache 保存 token id。”** K/V 是每层浮点表示，token id 和控制状态另有用途。
4. **“每层写完立即推进 cache 位置。”** 所有层共享本轮位置，最后一层才统一推进。
5. **“每次 decode 的 RoPE 都从 0 开始。”** 应从缓存位置续接，否则与历史 K 的位置关系不一致。
6. **“一次 prefill 的多样本生成不多占显存。”** 省的是前缀计算，当前仍复制 B 份缓存行。
7. **“独立采样保证输出不同。”** 随机抽样也可能相同；greedy 下同 prompt 通常相同。
8. **“token 一输出，就已经在 cache 中。”** 最新输出要在下一次 forward 才写入缓存；工具结果也一样，终止后则不必继续计算它。
9. **“max_tokens 只统计模型回答正文。”** 工具强制 token 和终止列也占生成预算，正文长度还取决于接口是否排除终止符。
10. **“空 builtins 和超时就能安全运行任意 Python。”** 当前工具仅允许有限表达式，也不是安全沙箱。
11. **“高 batch tok/s 等于单个用户生成更快。”** 总吞吐与单请求逐 token 延迟必须分别报告。

---

## 小结

主线只有一句话：**先一次性处理 prompt 并缓存 K/V，从最后一个位置选出首 token；之后只把新 token 送入模型，读取历史缓存并续接位置，重复直到结束。**

围绕它有几条分支：

- **缓存正确性**：各层保存实际 attention K/V，RoPE 按原序列位置续接，smear 还需上一 embedding；
- **多样本与采样**：同 prompt 只 prefill 一次，再复制 cache 到多行；每行按温度和 top-k 选择 token；
- **工具协议**：模型生成请求，环境强制插入结果；继续生成前，已选 token 送回 forward。来源 mask 为 RL 区分动作；
- **性能与安全**：cache 用显存换掉历史重复计算，TTFT / TPOT / 吞吐分别测；有限过滤器和同进程 eval 不能作为安全沙箱。

---

## 练习

1. **顺序追踪**：4-token prompt 生成 3 个 token。逐次写出输入内容、cache 起点、RoPE 位置，以及该次 logits 在预测哪个 token；区分理想缓存位置数和当前尾部 forward。
2. **显存账**：`L=12,C=768,H_q=6,H_kv=2,B=8,T_max=512,bf16`，求 head_dim、完整 K/V 形状和字节数。KV 头增加到 6 后，哪部分占用会变为 3 倍？
3. **多样本**：为什么同 prompt、temperature=0 时生成 16 条答案通常没有多样性？增大 B 为什么不等于零成本？
4. **工具追踪**：`python_start → 2+2 → python_end → output_start → 4 → output_end`，哪些 token 来源 mask=1？哪些 mask=0？哪些会送回模型？
5. **测试阅读**：`tests/test_engine.py` 的 `test_kv_cache_prefill`、`test_seed_reproducibility`、`test_multi_sample_first_token_diversity` 分别检查什么？为什么它们还不能代替真实 GPT 的 cached/uncached logits 对照？
6. **性能报告**：B=8 时测到 800 tok/s、TPOT=10 ms，这意味着单个用户每秒得到多少 token？如果 prompt 翻倍，还必须看哪个延迟指标？

---

## 前后章导航

- **上一章**：[第 09 章：监督微调——教会模型对话](./09-sft.md)。定义 assistant 与环境 token 的生成责任。
- **本章**：将 checkpoint 接入逐 token 生成、缓存和工具执行流程。
- **下一章**：[第 11 章：聊天评估——聊天模型会做题了吗？](./11-chat-evaluation.md)。用同一个 Engine 检验任务完成情况。
- **后续**：[第 12 章：强化学习](./12-rl.md)。对同 prompt 采样多条回答，使用来源 mask 区分模型动作和工具输出。
