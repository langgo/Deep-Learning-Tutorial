# 第 09 章　监督微调：教会模型对话

> **本章定位**　第三部分·从基础模型到聊天模型。给 base checkpoint 做监督微调（Supervised Fine-Tuning，SFT），让已有的续写能力开始遵守 user / assistant 对话协议。
>
> **中心问题**　模型仍然只预测下一个 token，为什么换成示范对话就能训练聊天行为？用户问题也在输入里，为什么只对 assistant 的回答计算 loss？
>
> **范围**　对话渲染、监督 mask、目标对齐、工具调用、数据打包与混合、checkpoint 继承和 SFT 训练闭环。逐 token 生成和 KV cache 在[第 10 章](./10-inference-engine.md)，聊天任务评估协议在[第 11 章](./11-chat-evaluation.md)。
>
> **先决条件**　读过[第 03 章](./03-tokenizer.md)的 tokenizer、[第 06 章](./06-base-train-loop.md)的 `x/y` 对齐与交叉熵，以及[第 07](./07-base-train-optim-scaling.md)–[08 章](./08-base-evaluation.md)的优化器和评估。
>
> **本章产物**　一条“**conversation → ids/mask → x/y → loss → 参数更新**”的主线；区分“不监督”和“不可见”，并理解工具、packing、数据混合和 optimizer warm-start 各自的职责。
>
> **源码入口**　`scripts/chat_sft.py`、`nanochat/tokenizer.py::RustBPETokenizer.render_conversation`、`tasks/common.py`、`tasks/smoltalk.py`、`tasks/mmlu.py`、`tasks/gsm8k.py`。损失仍由 `GPT.forward` 中的 `F.cross_entropy(..., ignore_index=-1)` 计算。

---

## 本章讲什么

前几章训练出了一个 base 模型。它擅长根据前缀续写，却不一定稳定遵循“用户提问、助手回答、回答结束”的聊天协议。例如，只输入“你好”，模型可能续写成广告、词典条目或一段小说，而不是以助手身份作答。

SFT 不给模型新增一个“聊天模块”，而是用示范教它：

1. **谁在说话**：输入用 user / assistant 边界区分角色；
2. **该怎么回答**：在用户问题之后，提高示范回答的预测概率；
3. **何时结束或调用工具**：学习应由 assistant 生成的协议标记。

算法仍然是“根据前文预测下一个 token”。nanochat 只是改变了训练题目：

- 把普通文本换成示范对话；
- 用户问题和工具结果继续放在上文里，供模型阅读；
- 只有 assistant 应该亲自生成的 token 才作为计分目标。

因此，“不监督用户问题”不是把问题删掉，而是**不要求模型复述问题；模型仍要读懂问题，才能把回答预测对**。

本章先用一条简单对话走通训练主线，再展开工程分支：

```text
示范对话 messages
  → 插入角色边界，得到 ids 与监督 mask
  → 构造输入 x、下一个 token 目标 y
  → 不监督的目标设为 -1
  → 计算 assistant 目标的交叉熵
  → backward → optimizer.step()
```

**读完本章你应该能**：

1. 解释 SFT 为什么仍是语言模型训练，而不是改变 GPT 架构；
2. 跟随同一条对话，从 `messages` 一直推到哪些 logits 参与 loss；
3. 区分监督 mask 与 attention mask，理解 mask=0 的上下文仍能影响学习；
4. 说明 assistant 正文、工具请求、工具输出和结束标记为什么采用不同 mask；
5. 解释 best-fit packing 和 `TaskMixture` 分别怎样组织数据；
6. 区分加载模型权重、继承 CLI 配置和恢复 optimizer state；
7. 说明 val BPB、ChatCORE 和 SFT checkpoint 在训练循环中的位置。

> **读法建议**：§1–§3 是完整主干，不需要跳读；§4–§6 展开工具协议、packing 和数据混合；§7–§8 再解释训练如何从 base checkpoint 接棒、评估并保存。

---

## 预备词汇

| 名称 | 含义 | 形状或代码对应 |
|---|---|---|
| `B` | 每张卡一次 forward 的序列行数，不是全局 token batch | `args.device_batch_size` |
| `T` | 每行模型输入的 token 数 | `args.max_seq_len` |
| `V` | 词表大小 | logits 最后一维 |
| `conversation` | 一条对话，含 `messages` 列表；每条消息有 `role` 和 `content` | Python dict |
| `ids` | 一条对话渲染后的 token id 序列 | 变长 `list[int]` |
| `mask` | 与 `ids` 等长的监督标记；1 表示要求模型预测这个目标 token | 变长 0/1 列表 |
| `row` | packing 后的一行 token，切片前长度为 `T+1` | `(T+1,)` |
| `batch_tensor` | `B` 行 token 组成的训练 batch | `(B, T+1)` |
| `x` / `inputs` | 模型读取的输入 token | `(B, T)` |
| `y` / `targets` | 输入后面一位的目标 token；不监督的目标替换为 `-1` | `(B, T)` |
| `mask_targets` | 与 `targets` 对齐的监督 mask | `(B, T)` |

表中的 `-1` 是给损失函数的“忽略这个目标”标记，不是输入 token。它怎样由 mask 产生、为什么不影响模型读取用户问题，会在 §3 的同一条对话示例中展开。

---

## 1. 预训练与 SFT：同一损失函数，选择不同预测目标

### 1.1 预训练：预测文本中每个下一个 token

对一段序列 `z_0, z_1, ..., z_N`，预训练把位置 `1...N` 都当作目标：

```text
L_pretrain = (1/N) · Σ_{j=1}^N -ln p(z_j | z_<j)
```

`z_<j` 表示 `z_j` 之前的所有 token。每个位置都在回答同一个问题：“已经看到前缀，接下来的真实 token 是什么？”

### 1.2 SFT：题目都看，只给 assistant 的答案计分

假设一条训练对话是：

```text
用户：法国首都是哪里？
助手：巴黎。
```

整段都送进模型，但 loss 只统计“巴黎。”和 assistant_end 等应由助手生成的目标。可以把它理解成：**题干保留，答案计分。**

数学上，只需从所有目标位置中选出集合 `A`：

```text
L_SFT = (1/|A|) · Σ_{j∈A} -ln p(z_j | z_<j)
```

- `j`：某个目标 token 在原序列中的位置；
- `A`：所有需要计分的位置，即 `mask=1` 的位置；
- `|A|`：这些目标的数量。

每个 token 的交叉熵公式没有变化，只是用户文本和工具返回不进入最后的平均。

### 1.3 哪些 token 计分，由“谁负责生成”决定

nanochat 采用“只监督助手目标”（assistant-only supervision）的策略。判断标准不是“普通 token 还是特殊 token”，而是：**推理时，这个 token 应由模型生成，还是由模板 / 用户 / 工具提供？**

| 来源 | 示例 | 监督标记 |
|---|---|---|
| assistant 正文 | 回答、说明文字 | `mask=1` |
| assistant 工具请求 | Python 表达式、`<\|python_start\|>`、`<\|python_end\|>` | `mask=1` |
| assistant 主动结束 | `<\|assistant_end\|>` | `mask=1` |
| 用户或模板提供的上下文 | 用户文本、user 边界、`<\|assistant_start\|>`、BOS | `mask=0` |
| 外部工具提供的内容 | 输出及 `<\|output_start\|>...<\|output_end\|>` | `mask=0` |

所以特殊 token 不能一概而论：模板放入的 assistant_start 不计分；模型应主动生成的 assistant_end、python_start 和 python_end 要计分。其他 SFT 实现也可能监督整段对话，那是另一种训练策略，不代表本章方案是唯一标准。

---

## 2. 把一条示范对话变成 `ids` 和 `mask`

### 2.1 渲染：把角色信息变成模型能读取的 token

原始数据是结构化对话：

```python
conversation = {"messages": [
    {"role": "user", "content": "法国首都是哪里？"},
    {"role": "assistant", "content": "巴黎。"},
]}
ids, mask = tokenizer.render_conversation(conversation)
```

模型不直接读取这个 dict。`render_conversation` 把消息正文编码为 token id，并插入边界 token：

```text
<|bos|>
<|user_start|> 法国首都是哪里？ <|user_end|>
<|assistant_start|> 巴黎。 <|assistant_end|>
```

边界让模型知道哪些文本属于用户、从哪里开始是 assistant 的回答。

### 2.2 同一条对话的逐 token 示例

为了能手算索引，**这里只做教学简化**：假设用户问题编码为一个 token `Q`，回答编码为两个 token `A₁`、`A₂`。表中的 **示例 token id（假设值）** 是为演示临时指定的编号，不是真实 tokenizer 的输出；实际切分和编号由词表决定。

**位置与 token id 不同**：例如位置 5 上的示例 token id 是 15；位置表示它在序列中的顺序，id 表示它在词表中的编号。

| 原始位置 | token 简写 | 示例 token id（假设值） | mask | 理由 |
|---:|---|---:|---:|---|
| 0 | BOS | 10 | 0 | 序列起点，由渲染器提供 |
| 1 | user_start | 11 | 0 | 用户边界，由渲染器提供 |
| 2 | `Q` | 12 | 0 | 用户问题，是回答的条件 |
| 3 | user_end | 13 | 0 | 用户边界，由渲染器提供 |
| 4 | assistant_start | 14 | 0 | 提示现在轮到 assistant |
| 5 | `A₁` | 15 | 1 | assistant 正文第一个 token |
| 6 | `A₂` | 16 | 1 | assistant 正文第二个 token |
| 7 | assistant_end | 17 | 1 | 模型应主动生成的结束标记 |

因此：

```text
ids  = [10, 11, 12, 13, 14, 15, 16, 17]
mask = [ 0,  0,  0,  0,  0,  1,  1,  1]
```

**mask 标记的是“这个 token 是否应该成为预测目标”**，还不是“同一个输入位置的 logits 是否监督”。两者之间差一位，下一节马上对齐。

### 2.3 为什么 assistant start 和 end 不对称

- `assistant_start` 是模板放进上下文的“现在轮到你”的提示，模型不必生成它；
- 模型读取这个提示后，应预测 `A₁`，所以 `A₁` 的 mask=1；
- `assistant_end` 需要模型自己生成，才能告诉引擎这次回答结束，所以它的 mask=1。

这个设计并不矛盾：**`assistant_start` 自身不作为目标，但它所在输入位置的 logits 用于预测第一个回答 token，因此会被监督。**

先记住这张表，下一节继续使用同一组 id，不换例子。

---

## 3. 从 `ids/mask` 到 loss：mask 必须跟目标对齐

### 3.1 先构造输入和目标

暂时只把 §2 的一条对话看成一行 `row`，令 `T=7`，则 `row` 长度是 `T+1=8`：

```text
row = [10, 11, 12, 13, 14, 15, 16, 17]
x   = [10, 11, 12, 13, 14, 15, 16]      # row[:-1]
y   = [11, 12, 13, 14, 15, 16, 17]      # row[1:]
```

模型在输入位置 `t` 的输出，预测的是后面一位的目标：

```text
y[t] = row[t+1]
```

`row[1:]` 从数组显示上是去掉第一项、向左对齐；“预测下一位”指的是目标在原序列中比输入靠后一位，不应仅凭“左移/右移”这个词记代码。

### 3.2 一行一行看：模型读到哪里、在预测什么

先不要急着看 mask。对输入位置 `t`，模型能读取从开头到当前位置的前缀：

```text
x[:t+1] = (x[0], ..., x[t])
```

它再用这段前缀预测下一项 `y[t]`。`x[<=t]` 可以表达数学直觉，但不是合法 Python 写法。

是否计算这项 loss，取决于**目标 `y[t]` 的 mask**，不是当前位置 `x[t]` 的 mask：

```text
mask         = [0, 0, 0, 0, 0, 1, 1, 1]
mask_targets = mask[1:] = [0, 0, 0, 0, 1, 1, 1]
mask_targets[t] = mask[t+1]
```

看同一条对话的完整对齐结果（后两行的 `...` 表示前面已有的全部前缀）：

| 输入位置 `t` | 预测时已有的前缀 `x[:t+1]` | 要预测的目标 `y[t]` | 目标 mask | 是否计算这项 loss |
|---:|---|---|---:|---|
| 0 | BOS | user_start | 0 | 否 |
| 1 | BOS, user_start | `Q` | 0 | 否 |
| 2 | BOS, user_start, `Q` | user_end | 0 | 否 |
| 3 | BOS, user_start, `Q`, user_end | assistant_start | 0 | 否 |
| 4 | BOS, user_start, `Q`, user_end, assistant_start | `A₁` | 1 | 是 |
| 5 | ..., assistant_start, `A₁` | `A₂` | 1 | 是 |
| 6 | ..., assistant_start, `A₁`, `A₂` | assistant_end | 1 | 是 |

最关键的是位置 `t=4`：模型已经读到用户问题和 assistant_start，要预测 `A₁`。assistant_start 自身不计分，但 `A₁` 要计分，所以**位置 4 的 logits 参与 loss**。

如果错用 `mask[:-1]`，得到的是输入位置的 mask `[0,0,0,0,0,1,1]`。这样位置 4 会被错误忽略，模型少学第一个回答 token；更复杂的多轮或工具边界也会整体错一位。

### 3.3 不监督的目标设为 `-1`，再计算交叉熵

源码在 packing 后按 batch 切片，核心是：

```python
# batch_tensor 保存 B 行 token；mask_tensor 保存相同布局的监督标记。
# 切片前两者形状都是 (B, T+1)。
inputs = batch_tensor[:, :-1]            # (B, T)，输入保留所有上下文 token
targets = batch_tensor[:, 1:].clone()     # (B, T)，复制目标，避免改写与 inputs 共享的存储
mask_targets = mask_tensor[:, 1:]        # (B, T)，与目标而非输入对齐
targets[mask_targets == 0] = -1           # 忽略对应的预测目标，不改 inputs
```

实际脚本通过 `.to(...)` 和 `.contiguous()` 处理设备、dtype 和连续存储；教学代码用 `.clone()` 明确隔离目标，避免设置 `-1` 时改坏输入。§2 的例子最终得到：

```text
x = [10, 11, 12, 13, 14, 15, 16]
y = [-1, -1, -1, -1, 15, 16, 17]
```

这里前四个目标被换成 `-1`，但输入 `x` 没变。`-1` 不是模型要生成的 token，而是交叉熵约定的“这一题不计分”标记：这四项不进入 loss，也不进入平均值的分母。

因此这一行只统计三项：

```text
L = (预测 A₁ 的 loss + 预测 A₂ 的 loss + 预测 assistant_end 的 loss) / 3
```

比如这三项分别是 `0.9、0.6、0.3 nats`，则 `L=0.6 nats`。它们分别来自输入位置 `4、5、6` 的 logits，不能按整行的 7 个输入位置求平均。

**答案也放在输入里，会不会偷看？** 不会，因果 attention 不允许位置 4 读取后面的 `A₁`；预测 `A₂` 时则允许读取已经出现的示范 `A₁`。训练时使用真实前缀，也叫 teacher forcing；推理时用的是模型已经生成的前缀。

**用户问题虽然不计分，仍然会影响学习。** 预测 `A₁` 时，模型需要读取 `Q`；如果没有理解问题，就很难把回答预测对。回答产生的 loss 会沿模型计算路径更新参数。因此：

```text
mask=0：不要求模型生成这个目标
不等于：模型看不到它，或它对回答没有作用
```

### 3.4 一个可运行的对齐自检

下面不加载模型，只用 §2 中假设的示例 token id 验证结果：

```python
row = [10, 11, 12, 13, 14, 15, 16, 17]
mask = [0, 0, 0, 0, 0, 1, 1, 1]

x = row[:-1]
y = row[1:]
mask_targets = mask[1:]
y_masked = [token if keep else -1 for token, keep in zip(y, mask_targets)]

assert x == [10, 11, 12, 13, 14, 15, 16]
assert y_masked == [-1, -1, -1, -1, 15, 16, 17]
assert [t for t, target in enumerate(y_masked) if target != -1] == [4, 5, 6]

# 错用输入的 mask：第一个回答 token 15 被漏掉。
y_wrong = [token if keep else -1 for token, keep in zip(y, mask[:-1])]
assert y_wrong == [-1, -1, -1, -1, -1, 16, 17]
```

到这里，主线已经闭合：**对话被渲染为 token；mask 按目标身份选择损失；损失推动同一个 GPT 更新参数。** 后面只是把这条主线扩展到工具、多条对话和真实训练配置。

---

## 4. 协议分支：工具请求、工具输出与消息边界

### 4.1 工具请求也是 assistant 回答的一部分

GSM8K 的示范解答含 `<<12/60=0.2>>` 这类计算记录。`GSM8K.get_example` 将叙述、表达式和结果转换成不同类型的 part（消息片段）：

```python
# assistant 的 content 可以不是字符串，而是按顺序排列的片段列表。
[
    {"type": "text", "text": "每分钟收入为"},
    {"type": "python", "text": "12/60"},
    {"type": "python_output", "text": "0.2"},
    {"type": "text", "text": "，再计算工作时长对应的收入。"},
]
```

这是结构示意；实际 `get_example` 保留原答案中计算标记前后的叙述。渲染后，各段的监督规则为：

| part 类型 | 渲染内容 | mask |
|---|---|---:|
| `text` | 直接编码正文 | 1 |
| `python` | `<\|python_start\|>12/60<\|python_end\|>` | 1 |
| `python_output` | `<\|output_start\|>0.2<\|output_end\|>` | 0 |

### 4.2 模型学会发出请求，工具负责返回结果

先看推理时真正发生的顺序：

```text
模型生成：<|python_start|> 12/60 <|python_end|>
程序执行 12/60
程序插入：<|output_start|> 0.2 <|output_end|>
模型读取 0.2，继续解释
```

所以训练也按同样的责任分工：

```text
工具调用整段   → mask=1，模型要学会自己生成
工具返回整段   → mask=0，程序会提供，模型只需读取
```

这不是只讨论 python_start 与 output_start 两个标记，而是包括表达式、结果及两侧边界。训练数据中已经写好了请求和结果；训练时不会每一步真的执行 Python。真正运行计算器并插入输出，是第 10 章推理引擎的工作。

如果把工具返回也设为 mask=1，模型就会额外学习“预测环境将返回什么”。nanochat 不这样做，是为了让训练行为与推理时的分工一致。

### 4.3 “不计 loss”不等于“看不到”

这里有两种完全不同的 mask：

| mask | 控制什么 | 本章例子 |
|---|---|---|
| 监督 / loss mask | 哪些目标参与交叉熵 | 用户目标、工具输出目标不参与 loss |
| attention mask | 哪些位置允许读取哪些上下文 | 不能偷看未来 token；滑动窗口还会限制可见范围 |

例如工具结果 `0.2` 的 loss mask=0：模型不需要生成它；但结果仍在输入序列中，后面的 assistant 可以读取它再继续回答。只有 attention mask 才决定一个位置能看见哪些上文。

一句话记忆：**loss mask 管“这题计不计分”，attention mask 管“这段上文看不看得见”。**

### 4.4 九个特殊 token 与编码边界

当前协议使用九个特殊 token：

```text
<|bos|>                              序列开始
<|user_start|> <|user_end|>           用户消息边界
<|assistant_start|> <|assistant_end|> 助手消息边界
<|python_start|> <|python_end|>       助手工具请求边界
<|output_start|> <|output_end|>       外部工具输出边界
```

它们由渲染器通过 `encode_special()` 显式插入，每个标记对应一个 id。普通用户正文走 `encode()`，其中的字面字符串 `"<|user_start|>"` 不会自动变成协议特殊 id。

这有助于避免把用户正文误当成角色分隔符，**但不等于解决所有 prompt injection（提示注入）问题**；正文仍可能通过语义影响模型。

### 4.5 System message 和角色交替

nanochat 没有独立的 `<|system_start|>`。如果第一条消息是 system，渲染器会复制对话，再把它的内容与第一条 user 消息合并：

```text
system 内容 + 两个换行 + 第一条 user 内容
→ 一条新的 user 消息
```

因此合并后的内容仍按 user 的 mask=0 处理。复制是为了不改动调用者的原始 dict。

去掉可选 system 后，消息要求依次为 `user、assistant、user、assistant...`，不符合会抛 `AssertionError`。工具调用不是一个额外的顶层 tool 角色，而是在 assistant `content` 中用 part 列表表达。这些是**当前渲染器支持的格式约束**，不是所有聊天模型必须使用的协议。

---

## 5. 数据布局分支：一行放多条对话

### 5.1 为什么做 best-fit packing

若 `T=2048`，一条渲染后的对话只有 100 个 token，一行只放它就会浪费大部分位置。packing（打包）尝试在一行里放多条完整对话，提高有效内容比例：

```text
[ BOS 对话 A ... ][ BOS 对话 B ... ][ BOS 对话 C ... ][ padding ... ]
```

这改变的是数据布局，不是 §3 的目标对齐规则。每条对话的 `ids` 与 mask 必须一起拼接，保证位置一一对应。

### 5.2 规则与一个明确限定的例子

`sft_data_generator_bos_bestfit` 的核心规则是：

1. 每行容量是 `T+1`，以便切出长度 `T` 的输入和目标；
2. 从数据集预取对话，维护默认 100 条的 buffer（候选缓冲区）；
3. 每次选择 buffer 中**能完整放入剩余空间的最长对话**，连同它的 mask 一起拼接；
4. 若没有候选能放下，就用 BOS 填满剩余位置，padding mask 全为 0。

例如容量 17，候选对话长度为 `[10,7,6,4]`。**为便于演示，先假设只考虑这四条，不补充 buffer**：

```text
第 1 行：放 10 → 剩 7 → 放 7 → 装满
第 2 行：放 6  → 剩 11 → 放 4 → 剩 7 → padding 7
```

真实 generator 会持续补充 buffer，第二行的剩余 7 也可能被新对话填上；只有补充后的候选都放不下时才 padding。

### 5.3 Padding 不计分，但相邻对话仍可能互相看见

先说 padding：它的目标 mask=0，因此目标会改为 `-1`。源码还显式排除从内容末尾指向第一个 padding token 的预测：

```python
targets[i, content_len - 1:] = -1
```

这里 `content_len-1` 对应“最后一个内容 token 预测第一个 padding token”的输入位置，所以也应忽略。conversation 开头的 BOS 自身 mask=0，不作为受监督目标。

**同一行里的多条对话没有被严格隔离。** 例如一行是：

```text
[BOS 对话 A][BOS 对话 B][padding]
```

当模型处理对话 B 时，因果 attention 允许它读取前面的 token，因此只要对话 A 还在当前层的可见窗口内，B 就可能看到 A 的内容。中间的 BOS 只是提示“这里开始了一段新对话”，不会让模型自动忘掉前文。

若要让 B 完全看不到 A，需要额外构造按对话分块的 attention mask；nanochat 当前没有这样做。这里的 loss mask 也只决定哪些目标计算损失，不控制哪些输入可见。不同 batch 行之间仍是独立的：第 0 行不会读取第 1 行。

### 5.4 渲染截断与 packing 的不同责任

`render_conversation` 默认先截取最多 2048 个 token，可能丢掉长对话尾部，甚至截在回答或工具块中间。packing 只是把**已经渲染好的序列**完整放入，不会继续拆开它：

```text
原始对话
  → 渲染：可能按 max_tokens 截断
  → packing：完整放入渲染结果，或选择其他候选 / padding
```

所以不能把 best-fit 说成“原始对话永不丢 token”。当前 SFT generator 也没有把 `args.max_seq_len` 传给渲染器；若显著缩小 `T`，要确认渲染后的对话能放进 `T+1` 的行容量。过长且无法放入的候选不会被 packing 自动切短。

### 5.5 多卡训练：数据怎样分，训练怎样一起停

多卡训练通常为每张 GPU 启动一个 Python 进程。进程编号叫 **rank**，进程总数叫 `world_size`。下面假设 `world_size=4`。

完整流程分为两部分：

```text
数据生成器：各 rank 读取不同索引 → 渲染 → 本地 best-fit → 返回 batch
训练循环：  用 batch 做 forward/backward → 同步结束信号 → 一起退出
```

#### 1. rank 先分到不同的样本索引

`TaskMixture` 已经把所有任务混合并打乱成一个样本列表。每个 rank 从自己的编号开始，每次把索引加 4：

```text
rank 0：0, 4,  8, 12, ...
rank 1：1, 5,  9, 13, ...
rank 2：2, 6, 10, 14, ...
rank 3：3, 7, 11, 15, ...
```

这里的数字是**混合列表中的位置**，不是 token id，也不是原始数据文件的行号。一个索引真正变成训练数据，要经过：

```text
TaskMixture[index]
  → 找到某个 Task 的某条样本
  → 返回 conversation
  → render_conversation 得到 ids/mask
  → 放进这个 rank 自己的 buffer
  → 在本地 buffer 中做 best-fit packing
  → 组成固定形状的 batch
```

所以，索引分片先决定“哪些样本归这个 rank”，best-fit 再决定“这些样本以什么组合进入训练行”。

#### 2. `cursor` 记录预取，`consumed` 记录装入 batch

每个 rank 都有自己的 `cursor` 和 `consumed`：

```text
cursor：下一次要预取哪个 TaskMixture 索引
consumed：有多少本地样本已经从 buffer 取出并装入 batch，换算到全局索引尺度
```

以 rank 0 为例，`cursor` 依次为 `0、4、8、12...`。读入 buffer 后，best-fit 可能先选择索引 0 和 8，索引 4 和 12 仍留在 buffer。因此“已经预取”不等于“已经用于准备训练 batch”。

源码每装入一条本地 conversation，就执行：

```python
consumed += ddp_world_size
```

为什么加 4 而不是加 1？因为每个 rank 只负责全局列表约四分之一的索引，而停止条件要与全局的 `dataset_size` 比较。若一个 rank 消费了 4 条本地样本：

```text
本地消费 4 条 × world_size 4 ≈ 全局进度 16 条
```

初始化值是 `consumed=rank`，因此它近似表示：

```text
consumed = rank + 本 rank 已装入 batch 的样本数 × world_size
```

它不是所有 rank 共同维护的精确计数，只是每个 rank 对全局进度的本地估算。

#### 3. `last_step=True` 是结束信号，不是立即退出

每次数据生成器准备好一个 micro-batch 后，会更新计数。如果满足任一条件：

```text
显式预算：it >= num_iterations
数据预算：consumed >= dataset_size
```

它就设置：

```python
last_step = True
```

但生成器仍会把刚准备好的 `(inputs, targets)` yield 给训练循环，不会在这里执行 `break`。所以刚触发结束条件的 batch 可能已经准备好，却还没有做 forward/backward。

训练循环的真实时序是：

```text
循环外：先 next(train_loader)，预取第一个 batch

正常一轮：
  当前 batch → forward → backward
  → next(train_loader)，预取下一 batch；这里可能设置 last_step=True
  → optimizer.step()，更新的是当前 batch 累积出的梯度

下一轮顶部：
  → 同步 last_step
  → 若为 True，只做最终评估与保存
  → break；刚预取的下一 batch 不再训练
```

例如 `grad_accum_steps=1`：训练 batch 1 后，预取 batch 2 时设置 `last_step=True`。batch 1 仍完成 optimizer step；batch 2 已准备但不会进入模型训练。

#### 4. 任意一个 rank 报告结束，所有 rank 一起停

不同 rank 拿到的对话长度不同，best-fit 每个 batch 装入的 conversation 数也不同，所以它们可能在不同时间达到停止条件。

下一轮循环顶部，所有 rank 对 `last_step` 执行 `all_reduce(MAX)`：

```text
同步前：[0, 1, 0, 0]    # rank 1 先达到停止条件
同步后：[1, 1, 1, 1]    # MAX=1，所有 rank 都收到结束信号
```

接着所有 rank 一起做最终 BPB / ChatCORE、保存 checkpoint，再一起 `break`。

为什么采用“任意一个到达就全体停止”？因为后续的 `reduce_scatter`、`all_reduce`、`all_gather` 都要求所有 rank 参加。若 rank 1 已退出，其他 rank 又进入下一次梯度通信，它们会一直等不到 rank 1，最终可能超时。

代价是：其他 rank 的 buffer 中可能仍有未使用样本，刚预取的 batch 也可能未训练。当前实现优先保证各 rank 不失步、通信不挂起，**不保证每条唯一样本恰好训练一次**。

这种同步只能处理正常结束。若某个进程突然 OOM、崩溃或被强制杀死，它无法参加结束同步，其他 rank 仍可能等待并报通信错误。

---

## 6. 数据来源分支：把三类任务混成一个训练集

### 6.1 三份数据各提供什么示范

| 数据集 | 当前使用的训练来源 | 主要示范内容 |
|---|---|---|
| SmolTalk | `smol-smoltalk`，约 460K 条 | 通用对话、回答风格 |
| MMLU | `auxiliary_train`，约 100K 条 | 四选题及单字母答案 |
| GSM8K | `main/train`，约 8K 条 | 数学解答、计算请求和结果 |

这些是当前配方的用途，不意味着某种能力只由这份数据决定。

### 6.2 统一接口，再渲染

每个 Task 的 `get_example(idx)` 都返回 conversation dict，再交给同一个 `render_conversation`：

```text
SmolTalk：已有 messages，检查内容类型与角色交替
MMLU：题目和选项成为 user 消息，正确字母成为 assistant 消息
GSM8K：问题成为 user 消息，解答转换成 assistant part 列表
```

MMLU 的真实 `render_mc` 格式例如：

```text
Multiple Choice question: What is the capital of France?
- Paris=A
- London=B
- Berlin=C
- Rome=D

Respond only with the letter of the correct answer.
```

assistant 示范回答为 `A`。这里字母放在选项之后，并使用无空格的 `=A`，是当前小模型配方的设计，避免答案字母与 prompt 中的编码形式不一致。

GSM8K 的表达式拆分是结构解析，不是又训练了一个 tokenizer；拆成 part 后才对各片段执行 token 编码。

### 6.3 TaskMixture：把多份索引混在一起

默认配置相当于把三份数据放进一个大列表：

```text
SmolTalk：460K 条 × 1 份 = 460K
MMLU：   100K 条 × 3 份 = 300K
GSM8K：    8K 条 × 4 份 =  32K
```

然后 `TaskMixture` 为每条记录建立“来自哪个 Task、在该 Task 中是第几条”的索引，并用固定 seed 42 打乱一次。训练时按这个列表读取，因此三类样本会交错出现，不是先训完一种再训下一种。

源码中的 `mmlu_epochs=3`、`gsm8k_epochs=4` 在这里表示**把同一数据重复加入 mixture 几份**，也叫过采样；不是外层训练循环完整跑了 3/4 个 epoch。

```python
train_tasks = [
    SmolTalk(split="train"),
    *[MMLU(subset="all", split="auxiliary_train")
      for _ in range(args.mmlu_epochs)],
    *[GSM8K(subset="main", split="train")
      for _ in range(args.gsm8k_epochs)],
]
train_dataset = TaskMixture(train_tasks)  # 内部固定打乱全部索引
```

按对话条数估算，三者约占 `58% : 38% : 4%`。这仍**不是 token 比例或 loss 权重**：一条数学解答可能比一条单字母 MMLU 回答长得多，mask=1 的 token 数也不同。

### 6.4 验证 mixture 不是最终聊天 benchmark

SFT val mixture 使用 SmolTalk test，加上 MMLU test 的 5200 条和 GSM8K test 的 420 条。后两者是**各 Task 固定 seed=42 shuffle 后的前若干条**，不是原文件顺序中的前几题。

源码注释说这些截样量用于接近某种训练混合比例，但它不保证对话、token 和监督目标三种比例都一致。这里主要用于周期性 val BPB 检查，不是完整的聊天任务评估。

还要留意评估独立性：这些 test 样本虽然不用于梯度更新，但训练中会反复查看它们的 BPB；若据此挑模型或调参，再报告同一测试集的任务分数，就不再是完全独立的一次最终测试。正式评估的协议与报告边界见[第 11 章](./11-chat-evaluation.md)。

---

## 7. 训练初始化：从 base checkpoint 接棒

SFT 启动时连续做四件事：

```text
1. 加载 base 模型参数和 metadata
2. 从 CLI / metadata 解析序列长度、batch 和基础 lr
3. 新建 SFT optimizer
4. 默认再加载 base optimizer 的历史状态，但恢复 SFT 新建的 lr
```

这四步不要混成一句“继续 base 训练”。模型参数、配置和 optimizer state 的继承方式并不相同。

### 7.1 先看各类内容怎样处理

| 内容 | SFT 的处理 |
|---|---|
| 模型参数 | 整体加载，随后继续更新 |
| tokenizer | 使用与 base 兼容的 token 映射 |
| seq len / batch | CLI 未指定时，从 metadata 读取；缺失时用 fallback |
| 三类 CLI 学习率 | CLI 未指定时，读取 base 的 `user_config` 基础值 |
| optimizer 历史张量 | 默认加载当前 rank 对应的 shard，做 warm-start |
| optimizer 参数组配置 | 加载 state_dict 时也会覆盖，之后显式恢复 SFT 新建的 lr |

`load_model("base",...)` 从 `base_checkpoints/` 读取模型权重和 metadata，再载入当前配置的 tokenizer，返回三个对象：

```python
model, tokenizer, meta = load_model(
    "base", device, phase="train", model_tag=args.model_tag, step=args.model_step,
)
```

加载器会 assert vocab size 一致，但**相同词表大小不代表相同 tokenizer**：同一个 id 对应的文本可能已改变。必须复用该模型训练时的 token 映射，不能只靠这个维度检查判断兼容。

### 7.2 继承的是 CLI 基础学习率，不是预训练末段 lr

SFT 的以下 CLI 参数默认是 `None`，表示“本次没有显式指定，请从 metadata 解析”：

- `max_seq_len`、`device_batch_size`、`total_batch_size`；
- `embedding_lr`、`unembedding_lr`、`matrix_lr`。

显式 CLI 值优先，metadata 中没有该项时用代码中的 fallback。`scalar_lr` 没有对应的 SFT CLI，`setup_optimizer` 使用默认值 `0.5`。

这里有三种不同的 lr，不能混用：

```text
base CLI 基础 lr
  ≠ base 按 batch、参数组缩放后的 lr
  ≠ base 最后一步 warmdown 后的 lr
```

SFT 读取的是第一种，再调用 `GPT.setup_optimizer` 做组内缩放。它**没有重新执行 base_train.py 的 batch 平方根缩放链**；改变 SFT batch 时也不会自动套用第 07 章的整套外推。当前继承规则是配置复用方式，不是学习率必然合适的保证。

### 7.3 Warm-start：保留优化历史，但不用 base 末段的小 lr

**warm-start** 的直觉是：模型换到 SFT 数据后，优化器不要把过去积累的“梯度通常往哪走、大小怎样”全部忘掉。默认 `--load-optimizer=1` 会加载这些历史状态；关闭后则是 **cold start**，模型权重仍来自 base，但优化器状态从零开始。

当前代码的顺序是：

```text
新建 SFT optimizer，记住各组 lr
  → 加载 base optimizer 的完整 state_dict
  → 把 lr 改回刚才记住的 SFT 值
  → 再乘 init_lr_frac=0.8
  → 训练时继续按 progress 调度 lr
```

为什么加载后还要改回 lr？因为完整 state_dict 会连参数组配置一起覆盖，而 base 训练末段的 lr 已经 warmdown 得很小；SFT 希望继承历史统计，不想照搬这个末段步长。

```python
# 教学化简：假设 base optimizer shard 存在。
optimizer = model.setup_optimizer(**sft_lrs)
fresh_lrs = [group["lr"] for group in optimizer.param_groups]

optimizer.load_state_dict(base_optim_state)   # 同时加载历史状态和参数组配置
for group, fresh_lr in zip(optimizer.param_groups, fresh_lrs):
    group["lr"] = fresh_lr * args.init_lr_frac
    group["initial_lr"] = group["lr"]
```

需要注意：日志虽写 “momentum buffers only”，实际加载的是完整 state_dict。betas、eps、weight decay、Muon 的 ns_steps/beta2 等也可能一起恢复，代码只显式改回 lr 和训练中的 Muon momentum。

因此也不能简单说“SFT 全部不做 weight decay”：新建 optimizer 时的 `weight_decay=0.0` 只给 Muon 组，AdamW 各组本来就有自己的固定值；加载 state_dict 后，Muon 的值还可能被 checkpoint 覆盖。

> **运行边界**：找不到当前 rank 的 optimizer shard 时会 cold start。当前没有通用的 optimizer 重分片逻辑；更改 GPU 数量时，历史 shard 的布局未必兼容。只加载完整模型权重不受这个 optimizer shard 限制。

### 7.4 调度：用训练进度控制更新力度

与第 07 章类似，SFT 每次更新前仍要设置 lr 和 Muon momentum。区别是 LR 函数使用归一化进度 `progress`：

```text
默认数据驱动：progress ≈ 已消费 conversation 数 / mixture 大小
指定 num_iterations：progress = generator 已产出的 micro-batch 数 / num_iterations
```

默认 `warmup_ratio=0`、`warmdown_ratio=0.5`，所以前半段保持 SFT initial_lr，后半段线性降到 0：

| progress | lr multiplier | 若 SFT initial_lr=0.016，当前 lr 为 |
|---:|---:|---:|
| 0 | 1 | 0.016 |
| 0.5 | 1 | 0.016 |
| 0.75 | 0.5 | 0.008 |
| 1 | 0 | 0 |

Muon momentum 则在前 300 个 **optimizer step** 从 0.85 升到 0.95，之后保持。

> **实现提醒**：这里的 `progress` 并不总等于 `optimizer step / num_iterations`。数据生成器每产出一个 micro-batch 就加一次内部计数，训练循环还会提前取下一 batch；有梯度累积时差异更明显。复现实验应以实际 step、梯度累积和日志为准。

---

## 8. 训练闭环：更新、评估、保存

### 8.1 参数仍按熟悉的循环更新

SFT 继续采用：

```text
得到 x/y → forward 计算监督目标的 loss
         → backward → 设置当前 lr / momentum
         → optimizer.step() → zero_grad()
```

梯度累积、fp16 GradScaler 和多 rank 通信沿用第 06–07 章的机制，梯度由 `MuonAdamW.step()` 聚合，不是 DDP backward hook。

梯度累积时还有一个数值细节：每个 micro-batch 先对自己的有效目标求平均，再除以累积次数。若不同 micro-batch 的有效目标数差很多，它们仍获得相同的 batch 权重，并不等于把全局所有监督 token 放在一起等权平均。

### 8.2 Val BPB：看正文预测，不直接检查协议行为

默认每 200 步做一次 val BPB，结束时也会评估。继续使用[第 08 章 §3.6](./08-base-evaluation.md#36-evaluate_bpb-的真实流程)的 `evaluate_bpb`：

```text
先跳过 targets=-1
  → 再排除字节数为 0 的特殊 token
  → 分别累积 total_nats 和 total_bytes
  → BPB = total_nats / (ln(2) × total_bytes)
```

所以它统计的是**mask=1 且有非零原始字节数的 assistant 正文、Python 表达式等普通 token**。即使 assistant_end、python_start、python_end 在训练 loss 中受监督，也因零字节而不进入 BPB。

因此 BPB 下降能支持“在这份 val mixture 上，普通监督 token 的条件预测更好”，但不能直接证明：

- 回答会正确停止；
- 工具边界生成正确；
- 工具调用可以执行；
- 新问题的答案正确。

### 8.3 ChatCORE：检查模型实际能否完成任务

默认每 200 步检查 ARC-Easy、ARC-Challenge、MMLU、GSM8K、HumanEval，并按随机基线中心化汇总为 ChatCORE。详细判分协议在[第 11 章](./11-chat-evaluation.md)，它和第 08 章的 base CORE 不是同一套评测。

默认分类任务不截样，生成任务每项最多 24 题，因此“训练中检查”不等于全量最终评估，生成任务的分数也可能有较大抽样波动。

源码使用未编译的 `orig_model`，与训练用的编译包装共享参数，避免变长评估频繁触发重编译；需要生成时交给 `Engine`。这是工程选择，不是另一份独立训练的模型。

> **本地运行注意**：Mac/MPS 内存不足时，可以用 `--chatcore-every=-1` 关闭训练中的 ChatCORE，结束后另做小 batch 评估。关闭周期性评估不意味着模型质量已经得到验证。

### 8.4 保存 SFT checkpoint

结束时保存到 `chatsft_checkpoints/<model tag>/`：

- 未指定 model tag 时使用 `d<depth>`；
- rank 0 保存完整模型和 JSON metadata；
- 每个 rank 保存自己的 optimizer shard；
- metadata 记录最终 step、val BPB、模型配置和原始 CLI 配置。

这个目录中的模型随后用于第 10 章推理、第 11 章聊天评估，以及第 12 章 RL 初始化。

---

## 常见误区

1. **“整段对话监督就不是 SFT。”** 整段监督也是一种 SFT 策略；本章讲 nanochat 的 assistant-only 选择，不是唯一标准。
2. **“mask=0 的用户 token 不参与学习。”** 它没有自己的预测损失，但作为上下文仍可影响回答，回答的 loss 也可能通过这条路径更新模型。
3. **“mask 跟输入对齐就可以。”** 是否计 loss 要看目标：`targets[t]=row[t+1]`，对应的标记必须是 `mask[t+1]`。
4. **“特殊 token 都不用监督。”** 模板提供的 assistant_start 不监督，模型生成的 assistant_end 和 Python 请求边界要监督。
5. **“每次 SFT 训练都会执行 Python。”** 训练读取示范里的请求和结果；真正运行工具是推理引擎的职责。
6. **“有 BOS 或 loss mask，就能隔离相邻对话的 attention。”** 不能。当前 packing 没有按对话边界新增注意力隔离，后面可能读取窗口内的前面内容。
7. **“1:3:4 是最终 token 比例。”** 这是数据集重复次数；条数占比取决于数据集大小，token 与监督目标占比还取决于长度和 mask。
8. **“vocab size 一样就能换 tokenizer。”** 大小相同不保证 id 到文本的映射相同，必须复用兼容的 tokenizer。
9. **“加载 optimizer 就照搬 base 的最后一步 lr。”** 完整 state_dict 会加载，但代码显式恢复 SFT 新建 lr，再乘初始比例和调度 multiplier。
10. **“val BPB 低就意味着协议和答案正确。”** BPB 不统计零字节特殊标记，也不等于任务成功率，需要 ChatCORE 和生成检查补充。

---

## 小结

**主干**：SFT 仍然训练同一个 GPT 预测下一个 token。nanochat 将对话渲染为 `ids/mask`，让 mask 和下一个 token 目标对齐，再通过 `ignore_index=-1` 只对选中的目标计算损失。

围绕主干有三类问题：

- **协议分工**：assistant 正文、工具请求和结束标记监督；用户、模板提示和工具输出不作为目标，但仍是可用上下文；
- **数据组织**：packing 提高行利用率，不保证原始对话未截断，也不自动隔离 attention；mixture 的重复次数不等于 token 权重；
- **训练工程**：复用 base 参数与兼容 tokenizer，区分 CLI lr 和 optimizer state，按进度调度，最后同时查看 BPB、ChatCORE 并保存 checkpoint。

如果只记一条数据流：

```text
messages → ids/mask → packed batch
         → inputs = batch[:, :-1]
         → targets = batch[:, 1:]，mask_targets = mask[:, 1:]
         → targets[mask_targets==0] = -1
         → cross_entropy → backward → optimizer.step()
```

---

## 练习

1. **同一条对话走到底**：按 §2 的示例 token id（假设值），标出原始位置 5、6、7 的目标，分别由哪个输入位置的 logits 预测？为什么位置 4 会有监督？
2. **对齐自检**：运行 §3.4，解释错误 mask 为什么漏掉目标 15，而不是把它替换成目标 14。
3. **监督与可见性**：用户 `Q` 的 mask=0，模型为什么仍能学习如何利用 `Q` 回答？再解释 loss mask 和 attention mask 的区别。
4. **工具分工**：对 `<<12/60=0.2>>`，列出请求、结果和各边界的 mask；指出训练与推理分别在哪一步真正执行计算。
5. **数据比例**：若 MMLU 重复次数从 3 改为 0，如何计算新 mixture 的条数占比？为什么不能只凭这个比例推算监督 token 权重？
6. **继承 lr**：追踪 `embedding_lr` 从 base 的 `user_config` 到 SFT 当前参数组 lr 的路径，指出哪里重新做组内缩放，哪里没有再执行 base batch 缩放。

---

## 前后章导航

- **上一章**：[第 08 章：基础模型评估——BPB、CORE 与生成样本](./08-base-evaluation.md)。界定 base 模型的续写评估。
- **本章**：用示范对话和监督 mask，让 base 模型开始遵守聊天协议。
- **下一章**：[第 10 章：推理引擎——KV 缓存与高效生成](./10-inference-engine.md)。将 SFT 模型用于逐 token 生成和工具执行；聊天评估与 RL 分别见第 11、12 章。
