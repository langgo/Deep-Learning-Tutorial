# 第 03 章：Tokenizer：从字节到 Token

> **本章位置**：[第 02 章：Dataset](./02-dataset.md) → **第 03 章：Tokenizer：从字节到 Token** → [第 04 章：GPT attention](./04-gpt-attention.md)
>
> **对应真实源码**：`nanochat/nanochat/tokenizer.py`、`nanochat/scripts/tok_train.py`、`nanochat/scripts/tok_eval.py`、`nanochat/tests/test_tokenizer.py`。本章还会顺着 `get_token_bytes()` 追到 `nanochat/nanochat/loss_eval.py`，并说明它在 `scripts/base_train.py`、`scripts/base_eval.py`、`scripts/chat_sft.py` 中的用途。

语言模型并不直接看见“你好”或 `print(x)`。进入 Transformer 前，一切都必须变成整数；离开模型后，整数又必须无损还原为文本。完成这个双向翻译的组件叫**分词器**（tokenizer）。

这件事看似是训练前的准备工作，实际却决定了模型的输入字典、上下文能装下多少文本、训练损失怎样比较，以及“用户的话”和“助手应学会说的话”如何区分。nanochat 使用的是一种现代而实用的组合：**用 `rustbpe` 训练字节级 BPE，用 `tiktoken` 高效编码和解码**。

---

## 学习目标

读完本章，你应能：

1. 说明字符、Unicode 字符串、UTF-8 字节和 token 的区别，并解释为什么 LLM 常从字节级 BPE 开始；
2. 手工执行几轮 BPE（Byte Pair Encoding，字节对编码）合并；
3. 沿着 `RustBPETokenizer.train_from_iterator` 读懂 nanochat 如何得到一个 32,768 词表的分词器；
4. 区分普通文本编码、显式特殊 token 编码与对话渲染；
5. 看懂 `render_conversation()` 返回的 `ids`、`mask`，以及它怎样让 SFT 只学习 assistant 输出；
6. 理解 `token_bytes.pt`、压缩率和 bits-per-byte（bpb）分别在回答什么问题。

## 前置概念：四个容易混掉的单位

先用一个小表把单位钉牢。它们都和“文本长度”有关，却不是同一件事。

| 单位 | 例子 | 是否固定宽度 | 谁关心它 |
|---|---|---:|---|
| 字符（更准确说 Unicode 码点） | `A`、`你`、`🙂` | 否 | 人、Python 的字符串操作 |
| UTF-8 字节 | `A` 常为 1 字节；`你` 为 3 字节 | 否 | 文件、网络、字节级分词 |
| token | 可能是一个字节、词的一部分、整个常见词或标点 | 否 | Transformer 的输入输出 |
| token id | 如 `1234` | 固定为一个整数类型 | PyTorch 张量、词嵌入表 |

Python 中：

```python
len("A".encode("utf-8"))   # 1
len("你".encode("utf-8"))   # 3
len("🙂".encode("utf-8"))  # 4
```

不要把 `len(text)` 当成字节数，更不要把“一个汉字等于一个 token”当规则。token 是**从训练语料统计出来的可变长片段**；同一句中文被不同词表编码，token 数也可能不同。

接下来模型只会处理形如 `[B, T]` 的整数张量：`B`（batch size）是一次并行处理的文本条数，`T`（sequence length）是每条文本的 token 位置数。第 04 章会把其中每个整数如何查表变成向量讲清楚。

---

## 1. 为什么不能直接按词、按字符，或按字节？

最经典的基线是**词级**分词：把 `I like apples` 切成 `I`、`like`、`apples`。直觉很好，但很快遇到麻烦：新词、拼写错误、网址、代码变量名、人名都会变成未登录词；若把所有可能词都放进词表，词表又会无限变大。中文没有天然空格，词边界本身还需额外判断。

另一条基线是**字符级**分词。它没有未知字符问题，却会让序列很长。英文一个常见单词要多个位置，中文、Emoji、代码和罕见 Unicode 也需要细致处理。Transformer 的计算和上下文长度都对 `T` 很敏感，过长不划算。

最后是**纯字节级**：把 UTF-8 文本直接变成 `0` 到 `255` 的字节编号。它有一个极漂亮的性质：只要输入能编码为 UTF-8，就没有未知符号；256 个初始 token 足以表示任何文本。但它又太细了：`hello` 至少要 5 个 token，常见模式没有被压缩。

BPE 取两者之长：

- 从 256 个原始字节起步，所以始终能无损表示任意 UTF-8 文本；
- 在训练语料里反复把常一起出现的相邻单位合成新 token；
- 高频片段变长 token，低频片段仍可退回为字节序列。

这不是让 tokenizer “理解”单词或中文词语。它只是用有限预算，把频繁的局部字节模式打包。语义学习仍是后面 Transformer 的工作。

### 一个必须保留的细节：单 token 不一定是合法字符串

UTF-8 中，某些非 ASCII 字符占多个字节。字节级 BPE 的某一个 token 可能恰好只包含其中一部分字节；单独把它解成 Python 字符串会失败或被替换，而和相邻 token 拼起来才是合法文本。

这正是源码在 `tok_train.py` 计算长度时调用：

```python
num_bytes = len(tokenizer.decode_single_token_bytes(token_id))
```

而不是先 `decode([token_id])` 再测字符串长度的原因。`RustBPETokenizer.decode_single_token_bytes()` 直接代理 `tiktoken` 的原始字节接口，避免破坏“不完整 UTF-8 token”。这是后面 bpb 指标正确性的关键。

---

## 2. 手算 BPE：从频次到 merge

先把正则切分暂时忽略，只观察一个教学化简的小语料：

```text
low low lower
```

为了易读，下面把字符当作初始单位；真实 nanochat 的初始单位是字节，训练实现由 `rustbpe` 完成。

初始切法：

```text
l o w   l o w   l o w e r
```

统计所有相邻对，`(l, o)`、`(o, w)` 各出现 3 次。假设规则在并列时选了 `(l, o)`，第一次 merge 后创建一个新 token `lo`：

```text
lo w   lo w   lo w e r
```

现在 `(lo, w)` 出现 3 次，第二次 merge：

```text
low   low   low e r
```

此时 `(low, 空格)`、`(空格, low)` 等对会出现；假设后续选择 `(e, r)`：

```text
low   low   low er
```

再进一步，语料若经常包含 `lower`，`(low, er)` 会被合成 `lower`。每轮都做同一件事：**统计当前序列的相邻对，选最常见者，为它分配一个新 id，并替换其全部出现位置**。训练结束后，merge 的先后次序（rank）就是编码时的规则。

可以把 BPE 写成下面的优化直觉：在固定词表大小预算下，优先为最常重复的片段分配一个 id，从而减少语料所需的 token 个数。它不是全局最优压缩算法，也不保证语言学上的“词”边界；它是一种简单、有效、可扩展的贪心算法。

真实源码没有手写 `get_stats` 或 `merge` 循环，而是把高性能实现交给 `rustbpe.Tokenizer()`。`scripts/tok_eval.py` 中嵌入的 `BasicTokenizer` 片段，正好保留了这类经典教学实现的骨架：`get_stats(ids)` 找频繁对，`merge(ids, pair, idx)` 替换，然后把两个旧 token 的字节串拼成新 token。

---

## 3. nanochat 的训练路径：语料 → 32K 词表 → 磁盘文件

调用链从 `nanochat/scripts/tok_train.py` 开始：

```text
parquets_iter_batched(split="train")
  → text_iterator()
  → RustBPETokenizer.train_from_iterator(..., vocab_size)
  → tokenizer.save(.../tokenizer.pkl)
  → 写入 .../token_bytes.pt
```

### 3.1 训练文本从哪里来，如何受限？

`text_iterator()` 使用 `nanochat.dataset.parquets_iter_batched(split="train")` 逐批读取 parquet 的 `text` 列。它不是把整个数据集一次装入内存，而是逐篇 `yield` 文本。脚本有两个明确的默认限制：

- `--max-chars=2_000_000_000`：最多供 tokenizer 训练 20 亿个 Python 字符；
- `--doc-cap=10_000`：每篇文档最多取前 10,000 个字符。

第二项避免超长文档主宰频率统计；第一项给训练成本划边界。注意变量名是 `nchars`，计数用的是 `len(doc_text)`，因此这里的 “chars” 是 Python 字符计数，不是 UTF-8 字节数。

### 3.2 `SPLIT_PATTERN`：先粗切，再做字节 BPE

`nanochat/nanochat/tokenizer.py` 定义了 `SPLIT_PATTERN`。这是一个使用 Unicode 属性的正则：`\p{L}` 表示字母，`\p{N}` 表示数字。它会把文本预先切成适合 BPE 的块，例如把字母串、标点与空白、换行、数字段区别对待。这样 BPE 不会任意跨越所有边界，把不该黏连的片段大量合并。

你不需要现在背下整条正则，但要看懂其中一个刻意选择：

```python
\p{N}{1,2}
```

它把数字预切成 1 到 2 位一组。源码注释明确说明，这偏离了 GPT-4 风格中常见的 `\p{N}{1,3}`：作者针对 **32K 词表**验证过，`1,2` 更合适；`1` 稍差，`1,3` 更差。这里不是普适定理，而是当前词表预算下的实验选择。数字切得更长不必然更好：词表容量有限，给数字组合更多空间，就少给自然语言、代码或标点模式空间。

注意“先正则块、后字节 BPE”并不推翻字节级的兜底能力。BPE 的基本原子仍是字节；正则只是规定哪些局部区域一起参与合并。

### 3.3 为什么是 32,768？特殊 token 占了哪里？

`tok_train.py` 的 `--vocab-size` 默认是 `32768`，即 \(2^{15}\)。传入 `RustBPETokenizer.train_from_iterator()` 后，源码先算：

```python
vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
```

`SPECIAL_TOKENS` 共 9 个：`<|bos|>`，以及 user、assistant、Python 调用和 Python 输出各自的开始/结束标记。也就是说，在默认总词表 32,768 中，`rustbpe` 训练的普通可合并词表预算是 32,759；其中前 256 个是原始字节，剩余预算才对应学习到的 merge token。源码还断言普通词表至少有 256，保证字节基底存在。

然后它取出 `rustbpe` 的 `pattern` 和 `get_mergeable_ranks()`，构造：

```python
enc = tiktoken.Encoding(
    name="rustbpe",
    pat_str=pattern,
    mergeable_ranks=mergeable_ranks,
    special_tokens=special_tokens,
)
```

这里可把 `mergeable_ranks` 理解为 `bytes → token id/rank` 的字典；`special_tokens` 则从普通词表末尾开始连续编号。**核心原理**是 BPE 词表和 rank；`rustbpe` 训练、`tiktoken` 推理是**工程优化**：前者完成训练，后者提供成熟而高效的编码/解码实现。它们不是另一种模型算法。

`tokenizer.save()` 用 pickle 保存编码对象为 `tokenizer.pkl`。训练脚本随后做一次真实的往返检查：`decode(encode(test_text)) == test_text`，其中测试文本包含数字、缩写、符号、中文和地球 Emoji。

---

## 4. 编码 API：普通文本不是特殊指令

`RustBPETokenizer` 把 `tiktoken.Encoding` 包了一层。最重要的几个接口如下：

| 源码符号 | 含义 |
|---|---|
| `encode(text, prepend=None, append=None)` | 普通文本编码；输入可为一个 `str` 或字符串列表 |
| `decode(ids)` | 一串 id 解回文本 |
| `encode_special(text)` | 把一个已注册的特殊 token 精确编码为单个 id |
| `get_bos_token_id()` | 取得 `<|bos|>` 的 id |
| `decode_single_token_bytes(token_id)` | 取得单个 token 的原始字节 |

一个安全设计尤其值得注意：`encode()` 使用的是 `self.enc.encode_ordinary()`，所以字符串 `"<|bos|>"` 出现在普通用户文本中时，**不会自动**变成 BOS 的单 id。只有内部代码显式调用 `encode_special("<|bos|>")`，或者经 `prepend` / `append` 参数指定，才会插入控制标记。

`tests/test_tokenizer.py::test_special_tokens` 专门断言了这件事：所有特殊 token 的 id 唯一，但 `tokenizer.encode("<|bos|>")` 结果长度大于 1。否则用户只要输入一段长得像控制符的文本，就可能意外改写模型上下文结构。

`from_pretrained("gpt2")` 和 `from_pretrained("cl100k_base")` 是评估比较用的适配入口。后者在 `tok_eval.py` 被称作 GPT-4 base tokenizer 的比较对象；它们的文档边界符名字是 `<|endoftext|>`，nanochat 的包装仍把它当 `bos_token_id` 使用。请不要因此推断本项目训练出的 tokenizer 就与这些预训练 tokenizer 词表兼容：编码器、词表和模型嵌入表必须成套匹配。

---

## 5. 最小张量实验：文本如何抵达模型

下面的代码是**教学检查代码，不是仓库新增 API**；它只调用真实已有接口。可在已准备好 tokenizer 文件的环境中运行：

```python
import torch
from nanochat.tokenizer import get_tokenizer

tok = get_tokenizer()
texts = ["你好，world!", "BPE works."]
rows = tok.encode(texts, prepend="<|bos|>")
print(rows)                         # 两个 list[int]，长度通常不同
print([tok.decode(r) for r in rows]) # 应还原原文（不显示 BOS 的语义）

# 模型要求矩形张量；这里只演示把同长度样本放进一个 batch 的形状
ids = tok.encode(["hello", "world"], prepend="<|bos|>")
x = torch.tensor(ids, dtype=torch.int32)
print(x.shape)  # (2, T)，T 取决于当前 tokenizer 的实际切分
```

最后一行不要期待一个跨词表固定的 `T`。唯一可靠的形状结论是：两条等长 token 序列形成 `x.shape == (2, T)`；张量里的每个元素都是范围在 `[0, vocab_size)` 的 token id。实际训练中，不同长度样本还需要打包、填充或切段；第 06 章会回顾 loader 输出如何进入训练循环。完整数据读取与打包已在第 02 章说明。

`encode` 也支持 `prepend` 和 `append`。例如 `encode("hello", prepend="<|bos|>", append="<|user_end|>")` 的首尾恰为相应特殊 id；这由 `test_encode_prepend_append` 覆盖。

---

## 6. 对话不是纯文本：`render_conversation` 与监督 mask

预训练阶段，模型通常学习预测文档中每一个后续 token。SFT（监督微调）不同：给定用户问题，我们希望模型学会生成**助手答案**，而不是把用户问题也当作它应复述的目标。

`RustBPETokenizer.render_conversation(conversation, max_tokens=2048)` 返回两个等长 Python 列表：

- `ids: list[int]`：渲染后的整段对话；
- `mask: list[int]`：同位置的训练开关，`1` 表示 assistant 应被监督学习的 token，`0` 表示忽略。

对一个普通双轮对话，概念上的布局为：

```text
<|bos|>                                      mask 0
<|user_start|> 用户问题 <|user_end|>          mask 0
<|assistant_start|> 助手回答 <|assistant_end|> mask: start 为 0，回答和 end 为 1
```

为什么 `assistant_start` 是 0，而回答第一个 token 是 1？自回归模型在位置 \(t\) 的表示预测位置 \(t+1\) 的目标。边界标记提供“现在该回答”的条件，真正希望模型预测的是后续回答内容，以及回答结束标记。

源码逐条处理消息并检查轮次：第 0、2、4… 条必须是 `user`，第 1、3、5… 条必须是 `assistant`。如果第一条是 `system`，它会 `deepcopy` 后把 system 内容加上两个换行，拼到随后的第一条 user 内容前；测试 `test_render_conversation_system_message_merged` 验证它与手工合并后的结果完全相同。这是当前格式的明确取舍，不等于模型拥有独立 system role token。

工具调用时 assistant 的 `content` 可以是 parts 列表：

- `type == "text"`：文本及其 token 的 mask 为 1；
- `type == "python"`：`<|python_start|>`、代码、`<|python_end|>` 的 mask 都为 1，模型需要学会何时和怎样调用工具；
- `type == "python_output"`：`<|output_start|>`、解释器输出、`<|output_end|>` 的 mask 都为 0，因为测试时这些内容来自 Python 解释器，不该让模型凭空背诵。

`test_render_conversation_tool_parts` 正在检查这个边界。最后，函数用 `ids[:max_tokens]`、`mask[:max_tokens]` 同步截断，默认最多 2048 token；这能限制内存，但也可能截掉末尾回答，不能误认为它是智能保留最后一轮的策略。

在 `scripts/chat_sft.py`，这些列表被拼进固定长度行后变成张量。`inputs = batch_tensor[:, :-1]`，`targets = batch_tensor[:, 1:]`，所以两者形状都是 `(B, T)`；再用对齐后的 `mask[1:]` 把 mask 为 0 的目标改成 `-1`（ignore index）。这正是“只对助手答案算损失”从 Python 列表落实到张量的地方。

RL 推理前缀使用兄弟接口 `render_for_completion()`：它复制对话、删掉最后一条 assistant 消息，先调用 `render_conversation`，最后追加 `<|assistant_start|>`，让生成器从答案位置继续。这由 `test_render_for_completion` 验证。

---

## 7. 压缩率与 `token_bytes`：为什么损失不能只按 token 平均？

`tok_eval.py` 对新闻、韩文、代码、数学公式、科学文本以及训练/验证数据做编码。对每段文本计算：

\[
\text{bytes per token} = \frac{\text{UTF-8 字节数}}{\text{token 数}}
\]

分子是 `len(text.encode("utf-8"))`，分母是 `len(tokenizer.encode(text))`。值越大，表示平均一个 token 覆盖的原始字节越多，压缩率越好。脚本比较当前 tokenizer 与 `gpt2`、`cl100k_base`，并在每种文本上先断言 encode/decode 往返一致。它没有预设“nanochat 一定更好”；不同语言和领域的结果应由实际运行得出。

压缩率影响效率：同一上下文窗口，token 更少通常能容纳更多原始文本；同样原文，训练与推理步骤也更少。但不能只追逐它。词表过大使嵌入层和输出分类更大，罕见 token 也更难训练；而且压缩好不直接保证下一个 token 更容易预测。

更细的比较发生在 `tok_train.py` 生成的 `token_bytes.pt`。它是一个 `torch.int32` 的一维张量，形状为：

```text
(vocab_size,)
```

第 `i` 项是 token id `i` 对应的原始字节长度；特殊 token 写为 0。`get_token_bytes(device="cpu")` 负责从 tokenizer 目录加载它。

`nanochat/nanochat/loss_eval.py::evaluate_bpb` 让模型返回每个目标位置的交叉熵 `loss2d`，形状 `(B, T)`。交叉熵默认单位是自然对数单位（nats）。若第 \(j\) 个有效目标 token 的损失为 \(\ell_j\)，它所代表的原始字节数是 \(b_j\)，则：

\[
\operatorname{bpb}
= \frac{\sum_j \ell_j}{\ln 2 \cdot \sum_j b_j}.
\]

其中 \(\ln 2\) 把 nats 转成 bits；分母的 \(\sum_j b_j\) 是所有被预测的原始字节数。特殊 token 的 `b_j=0`，不贡献分子和分母；SFT 中 target 为 `-1` 的位置也被跳过。源码首先把 `(B, T)` 展平，再以 `token_bytes[y]` 取每个目标的字节长度；有负 target 时改走安全分支，防止拿 `-1` 索引张量。

为什么不用“每 token 平均 loss”？因为 token 本身的粒度会变。假设同一段 100 字节文本，分词器 A 切成 100 个 token，B 切成 25 个 token；两者按 token 平均的交叉熵没有天然可比性。bpb 把它们拉回相同的原始字节尺度，因此修改词表大小或 tokenizer 后，仍能较公平地比较模型的压缩/预测质量。这不意味着 bpb 衡量了聊天质量、事实性或推理能力；它只是语言建模损失的一个 tokenizer 较稳健的归一化指标。

`evaluate_bpb` 最终会对分布式各 rank 的 `total_nats` 和 `total_bytes` 分别求和，再计算比值。`scripts/base_train.py` 初始化 tokenizer 时就同时加载 `get_tokenizer()` 和 `get_token_bytes()`；`scripts/base_eval.py` 与 `scripts/chat_sft.py` 也把它传给验证函数。于是 tokenizer 不只是数据预处理：它还参与整个项目的训练和验证度量闭环。

---

## 8. 测试告诉我们哪些契约？

`nanochat/tests/test_tokenizer.py` 训练了一个隔离的小 tokenizer：语料 `CORPUS` 包含英文、数字、Unicode、中文、Emoji 和一小段 Python；词表大小是 `256 + len(SPECIAL_TOKENS) + 35`。这避免测试依赖用户机器上的缓存 tokenizer，也恰好把设计契约写得很清楚。

- `test_vocab_size`：最终总大小必须包含 256 字节、9 个 special 和指定数量 merge；
- `test_encode_decode_roundtrip`：已见、未见文本都必须无损往返；
- `test_encode_batch`：字符串列表返回按输入逐项对应的 `list[list[int]]`；
- `test_render_conversation_masks`：监督 id 恰好等于两段 assistant 内容加各自 `<|assistant_end|>`；
- `test_render_conversation_truncation`：`max_tokens=32` 后，`ids` 和 `mask` 都正好是 32；
- `test_render_for_completion`：最后一条 assistant 答案被移除，结果末尾是 `<|assistant_start|>`。

阅读测试的方式不是把它当成附录，而是反问：如果我改动 tokenizer，是否还保持这些性质？尤其是 round-trip、特殊 token 不被普通文本触发，以及 `len(ids) == len(mask)`，都是后续流水线可以信赖的基础。

---

## 常见误区

### 误区 1：token 就是单词，或中文一个字就是一个 token

错误。BPE token 是训练语料的高频字节片段。可能是空格加英文词的一部分、完整汉字、多个字节的一部分、标点，或常见的多字符片段。必须用当前 tokenizer 实测，不能从肉眼猜 token 数。

### 误区 2：字节级 BPE 会把中文弄坏

恰恰相反，字节基底保证任何 UTF-8 文本可表达。真正要警惕的是：单独 token 的字节未必能解成合法 UTF-8 字符串。因此统计长度使用 `decode_single_token_bytes`，完整序列才使用 `decode`。

### 误区 3：特殊 token 字符串出现在输入中，就会被识别成控制标记

在 nanochat 的 `encode()` 中不会。它走 `encode_ordinary`；只有 `encode_special()` 或 `prepend` / `append` 明确指定时才产生特殊 id。这是重要的格式与安全边界。

### 误区 4：BPE merge 发生一次后，编码时就随意把相邻字符合并

错误。训练阶段学得的是固定词表和 rank；编码阶段必须依照这些规则确定切分。换一份训练语料或改一个词表大小，编码结果就可能不同。训练好的模型不能悄悄换 tokenizer：token id 的含义对应模型嵌入表的一行，二者必须匹配。

### 误区 5：压缩率高就代表 tokenizer 或模型一定更好

压缩率只描述文本被切得紧不紧。它忽略词表参数成本、领域覆盖、预测难度和下游能力。nanochat 的 32K 和数字 `1,2` 切分是当前配置下的选择，不应被神化为所有数据集的默认答案。

### 误区 6：mask 为 0 的 token 不会出现在模型输入里

错误。它们仍在 `ids` 中，作为预测 assistant 答案所需的上下文；mask 只决定该位置对应的**目标损失**是否计算。`chat_sft.py` 通过右移后的 `mask[1:]` 将目标置为 `-1`，而不是从输入删除用户文字。

### 误区 7：把本章的 BPE 视为“现代技巧”本身

BPE 与字节兜底是核心而成熟的基线；`rustbpe + tiktoken` 是工程上把训练与推理做好做快的组合；针对 32K 词表调整数字预切分则是实验验证后的具体速度/质量取舍。三者应分开看，避免把库选择或某个小调参误读成理论必然。

---

## 小结

1. LLM 看到的是 token id，不是 Python 字符串；token 与字符、字节都不是一一对应。
2. nanochat 从 256 个原始字节开始，以 BPE 高频相邻对合并获得可变长 token，因此能无损覆盖 Unicode、代码与未知文本。
3. `SPLIT_PATTERN` 先组织局部块，随后 `rustbpe` 训练；默认总词表是 32,768，其中 9 个位置预留给 `SPECIAL_TOKENS`，推理编码器由 `tiktoken.Encoding` 构建。
4. `encode_ordinary` 与 `encode_special` 严格分工；`<|bos|>`、对话边界和工具边界只能被显式插入。
5. `render_conversation()` 的 `(ids, mask)` 把结构化对话转换为训练材料；SFT 只监督 assistant 的文本、Python 调用及 assistant 结束符，不监督用户和工具输出。
6. `token_bytes.pt` 以原始字节长度为每个 token 计量，`evaluate_bpb` 用它把 token 级 nats 归一化成可跨 tokenizer 比较的 bits per byte。

下一章我们终于把 `(B, T)` 的整数矩阵送进模型：每个 id 怎样查到嵌入向量，注意力又怎样让每个位置从此前 token 中取信息。

---

## 练习

1. **纸上 BPE**：以 `banana banana` 为语料，把字符作为起始单位，手算三次最频繁相邻对 merge。每一步写出序列、候选 pair 的频次和新 token。若出现并列，任意选择其一，但说明选择会影响后续词表。
2. **字节观察**：在 Python 中打印 `"你🙂".encode("utf-8")`、其字节列表和长度。解释为什么一个基于 256 字节原子的词表无需“未知中文字符”占位符。
3. **追踪 mask**：不用运行训练，手工画出包含 user、assistant text、assistant python、python_output 的一轮对话，标出每类边界和内容应取 0 还是 1。再对照 `test_render_conversation_tool_parts` 核对。
4. **形状题**：若 `batch_tensor` 的形状为 `(4, 129)`，请写出 `inputs = batch_tensor[:, :-1]`、`targets = batch_tensor[:, 1:]` 以及 `mask_targets` 的形状。为什么 `mask` 要右移一位？
5. **指标题**：两套 tokenizer 在同一验证文本上产生不同 token 数。为什么不能直接比较其 mean loss？用本章 bpb 公式说明 `token_bytes` 改变的是哪一项。
6. **源码阅读**：阅读 `RustBPETokenizer.from_pretrained()`，解释为什么它传给构造器的是 `<|endoftext|>`，但属性名仍是 `bos_token_id`。回答时限定在源码注释给出的历史命名语境，不要把它扩展为未验证的模型行为。

---

## 前后章导航

- **上一章**：[第 02 章：Dataset](./02-dataset.md) 说明原始文本为何先于 tokenizer 存在，以及 loader 最终需要调用本章的编码器。
- **本章**：文本经 UTF-8 字节与 BPE 变成 token id；对话在 SFT 中还会得到监督 mask。
- **下一章**：[`04｜GPT attention：嵌入、注意力与 RoPE`](./04-gpt-attention.md) 将解释 token id 如何经过嵌入、位置编码和注意力，变成“预测下一个 token”的表示。
