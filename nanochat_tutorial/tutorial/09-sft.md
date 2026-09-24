# 第 09 章：Chat SFT：教会模型对话

> **本章源码地图**：训练入口是 `nanochat/scripts/chat_sft.py`；对话协议与 `ids/mask` 的产生在 `nanochat/nanochat/tokenizer.py::RustBPETokenizer.render_conversation`；数据任务及混合器在 `nanochat/tasks/common.py`、`nanochat/tasks/smoltalk.py`、`nanochat/tasks/mmlu.py`、`nanochat/tasks/gsm8k.py`。最终损失仍由 `nanochat/nanochat/gpt.py::GPT.forward` 中的 `F.cross_entropy(..., ignore_index=-1)` 计算；验证 BPB 在 `nanochat/nanochat/loss_eval.py::evaluate_bpb`，对话能力评估入口是 `nanochat/scripts/chat_eval.py::run_chat_eval`。

预训练结束后，我们得到的不是一个“聊天机器人”，而是一个很擅长把已有 token 接下去的 **base model**。给它一句“你好”，它也许会续写网页、小说或训练语料中的下一段；它并不知道这句话来自用户，也不知道何时该轮到自己回答。SFT（supervised fine-tuning，监督微调）解决的正是这个缺口：把高质量的“输入对话 → 应答对话”示范变成仍然熟悉的下一个 token 预测题，但只在模型应该负责的部分扣分。

本章的关键不是换了一个神秘算法。核心目标函数仍是交叉熵；真正决定模型会不会像助手一样行动的，是**对话协议、监督掩码和数据配比**。主线是：`conversation → render_conversation(ids, mask) → best-fit row → x/y 与 mask[1:] → targets=-1 → GPT.forward`。

```text
base checkpoint + 同一个 tokenizer
        │
Conversation（messages）
        │ render_conversation
        ▼
ids:  [BOS, user 标记与文本, assistant 标记与答案, ...]
mask: [ 0,          0,                   0/1, ...]
        │ best-fit 打包成长度 T+1 的 row，必要时 pad
        ▼
x = row[:, :-1]，y = row[:, 1:]；mask 也向左移一格对齐 y
        │ 把不应学习的位置设 y=-1
        ▼
GPT.forward(x, y) → 只对 assistant 目标 token 求交叉熵
```

## 学习目标

读完本章，你应该能够：

1. 解释为什么预训练的“续写能力”不会自动变成遵守角色边界的对话能力；
2. 沿真实代码说明 `render_conversation` 如何把 `messages` 变为等长的 `ids` 和 `mask`；
3. 手工检查 `x/y` 的一位 shift，并解释为什么训练掩码也必须使用 `mask[1:]`；
4. 说明为什么 assistant 正文和工具调用要监督，而用户提示、协议标记以及工具输出不应监督；
5. 看懂 SmolTalk、MMLU、GSM8K 如何变成统一的对话数据，以及 `TaskMixture` 怎样实现重复采样；
6. 区分 SFT 的核心原理、nanochat 的工程性装包/恢复设计，以及不能与正确性混为一谈的速度优化；
7. 读懂 SFT 的验证 BPB、checkpoint 保存路径，以及为什么聊天任务评估应交给第 11 章。

## 前置概念

建议先完成[第 06 章：Base Train：数据到损失与训练循环](./06-base-train-loop.md)和[第 08 章：Base Eval：BPB、CORE 与 sample](./08-base-evaluation.md)。本章在 base checkpoint 上定义聊天协议；其后第 10 章才讲如何实际生成，第 11 章专讲聊天任务评估。本章沿用下列符号：

| 符号 | 含义 | 在 SFT loader 中的形状 |
|---|---|---|
| `B` | 一个设备一个 micro-batch 中的 row 数 | `args.device_batch_size` |
| `T` | 模型输入长度 | `args.max_seq_len` |
| `V` | tokenizer 的词表大小 | logits 的最后一维 |
| `r` | 打包前的一条长度 `T+1` 的 token row | `(T+1,)` |
| `x` | 输入 token id | `(B, T)` |
| `y` | 下一个 token 目标；`-1` 表示忽略 | `(B, T)` |
| `m` | `render_conversation` 返回的 0/1 监督标记 | 与 `ids` 等长 |

`-1` 不是词表里的 token。它是 PyTorch 交叉熵的 `ignore_index`：这个位置既不产生 loss，也不产生梯度。

---

## 1. 从“会续写”到“会回答”：SFT 到底增加了什么？

预训练优化的是所有文本位置的条件概率。对 token 序列 `z_0, z_1, …`，它学习

\[
-\sum_t \log p_\theta(z_{t+1}\mid z_{\le t}),
\]

其中 \(\theta\) 是模型参数，\(p_\theta\) 是模型给下一个 token 的概率。这能积累语言、知识和模式匹配能力，却没有一个天然的变量叫“用户”或“助手”。普通文本中的问答也可能以各种格式出现。

SFT 仍然让模型预测下一个 token，只是选出“模型在部署时需要自己生成”的集合 \(A\)：

\[
\mathcal L_{\mathrm{SFT}}
= -\frac{1}{|A|}\sum_{t\in A}\log p_\theta(z_{t+1}\mid z_{\le t}).
\]

直觉上，用户的话是**条件**，不是模型应复制的答案；assistant 的话才是示范答案。模型在训练时仍可通过因果注意力读取用户 token，因而能学到“看到这种请求，应接这种回复”，但不会因为用户 prompt 本身被高频纳入目标而直接被奖励去复述 prompt。

这也解释一个常见误解：SFT 并非把模型改造成 encoder-decoder，更没有让它停止自回归。部署时仍是“给定已发生 token，预测一个 token，再把它接回去”。变化只是训练文本有了稳定、可被 tokenizer 识别的角色边界，且损失集中在期望生成区。

## 2. 对话协议：文字之外，模型还必须看见边界

`nanochat/nanochat/tokenizer.py` 的 `SPECIAL_TOKENS` 列出 SFT 所需的九个特殊 token：

```text
<|bos|>
<|user_start|>       <|user_end|>
<|assistant_start|>  <|assistant_end|>
<|python_start|>     <|python_end|>
<|output_start|>     <|output_end|>
```

它们在 tokenizer 训练后作为 special tokens 插入编码；不是普通字符串切分后碰巧得到的若干子词。`RustBPETokenizer.encode_special` 用 `encode_single_token` 取其 id。`<|bos|>` 也沿用全项目“一个新序列开始”的边界含义。

一条最普通的对话，在 `render_conversation` 中概念上被渲染为：

```text
<|bos|>
<|user_start|>用户问题<|user_end|>
<|assistant_start|>助手回答<|assistant_end|>
```

注意两个容易混淆的细节。

* `assistant_start` 是“现在轮到 assistant”的**条件标记**，源码给它 `mask=0`；回答正文与 `assistant_end` 给 `mask=1`。不要把这理解成第一个回答 token 没有监督：训练 target 会右移一位，所以“读到 `assistant_start` 后预测第一个正文 token”的 loss 取决于那个正文 token 的 `mask=1`。这样模型从该标记后的预测位置开始被要求学会生成答案，并学习何时结束。
* 如果原始数据第一条是 system message，当前实现不另设 system 标记：`render_conversation` 深拷贝该 conversation，把 system 内容、两个换行和随后的 user 内容拼在一起，再按 user 消息处理。它还断言 system 后必须是 user。

随后，代码断言消息严格交替：去掉可选 system 后，第 0、2、4… 条必须是 `user`，第 1、3、5… 条必须是 `assistant`。这不是形式主义；错位的角色会把“应当学习的 token”标错，模型却不会自动知道数据有问题。

### `ids` 和 `mask`：同一条序列的两层信息

`render_conversation(conversation, max_tokens=2048)` 返回两个等长 Python 列表。

* `ids` 是所有协议和内容的 token id；它完整提供上下文。
* `mask` 与 `ids` 一一对应，1 表示该 token 属于 assistant 需要学习的完成内容，0 表示只作条件或根本不参与监督。

函数最后会把两者都裁到最多 `max_tokens`，默认值是 2048。这里的裁剪只是防止过长对话带来内存问题；它不保证一条对话恰好等于训练时的 `args.max_seq_len`。后者由 loader 打包处理。

## 3. 工具调用：学会提出动作，不把环境回声当成自己要说的话

GSM8K 的答案文本会含 `<<表达式=结果>>`。`nanochat/tasks/gsm8k.py::GSM8K.get_example` 用正则切开这些片段：表达式成为 `{"type": "python", "text": expr}`，结果成为 `{"type": "python_output", "text": result}`，普通叙述成为 `text` part。于是 assistant 的 `content` 不一定是字符串，也可能是 part 列表。

对应地，`render_conversation` 的规则是：

```text
text:           文本 token                         mask=1
python:         <|python_start|>表达式<|python_end|> mask=1
python_output:  <|output_start|>结果<|output_end|>   mask=0
```

这条边界非常重要。模型在真实交互中应当决定“调用什么 Python 表达式”，所以调用标记和表达式要训练；但表达式的执行结果由 Python 工具在运行时送回，模型不应被训练成凭空预测工具输出。若把 `python_output` 也监督，离线 loss 可能看起来更低，却改变了责任边界：模型会被鼓励背诵或猜测环境反馈。反过来，mask 为 0 不代表模型看不见结果；后续 assistant token 仍可注意到它，并据此写出解释与最终答案。

这是一个可迁移原则：**监督谁负责产生的 token；把外部系统、用户和环境返回值留在上下文中但不计 loss。** 它不是“工具输出永远没有价值”，而是不要把它误标为模型自主动作。

## 4. 最关键的一格：target 和 mask 为什么都要 shift？

回忆预训练的行切分。若完整 row 是

```text
r = [r0, r1, r2, r3, r4, r5]          长度 T+1 = 6
x = [r0, r1, r2, r3, r4]              x.shape = (1, 5)
y = [r1, r2, r3, r4, r5]              y.shape = (1, 5)
```

模型在 `x` 的第 `j` 个位置输出 logits，预测的恰好是 `y[j] = r[j+1]`。因此若原始逐 token mask 为

```text
m = [0, 0, 0, 1, 1, 1]
```

正确的目标掩码必是 `m[1:] = [0, 0, 1, 1, 1]`，而不是 `m[:-1]`。后者会把“预测 `r3`”的损失错误地按 `r2` 的身份决定，整个监督边界右移一格。

```text
原 row:       user_start  用户文本  user_end  assistant_start  回答A  回答B  assistant_end
原 mask:          0          0        0            0            1      1         1

模型输入 x:   user_start  用户文本  user_end  assistant_start  回答A  回答B
预测目标 y:    用户文本    user_end  assistant_start  回答A     回答B  assistant_end
目标 mask:         0          0         0             1         1         1
                                                        ↑
                                     读到 assistant_start 后，开始监督回答
```

`chat_sft.py::sft_data_generator_bos_bestfit` 的真实处理正是：

```python
inputs = batch_tensor[:, :-1]       # (B, T)，转为 int32
targets = batch_tensor[:, 1:]       # (B, T)，转为 int64
mask_targets = mask_tensor[:, 1:]   # 与 targets 对齐
targets[mask_targets == 0] = -1
```

教学化简的最小实验如下，不需要真实 tokenizer；用字母只为看清索引：

```python
row  = [10, 11, 12, 13, 14, 15]
mask = [ 0,  0,  0,  1,  1,  1]
x = row[:-1]
y = row[1:]
y_masked = [token if keep else -1 for token, keep in zip(y, mask[1:])]
assert x == [10, 11, 12, 13, 14]
assert y == [11, 12, 13, 14, 15]
assert y_masked == [-1, -1, 13, 14, 15]
```

这不是仓库原样代码，而是对其切片逻辑的可运行缩影。实际 `GPT.forward` 先得到 `logits.shape == (B, T, V)`，再展平为 `(B*T, V)`；`targets` 展平为 `(B*T,)`，并调用 `F.cross_entropy(..., ignore_index=-1)`。所以只有 `13、14、15` 对这个玩具行贡献梯度。

## 5. 一行放多段对话：best-fit、padding 与不丢 token

一条对话通常比上下文短。SFT generator 设 `row_capacity = args.max_seq_len + 1`，以便切片后恰好得到 `(B,T)` 的 `x/y`。每一 row 都从一个完整 conversation 开始；它维护一个默认大小为 100 的 `(ids, mask)` buffer，并在剩余空间中寻找**能完整放下的最长 conversation**。这就是代码注释所称的 best-fit packing。

若没有任何 buffer 中的完整对话还能装入剩余位置，代码不截断这条候选对话，而用 `bos_token` 填满余位、以 0 填充 mask。随后还会按记录的 `content_len` 将对应 target 设为 `-1`。这样做的直接取舍是：少量位置浪费计算，但不会为了塞满 row 而把**已经进入 buffer 的完整 conversation**切断或丢弃 token。要区分这一点与前面 `render_conversation(..., max_tokens=2048)` 的独立长度上限：后者仍可能先截短超长原始对话。特殊的是，填充值选 BOS 并不意味着训练 BOS；padding mask 和显式的 `targets[i, content_len-1:] = -1` 都确保它没有 loss。

这部分是**工程数据布局**，不是 SFT 的数学定义。best-fit 能减少 padding、提升有效 token 比例；buffer 大小、填什么 token、是否允许跨样本拼接，都可以有别的实现。这里更不能把“吞吐高”误解为“语义一定更好”：正确性的底线是 ids/mask 同步、目标只含应监督 token、padding 不产生损失。

在多卡场景，`cursor` 从 `ddp_rank` 起步，并按 `ddp_world_size` 跳跃，使不同 rank 取不同 conversation；停止进度则按已消费样本而不是预取到 buffer 的数量近似计算，结束信号还以 `dist.all_reduce(..., MAX)` 同步，避免某张卡先结束而其他卡等待。

## 6. 三类数据如何合成一个课程

`Task` 是“可按索引取 conversation”的轻量抽象。`TaskMixture` 收集每个任务的 `(task_idx, local_idx)`，再以固定种子 `random.Random(42)` 打乱；因此来自不同任务的样本交错出现，而不是先练完一种再练另一种。

`chat_sft.py` 的训练 mixture 是：

* `SmolTalk(split="train")` 一份。源码注释标为约 46 万条通用对话，`SmolTalk` 检查可选 system message 和 user/assistant 交替，保留原 `messages`。
* `MMLU(subset="all", split="auxiliary_train")` 重复 `args.mmlu_epochs` 份，参数默认 3。`MMLU.get_example` 将四选题经 `tasks.common::render_mc` 组织为 user prompt，将正确的 `A/B/C/D` 单字母放进 assistant 回复。这里格式细节确实重要：`render_mc` 让 choice 后接 `=A`，并要求回答仅为字母，使 prompt 中答案字母的 token 化形式与 assistant 目标一致。
* `GSM8K(subset="main", split="train")` 重复 `args.gsm8k_epochs` 份，默认 4。它带来带步骤的数学解答以及上节的 Python 工具协议。

所谓“epochs”在这里不是一个通用训练框架自动推导出的概念，而是通过把同一 `Task` 实例重复放进 list 来过采样。重复次数改变的不是单一例子的标签，而是它在混合数据中的出现频率。通用对话、选择题格式、数学/工具格式之间的比例，是 SFT 配方的一部分；它值得实验，但没有任何一个默认比例能保证所有任务都最好。

验证集也构造成 `TaskMixture`：SmolTalk test、MMLU test 的前 `stop=5200` 条、GSM8K test 的前 `stop=420` 条。后两个限制在代码注释中说明是为接近训练比例，而不是完整测试集评估的替代品。

## 7. 从 base checkpoint 接着练：继承什么，重置什么？

SFT 入口通过 `load_model("base", device, phase="train", ...)` 读取预训练模型、tokenizer 和 metadata。`nanochat/nanochat/checkpoint_manager.py::load_model` 将 `"base"` 映射到 `base_checkpoints`；加载时还会检查 checkpoint 的 `vocab_size` 与当前 tokenizer 词表大小一致。这是必要兼容性检查：协议 special token 的 id 与 embedding 行必须对得上。

许多参数的命令行默认是 `None`，含义不是 Python 意义上的空配置，而是“优先继承 checkpoint”。`chat_sft.py` 从 metadata 或其 `user_config` 中解析 `max_seq_len`、`device_batch_size`、`total_batch_size` 及三组学习率；若旧 metadata 没有值，才使用代码中写明的 fallback。显式传参会覆盖继承值并打印提示。这样 SFT 通常延续预训练时已经匹配模型规模的 batch/优化配置，同时仍允许有意识地改动。

优化器则新建为 `model.setup_optimizer(...)`，且 SFT 设 `weight_decay=0.0`。在默认 `--load-optimizer=1` 下，脚本还尝试从 base checkpoint 读本 rank 的 optimizer shard，以保留动量等内部状态。不过 `load_state_dict` 会把预训练时的参数组元数据也覆盖进去；源码因此先保存新鲜 SFT 的各组学习率，加载后只恢复这些 LR。注释给出的原因是预训练末尾 warmdown 后 LR 接近零，直接继承会让 SFT 几乎不更新。

这是**训练连续性工程**，不是 SFT 必不可少的理论条件。保留动量可能更平滑；找不到 optimizer checkpoint 时脚本会警告并从新 optimizer 状态开始。模型权重仍来自 base checkpoint，两种情况都不是从随机模型重新训练对话。

学习率日程也适配“数据驱动的训练终点”：`get_lr_multiplier(progress)` 按 epoch 进度做线性 warmup、恒定段和线性 warmdown，默认 `warmup_ratio=0.0`、`warmdown_ratio=0.5`、`init_lr_frac=0.8`、`final_lr_frac=0.0`。Muon 参数组的 momentum 还在前 300 step 从 0.85 线性升到 0.95。这些是 nanochat 当前配方；不要把它们误读为所有 SFT 都必须采用的核心原理。

## 8. 训练闭环：该看哪些指标？

训练循环与预训练相同地做 `forward → backward → optimizer.step`，并按 `total_batch_size` 计算梯度累积次数。每个 micro-step 的 loss 会除以 `grad_accum_steps`，使累加梯度对应一个总 batch 的平均，而不是随累积次数放大。日志中的 `train/loss` 是 EMA 平滑值，适合看趋势，却不等于真实对话质量。

脚本有两层验证：

1. **验证 BPB**：每隔 `eval_every`（默认 200）步，`evaluate_bpb` 调用模型的 `loss_reduction='none'`，并只统计非负 target、且不是 special token 的字节数。因为 SFT y 中本来有大量 `-1`，它明确跳过这些位置。BPB 可监控模型对被监督 assistant 文本的条件预测是否恶化，但它不保证工具实际可用，也不能替代任务成功率。
2. **聊天任务检查**：脚本还会按 `chatcore_every`（默认 200）以未 `torch.compile` 的 `orig_model` 建立 `Engine`，调用 `run_chat_eval`。这是一项较慢的训练期行为检查；任务、受限分类、生成 verifier、ChatCORE、采样预算和 `chatcore_max_cat/chatcore_max_sample` 的含义全部见[第 11 章：Chat Eval](./11-chat-evaluation.md)。这里要保留的工程事实是：`torch.compile` 模型输入形状固定，而评估输入长度变化，因此脚本刻意使用原始模型；这是兼容性选择，不是能力差异。

验证 BPB 密集平滑，却容易与最终聊天体验脱节；聊天任务检查更接近“能否完成任务”，却慢且受其协议影响。应并排观察，但不要为了一个数字只调整答案格式。

训练结束时，`save_checkpoint` 写入 `chatsft_checkpoints/<model tag>/`；未指定 tag 时目录名为 `d<depth>`。rank 0 写模型与 JSON metadata，每个 rank 写自己的 optimizer shard。metadata 记录最终 step、最后验证 BPB、模型配置和原始用户 CLI 配置；后续评估或 RL 才能可靠地定位、重建这份模型。

## 常见误区

1. **“只要给整段对话做语言模型 loss 就行。”** 可以训练出某些格式，但会奖励模型复述 user、预测工具回声和协议条件。nanochat 明确将这些 target 设为 `-1`，只对 assistant 可行动作监督。
2. **“mask 对 `x` 对齐就行。”** loss 比较的是 logits 与 `y`，目标已经右移一位；必须用 `mask[:, 1:]`。这是最隐蔽也最致命的 off-by-one 错误之一。
3. **“assistant start 也必须 mask=1。”** 当前源码恰恰将 `<|assistant_start|>` 标为 0；它是启动生成的上下文，之后的回答文本和 `<|assistant_end|>` 才被监督。不要根据直觉私自改边界。
4. **“不监督工具输出，模型就不会利用输出。”** 不监督仅表示不要求模型生成它；输出仍留在 token 上下文中，后面的 assistant 文本可读取它。
5. **“best-fit 就是不 padding。”** 恰恰相反：无法完整放入下一条 conversation 时，当前实现选择 pad 剩余位置，优先保证不裁掉/丢弃候选对话。有效计算比例和数据完整性是取舍。
6. **“MMLU/GSM8K 各训练一次就公平。”** `TaskMixture` 的重复任务会改变采样权重；默认 MMLU 为 3 次、GSM8K 为 4 次。比较实验时必须记录 mixture，而不只记录模型结构。
7. **“验证 BPB 降了，就说明聊天一定更好。”** BPB 是受 mask 限定的 token 预测指标；聊天任务成功与 ChatCORE 是不同层面，评估口径见[第 11 章](./11-chat-evaluation.md)。
8. **“加载 optimizer 就原样沿用预训练学习率。”** 脚本仅意图 warm-start 动量，并在加载后恢复新建 SFT 的 LR，避免预训练 warmdown 的近零 LR 让微调停滞。

## 小结

SFT 可以用一句话概括：**把对话转写为带角色边界的 token 序列，并只对模型应该自己产生的下一个 token 反向传播。** 它保留了预训练的自回归 Transformer、交叉熵和绝大多数训练骨架，却通过协议和掩码赋予“何时回答、何时调用工具、何时停止”的行为接口。

在 nanochat 中，这条链路是：`SmolTalk/MMLU/GSM8K` 产出统一的 `conversation`；`TaskMixture` 按比例交错样本；`render_conversation` 产出同步的 `ids/mask`；`sft_data_generator_bos_bestfit` 打包、shift 并把非 assistant target 置为 `-1`；`GPT.forward` 用 `ignore_index=-1` 计算交叉熵；最后用 BPB、训练期聊天任务检查和 checkpoint 构成训练闭环。ChatCORE 的定义与聊天任务细节见第 11 章。这里最值得亲手验证的是 mask 的一位对齐，而最该谨慎调整的是数据责任边界与 mixture。

## 练习

1. 取一个只有 user/assistant 各一条消息的人工 `conversation`，在本地调用 `tokenizer.render_conversation`，打印 `ids`、`mask` 与 `visualize_tokenization(ids, mask, with_token_id=True)`。逐个标出 BOS、四个角色边界和哪些 token 为绿色（mask=1）。
2. 在不改仓库训练代码的前提下，写一个十行左右的纯 Python 小脚本，复现本章 `row → x/y → mask[1:] → -1` 的玩具例子。故意改成 `mask[:-1]`，列出哪一个目标的监督身份错位。
3. 阅读 `GSM8K.get_example`：对于 `<<12/60=0.2>>`，写出它会产生的两个 part，以及这两个 part 在 `render_conversation` 中各自的特殊 token 与 mask 值。
4. 假设想加入一个“检索工具”数据集。先不写代码，设计其 `content` parts，并回答：检索请求、检索结果、assistant 根据结果写的结论，哪几段应为 mask=1？理由是什么？
5. 改变实验设计而非盲改超参数：若把 `mmlu_epochs` 从 3 改为 0，你预期选择题格式能力、通用对话能力和总有效 assistant token 比例会怎样变化？写下假设，并说明应同时观察哪两类指标来检验它。

## 前后章导航

- 上一章：[第 08 章：Base Eval：BPB、CORE 与 sample](./08-base-evaluation.md)。它界定 base checkpoint 的续写评估边界；本章开始把这个 checkpoint 校准为聊天模型。
- 下一章：[第 10 章：Inference：KV 缓存与高效生成](./10-inference-engine.md)。SFT checkpoint 如何逐 token 生成、使用 KV cache 和提供 rollout，将在那里展开；第 11 章随后评估聊天任务，第 12 章才在 rollout 上做 RL。
