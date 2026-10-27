# 第 01 章　总览：从零到聊天模型的完整链路

> **本章定位**　序篇。先把全书的流程地图和源码地图搭起来。注意力、反向传播、优化器、RL 的具体公式留给后面章节。
>
> **中心问题**　一次"和聊天模型对话"背后到底包含了哪几个训练阶段、哪几个产物？这些阶段怎么串成一条真实的调用链？
>
> **范围**　只建立全局认识，不展开公式。每个阶段的推导分别在[第 02–13 章](./README.md)。
>
> **先决条件**　无。只要你愿意把 "ChatGPT" 这类名词**当成一条可以被逐段检查的流水线**，而不是一个神秘按钮。
>
> **本章产物**　一张能反复回查的流程图；一张 checkpoint 接力表；对"**同一个网络、不同训练数据和反馈，就得到不同能力**"这件事的最小化认识。
>
> **源码入口**　`runs/speedrun.sh`、`runs/runcpu.sh`、`nanochat/checkpoint_manager.py`、`nanochat/common.py`。

---

## 本章讲什么

初学者读 LLM 代码时最常卡在一件事：**不知道哪段代码对应哪个“概念阶段”**。读 `base_train.py` 时不知道它和 `chat_sft.py` 的关系；读 `chat_rl.py` 时不知道它从哪里拿模型；看到 `checkpoint_manager.py` 时不知道它存的东西在后面哪一步被谁读走。

**本章就做一件事：把全链路画一遍，让你在后面每一章都能回答“**这段代码在整条流水线的哪一步**”**。

读完你应该能做到：

1. 讲清一段原始文本是如何经过**分词 → 预训练 → 后训练**最终变成聊天回复的；
2. 分清 **base（预训练）** / **SFT（监督微调）** / **RL（强化学习）** 各自解决什么、各自不保证什么；
3. 顺着 `runs/speedrun.sh` 跟真实的调用链，并知道它当前**没有跑 RL**；
4. 找到 tokenizer、训练数据、checkpoint 分别在磁盘上的什么位置，理解阶段之间怎么交接；
5. 看懂 `--depth` 为什么是模型规模的主旋钮，并把**核心原理、工程优化、速度赛调参**三件事分开。

---

## 预备词汇

全书只依赖三条直觉：

- **参数**：训练过程中会被反向传播更新的那些数字。整个模型的"知识"全存在这些数字里。
- **token**：文本经分词器变成的整数。**不是**一个字 / 一个词——是从训练语料里**学出来的高频片段**，可能是 "ing"、"的"、"你好"、"print("。详见[第 03 章](./03-tokenizer.md)。
- **自回归**：生成文本的方式是"**根据已有 token 预测下一个，把它接回序列末尾，再预测下一个**"。每一步只多一个 token。

和贯穿全书的形状记号：

- **`B`**：batch 里的样本数（一次并行多少条序列）；
- **`T`**：本次 forward 每条序列送入的 token 数；训练 / prefill 时通常是一段序列，逐 token decode 时为 1；
- **`V`**：词表大小。

出现 `(B, T, V)` 的张量就是"**每个样本、每个位置、每个候选 token 各一个分数**"。

---

## 1. 一个聊天模型是一条流水线

**把 ChatGPT 这类模型拆成三件事**：

- 它要会**语言**——认得字、懂得语法、知道 "cat" 和 "dog" 都是动物；
- 它要会**对话**——分清用户和助手、知道什么时候该回答、用什么格式回；
- 它要**在可以打分的任务上表现好**——例如数学题答对、代码跑通。

**没有一个阶段能同时做这三件事。** nanochat 的做法是把它们串成三段：

```text
原始文本
   │ tokenizer       （把文本和 token id 来回翻译）
   ▼
文本 token
   │ base_train      ← 第 06 / 07 章
   ▼
base checkpoint      已经会"续写"，但不会聊天
   │ chat_sft        ← 第 09 章
   ▼
SFT checkpoint       会按 user / assistant 格式回答
   │ chat_rl         ← 第 12 章（可选）
   ▼
RL checkpoint        在某个可评分的目标上更偏好正确回答
   │ Engine + chat_cli
   ▼
一条逐 token 吐出的聊天回复
```

**关键认识**：这三段训练用的是**同一个 `nanochat.gpt.GPT` 网络**。网络从头到尾没变成什么"聊天网络"。**变的是训练数据、损失算在哪些位置、反馈从哪里来**。这是整本书反复要回到的一点。

接下来三小节分别讲三个阶段**解决什么**和**不解决什么**。

### 1.1 base：从原始文本学"续写"

**目标**：给前面的 token，预测下一个 token。

**数学形式**：给定前文 `x_1, ..., x_t`，参数 `θ` 下，模型给出下一个 token 的概率：

> `p_θ(x_{t+1} | x_1, ..., x_t)`

竖杠 `|` 的意思是"**在已知前文的条件下**"。训练目标是让模型给**原文真实的下一个 token** 分配尽可能高的概率。

**自监督就是这个意思**：训练数据根本不需要人工打标签，因为"下一个 token 是什么"**原文里已经写了**——训练器把文本右移一格，就得到了标签。

真实入口在 `scripts/base_train.py`，训练循环的三行核心代码就是：

```python
loss = model(x, y)         # x, y 都是 (B, T) 整数张量，y 是 x 右移一格
loss.backward()
optimizer.step()
```

关于数据怎么从 Parquet 里一路变成 `(B, T)`，看[第 02 章](./02-dataset.md)。

**base 的能力边界**：学完之后模型会**续写**——给它半句话，它能像预训练文本里出现过的那样接下去。**它不会聊天**：它不知道"你是谁"、不知道什么时候该回答、也不知道格式。

### 1.2 SFT：把续写校准成"回答"

**目标**：仍然是"预测下一个 token"，但：

- 数据换成**对话示范**（用户问 → 助手答）；
- **只在助手的 token 上算损失**——用户的话、工具的输出不要求模型模仿。

**调用链**（在 `scripts/chat_sft.py` 里）：

```text
load_model("base", ...)                      ← 从 base checkpoint 开始
    → TaskMixture(SmolTalk, MMLU, GSM8K)     ← 对话数据来源
    → tokenizer.render_conversation(...)     ← 产出 (ids, mask)
    → sft_data_generator_bos_bestfit(...)    ← 装成 (B, T)
    → model(x, y)                             ← target 里 mask=0 的位置被改成 -1
```

**mask 怎么工作**：`chat_sft.py` 把 `mask=0` 的 target 位置改写成 `-1`；而 `GPT.forward` 里的 `F.cross_entropy(..., ignore_index=-1)` 自动跳过这些位置。于是模型**能看到**用户的问题、角色边界、工具输出（作为上下文），但**只对助手该说的 token 计算损失**。详见[第 03 章](./03-tokenizer.md)和[第 09 章](./09-sft.md)。

**SFT 的能力边界**：擅长**教格式、教语气、教工具调用方式、教示范动作**。**不保证**凭空产生新知识，也**不保证**每道题都做对——它学的是示范数据里有的东西。

### 1.3 RL：用结果反馈再优化一次

**目标**：模型自己对一道题**生成完整回答**，奖励函数给回答打一个分 `r`。然后：

- 对于得分高的回答，模型生成的那些 token **概率要提高**；
- 对于得分低的回答，概率**要降低**。

**调用链**（在 `scripts/chat_rl.py` 里）：

```text
load_model("sft", ...)                       ← 从 SFT checkpoint 开始
    → GSM8K 题目                              ← 当前 RL 只针对数学题
    → 同一题采样 k 份答案
    → GSM8K.reward(...) 给每份 0.0 或 1.0    ← 看最终数字对不对
    → 计算相对优势、按优势加权更新
```

**相对优势**的意思是：同题 k 份答案的奖励算个均值 `μ`，每份答案的"**相对优势**"是：

> `A_i = r_i - μ`

这个和同一题的 baseline 比，比均值高的推上去、比均值低的压下去。详见[第 12 章](./12-rl.md)。

**源码里一个值得知道的边界**：nanochat 把这个算法叫做 "GRPO"，但**去掉了参考模型 KL 正则、PPO ratio 和 clipping**；源码注释说明它更接近 **REINFORCE 的 on-policy 版本**，不是严格意义的 GRPO。

**RL 的能力边界**：**只能优化奖励真正测量的东西**。奖励不完整、或者可以被钻空子（比如模型学会写错解释但凑对数字），结果就会偏。所以 RL 不是"更强的训练"，是"**更专的训练**"。

### 1.4 一张表对比三个阶段

| 阶段 | 从哪里开始 | 主要训练信号 | 主要得到什么 | 主要**不**保证什么 |
|---|---|---|---|---|
| **Base** | 随机初始化参数 | 原始文本的下一 token | 通用续写与语言基础 | 自动遵守聊天协议 |
| **SFT** | Base checkpoint | 助手示范 token 的交叉熵 | 对话格式、工具调用、示范风格 | 每道题都对 / 超越示范数据的能力 |
| **RL** | SFT checkpoint | 完整 rollout 的最终奖励 | 更偏向奖励认可的回答 | 奖励之外的能力和安全性 |

**三个阶段不能互相替代**：没有 base 的语言基础，SFT 教不出会聊天的模型；没有 SFT 的对话协议，RL 的 rollout 根本不像对话；没有 RL 的奖励反馈，模型在"示范里不写、但可以客观打分"的任务上无法进一步提升。

---

## 2. 跟 `runs/speedrun.sh` 走一遍真实调用链

概念图是方便记忆用的；**真正跑起来**用的是 `runs/speedrun.sh`，一个针对 **8×H100** 节点、注释里写明 ~1.5 小时的参考脚本。

按顺序走一遍：

| 阶段 | 脚本真实入口 | 作用 |
|---|---|---|
| 环境 | `uv venv`、`uv sync --extra gpu` | 建虚拟环境、装 GPU 依赖 |
| 数据首批 | `python -m nanochat.dataset -n 8` | 下载前 8 个 shard，够训 tokenizer |
| 分词 | `scripts.tok_train`、`scripts.tok_eval` | 训一个 32768 词表的 BPE、评估压缩率 |
| 数据剩余（后台） | `python -m nanochat.dataset -n 170 &` | 后台继续下载，让下载和 tokenizer 训练**同时跑** |
| 等后台 | `wait $DATASET_DOWNLOAD_PID` | 开始 base 前确认 170 个 shard 已下完 |
| base 预训练 | `torchrun ... scripts.base_train -- --depth=24 ... --fp8` | 8 进程并行预训练一个 d24 模型 |
| base 评估 | `scripts.base_eval` | 跑 CORE、BPB、采样生成样本 |
| SFT | `scripts.chat_sft` | 从 base checkpoint 继续微调 |
| chat 评估 | `scripts.chat_eval -- -i sft` | 评估 SFT 后的模型 |

**两件事特别值得关注**：

**① 后台下载 + `wait` 是刻意的并行**。tokenizer 训练 CPU 密集但对数据量要求低；base 训练要大量数据但要 GPU。于是脚本一边训 tokenizer，一边后台下更多 shard；进 base 之前再 `wait` 一下确保下完。读源码时如果没注意到 `&` 和 `wait`，会误以为 tokenizer 训练和数据下载是串行的。

**② speedrun 当前不跑 RL**。仓库里有 `scripts/chat_rl.py`，但 `speedrun.sh` 到 SFT 评估就结束了。看到网上文章说 "nanochat 完整覆盖 RL" 时，那说的是**仓库能力**，不是**默认参考流程**。注释里的 `chat_cli` 也只是示例，默认也不会跑。

脚本末尾两个评估：

- **`base_eval` 默认 `core,bpb,sample`**：BPB 衡量语言建模能力，CORE 汇总 **22 项 in-context learning 任务**，sample 用来人工看看模型在"胡说八道"的程度；
- **`chat_eval` 默认评 ARC、MMLU、GSM8K、HumanEval**：聊天场景的四个常见基准。

这些分数都依赖**任务选择、提示格式、判分规则**，不要当成唯一的"智力值"。详见[第 08 章](./08-base-evaluation.md)和[第 11 章](./11-chat-evaluation.md)。

---

## 3. 为什么训练目标要右移一格

这一节用一个最小代码片段把 §1 的 "`p_θ(x_{t+1} | x_1, ..., x_t)`" 转成**实际的张量操作**。

```python
import torch

# 用一个 5 token 的小行示意（真实训练用 (B, T+1) 的大矩阵）
# 数字只是示意 id，含义上表示 [BOS, 我, 喜欢, 猫, EOS]
row = torch.tensor([[101, 12, 34, 56, 102]])    # 形状 (B=1, T+1=5)

x = row[:, :-1]      # [[101, 12, 34, 56]]       形状 (1, 4) ← 输入
y = row[:, 1:]       # [[ 12, 34, 56, 102]]      形状 (1, 4) ← 目标
```

然后对照着看每一个位置：

| 位置 `t` | 模型看到的输入前缀 | 要预测的目标 |
|---|---|---|
| 0 | `[101]` | `12` |
| 1 | `[101, 12]` | `34` |
| 2 | `[101, 12, 34]` | `56` |
| 3 | `[101, 12, 34, 56]` | `102` |

**四个位置可以并行算损失**——每个位置的模型只能看到它左边的前缀。这就是 "**base 从一段文本一次性拿到 T 个训练信号**" 的具体操作。但代价是：训练时位置 `i` 必须严格看不到 `i+1` 及以后，否则等于偷看答案（具体机制叫 **causal mask**，详见[第 04 章](./04-gpt-attention.md)）。

**如果 `V = 32768`**：

- `GPT.forward(x)`（不传 `y`） → 返回 `logits`，形状 `(1, 4, 32768)`：4 个位置，每处对全词表的未归一化分数；
- `GPT.forward(x, y)`（传 `y`） → 内部对 `logits.view(-1, V)` 和 `y.view(-1)` 做 `F.cross_entropy(..., ignore_index=-1)`，返回标量 loss。

**整本书反复用的那条不变量**：

> **token id → logits → 某种训练目标。**

base 的目标是"下一个 token"，SFT 把用户和控制 token 的目标置为 `-1` 后仍然是"下一个 token"，RL 则是"把生成 token 的 loss 按优势加权"。**形状和 forward 的样子都不变**，变的只是损失加在哪里、如何加权。

---

## 4. Checkpoint：阶段之间传递的接力棒

### 磁盘上的目录结构

`nanochat/common.py::get_base_dir()` 返回中间产物的根目录。优先用环境变量 `NANOCHAT_BASE_DIR`，否则默认 `~/.cache/nanochat`。`speedrun.sh` 显式设成 `$HOME/.cache/nanochat`。

目录大致是：

```text
~/.cache/nanochat/
├── tokenizer/
│   ├── tokenizer.pkl              # BPE 词表 + 合并规则（第 03 章）
│   └── token_bytes.pt             # 每个 token 的字节数，用来算 bpb
├── base_data_climbmix/            # 预训练数据的 Parquet shard（第 02 章）
├── base_checkpoints/d24/          # base 阶段的 checkpoint
│   ├── model_000123.pt            #   模型参数
│   ├── meta_000123.json           #   step、模型配置、训练配置
│   └── optim_000123_rank0.pt      #   每个 DDP rank 自己的 optimizer 状态
├── chatsft_checkpoints/d24/       # SFT 阶段的 checkpoint
└── chatrl_checkpoints/d24/        # RL 阶段的 checkpoint
```

> `000123` 只是示意六位数字的 step 命名，不是真实 step。
> `d24` 是 "`--depth=24`" 的简称——`--depth` 决定模型规模（见 §5），不同规模训出来放在不同子目录。

### 阶段之间怎么接棒

**base → SFT**：`chat_sft.py` 一开始调 `load_model("base", ...)`，到 `base_checkpoints/` 底下找最新的那个 model。

**SFT → RL**：`chat_rl.py` 调 `load_model("sft", ...)`，到 `chatsft_checkpoints/` 下读模型。

**各阶段共用 tokenizer**：三条调用链都读同一个 `tokenizer/tokenizer.pkl`——tokenizer 是**项目开始时就固定下来的，之后 base、SFT、RL 都不能换**。换掉 token id 就对不上模型的嵌入表了。

### 几个容易踩的细节

**`load_model` 的默认选择**：不传 `model_tag` 就自动挑最大的 `d<number>` 目录；不传 `step` 就挑最大 step。**交互时方便，做实验对比时危险**——你以为对比的是改动前后两版，实际 `load_model("base")` 偷偷换到了新训的目录。做实验务必**显式指定 `tag` 和 `step`**。

**optimizer state 的分布式存法**：每个 rank 存自己的 `optim_<step>_rank<r>.pt`。模型参数只由 rank 0 存一份。恢复训练时每个 rank 读自己那份。

**SFT 的 optimizer "热启动"**：`chat_sft.py` 默认尝试从 base 的 optimizer state 加载**动量缓冲**（但重置学习率）。这让 SFT 一上来不是完全 cold start。详见[第 09 章](./09-sft.md)。

**RL 不保存 optimizer state**：源码里 `save_checkpoint` 这一调用显式传 `None`。RL 的 checkpoint 只有模型参数，没有 optimizer——因为 RL 阶段 iteration 少、重启代价低。

---

## 5. `--depth`：一个主旋钮联动训练规模

### 为什么把 `depth` 当主入口

传统 LLM 代码会同时暴露层数、宽度、头数、训练 token 数、batch size、学习率等许多超参数。nanochat 仍允许单独覆盖它们，但默认配方把 `--depth` 作为主入口，再沿一组经验关系推导其他值。

`scripts/base_train.py::build_model_meta(depth)` 先决定模型形状：

```text
base_dim  = depth × aspect_ratio                 # aspect_ratio 默认 64
model_dim = base_dim 向上对齐到 head_dim 的倍数
num_heads = model_dim / head_dim                 # head_dim 默认 128
```

所以 `--depth=24` 时：

```text
base_dim  = 24 × 64 = 1536
model_dim = 1536
num_heads = 1536 / 128 = 12
```

模型构造后，脚本再做三件事：

1. **统计 scaling parameters**：当前口径是 `transformer_matrices + lm_head`，不含输入 `wte`、`value_embeds` 和控制标量；
2. **推导目标 token 数**：默认 `--target-param-data-ratio=12`，即目标 token 预算约为 `12 × scaling parameters`；
3. **推导自动 batch、LR 和 Muon WD**：仅在用户没有覆盖相关参数时使用经验缩放规则。

所以 `--depth` 不只是“叠几层”，而是会通过模型参数量联动默认训练规模。但它不是唯一自由度，也不保证这些经验关系在所有模型和预算下都最优。

### 可以单独覆盖吗

可以。`--aspect-ratio`、`--head-dim`、`--max-seq-len`、`--total-batch-size`、`--num-iterations`、`--target-flops` 等都可以单独指定。`speedrun.sh` 就显式传了 `--target-param-data-ratio=8`（覆盖 argparse 默认的 12）和 `--device-batch-size=16`。

### 一张分层对比表

把 nanochat 里的东西按"**跟 LLM 核心数学的距离**"分成三层：

| 层次 | 内容 | 读的时候重点关注 |
|---|---|---|
| **核心原理** | token、因果 Transformer、next-token 目标、SFT mask、RL 奖励 | 一定要搞懂 |
| **工程优化** | ZeRO-2 风格优化器通信、梯度累积、`torch.compile`、显式 dtype、KV cache、prefetch | 懂原理之后再看 |
| **速度赛调参** | `--fp8`、`--depth=24`、`--target-param-data-ratio=8` | 这些是特定硬件 + 特定基准的结果，别外推 |

**README 的排行榜目标**：在 8×H100 上超过 GPT-2 的 CORE 参考分数 `0.256525`，看谁用的训练墙钟最短。它**只统计 base 训练的迭代时间，不包括评估、采样、日志、checkpoint IO**；辅报 val BPB。所以它既**不是**端到端耗时榜，也**不是**聊天能力榜。完整说明见[附录 C](./performance-research.md)。

---

## 6. 环境成本：跑通 ≠ 复现能力

`nanochat/pyproject.toml` 要求 Python `>= 3.10`。依赖链是 PyTorch、PyArrow、`rustbpe`、`tiktoken`、wandb 等。

**两种安装**：

```bash
uv sync --extra gpu    # 装 CUDA 版 PyTorch 等（给 H100 / A100 等）
uv sync --extra cpu    # 装 CPU 版（给 Mac / 无 GPU Linux）
```

`speedrun.sh` 用 `torchrun --standalone --nproc_per_node=8` 启 8 进程。显存不够时 README 建议降低 `--device-batch-size`，并用梯度累积维持总 batch——训练会变慢，但不会 OOM。

**学习入口 `runs/runcpu.sh`**：面向 CPU/MPS 的小配置：

- `--depth=6`、`--head-dim=64`、`--max-seq-len=512`、`--num-iterations=5000`；
- 整条链路（tokenizer → base → base_eval → SFT）都跑完；
- 脚本自己明说"**不要期待能得到强结果**"。

**它的用途**：让你**观察流程、张量、checkpoint 怎么生成、文件怎么分布**，不是在 Mac 上复现 8×H100 的能力。[第 13 章](./13-end-to-end.md)会讲它的 Mac/MPS 典型坑（例如 base_train 的 NaN）。

**计算精度**：`nanochat/common.py::COMPUTE_DTYPE` 在 CUDA SM 80+（A100 及以上）默认选 `bfloat16`，CPU/MPS 默认 `float32`。可以用环境变量 `NANOCHAT_DTYPE` 覆盖。它影响显存、速度、数值稳定性，但**不改变训练的目标和形状**。

---

## 常见误区

1. **"`speedrun.sh` 自动跑 RL。"** 不对——当前脚本在 SFT 评估结束。RL 是单独的 `scripts/chat_rl.py`，默认流程不触发。
2. **"SFT 从零创造新知识。"** 不对。SFT 从 base checkpoint 继续、以示范数据为师，**主要校准行为**（格式、语气、工具调用）；它不会凭空学到示范数据里没有的东西。
3. **"token 就是一个字 / 一个词。"** 不对。token 是从训练语料里学出来的高频字节片段，可能半个词、半个字节、也可能是整条 `print(`。详见[第 03 章](./03-tokenizer.md)。
4. **"checkpoint 就是模型参数文件。"** 不完整。它还包含 `meta_*.json`（配置 / step）；base 和 SFT 还有按 rank 保存的 optimizer state；RL 不存 optimizer。
5. **"`--depth` 只影响层数。"** 不对。它联动决定宽度、头数、训练 token 数、batch、学习率。见 §5。
6. **"FP8 / Flash-Attention / DDP 是 LLM 的核心数学。"** 不对，它们首先都是**效率工程**。换一个硬件后端，这些细节都可能变，但核心数学（next-token 目标、因果 attention、SFT mask、RL 奖励）不变。

---

## 小结

- **nanochat 是一条流水线**：tokenizer → base → base_eval → SFT → chat_eval → (RL) → Engine。
- **同一个网络 `GPT` 贯穿三段训练**，变的是数据、损失加在哪些位置、反馈从哪里来。
- **Base** 学续写、**SFT** 学对话格式和示范动作、**RL** 在可打分任务上追加偏好。三阶段不能互相替代。
- **`speedrun.sh` 是当前默认参考运行**：8×H100、~1.5 小时、跑到 SFT 评估结束、**不跑 RL**。
- **checkpoint 是接力棒**：每阶段的产物按 `base_checkpoints/d<N>/`、`chatsft_checkpoints/`、`chatrl_checkpoints/` 分别存放，`load_model(source)` 读上一阶段的最新模型。
- **`--depth` 是默认配方的主旋钮**：它联动宽度、头数、训练 token 数、batch 和学习率，但相关参数都可显式覆盖。
- 一句话收尾：**网络可以相同；阶段由数据和反馈定义；checkpoint 是接力棒；`--depth` 管规模；硬件技巧管成本**。

---

## 练习

1. 打开 `runs/speedrun.sh`，分别找到"**环境、分词、base、SFT**" 的 4 个入口命令。**哪一条**能证明脚本确实没跑 RL？
2. 把序列 `[9, 4, 7, 2, 8]` 当成一条 token 行，手写出训练用的 `x` 和 `y`。若 `V = 1000`，`GPT.forward(x)` 返回的 logits 形状是多少？
3. `load_model("sft", ...)` 不传 `tag` 和 `step` 时，它去哪个目录找？选哪个 model？若你同时训过 `d6` 和 `d24`，会读到哪个？想避免这个坑该怎么写？
4. 把 `runcpu.sh` 和 `speedrun.sh` 并排看，列出至少 3 处差异，并对每处判断：**主要影响容量 / 上下文 / 耗时 / 硬件兼容性**里的哪一类？
5. SFT 为什么**保留**用户问题作为输入，却**只训练**助手输出？请用 `p_θ(助手 token | 用户上下文)` 这个条件概率形式解释。
6. **概念分层**题。把下面这些项分成“**核心原理 / 工程优化 / 速度赛调参**”三类：next-token prediction、因果 mask、ZeRO-2 风格优化器通信、`--fp8`、`torch.compile`、SFT mask=-1、RL 相对优势 `A_i = r_i - μ`、`--target-param-data-ratio=8`、KV cache、softcap=15。

---

## 阅读本书时的三条校正

因为网上关于 nanochat 的二手资料跨多个 revision，容易让新读者混起来。看本书的时候始终记住：

1. "nanochat 覆盖 RL" 描述的是**仓库能力**，**不是** `speedrun.sh` 默认会跑 RL。
2. "一个 `--depth` 旋钮管一切" 描述的是**自动配置哲学**，不代表所有深度、数据、硬件下都已被数学证明最优。
3. 当前模型架构包含 **smear**、**backout**、**softcap=15**（第 05 章）；当前 SFT 数据是 **SmolTalk + MMLU + GSM8K**（第 09 章）；当前 ChatCORE 是五项（第 11 章）。这些可能和你在网上读到的老文章不一样——以本仓库源码和本书为准。

需要按源码主题反查的时候，翻[附录 B：源码主题与文件索引](./source-index.md)。

---

## 前后章导航

- **上一章**：无。本章就是入口。
- **下一章**：[第 02 章：数据集：从互联网文本到训练批次](./02-dataset.md)。先把 "**原始文本如何变成 `(B, T)` 整数张量**" 这一段讲透；也会解释 "**数据先于 tokenizer**" 为什么和 "**预训练时先加载 tokenizer**" 并不矛盾。
- **后面主线**：第 03 章训 tokenizer；第 04–05 章拆 `gpt.py` 的 attention 和 MLP；第 06–07 章讲 base 预训练；第 08 章 base 评估；第 09 章 SFT；第 10 章推理引擎；第 11 章 chat 评估；第 12 章 RL；第 13 章回到端到端。
