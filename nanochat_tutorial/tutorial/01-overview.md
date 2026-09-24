# 第 01 章：总览：从零到 ChatGPT 的全景图

> **本章位置**：第一章。先搭建流程和源码地图；attention、反向传播、优化器与 RL 的数学细节留给后文。

## 学习目标

读完本章，你应该能：

1. 说清文本怎样经过分词、预训练和后训练，最后成为聊天回复；
2. 区分 **base（预训练）**、**SFT（监督微调）**、**RL（强化学习）**分别解决的问题；
3. 沿 `nanochat/runs/speedrun.sh` 追踪真实流程，并知道它当前**没有运行 RL**；
4. 找到 tokenizer、数据与 checkpoint，理解阶段如何交接；
5. 理解 `--depth` 是主复杂度旋钮，区分核心原理、工程优化与速度赛配置。

## 前置概念

只需三条直觉：**参数**是训练中会更新的数字；**token**是文本编码后的整数，不保证等于一个字或词；**自回归**是“根据已有 token 预测下一个，再把它接回去继续预测”。

本书用形状记号：`B` 为 batch 中样本数，`T` 为每行 token 数，`V` 为词表大小。`(B, T, V)` 即每个样本、每个位置都有 `V` 个候选 token 的分数。

---

## 1. ChatGPT 是流水线，不是一个模型按钮

先把一个聊天模型分成三层能力：从大量文字中学到语言模式；学会用户与助手的交互格式；在可判分的目标上继续偏向较好的回答。nanochat 将其串为：

```text
原始文本
  → tokenizer：文本 <-> token id
  → base_train：预测下一个 token
  → base checkpoint：基础模型，擅长续写
  → chat_sft：学习对话示范
  → SFT checkpoint：会按 user/assistant 格式回答
  → chat_rl：采样、奖励、更新（可选）
  → RL checkpoint：针对奖励目标进一步调整
  → Engine + chat_cli：逐 token 对话
```

同一个 `nanochat.gpt.GPT` 网络贯穿三种训练。网络没有在 SFT 时变成“聊天网络”；变的是训练数据、计算损失的位置和反馈来源。

### base：从文本学习续写

给定 token 序列 \(x_1,\ldots,x_t\)，模型参数 \(\theta\) 给下一个 token 概率：

\[
p_\theta(x_{t+1}\mid x_1,\ldots,x_t).
\]

\(\mid\) 表示“已知前文的条件下”，\(\theta\) 是所有可学习参数。原文中下一个 token 就是答案，所以这叫自监督学习，不必给每句话人工标注。

真实入口是 `nanochat/scripts/base_train.py`：`get_tokenizer()` 读 tokenizer，`build_model_meta(depth)` 创建 `GPT`，`tokenizing_distributed_data_loader_with_state_bos_bestfit(...)` 从 `nanochat/nanochat/dataloader.py` 取 `x, y`，训练核心是：

```python
loss = model(x, y)
loss.backward()
optimizer.step()
```

预训练数据下载及 parquet 读取位于 `nanochat/nanochat/dataset.py`。base 模型会续写，但没有被专门教过谁是 user、何时 assistant 应答、怎样使用对话控制 token。因此，**会续写不等于会聊天**。

### SFT：把续写校准成回答

经典 SFT 基线仍是下一个 token 训练，只是数据换成高质量对话示范，并且通常只惩罚 assistant 输出部分。

`nanochat/scripts/chat_sft.py` 明确从 base 接棒：

```text
load_model("base", ...)
  → TaskMixture(SmolTalk, MMLU, GSM8K)
  → tokenizer.render_conversation(...)
  → sft_data_generator_bos_bestfit(...)
  → model(x, y)
```

训练数据来自 `nanochat/tasks/smoltalk.py`、`nanochat/tasks/mmlu.py`、`nanochat/tasks/gsm8k.py`。`nanochat/nanochat/tokenizer.py::RustBPETokenizer.render_conversation` 把对话渲染成 token 和同长度 mask。`chat_sft.py` 将 mask 为 0 的 target 写成 `-1`；而 `nanochat/nanochat/gpt.py::GPT.forward` 的交叉熵使用 `ignore_index=-1`。所以模型能看到用户问题、角色边界和工具输出作为条件，但只被要求预测 assistant 应说的 token。

SFT 很擅长教授格式、语气、工具调用形式和示范行为；它不会凭空产生可靠知识，也不保证每道题正确。

### RL：用结果反馈继续调整

RL 中，模型对问题生成完整回答，奖励函数给每份回答一个分数 \(r\)。高奖励回答的生成 token 概率应提高，低奖励回答则降低。

入口 `nanochat/scripts/chat_rl.py` 从 `load_model("sft", ...)` 开始，当前只针对 `tasks/gsm8k.py:GSM8K`。同一题采样多份答案，`GSM8K.reward(...)` 用最终数字是否匹配参考答案给 `0.0` 或 `1.0`。源码对同题样本计算：

\[
A_i=r_i-\mu,
\]

其中 \(r_i\) 是第 \(i\) 个回答的奖励，\(\mu\) 是同题样本平均奖励，\(A_i\) 是相对优势。随后将生成 token 的对数概率按优势加权。

注意准确边界：文件称它为“GRPO”，却移除了参考模型 KL 正则、PPO ratio 和 clipping；源码注释说明这是更简化、接近 REINFORCE 的 on-policy 版本。RL 只能优化奖励真正测量的东西；奖励不完整或可被投机，结果也会偏。

三阶段不能替代：base 提供广泛的语言基础；SFT 教对话协议；RL 针对可验证目标追加偏好。

| 阶段 | 从哪里开始 | 主要训练信号 | 主要得到什么 | 不保证什么 |
|---|---|---|---|---|
| Base | 随机初始化 | 原始文本的下一 token | 通用续写与语言基础 | 自动遵循聊天协议 |
| SFT | Base checkpoint | assistant 示范 token 的交叉熵 | 对话格式、工具调用形式与示范行为 | 每题正确或超越示范数据 |
| RL | SFT checkpoint | 完整 rollout 的结果奖励 | 更偏向奖励认可的回答 | 奖励之外的全面能力与安全性 |

第一次通读时，先记住这张表；优化器、采样与奖励公式可在后文第二遍补齐。

---

## 2. 跟随 `speedrun.sh` 看真实调用链

流程图是概念，`nanochat/runs/speedrun.sh` 才是当前参考运行。它面向空白的 **8×H100** 节点，注释写明约需 1.5 小时：

| 阶段 | 脚本真实入口 | 作用 |
|---|---|---|
| 环境 | `uv venv`、`uv sync --extra gpu` | 创建虚拟环境、安装 GPU 依赖 |
| 数据 | `python -m nanochat.dataset -n 8` | 下载 tokenizer 所需首批数据 |
| 分词 | `scripts.tok_train`、`scripts.tok_eval` | 训练 32,768 词表 BPE 并检查压缩 |
| 后台下载 | `nanochat.dataset -n 170 &`、`wait` | 下载与 tokenizer 训练重叠，训练前等待完成 |
| base | `torchrun ... scripts.base_train -- --depth=24 ... --fp8` | 八进程预训练基础模型 |
| base 评估 | `scripts.base_eval` | CORE、BPB、生成样本 |
| SFT | `scripts.chat_sft` | 从 base checkpoint 微调聊天行为 |
| chat 评估 | `scripts.chat_eval -- -i sft` | 评测 SFT 模型 |

两点特别重要：后台下载在 base 前由 `wait $DATASET_DOWNLOAD_PID` 同步；并且**当前 speedrun 没有调用 `scripts/chat_rl.py`**。仓库包含 RL 脚本，不代表默认参考流程跑了 RL。末尾的 `chat_cli` 命令也只是注释示例。

`base_eval.py` 默认运行 `core,bpb,sample`：BPB 衡量文本预测/压缩，CORE 汇总上下文学习任务，sample 供人工检查。`chat_eval.py` 评测 ARC、MMLU、GSM8K、HumanEval。分数都依赖任务、提示和判分规则，不能当作唯一“智力值”。

---

## 3. 最小张量实验：目标为何右移一格？

以下是**教学化简代码，不是仓库原样代码**。真实 loader 还处理分布式切分、文档打包、BOS 对齐和设备传输。

```python
import torch

# [BOS, 我, 喜欢, 猫, EOS]；数字只是示意 token id
row = torch.tensor([[101, 12, 34, 56, 102]])  # (B=1, T+1=5)
x = row[:, :-1]  # [[101, 12, 34, 56]]，形状 (1, 4)
y = row[:, 1:]   # [[ 12, 34, 56,102]]，形状 (1, 4)
```

模型看到 `101` 时应预测 `12`；看到 `101, 12` 时应预测 `34`，依此类推。若 `V=32768`，`GPT.forward(x)`（不传 targets）输出 logits，形状为 `(1, 4, 32768)`：四个位置，每处对全部词表的未归一化分数。传入 `y`，`GPT.forward(x, y)` 返回一个标量 loss。

真实 `GPT.forward` 最后由 `lm_head` 得到 `(B, T, V)`，再以 `F.cross_entropy(..., ignore_index=-1)` 得到损失。SFT 的 `x/y` 形状不变，只是用户和控制 token 对应的 `y` 被置为 `-1`；RL 也使用 token 序列，只把生成 token 的损失按优势加权。牢记不变量：**token id → logits → 某种训练目标。**

---

## 4. checkpoint：阶段间传递的接力棒

`nanochat/nanochat/common.py::get_base_dir()` 决定中间产物目录：设置 `NANOCHAT_BASE_DIR` 时使用它，否则默认 `~/.cache/nanochat`。speedrun 显式设置为 `$HOME/.cache/nanochat`。

`nanochat/nanochat/checkpoint_manager.py::save_checkpoint` 的布局可概念化为：

```text
~/.cache/nanochat/
├── tokenizer/
│   ├── tokenizer.pkl
│   └── token_bytes.pt
├── base_data_climbmix/          # parquet shard
├── base_checkpoints/d24/
│   ├── model_000123.pt          # 模型参数
│   ├── meta_000123.json         # step、模型和训练配置
│   └── optim_000123_rank0.pt    # 对应 DDP rank 的优化器状态
├── chatsft_checkpoints/d24/
└── chatrl_checkpoints/d24/
```

`000123` 仅示意六位 step 命名，不是某次训练的实际结果。多 GPU 下每个 rank 各自保存优化器状态；模型和 JSON 只由 rank 0 保存。

`load_model(source, ...)` 将 `base`、`sft`、`rl` 映射到三类目录。省略 `model_tag` 时优先选最大 `d<number>`，省略 `step` 时选最大模型 step。交互时方便，实验比较时却应显式记录 tag/step，以免悄悄加载错版本。

SFT 默认还会尝试加载 base optimizer state 以复用动量缓冲，并恢复新设的 SFT 学习率；当前 RL 保存 checkpoint 时传入 `None`，不保存 optimizer state。这是当前仓库策略，不能泛化为所有项目的规律。

---

## 5. `--depth`：一个旋钮背后的规模设置

传统脚本会暴露层数、宽度、头数、训练 token、batch、学习率等许多参数。nanochat 将 Transformer 层数 `--depth` 作为主复杂度旋钮：`base_train.py` 默认 20，speedrun 使用 24。

`base_train.py:build_model_meta(depth)` 先计算：

\[
\text{base\_dim}=\text{depth}\times\text{aspect\_ratio},
\]

其中 `aspect_ratio` 默认 64；再把宽度对齐到 `head_dim` 倍数，并令 `num_heads = model_dim / head_dim`。随后程序统计 scaling parameters，按 `--target-param-data-ratio` 推导目标训练 token；未指定总 batch 时也会推导 batch，并据此缩放学习率与权重衰减。故 depth 不只是“多叠几层”。

它并非封死其他参数：`--head-dim`、`--max-seq-len`、`--total-batch-size`、`--num-iterations` 都可以覆盖；speedrun 自己也覆盖了参数—数据比例与每设备 batch。学习时应分层：

- **核心原理**：token、因果 Transformer、下一个 token 目标、SFT mask、RL 奖励；
- **工程优化**：DDP、梯度累积、`torch.compile`、显式 dtype、KV cache、预取；
- **实验性速度赛技巧**：speedrun 的 `--fp8`、d24、特定数据比例，针对特定硬件和基准，非理解 LLM 的前置数学。

README 的榜单目标是在 8×H100 上超过 GPT-2 CORE 参考分数的墙钟时间；其中成本和时间是特定条件下的背景，不能直接外推到你的机器。

## 6. 环境成本：跑通不等于复现能力

`nanochat/pyproject.toml` 要求 Python `>=3.10`，依赖 PyTorch、PyArrow、`rustbpe`、`tiktoken`、wandb 等。GPU 安装用 `uv sync --extra gpu`，CPU/MPS 用 `uv sync --extra cpu`。speedrun 用 `torchrun --standalone --nproc_per_node=8` 启动八个进程。显存不足时，README 建议降低 `--device-batch-size`；梯度累积可维持总 batch，但通常更慢。

学习入口是 `nanochat/runs/runcpu.sh`：它使用 `--depth=6`、`--head-dim=64`、`--max-seq-len=512`、`--num-iterations=5000` 等小得多的设置，并继续走 tokenizer、base eval、SFT 调用链。脚本明确说不会得到强结果。它的用途是观察流程、张量和 checkpoint，不是复现 8×H100 的能力。

`nanochat/nanochat/common.py::COMPUTE_DTYPE` 会在 CUDA SM 80+ 默认选 bfloat16，CPU/MPS 默认 float32，也可由 `NANOCHAT_DTYPE` 覆盖。它影响显存、速度与数值行为，但不改变训练的核心目标。

## 常见误区

1. **“speedrun 自动完成 RL。”** 不对；当前脚本在 SFT 评估结束。
2. **“SFT 从零创造知识。”** 不对；它从 base checkpoint 接续，主要校准行为。
3. **“token 就是一个字。”** 不对；这里是可变长度字节 BPE 片段。
4. **“checkpoint 只有权重。”** 不完整；还有元数据，base/SFT 也可能有按 rank 保存的 optimizer state。
5. **“depth 只影响层数。”** 不对；源码还用它联动模型形状和训练规模推导。
6. **“FP8、Flash Attention、DDP 是核心数学。”** 不对；它们首先是效率工程。

## 小结

nanochat 的主线是：训练 tokenizer；base 用右移目标学习下一个 token；评估观察 BPB、任务和样本；SFT 用对话示范只监督 assistant 输出；可选 RL 用 GSM8K 奖励继续调整；`Engine` 与 CLI 将 checkpoint 变成逐 token 对话。

请记住：**网络可相同，阶段由数据和反馈定义；checkpoint 是接力棒；depth 管规模，硬件技巧管成本。**

## 练习

1. 在 `runs/speedrun.sh` 中找环境、分词、base、SFT 入口。哪一处证明 RL 没有被调用？
2. 对 `[9, 4, 7, 2, 8]` 手写 `x`、`y`。若 `V=1000`，logits 形状是什么？
3. 跟踪 `load_model("sft", ...)` 会读取哪个目录；未传 tag 和 step 时如何选择？
4. 比较 `runcpu.sh` 与 speedrun 至少三项差异，判断各自主要影响容量、上下文、耗时还是硬件兼容性。
5. 为什么 SFT 保留用户问题作输入，却只训练 assistant 目标？请用 \(p_\theta(\text{assistant token}\mid\text{用户上下文})\) 解释。

## 前后章导航

- **上一章**：无；这是路线图。
- **下一章**：[第 02 章：Dataset](./02-dataset.md)。先看原始文本如何成为训练 batch，以及“数据先于 tokenizer”是什么意思。
- **后续主线**：第 03 章用第 02 章的数据训练 tokenizer；第 04–05 章拆 `nanochat/nanochat/gpt.py`；第 06–07 章训练 base model；第 08 章只评估 base model；第 09–10 章依次进入 SFT 与推理；第 11 章才评估 chat 任务；第 12 章是依赖 inference rollout 的 RL；第 13 章回到端到端流程。
