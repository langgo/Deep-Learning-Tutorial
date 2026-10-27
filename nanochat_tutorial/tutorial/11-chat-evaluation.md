# 第 11 章　聊天评估：聊天模型会做题了吗？

> **本章定位**　第三部分·从基础模型到聊天模型。第 10 章让模型能够生成回答，本章判断这些回答是否完成了任务。
>
> **中心问题**　评估时模型到底看到了什么、怎样作答、谁来判断对错？为什么同一个模型会因为回答方式、候选数和判分规则不同而得到不同分数？
>
> **范围**　ARC / MMLU 的受限分类，GSM8K / HumanEval 的生成与判分，多卡计数，多候选成功率，ChatCORE，以及执行生成代码的安全边界。训练奖励与参数更新留给[第 12 章](./12-rl.md)。
>
> **先决条件**　读过[第 09 章](./09-sft.md)的聊天协议和[第 10 章](./10-inference-engine.md)的生成流程。ChatCORE 的中心化思想与[第 08 章](./08-base-evaluation.md)相同。
>
> **本章产物**　一条“**取出一道题 → 隐藏参考答案 → 模型作答 → 判分器给 0/1 → 汇总 accuracy**”的主线；能够解释分类和生成两条路径分别测了什么、没有测什么。
>
> **源码入口**　`scripts/chat_eval.py`、`tasks/common.py`、`tasks/arc.py`、`tasks/mmlu.py`、`tasks/gsm8k.py`、`tasks/humaneval.py`、`nanochat/tokenizer.py::render_for_completion`、`nanochat/execution.py`。

---

## 本章讲什么

先看最小的评估问题：

```text
题目：法国的首都是哪里？
A. 伦敦
B. 巴黎
C. 柏林
D. 罗马
参考答案：B
```

一次可信的评估必须回答三个问题：

1. **给模型看什么？** 给题目和选项，不能把参考答案 B 放进 prompt；
2. **允许模型怎样回答？** 可以只在 A/B/C/D 中选，也可以让它自由生成文本；
3. **什么算正确？** 由任务判分器按预先规定的规则决定。

nanochat 的五个默认聊天任务共用前半段流程，然后分成两条作答路径：

```text
Task 中的一道题（含参考答案）
  → render_for_completion 隐藏最后一条参考 assistant 消息
  → 得到模型真正读取的 prompt
      ├─ ARC / MMLU：只比较合法选项 token 的 logits
      └─ GSM8K / HumanEval：生成完整回答，再交给判分器
  → 该题得到 0 或 1
  → correct / total = accuracy
  → 五项任务进一步汇总为 ChatCORE
```

因此，评估分数不是“模型能力”单独产生的；它是下面几项共同作用的结果：

```text
模型 + prompt 模板 + 作答方式 + 生成参数 + 判分规则
```

本章先走通这条主线，再讨论多卡、多个候选和安全边界。

---

## 预备词汇

| 名称 | 本章含义 |
|---|---|
| `conversation` | Task 返回的一道题；通常同时保存 user 题目和最后一条参考 assistant 回答 |
| `prompt_ids` | 模型实际读取的 token；不包含当前题的参考回答 |
| `completion` | prompt 之后由模型新生成的文本 |
| verifier / 判分器 | 按任务规则判断回答是否通过的程序，即 `Task.evaluate` |
| categorical | 受限分类：只在合法选项中选择 |
| generative | 自由生成：先生成文本或代码，再判分 |
| `k` / `num_samples` | 每道生成题实际产生的候选回答数；与采样的 top-k 不是一回事 |
| accuracy | 通过题数除以评估题数 |

还要区分几个常见量：logits 是模型对词表 token 的未归一化分数；本章每道题的 score 是判分器给出的 0/1；accuracy 是所有题 score 的平均。分类路径直接比较候选 logits，不需要先计算训练 loss。

---

## 1. 第一步：从一道含答案的题，得到无答案的 prompt

### 1.1 Task 同时服务“出题”和“判分”

`Task` 子类主要提供两项能力：

```text
task[i]                         → 取出一道题
Task.evaluate(problem, answer)  → 判断一个回答是否通过
```

以首都题为例，Task 对象可以返回：

```python
conversation = {
    "messages": [
        {"role": "user", "content": (
            "Multiple Choice question: 法国的首都是哪里？\n"
            "- 伦敦=A\n- 巴黎=B\n- 柏林=C\n- 罗马=D\n"
            "\nRespond only with the letter of the correct answer."
        )},
        {"role": "assistant", "content": "B"},
    ],
    "letters": ["A", "B", "C", "D"],
}
```

这里保留参考答案 B 是合理的：之后判分时需要它。关键不是“Task 对象里有没有答案”，而是“送给模型的 prompt 里有没有答案”。

### 1.2 `render_for_completion` 在送入模型前隐藏答案

`render_for_completion` 会复制 conversation，删除最后一条参考 assistant 消息，再渲染剩余消息并追加 `assistant_start`：

```text
Task 中保存：
user_start + 题目 + user_end
assistant_start + 参考答案 B + assistant_end

模型实际看到：
BOS + user_start + 题目 + user_end + assistant_start
```

最后的 `assistant_start` 表示“现在轮到助手回答”。模型从它后面开始作答，但看不到参考 B。

原始 conversation 不会被修改，所以判分器稍后仍能读取参考答案：

```text
模型：只读取 prompt_ids
判分器：读取原 conversation 中的参考答案 + 模型回答
```

GSM8K 的参考 assistant 消息是由 text、python、python_output 等 part 组成的列表；HumanEval 还保存测试代码和入口函数名。这些内容留给判分器，不会因为存在 Task 对象中就自动进入 prompt。

> **长度边界**：当前渲染器会先截断已有对话，再追加 `assistant_start`。很长的题目可能被截断，因此复现实验应记录实际 prompt，而不只记录原始题目文本。

### 1.3 到这里，分类和生成才分叉

同一个无答案 prompt，可以采用两种不同作答协议：

| 路径 | 当前任务 | 模型怎样回答 | 怎样判分 |
|---|---|---|---|
| 受限分类 | ARC-Easy、ARC-Challenge、MMLU | 在合法标签 token 中选最高分 | 与参考标签比较 |
| 自由生成 | GSM8K、HumanEval | 用 Engine 逐 token 生成 completion | 抽取数值或运行测试 |

这两条路径测量的内容不同。分类路径主动排除了格式错误；生成路径更接近真实回答，但结果也更受采样、长度预算和判分器影响。

---

## 2. 分类路径：只在合法选项中选择

### 2.1 首都题怎样得到预测 B

首都题的 prompt 以 `assistant_start` 结束。模型 forward 后，最后一个真实位置的 logits 正在预测 assistant 的第一个回答 token。

评估代码不在整个词表中自由选择，而只取合法标签 A/B/C/D 的分数：

```text
A 的 logit：1.2
B 的 logit：4.8
C 的 logit：0.7
D 的 logit：2.1
          ↓
在四项中取最大值 → B
          ↓
与参考答案 B 比较 → 该题 score=1
```

这叫**受限分类**。即使全词表中“让我想想”的 token 分数更高，评估仍会强制在合法标签中选一个。

它适合测“模型在给定选项中偏向哪个答案”，但不能证明模型自由生成时会严格输出一个合法字母。

### 2.2 不同题长度不同，必须取各自的真实末位置

一批题会右侧补齐到相同长度。假设两个 prompt 的长度分别是 5 和 7：

```text
第 0 题：[BOS, user_start, Q, user_end, assistant_start]
补齐后： [BOS, user_start, Q, user_end, assistant_start, BOS, BOS]

第 1 题：[BOS, user_start, Q₀, Q₁, Q₂, user_end, assistant_start]
```

两题真正预测答案的位置分别是：

```text
answer_pos = [4, 6]
```

不能统一取张量最后一列：第 0 题的最后一列是 padding 后的输出，不是 `assistant_start` 后的答案分数。

源码因此先记录每题原始长度：

```python
answer_time_positions = [len(ids) - 1 for ids in prompt_ids]
```

再逐题读取：

```python
focus_logits = logits[row, answer_pos, letter_ids]
predicted_letter = letters[focus_logits.argmax().item()]
```

注意 `argmax()` 返回的是候选表中的索引。例如合法标签是 `[A,B,C,D]`，返回 1 表示选择 B；它既不是词表 token id，也不是字母本身。

下面用一个小张量验证“真实末位置”和“合法标签约束”这两个关键点：

```python
import torch

answer_pos = [4, 6]
letter_ids = [3, 7, 11, 15]  # A/B/C/D 在教学词表中的假设 id
letters = ["A", "B", "C", "D"]
logits = torch.zeros(2, 7, 20)

logits[0, 4, letter_ids] = torch.tensor([1.0, 9.0, 2.0, 0.0])  # 正确位置选 B
logits[0, 6, letter_ids] = torch.tensor([10.0, 0.0, 0.0, 0.0]) # padding 末位会选 A
logits[0, 4, 19] = 100.0                                      # 非法 token 更高
logits[1, 6, letter_ids] = torch.tensor([8.0, 1.0, 0.0, 3.0])  # 第 1 题选 A

predicted = [
    letters[logits[row, pos, letter_ids].argmax().item()]
    for row, pos in enumerate(answer_pos)
]
assert predicted == ["B", "A"]
assert letters[logits[0, -1, letter_ids].argmax().item()] == "A"
assert logits[0, 4].argmax().item() == 19
```

### 2.3 当前分类协议的边界

- MMLU 固定使用 A/B/C/D；ARC 使用数据行提供的 label，不能假设永远四选一；
- 每个 label 单独编码后必须正好是一个 token，否则当前单 token 打分方法不适用；
- 一批题只做一次 forward，没有逐 token decode，因此通常比自由生成便宜；
- 当前 padding 没有额外 attention mask，但 padding 位于真实 prompt 右侧，因果 attention 不允许真实位置读取未来 padding；评估也不会使用 padding 位置的 logits。

最重要的解释边界是：

> 分类 accuracy 衡量“在合法选项中选对的比例”，不是完整聊天回答的成功率。

---

## 3. 生成路径：先得到完整回答，再交给任务判分

GSM8K 和 HumanEval 没有一个可直接比较的单 token 标签。它们先让 Engine 生成 completion，再由任务自己的 verifier 判分。

### 3.1 一道生成题的完整流程

假设数学题的正确答案是 10：

```text
Task 中的一道题
→ render_for_completion，隐藏参考解答
→ Engine.generate_batch 生成 k 条候选
→ 从每条结果中切掉 prompt
→ decode 成 completion 文本
→ Task.evaluate 分别判分
→ 任一候选通过，则该题 score=1
```

对应核心代码可压缩为：

```python
prompt_ids = tokenizer.render_for_completion(conversation)
results, _ = engine.generate_batch(
    prompt_ids,
    num_samples=k,
    max_tokens=max_new_tokens,
    temperature=temperature,
    top_k=top_k,
)
completions = [
    tokenizer.decode(token_ids[len(prompt_ids):])
    for token_ids in results
]
outcomes = [task.evaluate(conversation, text) for text in completions]
question_score = int(any(outcomes))
```

`generate_batch` 返回的每行包含原 prompt，所以必须从 `len(prompt_ids)` 开始切片。这里丢弃了来源 mask：聊天评估只关心最终回答能否通过，不计算策略梯度。

当前实现一次处理一道生成题，同一道题的 k 个候选构成 Engine batch；不同题不会被动态合并进同一个生成 batch。

### 3.2 GSM8K：比较 `####` 后抽出的字符串

GSM8K 的 verifier 不会验证整段推导。它在参考答案和模型回答中都搜索：

```text
#### 数字
```

然后比较抽出的字符串。例如：

```text
参考：#### 10
预测：推导过程…… #### 10
结果：通过
```

当前正则是：

```python
r"#### (\-?[0-9\.\,]+)"
```

因此判分规则比“数学等价”简单得多：

```python
import re

pattern = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract(text):
    match = pattern.search(text)
    return match.group(1).replace(",", "") if match else None

assert extract("错误推导，但最后写 #### 10") == "10"
assert extract("答案是 10") is None
assert extract("#### 2.0") != extract("#### 2")
assert extract("#### 1,234") == extract("#### 1234")
assert extract("#### 1/2") == "1"       # 只匹配到斜杠前
assert extract("#### 3\n#### 10") == "3" # 取第一个匹配
```

这意味着：

- 推导错误但最终字符串正确，仍可能得分；
- 数学等价但格式不同，可能不得分；
- 若参考和预测都抽取失败，当前直接比较会出现 `None == None`，从而误判为通过。

所以 GSM8K accuracy 应解释为：

> 按当前 `####` 字符串抽取规则通过的题目比例。

它不是对推理过程正确性的完整验证。

### 3.3 HumanEval：运行生成程序，通过测试才得分

HumanEval 的 user prompt 通常包含函数签名和 docstring。但当前聊天评估不会自动把这个函数前缀拼回 completion；模型通常需要生成完整函数定义。

判分器构造：

```text
从原题开头提取的 import
+ 从 completion 提取的第一段代码
+ HumanEval 测试代码
+ check(entry_point)
```

例如：

```text
模型生成：
def add_two(x):
    return x + 2

测试：
def check(fn):
    assert fn(3) == 5

最后执行：check(add_two)
```

程序退出码为 0 才算通过。语法错误、函数名错误、断言失败或超时通常都会得到 0。

这比字符串比较强，但仍只说明：

> 生成程序在当前环境中通过了这组有限测试。

它不能证明程序对所有输入都正确。代码块提取也不是完整 Markdown 解析器：当前只取第一个匹配的围栏代码块；找不到围栏时使用整段 completion。

### 3.4 生成参数会改变评估协议

生成分数受以下参数影响：

```text
num_samples=k
max_new_tokens
temperature
top_k
Engine seed
```

默认 `temperature=0` 是 greedy。同 prompt 的多行通常完全相同，所以仅增大 k 未必带来新候选。使用非零温度后才有随机探索，但每道题调用 `Engine.generate` 时都会用默认 seed 42 重新初始化 RNG；当前 CLI 没有暴露 seed 参数。

此外，所有 Task 在加载后使用另一个固定 seed 42 打乱题目顺序。要区分：

```text
Task shuffle seed：决定评估哪些题、题目顺序
Engine seed：      决定同一道题怎样随机采样回答
```

二者不是同一条随机流。

---

## 4. 多卡评估：分题计算，再汇总正确数

### 4.1 多卡不是把模型包成 DDP

聊天评估没有反向传播。`torchrun` 启动多个进程后，每个进程各自加载一份模型，再评估不同题目。

生成任务按题号交错分配：

```text
world_size=4
rank 0：题 0,4,8,...
rank 1：题 1,5,9,...
rank 2：题 2,6,10,...
rank 3：题 3,7,11,...
```

分类任务先组成 batch，再按 batch 编号交错分配。这里没有 `DDP(model)`，也不需要同步梯度；分布式通信只在最后汇总计数。

### 4.2 为什么要先加正确数和题数

假设：

```text
rank 0：1/1 通过，局部 accuracy=100%
rank 1：50/100 通过，局部 accuracy=50%
```

直接平均两个局部 accuracy 会得到 75%，但全局正确率应是：

```text
(1+50) / (1+100) = 51/101 ≈ 50.5%
```

原因是两个 rank 处理的题数不同。正确做法是：

```text
all_reduce(SUM) 正确数
all_reduce(SUM) 总题数
最后再相除
```

源码正是分别聚合 `num_passed` 和 `total`。因此尾部不能整除 world size 时，各 rank 题数不同也不会影响全局 accuracy。

> **运行边界**：多卡命令用于 CUDA torchrun 路径；不要把它机械复制到多进程 CPU / MPS。进度日志可能由多个 rank 交错打印，最终摘要只由 rank 0 输出。

---

## 5. 从每题 0/1 到多候选分数与 ChatCORE

### 5.1 k 条候选中任一通过，该题就通过

生成任务对每题实际生成 k 条候选：

```text
候选 0：失败
候选 1：通过
候选 2：失败
        ↓
any([0,1,0]) = 1
```

因此：

```text
question_score = int(any(candidate_outcomes))
accuracy = 所有 question_score 的平均
```

当 `k=1`，它是单候选成功率；当 `k>1`，它表示“本次实际 k 次尝试至少一次通过”的题目比例。verifier 帮忙从 k 条回答中判断是否存在成功项，模型本身没有负责选出那条正确回答。

不同 k 的结果不能直接混为同一协议。greedy 下重复 k 次也不代表进行了 k 次独立探索。

### 5.2 当前实现不是论文中的组合 pass@k 估计器

论文常见做法是：一题先采样 n 条，其中 c 条通过，再估计随机选 k 条时至少一条通过的概率：

```text
1 - C(n-c,k) / C(n,k)
```

nanochat 没有计算这个公式。它只对**实际生成的 k 条候选**执行 `any`。

所以最准确的名称是“实际 k 候选的 any 成功率”或“pass@k 式分数”，不要把它不加说明地当成组合估计器。

### 5.3 ChatCORE 先消除任务基线，再按任务等权平均

五个任务的原始 accuracy 不在同一起点：四选题随机猜测约有 25%，开放生成任务的工程基线设为 0。

代码使用：

| 任务 | 基线 `r_i` |
|---|---:|
| ARC-Easy | 0.25 |
| ARC-Challenge | 0.25 |
| MMLU | 0.25 |
| GSM8K | 0 |
| HumanEval | 0 |

每项先中心化：

```text
c_i = (a_i - r_i) / (1 - r_i)
```

直觉是：

```text
达到基线 → 0
达到满分 → 1
低于基线 → 负数
```

再对五项 `c_i` 等权平均。比如原始 accuracy 为：

```text
[0.25, 0.25, 0.25, 0.50, 0.50]
```

中心化后是：

```text
[0, 0, 0, 0.5, 0.5]
```

因此 ChatCORE 为：

```text
(0 + 0 + 0 + 0.5 + 0.5) / 5 = 0.2
```

它按**任务**等权，不按各任务题数加权。MMLU 当前则在任务内部按题目计数，没有先按 subject 等权。

CLI 只在五个默认任务名都出现时输出 ChatCORE。但即使每项只通过 `-x` 评估少量题，任务名仍然齐全，因此仍会计算汇总；这不等于跑完了完整 benchmark。

ChatCORE 与第 08 章 base CORE 使用的任务和协议不同，不能直接当成同一个指标比较。

---

## 6. 怎样运行，并让结果可复现

### 6.1 常用命令

在 `nanochat/` 目录运行：

```bash
python -m scripts.chat_eval -i sft -a ARC-Easy -b 8

# 多任务名中的 | 必须加引号，避免 shell 把它当管道。
python -m scripts.chat_eval -i sft -a 'ARC-Easy|MMLU' -x 20

# CUDA 多卡：每个进程加载一份模型并分题评估。
torchrun --nproc_per_node=8 -m scripts.chat_eval -i sft -a ARC-Easy
```

| 参数 | 默认值 | 影响范围 |
|---|---:|---|
| `--batch-size` | 8 | 只影响分类任务的 forward batch |
| `--num-samples` | 1 | 只影响生成任务每题候选数 k |
| `--temperature` | 0.0 | 生成任务；0 表示 greedy |
| `--top-k` | 50 | 生成任务；temperature=0 时不会执行该分支 |
| `--max-new-tokens` | 512 | 生成任务的新增 token 预算 |
| `--max-problems` | None | 每个任务最多评估多少题 |

小 `-x` 适合调试接线，不应伪装成完整 benchmark。`max_problems=0` 还会导致最终除以零，应使用正数。

### 6.2 最少要记录哪些信息

```text
模型：source、model tag、checkpoint step、tokenizer
题集：任务名、split、数据缓存/版本、实际题数
协议：prompt 模板、分类或生成、判分器版本
生成：k、temperature、top_k、max_new_tokens、seed
结果：通过数/总数、accuracy、是否汇总 ChatCORE
运行：源码 commit + 未提交改动、硬件、dtype、attention 后端、world_size
```

只记录一个 accuracy 不足以比较两个模型。尤其是 k、温度、截样数量和判分器变化，都会改变分数含义。

当前五个 Task 都以 seed 42 打乱数据，但 hub 下载没有在 Task 参数中固定远端 revision。严格复现还应保存数据文件或版本标识。

---

## 7. HumanEval 执行防护：降低事故风险，不是安全沙箱

HumanEval 必须运行模型生成的程序才能执行测试，这比字符串判分危险得多。

### 7.1 当前代码做了什么

`execute_code` 将 guard、候选程序和测试代码交给新的 Python 子进程：

```text
父评估进程
  → 创建临时目录
  → 启动新的 Python 解释器
  → 设置部分资源和函数限制
  → 执行候选与测试
  → 捕获 stdout/stderr
  → 超时或结束后清理临时目录
```

主要防护包括：

- 默认 5 秒超时；
- stdin 关闭，环境变量只保留精简 PATH；
- stdout / stderr 被捕获；
- 非 macOS 时尝试用 rlimit 限制 256 MiB 内存；
- Python 层禁用 `os.remove`、`os.fork`、`subprocess.Popen` 等部分危险入口；
- 在临时工作目录运行，结束后删除该目录。

进程正常退出且退出码为 0 时，`success=True`。

### 7.2 为什么它仍不是安全沙箱

这些措施主要降低意外破坏，不能抵抗恶意代码：

- 临时 cwd 不等于文件系统隔离，绝对路径仍可能可见；
- 没有网络隔离、seccomp、容器或虚拟机边界；
- Python monkey-patch 可能被动态能力或更低层接口绕过；
- 超时杀死直接子进程，不保证清理所有可能的后代进程；
- 捕获输出没有显式大小上限；
- macOS 跳过当前 rlimit 内存限制。

因此不要在含凭证和重要文件的宿主环境中运行不可信候选。面对对抗性输入，应使用独立受控环境，并组合最小权限、只读文件系统、网络和系统调用隔离、CPU/内存/输出限制以及销毁流程。

本章中的安全自检只解析字符串或 AST，不执行模型生成的 HumanEval 程序。

---

## 常见误区

1. **“Task 里有参考答案，所以模型看到了答案。”** 不对。应检查 `render_for_completion` 产生的实际 prompt；参考答案保留给判分器。
2. **“分类 accuracy 等于自由回答成功率。”** 不对。分类只在合法标签中 argmax，主动排除了格式失败和其他 token。
3. **“batch 最后一列就是答案位置。”** 只有最长题如此；短题必须取补齐前的真实末位置。
4. **“GSM8K 得分说明推导正确。”** 当前 verifier 只比较 `####` 后抽出的字符串。
5. **“HumanEval 通过说明程序完全正确。”** 它只通过了当前有限测试。
6. **“多卡 accuracy 应平均各卡 accuracy。”** 应先 SUM 正确数和题数，再相除。
7. **“k 个候选 any 就是组合 pass@k 估计器。”** 当前实现没有计算组合公式。
8. **“ChatCORE 一定在 0 到 1 之间。”** 低于基线时可以为负。
9. **“子进程、超时和函数黑名单等于安全沙箱。”** 它们只降低部分风险。

---

## 小结

聊天评估的主线是：

```text
Task 保存题目和参考答案
→ 渲染时隐藏参考答案
→ 模型通过分类或生成作答
→ verifier 给每题 0/1
→ 汇总正确数和总题数
→ 得到 accuracy，必要时再算 ChatCORE
```

解释任何分数时，都要同时说明三层协议：

- **模型看到什么**：prompt 模板、截断和聊天标记；
- **模型怎样回答**：受限分类，还是带采样参数的自由生成；
- **怎样判断正确**：标签比较、数值字符串抽取，还是程序测试。

分数只能在这些条件一致时公平比较。

---

## 练习

1. **答案隔离**：用首都题说明参考 B 在哪一步被移除，又在哪一步被判分器重新读取。
2. **分类位置**：两个 prompt 长度为 5 和 7，补齐到 7。写出各自 `answer_pos`，解释为什么不能统一取 `logits[:, -1]`。
3. **协议区别**：说明“在 A/B/C/D 中选 B”和“自由生成字符串 B”分别可能避开或引入哪些错误。
4. **GSM8K 边界**：当前规则会怎样处理 `#### 1/2`、`#### 2.0` 对 `#### 2`、以及预测和参考都没有标记？
5. **多卡计数**：rank 0 为 1/1，rank 1 为 50/100。计算错误的卡平均和正确的全局 accuracy。
6. **ChatCORE**：对 `[0.25,0.25,0.25,0.5,0.5]` 手算中心化结果。为什么 k=8 的生成项不能直接与 k=1 结果比较？
7. **执行安全**：列出当前 HumanEval helper 已做的三项限制，以及仍缺失的三个系统隔离能力。

---

## 前后章导航

- **上一章**：[第 10 章：推理引擎——KV 缓存与高效生成](./10-inference-engine.md)。提供本章分类 forward 和生成 Engine。
- **本章**：把模型输出变成协议明确的任务 accuracy、多候选成功率和 ChatCORE。
- **下一章**：[第 12 章：强化学习——用结果奖励改进推理](./12-rl.md)。复用 GSM8K 判分器作为 reward，但训练 split、候选协议和参数更新属于另一层问题。
