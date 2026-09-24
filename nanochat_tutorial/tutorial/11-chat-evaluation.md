# 第 11 章：Chat Eval：聊天模型会做题了吗？

> **源码地图**：统一入口是 `nanochat/scripts/chat_eval.py`；任务的公共契约在 `nanochat/tasks/common.py`，具体题目适配在 `arc.py`、`mmlu.py`、`gsm8k.py`、`humaneval.py`；Python 程序执行保护在 `nanochat/nanochat/execution.py`。  
> **调用链**：`chat_eval.py → run_chat_eval → Task[i] → tokenizer.render_for_completion →` 分类路径 `model(prompt_ids)`，或生成路径 `Engine.generate_batch → Task.evaluate`；HumanEval 还会进入 `execute_code`。  
> **本章边界**：这里讨论**聊天模型**如何被评估，不重复 base model 的 CORE、BPB 或续写似然评估；它们属于[第 08 章](./08-base-evaluation.md)。

一个模型能把网页文本接得很顺，并不等于它会按聊天协议回答问题，更不等于它写出的函数能通过测试。进入 SFT 后，模型被训练为在 `user` 消息之后开始 `assistant` 回复；进入推理引擎后，它的回答又会经过采样、终止 token 和可能的工具状态机。于是评估也必须放在 **SFT 与 inference 之后**：我们要测的是“给一个聊天 prompt，系统最终交付的回答能否完成任务”，而不是训练集里某个 target token 的概率。

## 学习目标

读完本章，你应能：

1. 解释为什么 chat eval 不能直接沿用 base 续写评估；
2. 读懂 `Task`、`conversation` 与 `evaluate` 的数据契约；
3. 区分 ARC/MMLU 的受限分类打分与 GSM8K/HumanEval 的生成后验证；
4. 按张量形状跟踪多选题 batch 如何取得答案位置的 logits；
5. 说明 GSM8K 数值抽取、HumanEval 代码抽取与执行各自漏掉了什么；
6. 解释多卡题目切分、`all_reduce` 汇总、ChatCORE 和当前“pass@k 式”口径；
7. 知道 `execution.py` 的子进程防护为何不是对抗恶意代码的安全沙箱。

## 前置概念

建议已读[第 09 章：监督微调 SFT](./09-sft.md)和[第 10 章：Inference](./10-inference-engine.md)。本章使用：`B` 表示 batch 中题目数，`T` 表示补齐后的 prompt 长度，`V` 表示词表大小，`k` 表示每题生成的候选数。`logits[b,t,:]` 是第 `b` 道题在位置 `t` 预测**下一个** token 的 `V` 个未归一化分数。

还要先接受一个朴素但重要的事实：评估不是一个脱离任务的“智力仪表”。题集版本、shuffle、prompt 模板、输出约束、采样温度、最大生成长度、判分器和 `k` 都会影响数字。分数适合比较同口径下的模型或 checkpoint，不是模型的一张万能成绩单。

## 1. 为什么评估在 SFT 和 inference 之后？

base model 的自然行为是续写字节/token 序列；而 chat 模型要遵从角色边界、special token、回答起止格式，有时还要接收推理引擎注入的工具输出。`chat_eval.py` 对每题调用：

```python
encoded_prompt = tokenizer.render_for_completion(conversation)
```

这里的关键不是简单 `encode(question)`：`conversation` 内保留了完整的标准答案，但 `render_for_completion` 的目标是渲染“到 assistant 即将回答为止”的上下文，而不是把最后的标准 assistant 回复泄露给模型。随后分类任务直接看该位置的 logits；生成任务经 `Engine.generate_batch` 真正跑一次推理，再把文本 completion 交给判分器。

因此 chat eval 同时验证三件事：SFT 学到的对话格式是否可用、tokenizer 的聊天渲染是否正确、inference 的生成结果能否通过任务验证。它不是训练损失的替身。训练损失可以在已知参考回复上逐 token 算；实际聊天评估必须让模型面对一个没有参考回复的待答位置。

命令行入口的典型形式是：

```bash
cd nanochat
python -m scripts.chat_eval -i sft -a ARC-Easy
# 多卡时：torchrun --nproc_per_node=8 -m scripts.chat_eval -- -i sft -a ARC-Easy
```

`-i/--source` 要求 `sft` 或 `rl`，并由 `load_model(..., phase="eval")` 加载相应 checkpoint；默认不指定 `-a` 时依次评 ARC-Easy、ARC-Challenge、MMLU、GSM8K、HumanEval。先用 `-x` 限制题数排查流程是合理的，但小样本分数不能当正式结论，也不需要为了读懂本章运行昂贵评测。

## 2. Task/Conversation：题目和判分必须说同一种语言

`tasks/common.py::Task` 是一个很薄的抽象类。子类需要提供：

- `eval_type`：只能是 `'categorical'` 或 `'generative'`；
- `num_examples()` 与 `get_example(index)`：让 `Task` 通过 `__len__`、`__getitem__` 取题；
- `evaluate(problem, completion)`：给一题与一个候选回复，返回是否成功。

`Task` 还支持 `start/stop/step` 形成逻辑切片：`ds[0]` 不一定是底层数据的第 0 行，而是 `start + 0 * step`。`HubDataset` 则把 Hub 的 parquet 下载到本地缓存，使用固定 seed `42` 的 permutation 提供可复现 shuffle。多卡首次下载通过 manifest 文件锁避免每个 rank 重复下载。这些设计不使评估“天然公平”，但减少了题序、切片和数据获取带来的偶然差异。

每个 `get_example` 返回一个 Python dict，惯例核心字段是：

```python
conversation = {
    "messages": [
        {"role": "user", "content": "题目或提示"},
        {"role": "assistant", "content": "标准回答"},
    ],
    # 可选的、判分需要的元数据
}
```

这是一份双用途对象：渲染 completion prompt 时最后一条 assistant 是隐藏的参考答案；`evaluate` 时它又是 gold label。ARC 额外放 `letters`，MMLU 放 `subject` 与 `letters`，HumanEval 放 `entry_point`、`test`。GSM8K 的 assistant 内容尤其不同：它是由 `text`、`python`、`python_output` 组成的 parts 列表，以保留数据里的 `<<表达式=结果>>` 工具结构。

这就是契约的价值：评估循环无需知道 ARC 的 JSON 字段名或 HumanEval 的测试字符串，只依赖“拿到 conversation、渲染 prompt、最终调用 evaluate”。反过来，任务作者必须保证 prompt、参考答案和判分器匹配；比如多选答案必须是允许的字母，HumanEval 的 `entry_point` 必须和测试中的函数相符。

## 3. 路径一：分类题不生成整句话，而是在选项中比较

ARC 和 MMLU 的 `eval_type` 都是 `categorical`。二者都用 `render_mc(question, letters, choices)` 构造类似：

```text
Multiple Choice question: ...
- choice one=A
- choice two=B
...

Respond only with the letter of the correct answer.
```

选项文字后接 `=A`，以及 `=` 和字母之间没有空格，是刻意的 tokenization 细节：`"A"` 与 `" A"` 未必是同一 token。`run_categorical_eval` 会 `tokenizer.encode(letter)`，并断言每个字母恰好是一个 token；否则“在一个位置比较四个 token”的实现就不成立。

直觉上，既然正确格式规定只能回答 A/B/C/D，就没有必要让模型自由生成一大段解释，再用脆弱的字符串规则猜它选了什么。当前实现给模型一个较有利、也较稳定的受限选择：只比较合法答案字母的 logits。这应在报告中说清，因为它比“开放式聊天回答后抽取字母”更容易。

### 一次 batch 的形状旅行

设 `B=3` 道题，渲染后的 token 长分别是 `[7, 10, 8]`。代码用 BOS 当 pad，得到：

```text
prompt_ids before pad:  [len 7, len 10, len 8]
max_length:             10
padded prompt_ids:      (B, T) = (3, 10)
model output logits:    (B, T, V) = (3, 10, V)
answer positions:       [6, 9, 7]
```

第 `i` 题真正的末 token 位于 `len(ids)-1`，而不是 padding 后统一的 `T-1`。原因是 `logits[i, answer_pos]` 预测的正是 prompt 之后的第一个 assistant token；若错误地在 pad 末尾取 logits，就在问“BOS padding 后应该接什么”，分数没有意义。

每道题可用选项数也允许不同。若第 0 题的 `letters=['A','B','C','D']` 对应的词表 id 为 `[11,12,13,14]`，则：

```python
focus_logits = logits[0, 6, [11, 12, 13, 14]]  # (4,)
predicted_letter = letters[focus_logits.argmax().item()]
```

`letter_to_id_cache` 缓存重复字母的编码。最终 `ARC.evaluate` / `MMLU.evaluate` 断言预测确实在允许字母中，并与最后一条 assistant 的标准字母比较。这里的模型前向可以对 batch 并行，且没有 sampling，所以比逐题生成便宜。

### 最小评分/形状实验（不加载模型）

下面只复现“取合法选项并评分”的核心，适合在 Python 中观察形状；它不衡量语言能力：

```python
import torch

# B=2，补齐长度 T=5，词表 V=20
logits = torch.zeros(2, 5, 20)
answer_pos = [2, 4]
letter_ids = [3, 7, 11, 15]       # A/B/C/D 的假想 token id
letters = ["A", "B", "C", "D"]

# 让第 0 题偏向 C；第 1 题偏向 A
logits[0, 2, letter_ids] = torch.tensor([1.0, 2.0, 9.0, 0.0])
logits[1, 4, letter_ids] = torch.tensor([8.0, 1.0, 0.0, 3.0])

pred = [letters[logits[i, pos, letter_ids].argmax().item()]
        for i, pos in enumerate(answer_pos)]
gold = ["C", "B"]
print(logits.shape, pred, sum(p == y for p, y in zip(pred, gold)))
# torch.Size([2, 5, 20]) ['C', 'A'] 1
```

把第 0 题位置误改为 `4`，它会从全零 logits 中按并列规则选一个字母；这正是 padding 位置 bug 的缩小版。

## 4. 路径二：生成后再验证

GSM8K 和 HumanEval 的正确答案不是一个有限、固定的单 token 选项，所以标记为 `generative`。`run_generative_eval` 按 rank 分配题目，每题调用一次：

```python
results, _ = engine.generate_batch(
    encoded_prompt, num_samples=k, max_tokens=max_new_tokens,
    temperature=temperature, top_k=top_k,
)
completions = [tokenizer.decode(ids[prefix_length:]) for ids in results]
outcomes = [task_object.evaluate(conversation, text) for text in completions]
passed = any(outcomes)
```

`prefix_length` 很关键：引擎返回的 token 序列含 prompt，判分器应只看到新生成的 completion。该循环“逐题、每题 k 个候选”，而不像多选那样把不同 prompt 批量塞给模型，因为生成长度、结束时机与代码执行均可能不同。

### GSM8K：答案抽取并不是数学证明

`gsm8k.py` 用正则 `r"#### (\-?[0-9\.\,]+)"` 寻找最后答案格式中的一个数。它去掉逗号后，把预测数字字符串与参考答案最后 text part 中抽出的数字字符串直接比较。

这给出清晰、便宜、可复现的 exact-match 规则，但边界也很窄：没有 `####` 即使推理正确也得 0；`2.0` 与 `2` 是不同字符串；分数、科学计数法、单位、等价代数式不在模式内。反过来，碰巧输出正确 `#### 10` 即使过程胡乱也得 1。任务的原始答案会被解析为工具调用 parts，是为了 SFT 数据保留结构；当前 eval 对生成 completion 则只做上述文本数值抽取，并不重新检查推导过程。

### HumanEval：从文本到可运行程序

HumanEval 的 user prompt 是函数开头，标准 assistant 是 `prompt + canonical_solution`。判分时 `extract_program` 优先取第一个 Markdown ```python 或 ``` 代码块；没有代码块就把整个 completion 当 Python。它还从 prompt 开头连续收集 `import`/`from` 行，再拼成：

```text
prompt 中的 imports
+ 模型 completion 的程序
+ 数据集 test
+ check(entry_point)
```

`execute_code(program)` 成功退出才算通过。这个规则奖励“实现满足隐藏测试的函数”，不是和 canonical solution 的字符串相似。也因此它会受代码块抽取影响：模型在代码块外写的必要定义会丢失；若它返回半个函数、语法错误或测试失败，均为失败。测试集本身覆盖有限，pass 也不等于程序在所有输入上都正确。

## 5. 多卡：各做不同题，最后相加

无论哪条路径，先读 `get_dist_info()` 得到 `ddp_rank` 和 `ddp_world_size`。生成循环使用：

```python
for i in range(ddp_rank, num_problems, ddp_world_size):
    ...
```

分类循环先把题目按 `batch_size` 切成 batch，再让 rank 分配 batch index。这样 rank 0 做 `0, world_size, ...`，rank 1 做 `1, world_size+1, ...`；每题/每 batch 只评一次，尾部不整除时也不会凭空补题。

每个 rank 只维护本地 `num_passed, total`，结束后把两个一元素 long tensor 放到模型设备：

```python
dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
accuracy = num_passed / total
```

这里归约的是**计数**，不是各 rank accuracy 的平均。若各卡题数不同，直接平均百分比会错：`1/1` 与 `50/100` 的均值是 75%，全局实际是 `51/101`。`print0` 只让主 rank 输出最终结果；各 rank 的过程行仍可能交错，这是日志现象，不是重复计分。

## 6. ChatCORE 与 pass@k：数字前先问口径

当且仅当五个默认任务都已运行，脚本计算 ChatCORE。ARC-Easy、ARC-Challenge、MMLU 使用随机四选一基线 `r=0.25`；GSM8K 与 HumanEval 使用 `r=0`。对任务准确率 `a_i`：

\[
\operatorname{ChatCORE}=\frac{1}{5}\sum_i\frac{a_i-r_i}{1-r_i}.
\]

中心化让随机四选一对应 0、满分对应 1，避免多选任务因“瞎猜也有 25%”抬高均值；但它仍是五项**等权**的工程汇总，不是普适能力的数学定义。它也可能为低于随机基线的多选任务给出负贡献。只跑单个 `-a` 时脚本会打印该任务 accuracy，却不会打印 ChatCORE。

生成路径中，`num_samples=k` 时一题的 `outcomes` 只要有一个为真便记成功：

\[
\text{score}_q = \mathbb{1}[\exists j\in\{1,\ldots,k\},\;\text{verify}(c_j)=1].
\]

因此 `k=1` 是单次生成成功率；`k>1` 是常被称为 pass@k 的“至少一次成功”事件。当前 `chat_eval.py` 实际生成 k 个样本并 `any`，**没有**实现论文中“先取 n 个样本，再用组合公式无偏估计 pass@k”的另一种估计器。HumanEval 的外部测试和 GSM8K 的数值 verifier 使这种 oracle 口径有意义：若系统真的能运行/筛选候选，可以多试几次挑出正确者。

限制同样重要：更多样本通常更容易至少蒙对一次，也更耗时间和显存；温度为 0 时多个样本往往相同，`k` 变大并不制造多样性；自动 verifier 在真实聊天中未必存在。故 `pass@8` 不能直接与 `pass@1` 或另一个温度、top-k、max tokens 下的结果比较，更不能表述为“用户一次提问的准确率”。报告至少应带上 `k`、temperature、top-k、max_new_tokens、题目数和任务版本。

复制下面这张最小记录卡，比只贴一个分数更可复查：

```text
checkpoint/source/tag/step:
task + dataset version + split:
max_problems / num_samples(k):
temperature / top_k / max_new_tokens / seed:
prompt 与 verifier 版本:
各任务原始分数 + 汇总方法:
硬件、dtype、代码 revision:
```

## 7. 执行生成代码：有护栏，不是安全沙箱

`nanochat/nanochat/execution.py::execute_code` 在临时目录启动新的 Python 子进程。它使用 scrubbed 环境 `PATH`、关闭 stdin、捕获 stdout/stderr；父进程用默认 5 秒 timeout 杀掉超时子进程。非 macOS 上，guard 会尝试用 `rlimit` 设定默认 256 MB 的地址空间、数据段和栈限制；还把一批危险的 `os`、`shutil` 操作及 `subprocess.Popen` 置为 `None`，并删除部分模块引用。结果以 `ExecutionResult` 返回 `success`、stdout/stderr、错误、timeout 和 memory_exceeded。

这些措施很好地降低了评估时模型意外死循环、删除文件或 fork 的风险，也避免把待执行代码放进评估主进程内存；但文件开头已经明确声明：**它不是强安全沙箱**。网络没有被阻断，Python 的动态能力（例如 `ctypes`）可能绕过 monkey patch，没有 seccomp、容器或虚拟机等内核级隔离；macOS 还会跳过 rlimit 调用。不要把来自不可信用户或对抗模型的程序交给它执行。生产级运行应在权限最小化、网络隔离、资源配额严格且可销毁的容器/VM 中进行，并把语言和依赖面进一步收窄。

## 常见误区

1. **“SFT loss 低就不用 chat eval。”**不对。loss 是对参考 target 的训练/验证信号；chat eval 检查没有参考答案泄露时的生成与判分结果。
2. **“多选题必须自由生成 A/B/C/D 才真实。”**当前源码不是这样测：它只在合法字母 logits 中 argmax。这是受限分类，需如实说明。
3. **“padding 后每题都取最后一个位置。”**错。每题应取自己未补齐 prompt 的 `len(ids)-1`。
4. **“GSM8K 判对表示推理过程正确。”**错。当前是 `####` 后数字的字符串匹配。
5. **“HumanEval 运行通过就是代码绝对正确。”**错。它只说明通过该题附带测试和当前执行环境。
6. **“多卡时平均每张卡 accuracy 就行。”**错。必须先 `SUM` 正确数和总数，再相除。
7. **“ChatCORE 是通用智商。”**错。它是五项、特定基线和等权平均下的汇总。
8. **“pass@k 是一次回答的 accuracy。”**仅 `k=1` 接近这个含义；`k>1` 给了 oracle k 次机会和额外预算。
9. **“子进程 + 禁用几个函数 = 安全执行不可信代码。”**错。这里的设计防事故，不防有意攻击。

## 小结

chat eval 的位置在 SFT 与 inference 之后，因为评估对象是符合聊天协议、经真实生成系统产出的回答。nanochat 用统一 `Task → conversation → evaluate` 契约，把任务数据与公共循环拆开；ARC/MMLU 走批量、受限字母 logits 分类，GSM8K 走生成后的 `####` 数字抽取，HumanEval 则把程序与测试拼接到受防护的子进程执行。

分布式评估按 rank 切分题目，再归约正确计数和总计数。ChatCORE 是五个任务按随机基线中心化后的等权均值；`num_samples=k` 的生成得分是实际生成 k 次、任一验证通过即可的 pass@k 式事件。它们都很有用，但只能在清楚记录提示、判分器和采样预算的前提下解释。最后，HumanEval 的执行防护是评估工程的保险丝，不应被误用为恶意代码的安全边界。

## 练习

1. 用自己的话画出 ARC 的调用链，并在每一箭头旁标注数据类型：`conversation`、`list[int]` prompt、`(B,T,V)` logits、预测字母、布尔结果。
2. 设三道多选题长度为 `[12, 9, 15]`，`V=65536`。写出 padding 后输入和 logits 的形状、每题应读取的 answer position；说明若全部取位置 14 会怎样。
3. 不运行模型，改写本章最小实验，让第二题的合法选项只有 `['A','B','C']`；思考为什么评估循环逐题取 `letters`，而不是假定每题都是四选一。
4. 给 GSM8K 设计一个更宽松的数值判分器（如 Decimal 或有理数规范化）。列出它会修复的两个 false negative，以及可能新增的两个 false positive；不要直接假定“更宽松一定更公平”。
5. 假设某模型 `pass@1=0.30`、`pass@8=0.75`。分别描述“单次聊天”和“有外部 verifier 的八次候选筛选”中这两个数能说明什么、不能说明什么。
6. 为执行不可信 Python 列出至少四层生产隔离措施，并指出本仓库 `execute_code` 已覆盖哪几层、明确没有覆盖哪几层。

## 前后章导航

- ← [第 10 章：Inference：KV 缓存与高效生成](./10-inference-engine.md)：理解 `Engine.generate_batch` 如何实际生成这些候选。
- ↑ [教程目录](./README.md)：回看整条从 token 到聊天模型的学习路线。
- → [第 12 章：RL：用 GRPO 提升推理能力](./12-rl.md)：GSM8K 的同类结果验证如何进一步成为 rollout reward 与训练信号。
