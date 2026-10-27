# RLHF + PPO 从 0 到 1

> 目标：用尽量直观的方式讲清楚经典 RLHF + PPO：它要解决什么问题、每一步训练什么、数据长什么样、PPO 到底在优化什么，以及为什么它比普通 SFT 更复杂。

## 0. 先给结论

经典 ChatGPT 式训练可以粗略分成三层：

```text
预训练 Pretraining
  ↓
监督微调 SFT
  ↓
偏好对齐 RLHF + PPO
```

其中：

- **预训练**：让模型学会预测下一个 token，获得通用语言、知识和代码能力。
- **SFT**：让模型学会像助手一样回答问题。
- **RLHF + PPO**：让模型的回答更符合人类偏好，比如更有帮助、更真实、更安全、更少废话。

经典 RLHF + PPO 又可以拆成两件事：

```text
1. 训练 Reward Model：学会判断哪个回答更好
2. 用 PPO 训练 Assistant：让模型更倾向生成高分回答
```

一句话版：

> RLHF 先训练一个“裁判”，再让模型反复生成答案，用裁判打分，通过 PPO 稳定地把模型往高分答案方向推。

---

## 1. 为什么 SFT 之后还需要 RLHF？

先看 SFT。

SFT 数据通常长这样：

```text
User: 给我解释一下 Transformer。
Assistant: Transformer 是一种基于注意力机制的神经网络架构……
```

训练目标很简单：

```text
看到 user prompt 后，模仿 assistant 的标准答案。
```

这很有用，但有几个问题。

### 1.1 好答案不止一种

同一个问题可能有很多合理回答：

```text
问题：解释一下 RLHF
回答 A：短小直接，适合初学者
回答 B：公式完整，适合研究者
回答 C：啰嗦但大体正确
回答 D：看似自信但细节错了
```

SFT 只能模仿某个标准答案，但它不直接告诉模型：

```text
A 比 C 好
B 在专业场景比 A 好
D 虽然流畅但不应该选
```

### 1.2 人类更擅长“比较”而不是“写满分答案”

让标注员写一个完美答案很难。

但让标注员比较两个答案：

```text
A 和 B 哪个更好？
```

通常容易很多。

这就是 RLHF 的切入点：

> 不要求人类直接写出所有完美答案，而是收集人类偏好，让模型学习人类更喜欢什么。

### 1.3 模型不是天然想“成功”

语言模型的底层目标是模仿文本分布。

训练数据里有：

- 专家回答
- 普通人回答
- 错误回答
- 模糊回答
- 有害回答
- 废话回答

模型学到了所有这些模式。SFT 会把它拉向助手风格，但它仍然可能：

- 回答太啰嗦
- 编造事实
- 不知道什么时候拒答
- 在两个差不多的答案里选了人类不喜欢的那个

RLHF 试图把模型进一步拉向“人类更偏好的回答分布”。

---

## 2. 把语言模型看成一个 policy

为了理解 PPO，要先换一个视角。

普通语言模型可以写成：

```text
P(next_token | previous_tokens)
```

也就是：给定前文，预测下一个 token。

但在强化学习里，我们可以把它看成一个 **policy**：

```text
πθ(action | state)
```

对应到语言模型：

```text
state  = 当前上下文 tokens
 action = 下一个 token
πθ     = 当前语言模型
```

生成一整段回答，就是连续做很多次动作：

```text
prompt
  → token1
  → token2
  → token3
  → ...
  → 完整回答
```

所以：

> 语言模型生成文本，可以被看作一个强化学习 policy 在 token 空间里一步步采取 action。

这个视角一旦建立，PPO 就有位置了：

> PPO 是用来优化 policy 的算法。这里的 policy 就是 assistant 模型。

---

## 3. RLHF 的第一步：训练 Reward Model

### 3.1 偏好数据长什么样

Reward Model 的训练数据通常不是单个标准答案，而是成对比较：

```text
Prompt:
解释一下为什么天空是蓝色的。

Answer A:
天空呈蓝色主要是因为瑞利散射……

Answer B:
天空是蓝色的，因为海洋反射到了天空上……

Human preference:
A > B
```

其中：

- A 叫 chosen / preferred answer
- B 叫 rejected / dispreferred answer

### 3.2 Reward Model 做什么

Reward Model 输入：

```text
prompt + answer
```

输出一个标量分数：

```text
r(prompt, answer) = 2.3
```

分数越高，表示它越认为这个回答符合人类偏好。

它不是用来聊天的。它是一个打分器。

```text
Assistant Model：负责生成回答
Reward Model：负责评价回答
```

### 3.3 Reward Model 怎么训练

我们希望：

```text
r(prompt, chosen) > r(prompt, rejected)
```

常见损失是 Bradley-Terry / pairwise ranking loss：

```text
loss = -log sigmoid(r_chosen - r_rejected)
```

不用被公式吓到，它的意思很简单：

- 如果 chosen 分数比 rejected 高很多，loss 很小。
- 如果 chosen 分数不够高，loss 变大。
- 如果 rejected 反而更高，loss 很大。

所以 Reward Model 学到的是：

> 对同一个 prompt，人类更喜欢的答案应该拿更高分。

### 3.4 为什么不直接让 Reward Model 选答案？

可以，但不够。

Reward Model 只能评价已有答案。我们最终需要的是一个能主动生成好答案的 assistant。

所以接下来要做第二步：

```text
用 Reward Model 的分数，继续训练 Assistant Model。
```

这就是 PPO 阶段。

---

## 4. RLHF 的第二步：用 PPO 优化 Assistant

现在我们有三个模型概念：

```text
1. SFT Model
   初始助手模型，也是 PPO 的起点。

2. Reward Model
   冻结不动，只负责打分。

3. Policy Model
   正在被 PPO 更新的助手模型。
```

通常还会保留一个 reference model：

```text
Reference Model = 冻结的 SFT Model
```

它的作用是防止 policy 跑太远。

---

## 5. PPO 阶段的训练循环

PPO 的训练循环可以理解成下面几步。

### 5.1 取一批 prompts

例如：

```text
Prompt 1: 写一个 Python 函数判断回文。
Prompt 2: 解释什么是黑洞。
Prompt 3: 帮我总结这段文字。
```

这些 prompt 可以来自真实用户、人工构造、历史数据或混合数据。

### 5.2 当前 policy 生成回答

当前 assistant model 对每个 prompt 采样一个回答：

```text
Prompt: 写一个 Python 函数判断回文。
Policy answer: def is_palindrome(s): ...
```

这一步叫 rollout。

### 5.3 Reward Model 打分

把生成的回答交给 Reward Model：

```text
Reward Model(prompt, answer) → reward score
```

比如：

```text
score = 4.2
```

这个分数代表：Reward Model 认为这个回答有多好。

### 5.4 加上 KL 惩罚

如果只追求 Reward Model 高分，模型可能钻空子。

比如它发现某些奇怪模板能骗过 Reward Model，就不断生成这些模板。这叫 reward hacking。

所以 PPO 通常会加一个 KL penalty，让新模型不要离 SFT 模型太远：

```text
final_reward = reward_model_score - β * KL(policy || reference)
```

直觉：

```text
回答质量要提高，但不要变成奇怪的模型。
```

其中：

- policy 是当前正在训练的模型。
- reference 是冻结的 SFT 模型。
- β 控制“别偏太远”的力度。

### 5.5 估计 advantage

强化学习里不只关心 reward 高不高，还关心：

```text
这个回答比模型原本预期的好多少？
```

这个差值叫 advantage。

```text
advantage ≈ 实际 reward - 预期 reward
```

如果 advantage 是正的：

```text
这个回答比预期好，增加它的概率。
```

如果 advantage 是负的：

```text
这个回答比预期差，降低它的概率。
```

实际训练中通常会有一个 value head 来预测预期 reward，再用它计算 advantage。

#### value head 是什么？

value head 可以理解成接在 policy model 顶上的一个“小预测头”。

原来的语言模型输出的是：

```text
下一个 token 的概率分布
```

加上 value head 后，它还会额外输出一个数：

```text
从当前 token 位置继续生成下去，预计最终能拿多少 reward
```

所以同一个 Transformer backbone 上通常有两个头：

```text
LM head：预测下一个 token，用来生成文本
Value head：预测预期 reward，用来辅助 PPO 训练
```

Reward Model 是外部裁判，回答生成完后给实际分数。

Value head 是 policy 内部的小估分器，在每个 token 位置预测“后面大概能得多少分”。它的作用不是评价答案好坏本身，而是帮 PPO 判断：

```text
实际 reward 比 value head 预测高 → advantage 为正 → 增加概率
实际 reward 比 value head 预测低 → advantage 为负 → 降低概率
```

### 5.6 模型权重到底怎么更新

先区分两个阶段：

```text
rollout 阶段：policy 生成回答，只收集训练样本，不更新权重。
update 阶段：拿这些样本算 loss，反向传播，optimizer.step() 更新权重。
```

PPO 阶段通常只有这些参数会变：

```text
会更新：Policy Model 的权重 θ，以及它旁边的 value head
不更新：Reward Model
不更新：Reference Model，也就是冻结的 SFT Model
```

一次 PPO update 大概长这样：

```text
1. 用旧 policy 生成 answer
2. 记录每个 token 当时的概率 logprob_old
3. Reward Model 给完整 answer 打分
4. Reference Model 算 KL 惩罚
5. value head 估计“预期 reward”
6. 计算 advantage = 实际 reward - 预期 reward
7. 用 PPO loss 反向传播
8. optimizer.step() 改 policy 权重
```

注意：Reward Model 的分数通常是对完整回答给的，但更新权重时，梯度会分摊到这段回答里的 token 上。

直觉上就是：

```text
advantage > 0：这段回答比预期好 → 提高这些 token 路径的概率
advantage < 0：这段回答比预期差 → 降低这些 token 路径的概率
```

这里的“提高概率”不是把某句话写进模型，也不是改一个规则表，而是通过反向传播微调 Transformer 里的参数：embedding、attention、MLP 等权重都会被梯度轻微推动。

### 5.7 PPO 怎么限制更新幅度

如果只按 advantage 更新，模型可能一步改太猛。

PPO 会比较“更新前”和“更新后”同一个 token 的概率：

```text
ratio = π_new(token | context) / π_old(token | context)
```

然后把 ratio 限制在一个小范围里：

```text
clip(ratio, 1 - ε, 1 + ε)
```

直觉：

```text
这个回答不错，可以提高概率；
但这轮最多提高一点，不许一脚油门踩到底。
```

所以 PPO 的训练目标更像：

```text
policy_loss = 让高 advantage 的 token 更可能出现，但别超过 clip 限制
value_loss  = 让 value head 更准地预测 reward
kl_penalty  = 让 policy 别离 reference 太远
```

最后总 loss 反向传播：

```text
loss.backward()
optimizer.step()
```

这一步才是真正更新模型权重的地方。

---

## 6. PPO 到底在保护什么？

PPO 主要保护三件事。

### 6.1 防止训练不稳定

语言模型很大，输出空间巨大。如果更新太猛，很容易：

- 语言能力下降
- 输出格式崩坏
- 开始重复
- 变得极端啰嗦
- 过拟合 Reward Model

PPO 用 clipping 限制每次 policy 更新幅度。

### 6.2 防止模型偏离 SFT 太远

SFT 模型已经是一个不错的助手。RLHF 只是调优偏好，不应该把它训成另一个奇怪东西。

所以训练里通常有 KL 惩罚：

```text
离 reference model 越远，惩罚越大。
```

### 6.3 防止 reward hacking

Reward Model 不是人类本身，只是人类偏好的近似。

模型可能学会骗它。

比如 Reward Model 偏爱长答案，policy 可能开始输出超长废话。Reward Model 偏爱礼貌语气，policy 可能每句话都过度客套。

PPO + KL 不能彻底解决 reward hacking，但能让优化别太失控。

---

## 7. 把整个经典 RLHF + PPO 串起来

完整流程如下。

### 7.1 预训练

训练一个 base model：

```text
大规模文本 → next-token prediction → base model
```

它会续写文本，但不一定像助手。

### 7.2 SFT

用指令数据微调：

```text
prompt + ideal answer → cross entropy → SFT model
```

它开始像助手。

### 7.3 收集偏好数据

让 SFT model 或多个模型生成多个回答：

```text
prompt → answer A / answer B / answer C
```

人类排序：

```text
A > C > B
```

转换成 pairwise 数据：

```text
A > C
A > B
C > B
```

### 7.4 训练 Reward Model

```text
(prompt, chosen, rejected) → pairwise ranking loss → reward model
```

Reward Model 学会给好答案更高分。

### 7.5 PPO 优化助手

```text
policy 生成回答
  ↓
reward model 打分
  ↓
加 KL penalty
  ↓
PPO 更新 policy
```

最终得到 RLHF 后的 assistant。

---

## 8. 一个玩具例子

假设 prompt 是：

```text
用一句话解释什么是递归。
```

当前 policy 生成了三个可能回答：

```text
A: 递归就是函数在解决问题时调用自己。
B: 递归是一个很复杂的计算机概念，涉及很多东西。
C: 递归就是递归。
```

人类大概率偏好：

```text
A > B > C
```

Reward Model 学到：

```text
r(A) 高
r(B) 中
r(C) 低
```

PPO 阶段，如果 policy 生成了 A，reward 高，于是：

```text
“递归就是函数在解决问题时调用自己” 这类 token 路径概率略微提高。
```

如果生成了 C，reward 低，于是：

```text
“递归就是递归” 这类敷衍路径概率降低。
```

但 PPO 不会让模型一步到位地只会说 A，因为 clipping 和 KL 会限制更新幅度。

---

## 9. PPO 和普通 SFT 的区别

SFT 是模仿标准答案：

```text
给定 prompt，让模型最大化标准 answer 的概率。
```

PPO 是根据奖励调行为：

```text
模型自己生成 answer。
Reward Model 判断好坏。
好答案提高概率，坏答案降低概率。
但更新不能太大。
```

对比：

| 项目 | SFT | RLHF + PPO |
|---|---|---|
| 数据 | prompt + ideal answer | prompt + preference / reward |
| 训练信号 | 标准答案 token | 回答整体质量分数 |
| 模型行为 | 模仿示范 | 优化偏好 |
| 是否需要 Reward Model | 不需要 | 需要 |
| 是否采样模型输出 | 通常不需要 | 需要 rollout |
| 稳定性 | 相对稳定 | 更难、更贵、更不稳定 |

---

## 10. PPO 和 DPO 的关系

经典路线：

```text
SFT → Reward Model → PPO
```

DPO 路线：

```text
SFT → DPO
```

DPO 直接使用：

```text
prompt, chosen, rejected
```

不单独训练 Reward Model，也不跑 PPO。

所以 DPO 通常可以理解成：

> 用一个更简单的监督式目标，替代经典 RLHF 里的 Reward Model + PPO。

这也是为什么现在很多工程场景会优先尝试 DPO 类方法：简单、稳定、便宜。

但经典 RLHF + PPO 仍然重要，因为它奠定了很多对齐训练的概念框架。

---

## 11. RLHF + PPO 的常见坑

### 11.1 Reward Model 学偏了

如果偏好数据有偏，Reward Model 也会有偏。

例如标注员偏好长答案，Reward Model 就可能给长答案更高分，哪怕它更啰嗦。

### 11.2 Reward hacking

policy 可能找到 Reward Model 的漏洞，而不是真正变好。

```text
目标不是让人类满意，而是让 Reward Model 打高分。
```

这两者接近，但不相同。

### 11.3 KL 太强或太弱

KL 太强：

```text
模型几乎不变，RLHF 没效果。
```

KL 太弱：

```text
模型跑飞，语言质量或安全性下降。
```

### 11.4 PPO 调参复杂

PPO 有很多工程细节：

- reward normalization
- advantage estimation
- value loss
- policy loss
- KL coefficient
- clipping range
- rollout batch size
- sampling temperature
- stop sequence

这些都会影响最终结果。

### 11.5 评价很难

Reward Model 分数升高，不等于真实用户体验一定变好。

所以 RLHF 训练后还需要大量 eval：

- 人工评测
- 自动 benchmark
- 安全测试
- 红队测试
- 线上 A/B

---

## 12. 为什么 RLHF + PPO 曾经这么关键？

因为它解决了一个核心问题：

> 语言模型会说话，但我们想让它按照人类偏好说话。

预训练给了模型能力。

SFT 给了模型助手形态。

RLHF + PPO 进一步塑造它的行为：

- 更愿意遵循指令
- 更符合人类审美
- 更少输出有害内容
- 更像一个可靠助手
- 在“多个都还行”的答案中更常选人类喜欢的那个

它不是让模型突然拥有所有知识，而是改变模型在已有能力空间里的选择倾向。

---

## 13. 最后的心智模型

可以用一个简单类比记住 RLHF + PPO。

```text
SFT：看范文，模仿老师写答案。
Reward Model：训练一个阅卷老师，学会判断答案好坏。
PPO：学生反复答题，阅卷老师打分；学生根据分数调整答题习惯，但每次只许小幅调整，不能突然变成另一个人。
```

这就是经典 RLHF + PPO。

它的本质不是魔法，而是三件朴素的事：

1. 人类比较答案，提供偏好。
2. Reward Model 学会近似这些偏好。
3. PPO 用这个偏好信号稳定地更新语言模型。

最终目标只有一个：

> 让模型不只是会生成文本，而是更常生成“人类真正想要的文本”。
