# 第 07 章：Base Train：优化器与规模法则

> **本章源码地图**：训练循环在 `nanochat/scripts/base_train.py`；参数按用途分组在 `nanochat/nanochat/gpt.py::GPT.setup_optimizer`；AdamW、Muon 以及跨卡通信在 `nanochat/nanochat/optim.py::MuonAdamW`；设备、计算精度和进程组初始化在 `nanochat/nanochat/common.py`；FP8 线性层在 `nanochat/nanochat/fp8.py`；规模实验脚本是 `nanochat/runs/scaling_laws.sh` 与 `nanochat/runs/miniseries.sh`。

上一章结束时，我们已经得到了梯度：每个参数都有一个“往哪里改会让 loss 下降”的方向。但直接写 `parameter -= lr * gradient`，在 Transformer 的大规模训练中通常既不够快也不够稳。本章沿真实调用链解释 nanochat 接下来做的事：不同形状、不同职责的参数走不同优化器；学习率、动量和权重衰减随训练进程变化；低精度和多卡减少成本；最后用规模法则把许多手调旋钮收束到 `--depth`。

> **第一遍主线路线**：只想先理解 nanochat 全流程时，读 **§1（SGD → AdamW）→ §2 的参数分组结论 → §4（学习率日程）→ 小结** 即可。§3 的 Muon 数值实现、§5 的低精度、§6 的分布式通信和 §7 的规模法则可以第二遍再读；跳过它们不影响理解“loss.backward 后 optimizer 更新参数”的主线。

```text
x, y → model(x, y) → loss.backward()
    → base_train.py 设置各 param_group 的日程
    → MuonAdamW.step()
       ├─ AdamW：embedding / lm_head / 标量
       └─ Muon：Transformer 内的二维矩阵
    → zero_grad()
```

## 学习目标

读完本章，你应该能够：

1. 从 SGD 出发，解释 momentum、AdamW 和解耦权重衰减各自解决什么问题；
2. 说明为什么 `GPT.setup_optimizer` 不给所有参数使用同一组超参数，能在源码中定位每组参数；
3. 用矩阵形状理解 Muon 的“动量后正交化更新”，并区分其核心想法与当前实现的额外技巧；
4. 沿 `base_train.py` 读出 LR、Muon momentum、Muon weight decay 的实际调度；
5. 区分 bf16、fp16 的 `GradScaler` 与可选 FP8 矩阵乘，并知道它们并不改变 fp32 主权重更新；
6. 解释单卡、多进程同步与 ZeRO-2 式优化器状态分片的关系；
7. 根据源码说明参数—数据比、自动 batch/LR/WD、固定 FLOPs 扫描和 `--depth` 的含义与边界。

## 前置概念

请先读[第 06 章：Base Train：数据到损失与训练循环](./06-base-train-loop.md)。以下记号会反复出现：

| 符号 | 含义 | 本章源码线索 |
|---|---|---|
| \(\theta\) | 一个参数或参数矩阵 | `torch.nn.Parameter` |
| \(g_t\) | 第 `t` 次更新前的梯度 | `p.grad` |
| \(\eta_t\) | 学习率 | `group['lr']` |
| \(\lambda\) | 权重衰减系数 | `group['weight_decay']` |
| \(m_t,v_t\) | 一阶、二阶历史统计量 | AdamW 的 `exp_avg`、`exp_avg_sq` |
| \(B\) | 一次 optimizer 更新的全局 token batch | `total_batch_size` |
| \(D\) | 计划训练的总 token 数 | `target_tokens` / `total_tokens` |
| \(P\) | 模型参数规模；需先说明计数口径 | `num_scaling_params` |

这里的“step”始终指**一次 `optimizer.step()`**，而不是梯度累积中的一次 micro-step。

---

## 1. 经典基线：从 SGD 到 AdamW

最朴素的随机梯度下降（SGD）是：用当前 batch 梯度往下坡方向移动，

\[
\theta_{t+1}=\theta_t-\eta_t g_t.
\]

其中 \(\theta_t\) 是当前参数，\(g_t=\partial L/\partial\theta_t\) 是 loss 对参数的梯度，\(\eta_t\) 是步长。它的直觉正确，但 minibatch 梯度有噪声：这一步说向东，下一步可能说向北。不同参数的梯度量纲和典型大小也不同，一把固定尺子不够灵活。

### momentum：先把方向“平滑”

经典 momentum 维护梯度的指数滑动平均：

\[
m_t=\mu m_{t-1}+(1-\mu)g_t,
\qquad \theta_{t+1}=\theta_t-\eta m_t.
\]

\(\mu\) 是动量系数，接近 1 时记忆更长。沿着一致方向的梯度会叠加，左右摇摆的噪声会抵消。注意不同教材对 \(m_t\) 的归一化写法可不同，核心都是“带历史的一阶方向”。Muon 的第一步正是这种思想，后面会再加工这个方向。

### Adam：每个坐标有自己的步幅

Adam 同时维护一阶平均 \(m_t\) 与梯度平方的二阶平均 \(v_t\)。忽略初始偏差校正时，更新大意是：

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
\qquad
\theta_{t+1}=\theta_t-\eta\frac{m_t}{\sqrt{v_t}+\epsilon}.
\]

\(\beta_1,\beta_2\) 控制两种记忆，\(\epsilon\) 防止除零。若某一坐标一直有很大的梯度，分母变大，步子会缩小；反之也不会因极小分母爆炸。实际 Adam 还使用 \(1-\beta_1^t\)、\(1-\beta_2^t\) 校正起步时历史全为零的偏差。

`nanochat/nanochat/optim.py::adamw_step_fused` 正是这个基线的融合实现：先转 fp32，更新 `exp_avg`、`exp_avg_sq`，计算 `bias1`、`bias2`，再写回参数和状态。`@torch.compile` 与把可变超参数装在 0 维 CPU tensor 中，目的是避免 Python 开销和频繁重新编译；它们是**工程优化**，不是 Adam 数学的新定义。

### AdamW：把 weight decay 从梯度里拿出来

希望参数不要无边界增大时，可让每一步额外收缩参数。AdamW 的解耦写法是：

\[
\theta \leftarrow (1-\eta\lambda)\theta,
\qquad
\theta \leftarrow \theta-\eta\,\text{AdamDirection}.
\]

这与把 \(\lambda\theta\) 加进 Adam 梯度并不完全相同；对自适应分母而言，“正则项也被逐坐标缩放”会改变含义。源码明确先做：

```python
# optim.py::adamw_step_fused
p32.mul_(1 - lr_t * wd_t)
# 然后才更新 moment、做 bias correction、应用 Adam 更新
```

因此这里的 `weight_decay` 是解耦衰减，而不是某个 loss 函数里额外打印出来的 L2 项。

---

## 2. 不是“一种优化器管全家”：真实参数分组

`base_train.py` 创建模型后调用：

```python
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)
```

真正分组在 `nanochat/nanochat/gpt.py::GPT.setup_optimizer`。先把参数拆为：

- `transformer.h` 的参数：`matrix_params`，之后交给 Muon；
- token embedding `transformer.wte`、`value_embeds`、输出 `lm_head`：交给 AdamW；
- `resid_lambdas`、`x0_lambdas`、`smear_gate.weight`、`smear_lambda`、`backout_lambda`：也交给 AdamW，但各有不同配方。

这不是仅按“是否需要 weight decay”二分。embedding 是查表，`lm_head` 是输出投影，残差系数又是少量控制量；它们的尺度、功能和经验稳定区间不同。代码先检查所有参数恰好被覆盖一次，然后为 AdamW 组将 LR 乘上

\[
(\text{model\_dim}/768)^{-1/2}.
\]

这里 `model_dim` 是 `GPTConfig.n_embd`；这是一条以 768 宽度为参考的 **muP 风格外推配方**，不是从 SGD 定义必然推出的定理。

更重要的是：不要把命令行默认值误读为所有组的最终值。例如 `--embedding-lr` 默认 `0.3`，但 `value_embeds` 在组中又乘 `0.5`；`--scalar-lr` 默认 `0.5`，但 `resid_lambdas` 用 `scalar_lr * 0.01`，`smear_params` 则直接写 `0.2`。lm head、embedding、标量组也拥有不同的 `betas`、`eps` 和 decay。另一个容易困惑的例子是：`base_train.py` 的 CLI 默认 `--unembedding-lr=0.008`，`GPT.setup_optimizer` 函数签名默认是 `0.004`，RL CLI 也可使用 `0.004`；调用者显式传参时，函数签名默认不会生效。正确做法是沿调用链看实际实参，再看 `setup_optimizer` 构造出的 `param_groups`，而不是孤立地抄某一处默认值。

### 一个真实形状视角

假设某层 MLP 权重 \(W\) 形状为 `(768, 3072)`。它是二维、会对 token 流做矩阵乘，所以可以和其他同形状层权重堆叠。若该 Muon 组有 12 个矩阵，`muon_step_fused` 的注释所示输入形状可为：

```text
stacked_grads       : (12, 768, 3072)
stacked_params      : (12, 768, 3072)
momentum_buffer     : (12, 768, 3072)
second_momentum...  : (12, 1, 3072)  # 宽矩阵时按列统计
```

反过来，词表 embedding 常是 `(vocab_size, model_dim)`，最终投影和各种 0/1 维参数也不放进 Muon。`MuonAdamW` 的类说明明确警告：embedding、最终全连接层以及 0/1 维参数应走标准方法；当前仓库选择 AdamW。不要简单把“二维”当作绝对规则：这里的分组是这套模型的明确工程决定。

---

## 3. Muon：先有动量，再让矩阵更新更均衡

### 核心直觉：更新的“方向几何”也重要

对一个二维梯度矩阵 \(G\)，普通 momentum 得到 \(M\) 后直接更新，较大的奇异方向可能支配整个步子。Muon 的核心想法是：先做 momentum/Nesterov 风格组合，再将矩阵更新近似变为半正交（semi-orthogonal）的方向。若 SVD 为

\[
G=U\Sigma V^\top,
\]

理想化的正交化会将 \(\Sigma\) 替成 1，得到 \(UV^\top\)：保留左右子空间方向，压平不同奇异值的尺度。它不是在把**参数**强行正交化，而是在处理**本次更新**。

`muon_step_fused` 先做：

```python
momentum_buffer.lerp_(stacked_grads, 1 - momentum)
g = stacked_grads.lerp_(momentum_buffer, momentum)
```

随后不用昂贵的显式 SVD，而反复构造 `X @ X.mT`（宽矩阵）或 `X.mT @ X`（高矩阵），套用 `polar_express_coeffs` 的多项式迭代。`ns_steps` 在 `GPT.setup_optimizer` 的 Muon 组中设为 5。这相当于用矩阵乘法近似极分解/符号函数，适合 GPU。

### 当前 nanochat 的增量，不应偷换成 Muon 定义

`optim.py` 顶部讲得很坦白：本实现采用的是 `Polar Express` 系数，注释也指出结果不一定精确收敛为 \(UV^\top\)，而可能是奇异值在一定范围内的近似；它在经验上可用，不是精确 SVD。代码还增加了：

1. **MuonEq row equilibration**：按行范数重缩放，改善送入迭代的条件；
2. **Muon+ renormalization**：将 Frobenius 范数对齐到 \(\sqrt{\min(m,n)}\)；
3. **NorMuon 式 variance reduction**：按列或行维护因子化二阶统计量，让正交化后的神经元更新尺度更均匀；
4. **cautious weight decay**：只有 `g * stacked_params >= 0` 的元素才施加衰减掩码。

这些都是 nanochat 当前的现代、实验性优化配方；不能概括为“Muon 必然如此”。尤其代码注释明说某些关于 Muon 的 scaling 假设是从 AdamW 理论借来的，尚非严格验证。

Muon 更新前还将内部 `lr` 乘 \(\max(1,m/n)^{1/2}\)，其中矩阵形状为 `(m,n)`。这同样是本实现的形状校正。读优化器时要把三层分开：**原理**是动量加更新正交化；**数值实现**是迭代近似而非 SVD；**实验配方**是均衡、重归一化、方差缩放与谨慎衰减。

---

## 4. 日程：同一个 LR 并不贯穿训练

有梯度和优化器还不够；步长在训练早期、主体和末尾通常应不同。`base_train.py` 每次真正更新前计算：

```python
lrm = get_lr_multiplier(step)
muon_momentum = get_muon_momentum(step)
muon_weight_decay = get_weight_decay(step)
for group in optimizer.param_groups:
    group['lr'] = group['initial_lr'] * lrm
    if group['kind'] == 'muon':
        group['momentum'] = muon_momentum
        group['weight_decay'] = muon_weight_decay
```

### LR：线性 warmup、平台、线性 warmdown

`get_lr_multiplier` 使用 `warmup_steps`、`warmdown_ratio`、`final_lr_frac`。前 `warmup_steps`，乘数从 \(1/\text{warmup}\) 线性升至 1；中段恒为 1；最后 `round(warmdown_ratio * num_iterations)` 步线性降到 `final_lr_frac`。默认 CLI 分别是 40、0.65、0.05，但它们可被用户覆盖，且短跑时日程比例会和长跑非常不同。

warmup 的直觉是：刚初始化时激活、梯度统计和优化器状态都还未稳定，过大的第一步容易破坏训练；warmdown 则在预算末尾用更小步子细化。它是常见训练配方，不是模型架构的一部分。

### Muon momentum 与 WD：也是被调度的

`get_muon_momentum` 在前 400 step 从 0.85 线性升到 0.97；在 LR warmdown 区间由 0.97 降到 0.90；中间保持 0.97。它只写入 Muon 组，AdamW 的 betas 在组创建时固定。

`get_weight_decay` 让 Muon 的 `weight_decay_scaled` 在整个训练期间余弦衰减至 0：

\[
\lambda_t=\lambda_0\cdot\tfrac12\left(1+\cos(\pi t/N)\right),
\]

\(N\) 是 `num_iterations`。注意 AdamW 各组的 decay 没有在这段循环中重新赋值；此处“WD 调度”具体指 Muon 组。

### 最小实验：把日程当函数看，而非相信曲线名字

下面是**教学化简代码**，只复刻源码的 LR 乘数公式；它不会启动训练：

```python
def lr_multiplier(it, total=1000, warmup=40, warmdown_ratio=0.65, final=0.05):
    warmdown = round(warmdown_ratio * total)
    if it < warmup:
        return (it + 1) / warmup
    if it <= total - warmdown:
        return 1.0
    progress = (total - it) / warmdown
    return progress + (1 - progress) * final

for it in (0, 39, 40, 350, 999, 1000):
    print(it, round(lr_multiplier(it), 4))
```

检查点：`it=0` 不是 0，而是 `1/warmup`；`it=40` 已到平台；末端趋向 `final`。若把 `total` 改成很小的数，再观察 warmup/warmdown 是否重叠或主体平台变短，这正是不要把长训练默认日程照搬给玩具实验的原因。

---

## 5. 精度：bf16、fp16 GradScaler 与 FP8 各在何处

### 计算 dtype 与主权重

`nanochat/nanochat/common.py::_detect_compute_dtype` 先看 `NANOCHAT_DTYPE`，否则 CUDA SM 8.0 及以上自动选 `torch.bfloat16`，较旧 CUDA 自动选 fp32，CPU/MPS 默认 fp32。`COMPUTE_DTYPE` 是矩阵乘、激活等**计算**精度；`Linear` 的主权重仍为 fp32，并在 forward 转为输入 dtype，而 embedding/value embedding 则会在初始化时转为 `COMPUTE_DTYPE`（强制 fp16 时例外，保留 fp32 以便 GradScaler unscale）。这是一套显式 dtype 管理，不依赖通用 autocast。

bf16 与 fp32 有相同指数范围，数很小或很大时较不易溢出；代价是尾数精度较低。fp16 的指数范围较窄，反传梯度可能下溢为 0 或溢出为 inf。`chat_sft.py` 也实现同一 fp16 `GradScaler` 分支；`chat_rl.py` 当前没有这条 scaler 路径，因此不应把“fp16 训练已支持”泛化到 RL。

### GradScaler 只为 fp16 路径兜底

因此 `base_train.py` 仅在 `COMPUTE_DTYPE == torch.float16` 时创建：

```python
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
```

它在 backward 前放大 loss，更新前 `unscale_` 梯度；若检测到 inf/nan，`scaler.step(optimizer)` 会跳过不安全更新并调整 scale。多进程时每张卡可能各自发现异常，代码对 `scaler._found_inf_per_device(optimizer)` 做 `dist.all_reduce(MAX)`，确保任一 rank 失败时所有 rank 一致跳过。bf16/fp32 不走这个 scaler 分支。不要把 GradScaler 当成“让低精度更准确”的一般开关；它主要避免 fp16 梯度范围问题。

### FP8：可选地加速 Linear 的三次 GEMM

传 `--fp8` 时，`base_train.py` 仅在 CUDA 上尝试转换适合的 `nn.Linear`：输入/输出维度都能被 16 整除，且较小维度至少 128。`convert_to_float8_training` 将这些模块替成 `Float8Linear`，但共享原来的 `weight`、`bias`，不复制主参数。

`fp8.py::_Float8Matmul` 对 Linear 的三个矩阵乘分别处理：

```text
forward:      input @ weight.T
backward:     grad_output @ weight          → grad_input
grad_output.T @ input                       → grad_weight
```

它按 tensor 最大绝对值求一个动态 scale，将 input/weight 量化为 `float8_e4m3fn`，梯度用范围更大的 `float8_e5m2`，调用 PyTorch 内建 `torch._scaled_mm`，输出仍回到计算 dtype。tensorwise 的意思是“每个张量一个标量 scale”，速度好但不如按行缩放细致。

这里有一个值得亲自核对的源码事实：CLI 的 `--fp8-recipe` choices 列出 `rowwise`、`tensorwise`，帮助文字也提到两者；但当前 `Float8LinearConfig.from_recipe_name` 实际只接受 `tensorwise`，否则抛出 `ValueError`。因此不要据此文档或参数 choices 声称仓库已实现 rowwise。FP8 还要求合适硬件和 CUDA；评估时 `disable_fp8(model)` 临时换回普通 `Linear`，以 bf16 路径评估，之后恢复 FP8 模块。它是**速度/吞吐实验技巧**，不是让模型多学信息的原理。

---

## 6. 多卡：启动像 DDP，同步在优化器里；状态按 ZeRO-2 思路分片

`torchrun --nproc_per_node=8 -m scripts.base_train` 会提供 `RANK`、`LOCAL_RANK`、`WORLD_SIZE`。`common.py::compute_init` 检测到这些变量且设备为 CUDA 后，选择本地 GPU、以 NCCL 初始化 process group，并返回 rank/world size。每个进程各自完成前向和反向，因而首先各有一份本地梯度。

一个容易误读的细节是：代码注释明确写着 **nanochat does not use DDP**。也就是说，它没有把模型包成 `torch.nn.parallel.DistributedDataParallel` 让 DDP 的 backward hook 做梯度 all-reduce；`MuonAdamW.step()` 自己承担梯度通信和参数同步。因此这里“DDP”可指 `torchrun` 多 rank 运行/分布式语境，但不是 PyTorch `DDP(model)` 那条实现路径。

对 AdamW 大参数，`_reduce_adamw` 采用：

```text
本地完整梯度 → reduce_scatter(AVG) → 本 rank 的梯度切片
→ 只更新对应参数切片及其 exp_avg / exp_avg_sq
→ all_gather → 每张卡重新拥有完整、相同的参数
```

小于 1024 个元素的小参数则直接 all-reduce，完整更新，因状态本来很小。大参数要求第 0 维可被 `world_size` 整除，这是源码 `assert` 的实际限制。

Muon 按**同形状参数的列表维**堆叠、补零，再将一组矩阵分给不同 rank：`_reduce_muon` 产生 `(K, *shape)` 的 stack，必要时 padding 到 `ceil(K/world_size) * world_size`；每个 rank 更新自己拥有的 chunk，最后 all-gather 并 `torch._foreach_copy_` 回原参数。每张卡仅保存自己 chunk 的 momentum/二阶状态。

这叫“**ZeRO-2 式**”是因为优化器状态（并且更新计算所需梯度切片）分片，而训练后每个 rank 仍有完整参数；它并非完整复刻某个 ZeRO 系统，也不意味着参数本身长期分片。实现分三阶段：异步启动 reduce；等待各组 reduce、更新并启动 gather；等待 gather、复制结果。目的是把通信和后续计算重叠。单卡时 `world_size=1`，通信分支被跳过，代码自然退化为普通完整参数更新。

---

## 7. 规模法则：把“训多久、多大 batch”变成可检验假设

### 先声明 P 的口径

“模型有多少参数”并不总有唯一答案。`GPT.num_scaling_params()` 分别报告 `wte`、`value_embeds`、`lm_head`、`transformer_matrices`、`scalars`、`total`。`base_train.py::get_scaling_params` 当前选择：

```python
transformer_matrices + lm_head
```

作为 scaling 参数。原因是代码注释称这种口径在其分析中得到较干净的 scaling law；这不是所有论文的统一定义。比如历史上 Kaplan 与 Chinchilla 对 embedding 是否计入就有不同约定。比较 \(D/P\) 前必须固定此口径。

### 参数—数据比决定默认训练 horizon

脚本默认 `--target-param-data-ratio=12`。设当前口径下参数数为 \(P\)，则目标 token 数：

\[
D=12P.
\]

命令行帮助同时注明 Chinchilla 常被概括为约 20；nanochat 的默认 12 是当前工程配方，不应写成 Chinchilla 的默认值。若显式给 `--num-iterations`，它优先；否则 `--target-flops` 次之；最后才用这个参数—数据比。总步数分别为：

\[
N=\frac{\text{target FLOPs}}{\text{FLOPs/token}\times B}
\quad\text{或}\quad
N=\left\lfloor D/B\right\rfloor.
\]

其中 `model.estimate_flops()` 给出每 token 的估算训练 FLOPs：线性矩阵参数约按 forward 2、backward 4 合计 6 FLOPs/参数，再加注意力 QK 等项；这是估算尺度，不是硬件计时。

### 自动 batch、LR、WD：链条而非独立魔法数

参考模型是 meta device 上构建的 depth 12，`D_REF = ratio * P_ref`，`B_REF = 2**19`。若用户没有覆盖 `--total-batch-size`，代码按经验关系

\[
B_{opt}\propto D^{0.383}
\]

预测 batch，并四舍五入到最近的 2 的幂。然后对 AdamW 和 Muon 都使用

\[
\eta=\eta_{ref}\sqrt{B/B_{ref}}.
\]

源码明确说 SGD 的线性缩放是常见基线、AdamW 的平方根缩放是常见做法，而把同样平方根法则用于 Muon 是“未仔细研究的 assumption”。这正是应报告为假设、而不是理论保证的地方。

最后使用 `T_epoch = B/(ηλD)` 保持不变的框架，得到 Muon decay 的缩放：

\[
\lambda=\lambda_{ref}\sqrt{B/B_{ref}}\,(D_{ref}/D).
\]

注意后续它还会随训练余弦衰减至零。这一串自动化减少了调参维度，但没有消灭实验：数据分布、硬件可承受 batch、模型结构和优化器改动都可能让参考关系失效。

### `--depth` 为什么像唯一复杂度旋钮

`build_model_meta(depth)` 中先令 `base_dim = depth * aspect_ratio`，默认 `aspect_ratio=64`，再向上取整到 `head_dim` 的倍数；默认 `head_dim=128`。`num_heads = model_dim // head_dim`。因此 depth 增加通常同时带来更深、更宽、更多 attention heads 的模型；它不是“只多一层而其余绝对不变”。meta device 只构造形状、不分配真实数据，正适合先数参数、估 FLOPs、推导 batch/预算。

### 两个实验脚本分别问什么？

`runs/scaling_laws.sh` 对多个固定 `FLOPS_BUDGETS` 与多个 `DEPTHS` 组合训练。它传 `--target-flops` 和 `--target-param-data-ratio=-1`，让 `base_train.py` 按同一计算预算算步数；每次从日志取参数、iterations、batch、最终 val BPB、CORE、时间，追加 CSV。它回答的更接近：**同一预算下哪个规模更好？**

`runs/miniseries.sh` 则遍历一串 depths，直接按训练脚本默认参数—数据比运行，并保存 depth、参数、scaling 参数、token、ratio、BPB、CORE、时间。它更像：**沿默认配方，模型随深度如何变化？** 两者都以 `torchrun` 多进程启动，且只在日志设定的最终时机测 CORE；它们是昂贵实验编排，读者不需要为理解本章而运行。

另外，脚本按 depth 降低 `--device-batch-size` 来避免 OOM。这不等于改变全局 `total_batch_size` 的理论目标：训练脚本会用梯度累积凑全局 batch，前提是它能被每次全局 micro-batch token 数整除。

---

## 常见误区

1. **“AdamW 就是 Adam 加 L2 loss。”** AdamW 在代码中先直接缩参数，再做自适应更新；与把 L2 梯度交给 Adam 缩放不完全等价。
2. **“Muon 把模型权重正交化。”** 它处理的是每步二维矩阵的更新方向；参数不是被强制保持正交。
3. **“所有二维参数都该用 Muon。”** 当前实现明确将 embedding、lm head 和标量交给 AdamW；分组是架构相关选择。
4. **“`--embedding-lr` 就是 embedding 相关所有张量的最终 LR。”** `setup_optimizer` 还会做宽度缩放和 value embedding 的额外 0.5 缩放。
5. **“WD 只要设一次。”** Muon 的 decay 既受规模缩放，又在训练中余弦降至零；AdamW 组的 decay 在当前循环中不走该 scheduler。
6. **“bf16 也必须开 GradScaler。”** 当前代码仅对 fp16 创建 scaler；bf16 指数范围与 fp32 相同。
7. **“`--fp8-recipe=rowwise` 已可用。”** CLI choices 虽列出它，当前 `fp8.py` 实现只接受 `tensorwise`。
8. **“用了 torchrun 就一定用了 `DistributedDataParallel`。”** 本仓库初始化 process group，但梯度同步由 `MuonAdamW` 的通信代码完成，并未包装 DDP 模型。
9. **“ZeRO-2 式分片后每卡没有完整参数。”** 此实现分片的是优化器状态/更新切片，all-gather 后每卡仍得到完整同步参数。
10. **“12 token/parameter 是普适最优定律。”** 它是此仓库、此参数计数口径与实验配方下的默认假设；不同数据、预算与架构需重新验证。

## 小结

训练循环得到梯度后，SGD 给出最基本的更新方向；momentum 平滑方向；AdamW 用一阶、二阶统计量自适应缩放，并将权重衰减解耦。nanochat 在 `GPT.setup_optimizer` 中让 embedding、输出头和控制标量走 AdamW，让 Transformer 内同形状矩阵分组走 Muon。Muon 的核心是动量更新后近似正交化；Polar Express、行均衡、重归一化、方差缩放和谨慎衰减则是当前实现的额外配方。

`base_train.py` 将 LR 做 warmup—平台—warmdown，单独调度 Muon momentum 和 decay。bf16/fp16/FP8 主要是计算效率与数值范围的选择，主权重更新仍强调 fp32；多卡通过优化器内部异步 reduce/gather 实现同步与 ZeRO-2 式状态分片。最后，规模法则将目标数据量、batch、LR、WD 关联起来，让 `--depth` 成为主要复杂度入口，但每个关系都应视为可测量、可推翻的经验假设。

## 练习

1. 对一维参数 \(\theta=10\)、梯度 \(g=2\)、\(\eta=0.1\)、\(\lambda=0.01\)，按 AdamW 的“先 decay、再梯度更新”顺序（将 Adam direction 简化为 \(g\)）算一次更新。再比较把 \(\lambda\theta\) 混入梯度的 SGD 写法。
2. 打开 `gpt.py::GPT.setup_optimizer`，列出所有 AdamW group 的参数、LR 来源、betas 和 decay。哪些项来自命令行，哪些是函数中硬编码的？
3. 对形状 `(12, 768, 3072)` 的 Muon stack，说明三维各自表示什么。为什么宽矩阵的 `second_momentum_buffer` 可以是 `(12, 1, 3072)`？
4. 运行本章的教学化简 LR 实验（不训练）。将 `total` 改为 80，画或打印每 10 步的 multiplier；解释长训练默认 warmdown 比例在这种短跑中的含义。
5. 假设 `P=50,000,000`、ratio=12、自动得到全局 `B=524,288`。计算目标 token 数和按参数—数据比得到的完整 step 数（忽略整数截断）。若 8 个 rank、每 rank `device_batch_size=16`、`T=2048`，梯度累积是多少？
6. 设计一个不昂贵的 scaling-law 验证计划：固定哪些数据、tokenizer、评估集、FLOPs 口径和随机种子？为什么不能只比较“训练时间最短”的运行？

## 前后章导航

- ← 上一章：[第 06 章：Base Train：数据到损失与训练循环](./06-base-train-loop.md)。那里构造 `x/y`、交叉熵和梯度；本章说明梯度如何成为稳定、高效的参数更新。
- → 下一章：[第 08 章：Base Eval：BPB、CORE 与 sample](./08-base-evaluation.md)。优化配方和规模选择最终要由验证 BPB、CORE 与任务评估检验，而不能只看训练 loss。
