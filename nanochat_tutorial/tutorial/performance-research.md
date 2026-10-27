# 附录 C：性能、排行榜与研究方法

nanochat 的性能系统、Time-to-GPT-2 排行榜和 `dev/LOG.md` 各自回答不同问题。它们值得独立学习，因为“模型更准”“训练更快”“推理更便宜”是三种不同结论。

## 1. Time-to-GPT-2 到底测什么？

当前排行榜目标是在统一的 **8×H100** 条件下，让 base model 的 DCLM CORE 超过 GPT-2 基线 `0.256525`，并按 `total_training_time` 排序。这个时间只累计训练 iteration，排除评估、sample、日志和 checkpoint。验证 BPB 是辅助质量信号。

因此一项修改只有在以下链条同时成立时，才能称为排行榜进步：

```text
相同硬件与协议
  → CORE 超过阈值
  → 训练 iteration 墙钟更少
  → val BPB 没有显示异常退化
  → 代码 revision、命令与日志可复查
```

它不等价于：参数最少、FLOPs 最少、聊天最好、推理最快或训练总账单最低。数据下载、tokenizer、SFT 和评估虽然属于产品流水线，却不计入排行榜主时间。

## 2. 为什么还要 miniseries？

nanochat 希望 `--depth` 是主复杂度旋钮。一项只在 d24 的一个 seed 上偶然获胜的技巧，可能是对单点的过拟合。`runs/miniseries.sh` 沿多个 depth 检查默认配方是否仍呈合理缩放；`runs/scaling_laws.sh` 则在多种 FLOPs budget 下比较规模选择。

| 实验 | 固定什么 | 改什么 | 回答什么 |
|---|---|---|---|
| 单次 d12 快速迭代 | 数据、seed、预算、评估 | 一个实现或配方 | 值得继续测吗？ |
| miniseries | 默认训练规则 | depth | 改动能跨规模迁移吗？ |
| scaling laws | FLOPs budget 与协议 | depth、预算 | 同算力下哪个规模更合适？ |
| speedrun | 8×H100、CORE 阈值 | 完整参考配方 | 达标需要多少训练墙钟？ |

“一个复杂度旋钮”是减少用户配置和约束研究改动的设计哲学，不是自动保证全局 compute-optimal 的数学证明。

## 3. 训练性能：tokens/s 与 MFU

一次训练 step 处理 `total_batch_size` 个 token，耗时为 `dt`：

\[
\text{tokens/s}=\frac{B_{tokens}}{dt}.
\]

模型用 `estimate_flops()` 估计每 token 的 forward+backward FLOPs，再与 GPU 理论峰值比较：

\[
\mathrm{MFU}
=\frac{\text{FLOPs/token}\times B_{tokens}/dt}
       {\text{peak FLOPs/GPU}\times\text{world size}}.
\]

MFU（Model FLOPs Utilization）回答“离理论算力上限多远”。它依赖 FLOPs 口径与设备峰值表；未知 GPU 时数字可能没有意义。高 MFU 也不代表模型质量高：删除评估、缩短上下文或改变模型都可能让吞吐变好，却改变任务本身。

## 4. 推理性能：TTFT、TPOT、MBU

`scripts/infer_bench.py` 把推理分成两个 regime：

- **Prefill**：整段 prompt 并行处理，大矩阵乘多，通常更偏 compute-bound；重点看 TTFT（首 token 时间）与 prefill MFU。
- **Decode**：每步只产生一个 token，却需重读权重和增长中的 KV，常更偏 memory-bandwidth-bound；重点看 TPOT（每输出 token 时间）、总 tok/s 与 MBU。

MBU（Model Bandwidth Utilization）的近似为：

\[
\mathrm{MBU}
=\frac{(\text{weight bytes}+B\times\text{KV read bytes})/\mathrm{TPOT}}
       {\text{peak memory bandwidth}}.
\]

脚本会扫 decode batch size，得到延迟—吞吐曲线。增大 batch 往往提高总 tok/s，却会增加显存、排队和单请求延迟；因此不能只挑最高吞吐点。

当前 benchmark 的准确边界：

- 仅支持 **单张 CUDA GPU**，不能在 `torchrun` 下运行；
- 第一次 `next(generator)` 的 TTFT 包含 prefill、cache 复制和首次采样，不是纯 kernel prefill；
- TPOT 使用 decode step 时间的中位数；
- prompt + decode 被限制在 `sequence_len` 内；
- 架构静态估算不是实测速度。

## 5. 研究日志为什么记录失败？

`dev/LOG.md` 展示了 nanochat 的目标函数：不是“每 step loss 最低”，而是**在代码仍简洁、跨规模可迁移的前提下，更快达到能力阈值**。这解释了很多负结果：

| 尝试 | 观察 | 为什么没保留 |
|---|---|---|
| DyT | 吞吐约低 10%，质量未胜基线 | 增加参数和计算但无净收益 |
| MoE | per-step 可能更好 | GPT-2 规模下 dispatch 开销吃掉墙钟收益 |
| SwiGLU | 现代常见 | 当前规模/配方下不如 ReLU² 的 step 与墙钟效率 |
| multi-token prediction | 增加监督 | 显存从约 34GB 到 47GB，墙钟更差 |
| varlen 文档隔离 | 减少跨文档 attention | 改善太小，不值得复杂度 |
| LeakyReLU² / XSA | 部分 per-step 改善 | 单步更慢，Time-to-GPT-2 反而更差 |
| gradient clipping | 常规稳定技巧 | 梯度本来未触阈值，却增加通信/时间 |
| ClimbMix 替换 FineWeb-EDU | 数据质量提升 | **保留**；这是显著减少达标墙钟的改动 |

负结果不是“这些技术永远无效”。它们只说明：在记录的 revision、规模、数据、预算和评估下，没有击败本项目的目标函数。换成更大模型、不同硬件或不同下游目标，结论可能变化。

## 6. 一个最小、可信的改动实验

1. 写清假设，例如“窗口模式 X 在不降低最终 BPB/CORE 的情况下提高 tokens/s”。
2. 固定 revision、数据 shard、tokenizer、depth、训练 token/FLOPs、seed、评估 bundle。
3. 一次只改一个因素；先用 d12 做至少基线与候选对照。
4. 同时记录：train loss、val BPB、CORE、tokens/s、MFU、peak VRAM、总训练时间。
5. 若涉及推理，再记录固定 prompt/decode 长度下的 TTFT、TPOT、tok/s、MBU 与 batch sweep。
6. 结果接近噪声时重复 seed，不挑最好的一次。
7. 候选通过后再跑 miniseries；最后才值得跑完整 speedrun。
8. 保留失败结果和回滚理由，避免后来者重复支付成本。

## 7. 报告模板

```text
revision / diff:
hardware / driver / torch / dtype / FA3 or SDPA:
data shards / tokenizer hash:
model depth, dim, heads, window pattern:
training horizon / global batch / CLI:
quality: min val BPB, final CORE, task details:
training: total_training_time, tok/s, MFU, peak VRAM:
inference: prompt/decode lengths, batch sweep, TTFT, TPOT, tok/s, MBU:
seeds / run count / spread:
known confounders and unrun checks:
```

这张表的意义是防止三类偷换：用吞吐代替质量、用 pass@k 代替单次回答、用一个硬件上的速度代替算法普适性。

---

- 返回[教程目录](./README.md)
- 查阅[附录 B：源码主题与文件索引](./source-index.md)
