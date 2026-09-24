# 附录 B：术语与符号速查

这里给出“一句话定义 + 首次深入章节”。定义刻意简短；遇到实现细节，请回到对应章节与当前源码。

## 常用形状符号

| 符号 | 含义 |
|---|---|
| `B` | batch size，并行处理的序列数 |
| `T` | 当前序列的 token 数 |
| `V` | tokenizer 的词表大小 |
| `C` | 模型宽度，即 embedding 通道数 `n_embd` |
| `H` / `H_kv` | query head 数 / key-value head 数 |
| `D` | 单个 attention head 宽度，通常为 `C/H` |
| `L` | Transformer 层数；部分章节也用它表示生成轨迹长度，以上下文为准 |

## 数据、模型与训练

| 术语 | 一句话定义 | 深入章节 |
|---|---|---|
| shard | 大数据集切成的一个文件分片；nanochat 当前使用 Parquet | [02](./02-dataset.md) |
| row group | Parquet 内可独立读取的一块行数据，nanochat 也用它做多 rank 分片 | [02](./02-dataset.md) |
| BPE | 从字节原子开始，反复合并高频相邻片段的 tokenizer 训练方法 | [03](./03-tokenizer.md) |
| BOS | 文档/序列开始标记 `<|bos|>` | [03](./03-tokenizer.md) |
| embedding | 把离散 token id 查表映射为连续向量 | [04](./04-gpt-attention.md) |
| causal attention | 每个位置只能读取自己及过去，不能偷看未来 | [04](./04-gpt-attention.md) |
| RoPE | 通过旋转 Q/K 通道对注入位置信息 | [04](./04-gpt-attention.md) |
| QK norm | 在注意力点积前规范化 Q/K，改善训练稳定性的一种架构选择 | [04](./04-gpt-attention.md) |
| GQA | 多个 query head 共享较少的 KV head，以减少 KV cache | [04](./04-gpt-attention.md) |
| RMSNorm | 用均方根归一化 token 表示；nanochat 版本没有可学习缩放参数 | [05](./05-gpt-mlp-tricks.md) |
| residual stream | 各层通过加法持续读取和写回的 `(B,T,C)` 表示主干 | [05](./05-gpt-mlp-tricks.md) |
| value embedding | nanochat 的 ResFormer 风格实验增量：从 token id 额外生成 V 内容 | [05](./05-gpt-mlp-tricks.md) |
| smear | 门控地把上一 token 的归一化 embedding 混入当前位置 | [05](./05-gpt-mlp-tricks.md) |
| backout | 最终 norm 前减去一部分中层残差的实验性技巧 | [05](./05-gpt-mlp-tricks.md) |
| softcap | 用 `tanh` 平滑限制 logits 的绝对值 | [05](./05-gpt-mlp-tricks.md) |
| gradient accumulation | 多次 forward/backward 累积梯度后才更新一次，换时间省显存 | [06](./06-base-train-loop.md) |
| AdamW | 带一阶/二阶统计量、解耦权重衰减的优化器 | [07](./07-base-train-optim-scaling.md) |
| Muon | 对二维矩阵的动量更新做近似正交化的优化器思路 | [07](./07-base-train-optim-scaling.md) |
| muP 风格缩放 | 跨模型宽度迁移超参数的经验配方；本教程只解释仓库中的具体用法 | [07](./07-base-train-optim-scaling.md) |
| ZeRO-2 式分片 | 分片梯度更新与优化器状态，而每个 rank 最终仍持有完整参数 | [07](./07-base-train-optim-scaling.md) |
| FP8 | 用 8 位浮点格式加速合适硬件上的矩阵乘；不是模型能力原理 | [07](./07-base-train-optim-scaling.md) |

## 评估、推理与后训练

| 术语 | 一句话定义 | 深入章节 |
|---|---|---|
| BPB | `总负对数概率 / (ln2 × 原始有效字节数)`，即每字节 bit 损失 | [08](./08-base-evaluation.md) |
| perplexity | `exp(平均每 token NLL)`；依赖 tokenizer 切分，跨词表比较要谨慎 | [08](./08-base-evaluation.md) |
| CORE | base 模型在固定 ICL 协议下的多任务中心化汇总 | [08](./08-base-evaluation.md) |
| SFT | 用带答案的对话示范继续做监督学习 | [09](./09-sft.md) |
| loss mask | 指定哪些 target token 产生 loss；上下文仍可见但可不受监督 | [09](./09-sft.md) |
| prefill | 一次处理完整 prompt，并写入增量推理状态 | [10](./10-inference-engine.md) |
| decode | prefill 后逐 token 生成的阶段 | [10](./10-inference-engine.md) |
| KV cache | 保存历史 token 在每层的 key/value，避免重复投影历史 | [10](./10-inference-engine.md) |
| temperature | 采样前缩放 logits 的参数；0 在本实现中表示 greedy | [10](./10-inference-engine.md) |
| top-k | 只在分数最高的 k 个 token 中采样 | [10](./10-inference-engine.md) |
| ChatCORE | 五个 chat 任务按随机基线中心化后的等权工程汇总 | [11](./11-chat-evaluation.md) |
| pass@k | 前 k 个候选至少一个通过 verifier 的成功事件 | [11](./11-chat-evaluation.md) |
| rollout | 策略针对 prompt 采样出的一整条回答轨迹 | [12](./12-rl.md) |
| reward | 对整条 rollout 的标量结果反馈 | [12](./12-rl.md) |
| advantage | 奖励相对 baseline 的差值；当前代码用 `reward - group mean` | [12](./12-rl.md) |
| REINFORCE | 用 `log probability × return/advantage` 更新采样策略的策略梯度方法 | [12](./12-rl.md) |
| GRPO 风格 | 本仓库借用同题多样本组内比较，但不是包含所有常见组件的完整配方 | [12](./12-rl.md) |
| DAPO 风格 token-level normalization | 按有效 token 总数归一化策略目标；不代表得到逐 token 奖励 | [12](./12-rl.md) |
| on-policy | 用当前策略刚采集的数据更新当前策略；当前脚本每批 rollout 只更新一次 | [12](./12-rl.md) |
| KL regularization | 限制训练策略偏离 reference policy 的稳定机制；当前 RL 未使用 | [12](./12-rl.md) |
| PPO ratio/clip | 修正旧策略数据复用并限制更新幅度的机制；当前 on-policy 单次更新未使用 | [12](./12-rl.md) |

---

- 返回[教程目录](./README.md)
- 动手验证：[附录 A：零数据 Toy Checks](./toy-checks.md)
