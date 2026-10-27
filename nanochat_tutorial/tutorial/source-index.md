# 附录 B：源码主题与文件索引

> 本附录按主题反查源码。想“先理解再查代码”的读者，请从[第 01 章：总览](./01-overview.md)开始；本附录面向已经在读代码、想快速定位某个具体行为的工程师与维护者。全书所有判断仍以本仓库固定的 nanochat 源码 revision `92d63d4` 为准。

## 1. 面向常见工程问题的反向索引

| 想查的问题 | 对应源码入口 | 本书章节 |
|---|---|---|
| 完整端到端如何编排 | `runs/speedrun.sh`、`runs/runcpu.sh` | [01](./01-overview.md)、[13](./13-end-to-end.md) |
| ClimbMix 下载、Parquet 读取与多-rank分片 | `nanochat/dataset.py`、`nanochat/dataloader.py` | [02](./02-dataset.md) |
| BOS-aligned best-fit packing 与近似恢复 | `nanochat/dataloader.py::tokenizing_distributed_data_loader_with_state_bos_bestfit` | [02](./02-dataset.md) |
| 字节级 BPE、特殊 token 与对话渲染 | `nanochat/tokenizer.py`、`scripts/tok_train.py` | [03](./03-tokenizer.md)、[09](./09-sft.md) |
| 因果多头注意力、RoPE、QK norm、GQA | `nanochat/gpt.py::CausalSelfAttention`、`apply_rotary_emb` | [04](./04-gpt-attention.md) |
| 滑动窗口与 Flash Attention 3 / SDPA 回退 | `nanochat/flash_attention.py`、`nanochat/gpt.py::_compute_window_sizes` | [04](./04-gpt-attention.md)、[07](./07-base-train-optim-scaling.md) |
| MLP、残差流、value embedding、smear、backout、softcap | `nanochat/gpt.py::MLP`、`Block`、`GPT.forward` | [05](./05-gpt-mlp-tricks.md) |
| 权重初始化与 meta device 构造 | `nanochat/gpt.py::GPT.init_weights`、`scripts/base_train.py::build_model_meta` | [05](./05-gpt-mlp-tricks.md)、[07](./07-base-train-optim-scaling.md) |
| 基础训练循环、梯度累积、评估调度 | `scripts/base_train.py` | [06](./06-base-train-loop.md) |
| 交叉熵、`ignore_index=-1` 与右移目标 | `nanochat/gpt.py::GPT.forward`、`scripts/base_train.py` | [06](./06-base-train-loop.md)、[09](./09-sft.md) |
| AdamW + Muon 参数分组与融合 kernel | `nanochat/optim.py`、`nanochat/gpt.py::GPT.setup_optimizer` | [07](./07-base-train-optim-scaling.md) |
| 学习率、Muon momentum 与权重衰减调度 | `scripts/base_train.py::get_lr_multiplier`、`get_muon_momentum`、`get_weight_decay` | [07](./07-base-train-optim-scaling.md) |
| 复杂度旋钮与规模法则 | `scripts/base_train.py::build_model_meta`、参数—数据比与批量推导段 | [07](./07-base-train-optim-scaling.md)、[附录 C](./performance-research.md) |
| 分布式优化器通信（ZeRO-2 风格） | `nanochat/optim.py::MuonAdamW`（`_reduce_adamw`、`_reduce_muon`、`step`） | [07](./07-base-train-optim-scaling.md) |
| FP8 训练与 `disable_fp8` 评估切换 | `nanochat/fp8.py`、`scripts/base_train.py::disable_fp8` | [07](./07-base-train-optim-scaling.md) |
| 显式精度管理与自定义 `Linear` | `nanochat/common.py::COMPUTE_DTYPE`、`nanochat/gpt.py::Linear` | [05](./05-gpt-mlp-tricks.md)、[07](./07-base-train-optim-scaling.md) |
| Base 评估：BPB、CORE、sample | `nanochat/loss_eval.py::evaluate_bpb`、`scripts/base_eval.py`、`nanochat/core_eval.py` | [08](./08-base-evaluation.md) |
| checkpoint 保存与恢复（含 approximate resume） | `nanochat/checkpoint_manager.py`、`scripts/base_train.py` | [01](./01-overview.md)、[06](./06-base-train-loop.md)、[13](./13-end-to-end.md) |
| SFT 对话协议、mask 右移与 best-fit padding | `scripts/chat_sft.py`、`nanochat/tokenizer.py::RustBPETokenizer.render_conversation` | [09](./09-sft.md) |
| 任务数据与混合 | `tasks/common.py`、`tasks/smoltalk.py`、`tasks/mmlu.py`、`tasks/gsm8k.py` | [09](./09-sft.md) |
| 推理引擎、KV cache 与 smear 前一 embedding | `nanochat/engine.py::Engine`、`KVCache` | [10](./10-inference-engine.md) |
| 采样策略、工具状态机与计算器 | `nanochat/engine.py::sample_next_token`、`RowState`、`use_calculator` | [10](./10-inference-engine.md)、[11](./11-chat-evaluation.md) |
| 聊天任务评估、ChatCORE、pass@k | `scripts/chat_eval.py`、`tasks/arc.py`、`tasks/humaneval.py` | [11](./11-chat-evaluation.md) |
| HumanEval 代码执行防护（非安全沙箱） | `nanochat/execution.py` | [11](./11-chat-evaluation.md) |
| 简化的 GRPO/REINFORCE RL | `scripts/chat_rl.py`、`tasks/gsm8k.py::GSM8K.reward` | [12](./12-rl.md) |
| 推理性能：权重/KV 字节、TTFT、TPOT、MBU | `scripts/infer_bench.py`、`nanochat/gpt.py::kv_bytes_per_token`、`kv_read_bytes` | [10](./10-inference-engine.md)、[附录 C](./performance-research.md) |
| 设备、分布式、精度与路径工具 | `nanochat/common.py`：`compute_init`、`autodetect_device_type`、`get_base_dir`、`download_file_with_lock`、`get_peak_flops`、`get_peak_bandwidth` | [01](./01-overview.md)、[07](./07-base-train-optim-scaling.md) |
| 依赖、脚本入口与开发测试配置 | `pyproject.toml`、`runs/`、`tests/` | [13](./13-end-to-end.md) |

## 2. 推荐的源码阅读顺序

想从“读本书”进阶到“改代码”，建议按真实调用链而不是文件名顺序：

1. `runs/speedrun.sh`：找到真实阶段边界和命令行覆盖；
2. `scripts/base_train.py`：看模型、数据、优化器、评估、checkpoint 如何编排；
3. `nanochat/dataloader.py` → `nanochat/tokenizer.py`：看 `(B,T)` 从哪里来；
4. `nanochat/gpt.py` → `nanochat/flash_attention.py`：看 `(B,T)` 如何变成 loss / logits；
5. `nanochat/optim.py`：看梯度何时通信、参数如何更新；
6. `scripts/chat_sft.py` → `tasks/`：看对话协议与 mask；
7. `nanochat/engine.py` → `scripts/chat_eval.py`：看真实生成与 verifier；
8. `scripts/chat_rl.py`：最后再看 rollout 如何回到梯度；
9. `tests/` 与 `dev/LOG.md`：分别回答“必须保持什么”和“试过什么但没有保留”。

## 3. 关键测试文件速查

| 测试文件 | 保证的契约 | 修改哪些代码后至少必跑 |
|---|---|---|
| `tests/test_tokenizer.py` | round-trip、特殊 token、对话 mask | tokenizer、SFT 协议 |
| `tests/test_attention_fallback.py` | FA3/SDPA、GQA、滑窗、KV cache | attention、kernel fallback |
| `tests/test_engine.py` | cache 复制、采样、seed、多候选 | inference engine |
| `tests/test_optim.py` | AdamW 对齐、Muon 更新、收敛 | optimizer |
| `tests/test_execution.py` | timeout、内存、环境隔离 | HumanEval 执行 |
| `tests/test_tasks.py` | task slicing、mixture、选项格式 | SFT/eval task |

## 4. 两层验证

写代码或读完一章后，可以用三条命令验证不同层面的不变量：

```bash
# 1) 教程对当前源码的关键事实：无第三方依赖
python3 tutorial/source_checks.py

# 2) 上游真实测试：需先进入 nanochat 并安装 dev 依赖
cd nanochat
uv sync --extra cpu --group dev   # 或 --extra gpu
# `nanochat/` 未被安装为包，必须用 `python -m pytest` 将当前目录加入 sys.path
uv run python -m pytest
# macOS 上跳过内存沙箱测试（RLIMIT_AS 在 Darwin 不生效）：
# uv run python -m pytest --deselect tests/test_execution.py::test_memory_limit
```

平台说明：

- CUDA / Flash-Attention-3 / Hopper 特性的测试在 CPU 或 Mac 环境下会被 `skip`，属预期行为；
- `test_memory_limit` 依赖 `resource.RLIMIT_AS`，Darwin 内核将其忽略，因此该测试在 macOS 上必失败。

两层不能互相替代：

- Source Checks 发现教程文本与当前源码之间的事实漂移；
- `pytest` 才覆盖 tokenizer、Engine/KV cache、FA3/SDPA、optimizer、task 与执行隔离等真实实现。

模型质量还需 BPB、CORE、ChatCORE 和固定协议下的实验，这些不在上述两层内。相关记录模板参见[附录 C：性能、排行榜与研究方法](./performance-research.md)。

## 5. 复查记录卡

教程或上游升级时，建议至少记录：

```text
nanochat revision:
被修改章节:
源码与测试证据（文件 / 行 / 测试名）:
source_checks / pytest 结果:
仍未运行的 GPU、数据或能力评估:
```

这样可以把“文档看起来完整”升级为“结论有来源、版本和失败条件”。

---

- 返回[教程目录](./README.md)
- 下一项实践：[附录 C：性能、排行榜与研究方法](./performance-research.md)
