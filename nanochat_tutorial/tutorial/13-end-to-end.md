# 第 13 章：端到端：speedrun 全流程与动手实践

> **本章位置**：[`12｜RL`](./12-rl.md) → **13｜端到端：从脚本到可验证的小实验** → 返回[教程目录](./README.md)。
>
> **本章源码地图**：完整 GPU 路径在 `nanochat/runs/speedrun.sh`，CPU/MPS 演示路径在 `nanochat/runs/runcpu.sh`；训练入口依次是 `nanochat/scripts/tok_train.py`、`nanochat/scripts/base_train.py`、`nanochat/scripts/base_eval.py`、`nanochat/scripts/chat_sft.py`、`nanochat/scripts/chat_eval.py`，交互入口是 `nanochat/scripts/chat_cli.py`。产物位置由 `nanochat/nanochat/common.py::get_base_dir` 与 `nanochat/nanochat/checkpoint_manager.py` 决定。

前十二章把每个部件拆开了；本章再把它们接回真实的 shell 调用链。目标不是承诺某台机器能在固定时间、固定预算下复现某项能力，而是建立一条**可检查、可中断、可缩小**的工作流：每一步先知道读什么、写什么、如何验证，再进入下一步。

先校正一个常见误解：当前 `runs/speedrun.sh` 的端到端路径是 **tokenizer → base 预训练 → base 评估 → SFT → chat 评估**，它**没有**调用 `scripts/chat_rl.py`。RL 是仓库中的独立可选阶段；要运行它，需要先有 SFT checkpoint，并单独执行第 12 章对应的命令。

## 学习目标

完成本章后，你应能：

1. 从当前 `speedrun.sh` 和 `runcpu.sh` 还原端到端阶段、命令、输入与 checkpoint 交接；
2. 区分“完整 GPU 参考脚本”“CPU/MPS 流程演示”和“最小可检查命令”，不把它们的成本或结果混为一谈；
3. 用 `NANOCHAT_BASE_DIR`、model tag、step 和 checkpoint 元数据定位实际产物；
4. 在不修改 `nanochat/` 的前提下，安全地缩小 base/SFT 实验，并为每步设置可验证信号；
5. 说明如何把可选 RL 接在 SFT 后面，以及为何不能把 reward 或 pass@k 当作通用能力证明；
6. 对照当前源码发现脚本注释、默认值和实际 CLI 参数的差异。

## 前置概念

请先完成第 01–12 章，尤其是：第 06 章的 `x/y` 右移训练、第 07 章的 batch 与 dtype、第 09 章的 SFT mask、第 10 章的 checkpoint/推理、第 11 章的 chat eval 协议，以及第 12 章的 rollout 与 outcome reward。

本章用下表跟踪每个阶段的交接对象：

| 阶段 | 主要输入 | 主要输出 | 最小检查 |
|---|---|---|---|
| tokenizer | 下载的文本 shard | `tokenizer.pkl`、`token_bytes.pt` | `tok_eval` 能读取并报告 tokenizer 指标 |
| base | tokenizer、预训练 shard | `base_checkpoints/<tag>/` | BPB/CORE/样本命令能加载 checkpoint |
| SFT | base checkpoint、对话 task | `chatsft_checkpoints/<tag>/` | ChatCORE/`chat_eval` 或 CLI 能加载 SFT checkpoint |
| 可选 RL | SFT checkpoint、GSM8K rollout | `chatrl_checkpoints/<tag>/` | 独立 pass@k 日志与 RL checkpoint |
| chat | SFT 或 RL checkpoint | 流式 token 输出 | 明确 source/tag/step 后可复现加载路径 |

这些检查只证明接口和流程在当前配置下工作；它们不证明模型“聪明”、安全、无偏或可泛化。

---

## 1. 主线：脚本真正串了什么？

`speedrun.sh` 的执行顺序可以压缩为：

```text
uv 环境
  → 下载 8 个 shard，训练/评估 tokenizer
  → 后台下载至 170 个 shard，等待下载完成
  → 8 rank base_train（d24、FP8）
  → base_eval
  → 8 rank chat_sft
  → chat_eval -i sft
```

对应的关键命令是脚本中的原样调用：

```bash
python -m nanochat.dataset -n 8
python -m scripts.tok_train
python -m scripts.tok_eval

python -m nanochat.dataset -n 170 &
wait $DATASET_DOWNLOAD_PID

torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=$WANDB_RUN

torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
```

注意两层边界：

- `speedrun.sh` 面向 8×H100 的 GPU 环境并安装 `uv sync --extra gpu`；不要在 CPU/MPS 环境盲目复制它。
- `--target-param-data-ratio=8` 是该脚本**显式传入**的值。`base_train.py` 当前 argparse 默认值是 12；脚本附近关于旧默认值的注释不是运行时真相。命令行实参优先。

脚本的后台下载不是并行训练：它只与 tokenizer 训练重叠，`wait $DATASET_DOWNLOAD_PID` 保证 base 训练开始前下载任务结束。若在这里删除 `wait`，后续失败或数据量不足时就不再是同一条可审计路径。

## 2. 源码解读：环境、目录和 checkpoint 如何交接

### 2.1 先固定产物目录

两个运行脚本都设置：

```bash
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"
```

`common.py::get_base_dir()` 的规则是：若 `NANOCHAT_BASE_DIR` 已设置就使用它，否则使用 `~/.cache/nanochat`。这意味着实验不应只靠“最近一次运行”的隐式目录。给每次实验独立目录更容易回溯，例如：

```bash
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat-tutorial-check"
```

这不会改动仓库代码；它只把下载、tokenizer 和 checkpoint 产物隔离。代价是新目录需要重新准备相应数据和 tokenizer，不能把它误当成自动复制已有产物。

`checkpoint_manager.py::load_model` 将 source 映射为：

```text
base → base_checkpoints
sft  → chatsft_checkpoints
rl   → chatrl_checkpoints
```

若未给 `model_tag`，它优先选名称形如 `d<number>` 中 depth 最大的目录；若未给 `step`，选该目录模型文件中 step 最大者。这方便交互，但会使多次实验的加载不可见。比较实验和排错时，始终显式传 tag/step。

### 2.2 tokenizer 是 base 的前置契约

`tok_train` 生成的 tokenizer 不只是“可有可无的文本工具”。base checkpoint metadata 保存模型的 `vocab_size`；`checkpoint_manager.py::build_model` 加载时断言当前 tokenizer 的词表大小必须匹配模型配置。更重要的是，即使词表大小碰巧相同，token id 的语义也必须对应同一 tokenizer 文件。

因此一条可靠的顺序是：先完成 tokenizer，再训练/加载 base；之后 SFT、RL 和 CLI 都继续使用同一套 tokenizer。不要在已有 checkpoint 上无记录地重训 tokenizer，再期待 embedding 行仍表示原来的 token。

### 2.3 base、SFT 与 RL 的接力

`base_train.py` 保存模型参数、JSON metadata 和每个 rank 的 optimizer shard；`chat_sft.py` 通过 `load_model("base", ...)` 读取 base，并可尝试 warm-start base optimizer shard；`chat_rl.py` 通过 `load_model("sft", ...)` 读取 SFT。当前 RL 保存模型和 metadata，但传给 `save_checkpoint` 的 optimizer state 为 `None`。

这解释了为什么“某文件存在”不足以说明能续训：要检查正确的 source 目录、model/meta 是否成对、需要 optimizer 时本 rank shard 是否存在，以及模型配置与 tokenizer 是否兼容。

---

## 3. 形状例：一条样本在端到端中如何改名不改本质？

不同脚本的输入看起来不同，最终都回到自回归模型的 `(B,T)` token id 与目标。设训练上下文 `T=8`，一个 token row 长 `T+1=9`：

```text
预训练 row: [BOS, a, b, c, d, e, f, g, h]
x:           [BOS, a, b, c, d, e, f, g]    # (B, 8)
y:           [a,   b, c, d, e, f, g, h]    # (B, 8)
```

base loader 的每个有效 target 都参与交叉熵。SFT 的 row 同样右移，但对用户、协议、工具输出和 padding 对齐的 target 写 `-1`：

```text
SFT targets: [-1, -1, -1, answer_1, answer_2, ..., assistant_end]
```

RL rollout 也构造 `inputs=ids[:, :-1]`、`targets=ids[:, 1:]`，再用生成 mask 的 `mask[:, 1:]` 把 prompt、强制工具输出与 padding 设为 `-1`；对剩余位置的 NLL 取负得到 `logp`，乘同题 advantage。故端到端的共同骨架不是“每一步都一样”，而是：

```text
离散 token → GPT.forward → logits / token loss
                         ├─ base/SFT：最小化交叉熵
                         └─ RL：最大化 masked logp × advantage
```

推理阶段没有 targets：`Engine.generate` 先 `(1,P)` prefill，再以 `(B,1)` decode；它读取 SFT 或 RL checkpoint，不会把聊天内容写回训练权重。

---

## 4. 实验：从低风险检查到小规模训练

> **本节所有命令的共同前提**
>
> - 从仓库中的 `nanochat/` 目录执行，并已按设备完成 `uv sync --extra cpu` 或 `uv sync --extra gpu`、激活虚拟环境；
> - 明确当前 `NANOCHAT_BASE_DIR`。除纯 `--help` 外，后续阶段还需要该目录中与阶段匹配的 tokenizer、数据 shard 或 checkpoint；
> - **验证目标只是代码路径和接口**，不是模型能力；
> - 失败先按 `cwd / venv → NANOCHAT_BASE_DIR → tokenizer/data → source/tag/step → batch 整除 → dtype/FA3/OOM` 排查。

### 4.1 只读检查：先确认环境与脚本事实

在 `nanochat/` 目录执行以下命令只会显示帮助或文本，不会训练：

```bash
python -m scripts.base_train --help
python -m scripts.chat_sft --help
python -m scripts.chat_rl --help
python -m scripts.chat_cli --help
```

然后核对三件事：当前 `--window-pattern` 默认是 `SSSL`（`S` 为约四分之一上下文再向上对齐到 128；不要采信 `gpt.py` 中与实现矛盾的 `2048 -> 768` 注释）；`base_train.py` 将基础训练的 `n_kv_head` 设为 `num_heads`；实际 `COMPUTE_DTYPE` 由硬件或 `NANOCHAT_DTYPE` 决定，而不是教程中写死的 bf16。启动 base/SFT 脚本时会打印选中的 dtype 及其原因。

在没有兼容 FA3 的环境，`base_train.py` 可能提示 SDPA fallback；对 CPU/MPS 小实验，`runcpu.sh` 明确使用 `--window-pattern=L`，避免交替滑窗的低效路径。不要因此反推全局默认也是 `L`。

### 4.2 CPU/MPS 脚本：流程演示，而不是“几分钟模型”

当前 `runcpu.sh` 的实际配置包括：

```bash
python -m scripts.base_train \
  --depth=6 --head-dim=64 --window-pattern=L --max-seq-len=512 \
  --device-batch-size=32 --total-batch-size=16384 \
  --num-iterations=5000 --run=$WANDB_RUN
```

它在前面仍下载 8 个 shard，并将 tokenizer `--max-chars` 设为 `2000000000`；这不是一个无需数据、几分钟必完成的 toy command。其注释也只将它定位为 CPU/MPS 上覆盖代码路径的演示。真实耗时取决于机器、磁盘、网络、PyTorch 后端和已有缓存，不能从脚本或教程推断固定数值。

要跑原脚本，使用：

```bash
cd nanochat
bash runs/runcpu.sh
```

但开始前先读脚本，确认数据下载和 5000 次更新符合你的预算。脚本随后会跑 SFT；它没有运行 RL。

### 4.3 最小 base 命令：缩短训练，不伪造完整数据

若 tokenizer 和所需训练数据已经准备好，可用 `base_train.py` 顶部示例所依据的缩小原则：减小 depth、序列长度、每设备 batch、验证 token 和 iteration；关闭昂贵 CORE。一个与当前 CLI 匹配的例子是：

```bash
cd nanochat
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat-tutorial-check"
python -m scripts.base_train \
  --device-type=cpu \
  --depth=4 --head-dim=64 --window-pattern=L --max-seq-len=128 \
  --device-batch-size=1 --total-batch-size=128 --num-iterations=2 \
  --eval-every=-1 --core-metric-every=-1 --sample-every=-1 \
  --run=dummy --model-tag=tutorial-d4
```

这条命令只适用于该目录中已具备 tokenizer 和数据的情况；否则它会在数据/tokenizer 准备步骤失败或下载，而不是神奇地训练。`total_batch_size=128` 满足单 rank 下 `1×128` micro-batch 的整除要求，因此 `grad_accum_steps=1`。它的价值是检查模型初始化、`x/y`、forward/backward、checkpoint 保存等接口，不是评估语言能力。

若想开启一次 BPB 检查，必须选择与 batch/序列长度相容的 `--eval-tokens`，并确保验证数据已准备好；不要为了得到一个数字而把未完成或不匹配的配置当作基准结果。

### 4.4 SFT、chat 与可选 RL：按 checkpoint 逐步接上

在同一 `NANOCHAT_BASE_DIR` 下、base tag 已存在时，SFT 可显式指定 base tag：

```bash
cd nanochat
python -m scripts.chat_sft \
  --device-type=cpu --model-tag=tutorial-d4 --num-iterations=1 \
  --device-batch-size=1 --total-batch-size=128 --max-seq-len=128 \
  --eval-every=-1 --chatcore-every=-1 --run=dummy
```

这会让 `chat_sft.py` 从 base checkpoint 继承未覆盖的配置；显式传入的值优先。它仍依赖 SFT task 数据可用，且默认 `--load-optimizer=1` 会尝试读取 base optimizer shard；小实验若不需要 warm-start，可传 `--load-optimizer=0`，但这改变的是训练状态，不是“修复”数据或 checkpoint 问题。

使用明确 source/tag/step 进行单次对话的命令形式为：

```bash
python -m scripts.chat_cli -i sft -g tutorial-d4 -s <SFT_STEP> \
  --device-type=cpu -p "What is 2 + 2?"
```

把 `<SFT_STEP>` 替换为实际保存的 `model_XXXXXX.pt` 中的数字。一次极小训练后的输出无论好坏都只是流程证据；不要把它摘录为模型质量结论。

若 SFT checkpoint 已准备好，RL 是**单独的**命令：

```bash
python -m scripts.chat_rl --device-type=cpu \
  --model-tag=<SFT_TAG> --model-step=<SFT_STEP> \
  --num-epochs=1 --examples-per-step=16 --num-samples=16 --device-batch-size=8 \
  --run=dummy
```

这是当前参数约束下的示例：`num_samples` 必须能被 `device_batch_size` 整除，`examples_per_step` 也须能被 world size 整除。它仍可能昂贵，因为每个训练题要生成多条 rollout；训练脚本没有 fp16 GradScaler 路径。应先在第 12 章理解 reward、mask 和 on-policy 单次更新，再决定是否运行。结果只针对 GSM8K 的 `####` 数字 outcome reward，不能直接声称“推理能力提升”。

---

## 5. 常见误区与排错顺序

1. **“speedrun 是 tokenizer、base、SFT、RL 的完整默认链。”** 不对。当前脚本停在 chat SFT/eval；RL 需单独运行。
2. **“`runcpu.sh` 是一个极小、固定几分钟的实验。”** 不对。它仍有大量 tokenizer 字符、数据下载和 5000 次 base 更新；只承诺演示路径，不承诺时间或质量。
3. **“base 脚本的默认值就是 speedrun 的实际值。”** 不对。脚本传入的 CLI 参数覆盖 argparse 默认；例如 speedrun 明确传 `--target-param-data-ratio=8`。
4. **“换一个 tokenizer 后，只要 vocab size 相同就能加载。”** 不对。embedding 行和 token id 的语义必须保持一致；词表大小断言只是最低限度检查。
5. **“不写 tag/step 更省事。”** 交互时可以，比较实验时不可以。自动选择最大 depth/最后 step 会隐藏真实模型来源。
6. **“一个 checkpoint 文件即可恢复所有状态。”** 不完整。训练续接还可能需要 metadata、正确 tokenizer、按 rank 的 optimizer shard 和相同配置。
7. **“极小训练生成一句通顺的话，就验证了模型方案。”** 不对。它最多验证代码路径和接口；质量结论要固定数据、训练 token、评估协议和随机性。
8. **“pass@k 上升证明单次对话更好。”** 不对。它是多次尝试至少一次成功的 verifier 指标，且当前 reward 可被格式/数字投机。

推荐排错顺序是：先确认当前工作目录与 Python 环境；再确认 `NANOCHAT_BASE_DIR`；检查 tokenizer、模型和 metadata 是否齐全且 tag/step 正确；只后才看 dtype、FA3/SDPA、batch 整除或 OOM；最后才讨论 loss、BPB、ChatCORE 或 pass@k。这样不会把“加载错 checkpoint”误诊为优化器或模型问题。

## 小结

端到端并不等于一条黑箱 shell 命令。当前 nanochat 的参考 GPU 脚本完成 tokenizer、base、评估和 SFT/chat 评估，但不执行 RL；CPU/MPS 脚本是相同主线的流程演示，成本和结果必须由实际机器与配置验证。

最可靠的实践是固定产物目录，记录 tokenizer、source/tag/step、CLI 参数和 dtype；用 `x/y`、mask、checkpoint 和显式加载命令检查每一个交接点；只在这些契约成立后再讨论训练质量。小实验用于验证路径，不用于虚构能力或基准结果。

## 练习

1. 打开 `runs/speedrun.sh`，列出从 `nanochat.dataset -n 8` 到 `chat_eval` 的每个命令，并标出哪一步写 tokenizer、base checkpoint、SFT checkpoint。指出 RL 缺在哪一行。
2. 对 `runcpu.sh` 的 base 命令，计算单 rank 的 micro-batch token 数 `32×512`，并解释其 `total_batch_size=16384` 对应多少个梯度累积 micro-step。
3. 在不训练的前提下，设置一个新的 `NANOCHAT_BASE_DIR`，运行四个 `--help` 命令；记录当前 `base_train` 的 `--window-pattern`、`--target-param-data-ratio` 和 `--depth` 默认值。再与 `speedrun.sh` 的显式参数比较。
4. 为一次本地实验写一张最小记录卡：git revision、设备、`COMPUTE_DTYPE` 原因、base/SFT/RL 的 tag/step、tokenizer 来源、完整 CLI、数据状态、预先声明的成功/失败检查。为什么这比只保存 loss 截图更有用？
5. 读 `chat_rl.py::get_batch`，说明为何 `num_samples=10, device_batch_size=8` 不满足当前实现的 rollout 切分假设；将它改为哪个数值组合可以满足整除，同时会如何影响 rollout 成本？

## 前后章导航

- ← [第 12 章：RL：用 GRPO 提升推理能力](./12-rl.md)：RL 需要先加载 SFT checkpoint；本章把它放回完整目录、脚本和实验约束中。
- **本章**：从 `speedrun.sh`/`runcpu.sh` 出发，按 tokenizer → base → SFT → 可选 RL 的 checkpoint 契约组织一条可验证路径。
- → 返回[教程目录](./README.md)：可按主题回读各章，并以当前 `nanochat/` 源码为最终依据。
