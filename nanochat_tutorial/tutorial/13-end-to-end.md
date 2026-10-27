# 第 13 章　端到端：从脚本到可验证实验

> **本章定位**　第五部分·实践。把模型、训练与评估接成一条能追踪输入、输出和证据的实验链。
>
> **中心问题**　没有 8×H100 时，应该验证什么、按什么顺序验证？怎样避免把“命令结束了”误当成“模型有效”？
>
> **范围**　参考脚本、资源档位、tokenizer / 数据 / checkpoint 依赖、静态检查、有条件的小规模 base → SFT → chat 实验与分层排错。低成本链不含完整 benchmark 或 RL；本章列出的安装、下载、训练和生成命令不是文档检查步骤。
>
> **先决条件**　读过[第 09 章](./09-sft.md)的监督与 packing、[第 10 章](./10-inference-engine.md)的生成，以及[第 11](./11-chat-evaluation.md)–[12 章](./12-rl.md)的评估与 RL 边界。
>
> **本章产物**　一张依赖接力表、一组按成本分级的命令模板，以及能区分流程证据和能力证据的实验记录卡。
>
> **源码入口**　`runs/speedrun.sh`、`runs/runcpu.sh`；`scripts/tok_train.py`、`tok_eval.py`、`base_train.py`、`base_eval.py`、`chat_sft.py`、`chat_eval.py`、`chat_cli.py`；`nanochat/common.py`、`checkpoint_manager.py`。文档检查入口为 `tutorial/source_checks.py`。

---

## 本章讲什么

端到端不等于“一次运行所有脚本”，而是**让下一阶段确实读到上一阶段预期的产物，并为每一步预先声明成功标准**。

完整的数据流先看这一条：

```text
预训练文本 → tokenizer 与 token_bytes
           → base 训练 → base checkpoint
示范对话 + 同一 tokenizer + base checkpoint
           → SFT → SFT checkpoint
用户问题 + 聊天历史 + 同一 tokenizer + SFT checkpoint
           → Engine → 回答
                       └→ 独立任务判分（另一个成本与风险档位）
```

本章按执行决策组织，而不是按脚本名堆命令：

1. 选目标：只查文档、查 CPU 数值路径，还是实际训练；
2. 固定目录、tokenizer、数据和 checkpoint 身份；
3. 先做不导入训练入口的静态检查；
4. 条件满足后，才做有界 base、SFT 和生成；
5. 区分产物存在、数值健康、任务成功，按层排错。

> **读法建议**：§1–§3 可在没有训练数据时完成；§4 是参考脚本阅读；§5 是需要另行准备资源的实验模板，不属于零成本检查；§6–§7 解释证据与排错。

---

## 预备词汇

| 名称 | 含义 |
|---|---|
| `source` | checkpoint 阶段：`base`、`sft`、`rl`，分别映射到三个目录 |
| `model tag` | 阶段目录下的模型子目录名；默认为 `d<depth>`，显式 tag 不一定代表真实深度 |
| `step` | checkpoint 文件的循环编号；文件形如 `model_000002.pt`，各脚本与已完成更新次数的关系不同 |
| `B`、`T`、`W` | 每 rank 的序列行数、每行输入 token 数、rank 数量 |
| micro-batch | 每 rank 一次 forward/backward 处理的 `(B,T)` 输入 |
| 梯度累积次数 | 一个 optimizer step 包含多少个 micro-batch |
| `NANOCHAT_BASE_DIR` | 模型、tokenizer、数据等产物的共同根目录，不是 Python 源码目录 |
| smoke test | 有界代码路径检查；通过不代表模型能力、稳定性或安全性通过 |

`torchrun` 启动多个进程并提供 rank 环境变量，**不会自动把 GPT 包成 `DDP(model)`**。当前训练的梯度通信由自定义优化器承担，见第 07、12 章。CPU/MPS 模板用单进程 Python，且不要继承旧的 `RANK/LOCAL_RANK/WORLD_SIZE`：`compute_init` 只为 CUDA 初始化进程组，不能把这些模板改成 CPU/MPS torchrun 就期待正常通信。

---

## 1. 先选验证目标与资源档位

| 档位 | 做什么 | 能支持什么结论 | 主要成本 / 风险 |
|---|---|---|---|
| A：静态 | 读源码、解析 Python、检查链接与 diff | 指定源码契约和文档结构仍成立 | 不运行模型；不能证明数值语义 |
| B：小型 CPU | 已审阅的内存测试、张量 / 索引自检 | 指定小输入上的实现行为正确 | 需要已有依赖；不能覆盖 CUDA、MPS 或长上下文 |
| C：小规模实际链 | 少量 base 更新、SFT 更新、加载与有限生成 | 当前配置下流程可通、观测数值有限 | 数据准备、编译、词表投影、保存仍有成本 |
| D：参考训练与评估 | GPU 配方、完整或明确截样的 benchmark | 按固定协议报告能力 | 训练资源、下载；HumanEval 还涉及程序执行 |

**先写停止条件，再选命令。** C 档至少要求 loss 有限、有有效监督目标、模型与 metadata 能加载。NaN、空监督 batch 或缺文件时停止；这些是实验验收要求，不是当前脚本都已实现的自动拦截。异常 checkpoint 不继续传给 SFT。

`speedrun.sh` 面向空白 8×H100 节点，脚本注释给出约 1.5 小时的参考；这不是本章实测或跨机器保证。`runcpu.sh` 是 CPU/MPS 流程演示，不保证聊天质量，也不是几分钟的最小检查。无需证明能力时，不要先跑 D 档。

---

## 2. 把依赖和产物身份固定下来

### 2.1 cwd 与产物目录是两件事

后面的 `python -m scripts...`、`python -m nanochat.dataset...` 都从**源码仓库 `nanochat/`**执行；文档检查从其父目录 `nanochat_tutorial/` 执行。以下只设置 shell 变量，不安装或训练：

```bash
# 起点：nanochat_tutorial/；后续可用这两个绝对路径切换。
TUTORIAL_ROOT="$PWD"
NANOCHAT_REPO="$TUTORIAL_ROOT/nanochat"
cd "$NANOCHAT_REPO"
```

当前 `get_base_dir()` 的真实规则是：

- 环境变量**非空**时原样采用；未设置或空字符串时回退到 `~/.cache/nanochat`；
- 默认路径会展开 home；显式值不会再执行 `expanduser` / `expandvars`；
- 相对路径相对于当时 cwd；函数还会 `makedirs(..., exist_ok=True)`，不是纯查询。

所以 `export NANOCHAT_BASE_DIR='~/experiment'` 不会自动指向 home，空值也不会指向当前目录。优先用 shell 已展开的绝对路径。

实际实验应新建独立目录，而非固定复用 `nanochat-tutorial-check`：

```bash
# 仅在准备实际实验时执行；本次实验拥有此目录，旧产物不会被覆盖。
EXPERIMENT_DIR=$(mktemp -d "${TMPDIR:-/tmp}/nanochat-exp.XXXXXX") || exit 1
# 创建失败就停止；避免空值让 get_base_dir() 回退到用户的默认缓存。
export NANOCHAT_BASE_DIR="$EXPERIMENT_DIR"
export NANOCHAT_DTYPE=float32
export OMP_NUM_THREADS=1
MODEL_TAG=tutorial-smoke
```

新目录**不自动复制**旧数据或 tokenizer。需要经授权复制兼容的已准备产物，或另行下载与训练。不能在这个根目录重新训练 tokenizer 后，继续使用旧 token 映射的 checkpoint。两份词表大小相同，也可能 id 语义完全不同。

### 2.2 接力表：谁读谁、谁写什么

| 阶段 | 主要输入 | 主要输出 / 副作用 |
|---|---|---|
| `tok_train` | 本地预训练 train shard | `tokenizer/tokenizer.pkl`、`token_bytes.pt`；会覆盖同目录文件 |
| `tok_eval` | 当前 tokenizer、train/val shard、GPT-2/GPT-4 参照 tokenizer | 压缩率与 round-trip 输出；参照资源未缓存时可能下载 |
| `base_train` | 当前 tokenizer、token_bytes、train shard | `base_checkpoints/<tag>/`：模型、metadata、各 rank optimizer shard |
| `base_eval` | base checkpoint、同一 tokenizer；按模式读取数据 / eval bundle | stdout；**仅 CORE 分支**写 `base_eval/base_model_<step>.csv` |
| `chat_sft` | base 模型、当前 tokenizer/token_bytes、SmolTalk/MMLU/GSM8K | `chatsft_checkpoints/<tag>/`：模型、metadata、optimizer shard |
| `chat_eval` | 指定模型、任务数据与判分器 | stdout 中的任务结果；不保存新模型 |
| `chat_cli` | 指定模型与当前 tokenizer | 流式回答；不保存新模型 |
| 可选 RL | SFT 模型、GSM8K 与 rollout | `chatrl_checkpoints/<tag>/`；当前不保存 optimizer state |

任务缓存位于 `task_data/<repo slug>/<subset>/<split>/`，slug 将仓库名中的 `/` 换为 `--`。SFT 即使训练很短，仍会构造训练和验证 Task；有 manifest 时读取其中列出的 parquet，缺 manifest 时可能下载。**减少 iteration 不等于缩小数据准备成本**。

预训练读取器也不按文件名查找固定 val：它将排序后的**最后一个 parquet**当 val，其余当 train。下载器虽请求 `shard_06542.parquet`，读取器却不验证它存在；缺它时，最后一个现有 train shard 会被当作 val。因此接力前要查真实文件清单，而非只看下载命令。

### 2.3 source、tag、step：不要交给加载器猜

`load_model` 先按 source 选阶段目录，然后：

1. 没给 tag：优先匹配名字以 `d` 加数字开头的目录，取解析深度最大者；没有这种名字时才取目录修改时间最新者。这不是真正统计参数量，也不是“最新 tag”；
2. 没给 step：在该 tag 中取 `model_<step>.pt` 的最大数值，不检查该步质量，也不预先保证 metadata 齐全；
3. 读取 model 与同 step 的 meta，构建模型，再从**当前根目录的 `tokenizer/`**读取 tokenizer。

checkpoint **没有自动打包自己的 tokenizer**。加载器只检查 vocab size，不能证明 token 映射一致。模型与 optimizer 的加载也分别选择文件，实验期间不要并发向同一目录写入。

不同入口的 step 参数不同：

| 入口 | 模型身份参数 |
|---|---|
| `base_train` | 输出 `--model-tag`；恢复用 `--resume-from-step`，还需兼容配置与 optimizer shard |
| `base_eval` | `--model-tag`、`--step`，source 固定为 base |
| `chat_sft` | `--model-tag`、`--model-step`，输入固定为 base；同 tag 写入 SFT 阶段 |
| `chat_eval` / `chat_cli` | `-i source -g tag -s step` |

仅推理需模型、metadata、兼容 tokenizer；恢复优化还需要对应 rank 的 optimizer state 和兼容配置。base 的 dataloader 恢复是近似恢复，不应承诺逐 token 无缝复现。

文件编号也要按阶段解释（正常有限值路径，未跳过更新时）：

| 阶段 | 保存位置 | 编号与更新次数 |
|---|---|---|
| base | 外层循环检查时；终点编号为 num_iterations | 新训练 step=2 时已完成两次更新 |
| SFT | 检查到 last_step 后，最终 BPB 之后 | step 是已完成更新循环次数，不是 generator 产出次数 |
| RL | 本步 optimizer.step() 之后 | 零基 step=60 时已调用 61 次更新；不是 SFT 的计数口径 |

模型和 metadata 是顺序写入，不是原子提交的一对文件。中断可留下只有 model 的最新 step；显式指定经核对的完整文件对，不要让“取最大 step”代替验收。

---

## 3. 先做静态检查，再选择安全的小型测试

### 3.1 不导入训练入口的检查

```bash
cd "$TUTORIAL_ROOT"
python3 tutorial/source_checks.py
git diff --check
```

`source_checks.py` 使用标准库，当前包含八组源码 / 文档检查。源码部分主要检查**字面量是否存在**，链接部分检查目标和锚点；它不运行模型，也不证明注释中的 on-policy 等结论正确。PASS 之后仍需独立审阅行为与数值语义。

还可以解析源码语法，不执行其顶层代码：

```python
import ast
from pathlib import Path

# 从 nanochat_tutorial/ 执行；解析不是 import，不构造模型或 Task。
repo = Path("nanochat")
entries = ["tok_train", "tok_eval", "base_train", "base_eval",
           "chat_sft", "chat_eval", "chat_cli"]
for name in entries:
    path = repo / "scripts" / f"{name}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assert isinstance(tree, ast.Module)
print("7 个脚本语法可解析；未执行脚本")
```

失败时按证据核对源码和教程；这次章节编辑**不改 `source_checks.py`、源码或配置**。若问题必须靠改这些文件解决，应停下询问，而不是删断言或回退工作树。

### 3.2 `--help` 和 pytest 为什么不是默认零风险

argparse 通常在开始训练前处理 `--help`，但入口**先执行 import**：训练脚本会设置环境变量，`base_train` 还打印 banner；attention 模块在兼容 CUDA 上可能尝试获取 kernel。因此 `--device-type=cpu --help` 也不能阻止所有导入期 CUDA 探测。`tok_eval` 没有 help 解析，不能用 `--help` 阻止它实际评估。

同样，`uv sync` 是依赖安装，可能联网并修改环境；`uv run` 也不应未经核对就当作“只用现有 venv”。本章检查优先使用已有 `.venv/bin/python`，不自动修复依赖。

已有小型测试可选择 `tests/test_tasks.py`：测试函数使用内存 Arrow table 和本地 Task，不调用 Hub 下载；其导入链仍加载 `nanochat.common`，涉及 torch、dtype 探测和日志初始化。`test_tokenizer.py` 在内存训练极小 BPE；`test_engine.py` 使用 mock 模型。选用前要审阅当前测试及导入链，不能只凭文件名判定安全。

```bash
# 已有依赖、已审阅这些测试且确认当前主机无 CUDA 导入下载分支时，才执行。
cd "$NANOCHAT_REPO"
NANOCHAT_DTYPE=float32 OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_tasks.py
```

不要默认运行整个 pytest：`test_execution.py` 含无限循环、1 GiB 内存申请与对抗性执行输入。Darwin 的 `RLIMIT_AS` 与 Linux 不同，删掉某个 Mac 内存用例也不能把剩余执行测试变成安全沙箱，见[第 11 章 §7](./11-chat-evaluation.md#7-humaneval-执行防护降低事故风险不是安全沙箱)。

测试结论必须对应实际路径：Task 测试不验证 SFT packing / 停止循环，渲染测试不验证真实 GPT，mock Engine 不替代 cached/uncached logits 对照；CPU 通过也不覆盖 MPS、FA3 或 CUDA optimizer。

---

## 4. 阅读参考脚本：真实顺序与失败边界

### 4.1 speedrun：下载和 tokenizer 重叠，base 在 wait 后

以下是**阅读用摘录，不是本章执行清单**；cwd 为 `nanochat/`。这里仅纠正 torchrun 命令中多余的 `--`，其余保留阶段顺序；并未改源码脚本。实际脚本还包含 uv 安装检测、venv 创建/激活，并把根目录显式设为 `$HOME/.cache/nanochat`。

```bash
uv sync --extra gpu
source .venv/bin/activate
# 实际脚本在 WANDB_RUN 为空时设 dummy，关闭 wandb 上传。
WANDB_RUN=${WANDB_RUN:-dummy}

python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval
wait "$DATASET_DOWNLOAD_PID"

torchrun --standalone --nproc_per_node=8 -m scripts.base_train \
    --depth=24 --target-param-data-ratio=8 --device-batch-size=16 \
    --fp8 --run="$WANDB_RUN"
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval \
    --device-batch-size=16
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft \
    --run="$WANDB_RUN"
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -i sft
```

关键不是“后台训练”，而是：

```text
首次下载结束 → 启动后台扩充下载 ──────────→ wait 返回 → base
            └→ 前台 tokenizer 训练 → tok_eval ┘
```

- `-n 8` 是请求 8 个 **train** shard，另加固定验证 shard `shard_06542.parquet`；`-n 170` 请求前 170 个 train shard加同一验证 shard，已存在文件跳过，不是额外下载 170 个。
- `wait` 等待后台进程结束，不验证所有文件齐全。当前 downloader 将单文件重试失败记为 `False`、打印成功数量，但没有据此返回非零退出码。
- 原脚本四条 torchrun 命令在模块名后都有额外 `--`。这些入口用普通 `parse_args()`，不会主动移除该标记；后续选项因而成为未识别参数。读者不能把原脚本当作已验证可直接跑通的命令。本摘录去掉标记，并不验证后续训练成功。
- 脚本没有 `set -e` 或逐阶段显式失败拦截；即使某命令或 wait 返回非零，后续命令仍可能继续。**不能把这个脚本描述成 fail-fast 流水线。** 手动执行时应逐阶段核对退出状态、成功数量与实际文件，再继续。
- base 的 ratio 实际是显式传入的 8；argparse 默认是 12，邻近注释的“10.5 (default)”不是实际运行值。这是速度配方，不是理论最优结论。
- base_eval 默认跑 CORE、BPB、sample；SFT 与最终 chat_eval 都涉及五项聊天任务，包括 HumanEval。默认链**不运行 RL**，但绝非低风险 smoke test。

不要以为在外部导出独立根目录再调用原脚本就能隔离实验：两个参考脚本都会覆盖该环境变量。若不改源码，应经资源授权后手动逐条执行经过核对的命令，并保留自己的目录、tag 与日志。

### 4.2 runcpu：CPU 依赖不等于强制 CPU 设备

当前脚本依次执行：下载 8 个 train shard加验证 shard、tokenizer 训练/评估、base、base_eval、SFT。末尾 chat_cli 是注释示例，没有实际调用，脚本也没有独立 chat_eval。

| 部分 | 实际参数 / 行为 |
|---|---|
| tokenizer | `--max-chars=2000000000`，默认词表 32768；max_chars 是预算，不保证实际字符量恰好达到 |
| base | depth=6、head_dim=64、window=L、T=512、B=32、全局 token batch=16384、5000 次更新 |
| base 的附加工作 | 每 100 步 val BPB 与 sample；CORE 关闭；默认只在结束保存 |
| base_eval | batch=1、split_tokens=16384、每 CORE 任务最多 16 例，但仍默认启用三个模式 |
| SFT | eval_every=200、eval_tokens=524288、num_iterations=1500；ChatCORE 默认开启 |

安装 `--extra cpu` 不会指定 `--device-type=cpu`。脚本没有强制设备，自动优先 CUDA，再 MPS，再 CPU。`COMPUTE_DTYPE` 在导入时选择：无 CUDA 默认 fp32，可用 `NANOCHAT_DTYPE` 覆盖；即使明确选 CPU，主机有 CUDA 时自动 dtype 判断仍可能基于 CUDA。需要固定的 CPU 自检时显式设 `NANOCHAT_DTYPE=float32`。

checkpoint loader 会把 CPU/MPS 上载入的 bf16 权重转为 fp32，但这**不等于**覆盖全局计算 dtype。attention 通常走 SDPA fallback；CPU/MPS 的 MFU 或 CUDA 内存统计不能当作真实硬件利用率证据。

注释中的 M3 Max 时间（tokenizer 约 34 秒、base 约 30 分钟、SFT 约 10 分钟）只是历史参考，不保证其他机器的耗时或结果。OOM、NaN、睡眠暂停按 §7 排查，不是演示的成功标准。

---

## 5. 有条件的小规模 base → SFT → chat 模板

### 5.1 前置条件与 batch 账

**以下命令未作为本章验证运行。** 只有明确允许实际训练、数据已准备、磁盘/内存充足时才使用。确认：

- 使用已有 venv，根目录是 §2.1 的独立实验目录；本节开始先 `cd "$NANOCHAT_REPO"`，不要沿用 §3 的教程 cwd；
- `tokenizer/` 两个文件和至少一个 train shard、固定 val shard已准备且来源可追踪；
- SFT 的 SmolTalk train/test、MMLU auxiliary_train/test、GSM8K train/test 已缓存。关闭训练重复份数也不会取消 val Task 构造；
- 每次保存前确认对应阶段 tag 目录尚不存在，不覆盖已有实验。

若需从空目录准备数据，`python -m nanochat.dataset -n 2` 请求 **2 train + 1 val** 文件；`python -m scripts.tok_train --max-chars=50000000` 是 5000 万字符预算示例，不是“这些材料必足够”的保证。这两条涉及下载/训练，需单独授权；也不要自动接 tok_eval，它可能获取参照 tokenizer。

一个更新的 token batch 需满足：

```text
每 rank micro-batch tokens = B × T
所有 rank micro-batch tokens = B × T × W
grad_accum_steps = total_batch_size / (B × T × W)
```

分子必须为正且整除分母；仅“余数为零”还不能排除零 batch。下面以 `B=1,T=2048,W=1,total_batch_size=2048` 为例，累积次数为 1。

为什么不把 T 缩到 128？base 单独检查可以更短，但 **SFT generator 没把自己的 T 传给渲染器**：渲染先截断到默认 2048 token，packing 再尝试把整个候选放进容量 `T+1` 的行。T=128 时可能全 buffer 都放不下，产出全 padding、无监督 batch 并停滞消费。本链保留 T=2048，使每个已渲染候选都能放进空行，再缩小模型与更新预算。这不保证渲染后仍有 assistant 目标，也不保证低峰值内存；原始长对话的尾部可能已在渲染时丢失，见第 09 章。

### 5.2 Base：两次更新，关闭所有训练内评估与采样

```bash
cd "$NANOCHAT_REPO"  # §3 已切回教程目录；训练模块必须从源码仓库执行。
# 使用 §2.1 的独立根目录与 MODEL_TAG；开始前确认 base tag 目录不存在。
.venv/bin/python -m scripts.base_train \
    --device-type=cpu --depth=2 --head-dim=64 --window-pattern=L \
    --max-seq-len=2048 --device-batch-size=1 --total-batch-size=2048 \
    --num-iterations=2 --eval-every=-1 --core-metric-every=-1 \
    --sample-every=-1 --run=dummy --model-tag="$MODEL_TAG"
```

沿用 `B=1,T=2048,W=1`：`x/y` 均为 `(1,2048)`，每次更新一个 micro-batch。默认 aspect_ratio=64，使 depth=2 的宽度为 128、attention 头数为 2。base 的 num_iterations 控制更新循环：这里两次 forward/backward，另有预取、初始化、编译与保存成本。默认 warmup_steps=40，两个更新的 LR multiplier 仅为 `1/40、2/40`；这是路径检查，不是合理的短程学习配方。

预期完成后有 `base_checkpoints/tutorial-smoke/model_000002.pt`、`meta_000002.json` 和 `optim_000002_rank0.pt`。检查日志中的 loss 是有限数，不期待两步训练明显下降；禁用 BPB 时 metadata 没有有效 val BPB。只看保存日志还不足以证明参数全有限或 checkpoint 可以重新加载。

若只核对加载与正文预测，可以另行选择 **BPB-only**，不跑 CORE / sample：

```bash
.venv/bin/python -m scripts.base_eval --device-type=cpu \
    --model-tag="$MODEL_TAG" --step=2 --eval=bpb \
    --device-batch-size=1 --split-tokens=2048
```

它分别读取 train、val 各一个 batch，不写 CORE CSV。BPB 是否有限取决于有效字节目标；一个小样本结果不是可靠能力比较。预算小于 `B×T×W` 时会向下取整到零步，因此不能盲目把 split_tokens 继续减小。

### 5.3 SFT：iteration 按产出的 micro-batch 计，结束 BPB 仍在

```bash
# 开始前人工确认 SFT tag 目录不存在；输入 base step 明确指定为 2。
.venv/bin/python -m scripts.chat_sft --device-type=cpu \
    --model-tag="$MODEL_TAG" --model-step=2 --load-optimizer=0 \
    --max-seq-len=2048 --device-batch-size=1 --total-batch-size=2048 \
    --num-iterations=4 --eval-every=-1 --eval-tokens=2048 \
    --chatcore-every=-1 --run=dummy
```

这里 cold start optimizer，是为了只检查 SFT 路径，**不是推荐的质量配方**。模型权重仍来自 base。

当前实际顺序为：generator 每 yield 前增加内部计数 → 预取首 batch → 检查结束 → 更新时又预取下一 batch。若数据消费没有提前结束且累积次数为 1：

```text
预取 batch 1
更新 1：使用 batch 1，预取 batch 2，progress=2/4
更新 2：使用 batch 2，预取 batch 3，progress=3/4
更新 3：使用 batch 3，预取 batch 4，标记结束，progress=4/4
下一轮：最终 BPB → 保存 step=3 → 退出；batch 4 未用于训练
```

默认 warmdown_ratio=0.5、final_lr_frac=0，三次更新的 LR multiplier 为 `1、0.5、0`；实际 LR 还乘以 `initial_lr`（已含 init_lr_frac=0.8）。第三次虽做 backward 和 optimizer.step，却没有非零 LR 的参数学习；momentum 等 optimizer 状态仍可能改变。相同条件下 num_iterations=2 的唯一更新 multiplier 就是 0，所以不能据“有一次更新”推断学到了东西。

下面用标准库复核预取与 LR 账；不导入 SFT，不模拟数据消费提前结束、张量或优化器：

```python
def count_sft_updates(horizon, accumulation):
    assert horizon > 0 and accumulation > 0
    emitted = 1                         # 循环外已预取第一个 micro-batch
    updates = 0
    progress_after_update = []
    while emitted < horizon:            # 结束条件在外层检查
        for _ in range(accumulation):
            emitted += 1                # backward 后预取；内层不检查结束
        updates += 1
        progress_after_update.append(emitted / horizon)
    return updates, emitted, progress_after_update

assert count_sft_updates(4, 1) == (3, 4, [0.5, 0.75, 1.0])
assert count_sft_updates(4, 2) == (2, 5, [0.75, 1.25])
assert count_sft_updates(1, 1) == (0, 1, [])
# 默认后半段线性降到零；只复核上面累积次数为 1 的例子。
progress = count_sft_updates(4, 1)[2]
lr_multipliers = [1.0 if p <= 0.5 else 2.0 * (1.0 - p) for p in progress]
assert lr_multipliers == [1.0, 0.5, 0.0]
print("3 次更新调用，4 个 batch 已产出，LR multiplier =", lr_multipliers)
```

累积中途的结束标记不会打断内层，进度可超过 1，默认调度可算出负 LR multiplier；数据消费也可先触发停止。本例固定累积次数为 1，实际 step 仍以日志和文件为准。

`--eval-every=-1` 只关闭周期 BPB：判断是 `last_step or periodic_condition`，因此**结束时仍评估，评估调用返回后才保存**。源码没有以“BPB 有限”为保存前置断言，即使返回 inf 也可能保存；教程验收应自行拒绝这种结果。本例 eval_tokens=2048 对应一个 `(1,2048)` val batch；若有效目标没有原始字节，BPB 返回 `inf`，不能算通过。`--chatcore-every=-1` 才关闭训练内 ChatCORE。有限 BPB 也不证明停止标记或工具行为正确。

### 5.4 Chat：先查真实 step，再决定是否生成

```bash
ls "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$MODEL_TAG/"
# 只有实际存在 model_000003.pt 和 meta_000003.json 时才使用下面的 step=3。
.venv/bin/python -m scripts.chat_cli --device-type=cpu \
    -i sft -g "$MODEL_TAG" -s 3 -t 0 -p "What is 2 + 2?"
```

本例的问题固定为 “What is 2 + 2?”，预期观察的是加载与有限生成，**不是承诺回答 4**。CLI 硬编码最多新增 256 列（含工具插入及终止标记），没有 `--max-new-tokens` 参数；温度 0 走 greedy，不执行 top-k。需要更小预算时，应另行设计经批准的检查，不能凭空加 flag。

**主线到此结束，以下不默认执行**：CLI 可能调用计算器，简单问题也不保证没有工具请求；过滤与超时不是沙箱。默认 chat_eval 会加载任务，并可能下载、生成或执行程序。另行授权分类检查时可选 MMLU、`-b 1 -x 2`，但两题 accuracy **不是 ChatCORE**。RL 的 `--num-epochs=1` 按 GSM8K 题数计算更新预算，不是一个小型 max-steps 设置，见第 12 章。

---

## 6. 从产物到证据：不要把各种数字都叫 score

### 6.1 证据逐层收集，但不互相替代

| 观察 | 可以说明 | 不能说明 |
|---|---|---|
| source checks、AST、链接通过 | 被检查的源码契约与文档结构成立 | 数值、设备或任务正确 |
| checkpoint 文件存在 | 保存路径已执行 | 参数有限、tokenizer兼容、恢复成功 |
| 加载成功且固定小输入输出有限 | 此身份与输入下路径可运行 | 长序列、其他设备或真实能力 |
| train loss / val BPB 有限 | 指定目标集合上的预测数值可计算 | 答案正确、主动停止或工具安全 |
| 按固定协议的任务成功率 | 此数据、采样、判分规则下的任务结果 | 未测泛化、用户体验或全面安全 |

logp 是目标的对数概率，NLL 是其负数 `-log p(target)`；训练 loss 按选定目标归约，BPB 按有效原始字节归一化。分类评估比较合法字母 logits 并统计 accuracy，不是把 loss 换个名字。SFT 日志中的平滑 loss 也不是当前 batch 的原始均值，别只凭一行日志判定数值健康。

ChatCORE 是各任务成功率按代码约定基线中心化后汇总，**中心化可为负**；ARC 的固定 0.25 并不按实际选项数调整。生成任务多候选任一通过是实际 k 次尝试的成功率，不等于单次回复质量。报告时保留温度、top-k 和 k；当前模型刚采样不保证行为分布等于 raw model 分布，RL 的 on-policy 限定见第 12 章。

### 6.2 最小实验记录卡

记录这些字段，比只截图 loss 更容易复核：

```text
源码：HEAD + 工作树差异 + 日期（不能只记 HEAD）
运行：cwd、解释器路径、完整 CLI、退出状态
资源：设备、NANOCHAT_DTYPE / 实际 COMPUTE_DTYPE、attention 后端
依赖：绝对 BASE_DIR、tokenizer 来源与文件摘要、数据 split 与 shard
身份：每阶段 source / tag / 输入 step / 实际输出 step
证据：实际更新与 LR、有效目标、loss/BPB、模型/meta/optimizer 文件
结论：预先声明的通过标准、未运行项目、失败与停止原因
```

CORE CSV 名称只包含 step，没有 tag；同根目录、不同 tag、同 step 的 CORE 评估可能覆盖同名文件。训练目录按 tag 隔离也不自动隔离所有评估结果，应另存带完整身份的日志。

---

## 7. 按层排错：先查便宜且确定的原因

| 层 | 首先核对 | 不要过早得出的结论 |
|---|---|---|
| 环境与身份 | cwd、解释器、绝对 BASE_DIR、是否被参考脚本重设 | “模型有问题” |
| 数据 | 请求 train 数与固定 val、下载成功数、真实 shard / task manifest | “wait 完成所以数据必齐” |
| tokenizer | 两个文件、id 映射来源、checkpoint vocab | “大小相同就兼容” |
| checkpoint | source/tag/实际 step、同 step model/meta、恢复所需 rank shard | “加载器会自动找正确模型” |
| batch 与 packing | 正数整除、T 与渲染上限、batch 中是否有监督目标 | “把 T 调小总会更省且正确” |
| dtype 与后端 | 实际打印的 dtype、是否走 SDPA、设备与可用内存 | “装 CPU extra 就一定在 CPU” |
| 数值与能力 | loss/梯度/参数有限、BPB目标、任务协议与 sample | “loss 降了所以聊天正确” |

Mac 还要查统一内存压力和睡眠。ChatCORE OOM 可通过关闭训练内评估降低成本，但同时放弃了该能力证据；CPU/MPS 日志的 CUDA peak-memory=0 也不表示没有内存开销。

需要改源码才能满足预算、增加生成上限或修复运行问题时，先停止并单独授权，不把源码改动混入教程优化。

---

## 常见误区

1. **“端到端就是完整脚本照跑。”** 应先选验证目标；默认 benchmark、下载和执行风险不是必要的静态检查。
2. **“wait 或退出码为 0 就说明下载齐了。”** 当前单文件失败只进入统计，脚本也不 fail-fast；还需核对实际产物。
3. **“换根目录 / tag 就能隔离一切。”** 原脚本重设根目录，tokenizer 仍是根目录共享资源，CORE CSV 也不含 tag。
4. **“SFT iteration 就是 optimizer step，-1 就关掉所有评估。”** micro-batch 计数与预取改变更新数量；最终 BPB仍会执行。
5. **“mask=0 的输入不可见、完全无梯度。”** 它不是监督目标，但仍可影响后续回答及其梯度；全 padding 则可能没有有效训练信号。
6. **“所有分数都在 0–1，越高越好。”** loss / BPB 通常越低越好，accuracy 与中心化指标不同，中心化还能为负。
7. **“pytest 或有限工具过滤证明安全。”** 测试仅覆盖选定行为，超时、黑名单和进程执行防护都不能替代安全隔离。

---

## 小结

端到端实验的主干是：**先声明目标与资源，再固定依赖和身份，静态检查通过后才进入小规模 base → SFT → chat，最后按证据范围解释结果。**

参考脚本展示流程，不替你处理所有失败、资源隔离和能力证明。特别要记住三件事：tokenizer 从当前全局目录读取；SFT 的停止、LR 和最终 BPB要看循环而非 help；“能保存、能加载、能答对”是不同层级的证据。

---

## 练习

1. **接力表**：为 base → SFT → CLI 标出每个输入目录、source/tag/step 和 tokenizer 来源。为什么仅复制 model 文件不能复现？
2. **并发与失败**：画出 speedrun 的下载、tokenizer、wait 顺序。一个 shard 返回 False 时，为什么 wait 不一定报告失败？只阅读源码，不注入训练命令。
3. **batch 账**：runcpu 的 `B=32,T=512,W=1,total_batch_size=16384`，计算累积次数和 5000 次更新的输入 token 数；为什么这不是唯一文本 token 数？
4. **SFT 计数**：按 §5.3 推导 num_iterations=4 的实际预取、更新次数和 LR multiplier。累积次数变为 2 后，哪一层仍会继续工作？
5. **安全检查计划**：只用 AST 和 `tests/test_tasks.py` 能证明什么？哪些理由使 `tok_eval --help`、默认 chat_eval 和整套 pytest 不应加入零风险清单？
6. **记录卡**：给一次未实际运行的实验填写计划卡，分别定义流程、数值、能力的成功标准；将未知项标为“待验证”，不要用预期日志替代实测。

---

## 前后章导航

- **上一章**：[第 12 章：强化学习](./12-rl.md)。可选 RL 需要独立资源预算、采样与策略边界，不属于本章低成本链。
- **本章**：全书收尾，把数据、训练、推理与证据接成可审计的实验。
- **返回**：[教程目录](./README.md)；按问题回读[第 09 章](./09-sft.md)的监督与 packing、[第 10 章](./10-inference-engine.md)的生成、[第 11 章](./11-chat-evaluation.md)的任务评分。
