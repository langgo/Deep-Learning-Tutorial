# 第 02 章：Dataset：从互联网文本到训练批次

> **本章位置**：[第 01 章：总览](./01-overview.md) → **第 02 章：Dataset：从互联网文本到训练批次** → [第 03 章：Tokenizer](./03-tokenizer.md)  
> **对应源码**：`nanochat/nanochat/dataset.py`、`nanochat/nanochat/dataloader.py`、`nanochat/dev/repackage_data_reference.py`、`nanochat/scripts/base_train.py`。

Transformer、优化器和 GPU 往往最吸引注意力，但模型见到的第一个对象不是网页，也不是句子，而是一个固定形状的整数张量。中间要完成一条很长的翻译链：下载文本、按块读取、把工作分给多张卡、分词、给文档加边界、把长短不同的文档塞进等长行，最后构造“输入”和“下一个 token 目标”。

nanochat 当前的 base 训练数据链路可以先记成一行：

```text
ClimbMix Parquet shards
  → row group（DDP 分片）→ list[str]
  → tokenizer + 每篇文档的 BOS → list[list[int]]
  → BOS-aligned best-fit packing → rows: (B, T + 1)
  → 左右错开一格 → x, y: (B, T)
```

本章只讲数据如何到达 `x, y`；交叉熵、反向传播和完整训练循环属于[第 06 章](./06-base-train-loop.md)。

## 学习目标

读完本章，你应能：

1. 解释“数据先于 tokenizer”与“预训练时先加载 tokenizer”为什么并不矛盾；
2. 找到 ClimbMix shard 的下载入口，并说明当前 train/val 切分规则；
3. 说明 Parquet、列和 row group 分别是什么，以及 DDP 为什么按 row group 分工；
4. 沿真实 loader 追踪文档批量如何变成带 BOS 的 token 文档；
5. 手工执行一次 BOS-aligned best-fit packing，区分利用率和数据保留率；
6. 写出 `(B, T+1)`、`x`、`y` 的形状，并理解 loader resume 为什么只是近似恢复。

## 前置概念：三种不同层次的数据

先把最容易混淆的三个对象分开：

| 层次 | 例子 | 主要用途 |
|---|---|---|
| 原始文本 | `"The sky is blue."` | 训练 tokenizer，也作为预训练语料 |
| token id | `[314, 905, 17]` | 文本的整数编码 |
| 模型 batch | `torch.LongTensor`，形状 `(B, T)` | 直接喂给 GPT |

**数据先于 tokenizer**，指的是 tokenizer 不能凭空训练。BPE 要先从真实文本中统计哪些字节片段常一起出现。因此当前仓库升级到 ClimbMix 时，`dataset.py` 的提示顺序是：先下载数据，再运行 `scripts.tok_train` 重训 tokenizer。

但在**预训练运行时**，顺序恰好是先得到训练好的 tokenizer，再创建 dataloader：`base_train.py` 先调用 `get_tokenizer()`，随后 dataloader 才能把字符串转换成 id。两句话说的是 tokenizer 生命周期的不同阶段，并不冲突。

本章记号：`B` 是一张设备一次产出的样本行数，即 `--device-batch-size`；`T` 是每行交给模型预测的位置数，即 `--max-seq-len`。所有 token id 都是整数，最终 `x` 和 `y` 都是 `(B, T)`。

---

## 1. ClimbMix：shard 下载与 train/val 切分

`nanochat/nanochat/dataset.py` 把当前预训练集定义为 Hugging Face 上的 ClimbMix 重打包版本：

```python
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542
DATA_DIR = os.path.join(get_base_dir(), "base_data_climbmix")
index_to_filename = lambda index: f"shard_{index:05d}.parquet"
```

所以文件名形如 `shard_00000.parquet`，最后一个索引为 `6542`。下载命令例如：

```bash
python -m nanochat.dataset -n 8
```

`-n 8` 表示请求编号 `0` 到 `7` 的 **8 个训练 shard**；代码还会无条件追加 `6542`，下载验证 shard。`-n -1` 才表示请求所有训练 shard。下载使用 `requests.get(..., stream=True)`，按 1 MiB chunk 写到 `.tmp` 临时文件，完成后 `os.rename` 为正式文件；失败最多重试 5 次，并按指数退避等待。多个文件由 `multiprocessing.Pool` 并行下载。

### 当前切分的准确含义

`list_parquet_files()` 从本地目录中取全部 `.parquet` 文件（排除 `.tmp`），按文件名字典序排序。之后读取端使用：

```python
parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
```

即：**本地排序结果的最后一个 parquet 做验证集，其他做训练集。** 正常使用下载入口时，最后文件就是 `shard_06542.parquet`，所以这个约定成立。

这里有一个实际边界：若你手工拷贝了一批 shard 却没有 `shard_06542.parquet`，代码不会验证“最后一个是否真是 6542”，而会把你本地最大的那个编号划给 val。不要把它误解为随机比例切分。找不到新目录时，代码还有从旧 `base_data` 回退的兼容逻辑；这是 FinewebEdu-100B 升级到 ClimbMix-400B 的过渡支持，不是把两套数据混合训练的策略。

### 这些 Parquet 文件从哪里来？

`dev/repackage_data_reference.py` 不是运行时依赖，而是数据准备的参考脚本。当前 `dataset_tag = "climbmix"` 时，它读取 `nvidia/Nemotron-ClimbMix` 的 train split。源列是 GPT-2 tokenizer 编码的 `tokens`；脚本先用 tiktoken 将它们 decode 回**字符串**，再写出只有 `text` 列的 Parquet。因此运行时并不直接使用源数据的 GPT-2 token，而是会用 nanochat 自己训练好的 tokenizer 重新编码文本。

脚本先固定种子打乱：

```python
ds = ds.shuffle(seed=42)
```

随后积累约 `250_000_000` 个字符，并且只在文档数是 `row_group_size = 1024` 的倍数时写一个 shard。写入参数包括 `compression="zstd"`、压缩等级 3、关闭 dictionary encoding 和统计信息；参考注释说明压缩后的 shard 约为 100 MB。这个预先 shuffle 很重要：运行 loader 时基本按文件和 row group 顺序读取，数据的宏观随机性主要已在打包时建立。

---

## 2. Parquet 与 row group：一次读一块，不是一次读全库

Parquet 是列式文件格式。一张表可以有多列；nanochat 运行时只取 `text` 列。一个 Parquet 文件内部又分成多个 **row group**：每个 row group 是一批连续的行，可独立读取、解压和转换。

最小读取逻辑在 `parquets_iter_batched()`：

```python
pf = pq.ParquetFile(filepath)
rg = pf.read_row_group(rg_idx)
texts = rg.column('text').to_pylist()
yield texts
```

这比把整个文件 `read()` 到内存更适合流式训练，也比逐文档做磁盘访问更有效率。参考打包脚本把一组定为 1024 篇文档；读出后才变成 Python 的 `list[str]`。

### DDP 按 row group 分片

DDP（Distributed Data Parallel）会运行多个训练进程。每个进程有 `rank`，总进程数是 `world_size`。如果每个 rank 都读取同一批文本，多卡只是在重复计算。

`dataloader.py` 的 `_document_batches()` 将 row group 轮流分给 rank：

```python
rg_idx = ddp_rank
while rg_idx < pf.num_row_groups:
    rg = pf.read_row_group(rg_idx)
    ...
    rg_idx += ddp_world_size
```

例如 `world_size = 4`：

```text
rank 0: 0, 4, 8, ...
rank 1: 1, 5, 9, ...
rank 2: 2, 6, 10, ...
rank 3: 3, 7, 11, ...
```

分片单位选 row group 而不是单篇文档，是很实用的工程边界：每张卡读取不同的大块，不必为了每一行文本进行跨进程协调；同时 1024 篇文档又不会大到必须一次把整个 shard 放进内存。读取一个 row group 后，代码还按 `tokenizer_batch_size`（默认 128）切成较小的字符串批次。这些只是**分词批次**，还不是模型的 `(B, T)` batch。

所有文件遍历完后，生成器从头再来并令 `epoch += 1`，所以它是无限数据流。这里的 epoch 表示 loader 绕过本地数据列表的次数，不等同于严格、不重复的 token 计数。

---

## 3. 文档批量 → tokenizer → 每篇文档的 BOS

`base_train.py` 中真正的训练入口是：

```python
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len,
    split="train", device=device,
    resume_state_dict=dataloader_resume_state_dict,
)
x, y, dataloader_state_dict = next(train_loader)
```

在 loader 内部，`refill_buffer()` 从 `_document_batches()` 获得 `list[str]`，然后批量编码：

```python
bos_token = tokenizer.get_bos_token_id()
token_lists = tokenizer.encode(
    doc_batch, prepend=bos_token, num_threads=tokenizer_threads
)
```

默认可用 4 个 tokenizer 线程。关键不是“行开头有一个 BOS”，而是**每篇文档**都在编码时 prepend 同一个 BOS（beginning-of-sequence）id。这样即使一行塞进多篇文档，后面每篇也有自己的明确开端。

例如 token 文档可能是：

```text
文档 A: [BOS, 10, 11]
文档 B: [BOS, 20]
文档 C: [BOS, 30, 31, 32]
```

`doc_buffer` 保存这些长度不同的 `list[int]`，默认尝试维持至少 1000 篇文档。它故意还不是规则矩阵，因为不同文档长度正是 packing 要处理的问题。

BOS 的直觉是“新文档从这里开始”。一篇文档中的 token 向左看时，至少能看到本篇的 BOS，而不是把上一文档的末尾当作自己的自然开头。这是源码中 **BOS-aligned** 的含义；它不等于在这里额外插入 EOS，当前路径只显式 prepend BOS。

---

## 4. BOS-aligned best-fit packing：固定形状的代价

GPT 需要规则的张量形状，真实文档却有长有短。逐文档 pad 到 `T` 会浪费大量位置；简单按到达顺序拼接虽容易填满，也可能留下较差的组合。

nanochat 使用 BOS-aligned best-fit packing。每个训练行先按下面的容量构造：

```python
row_capacity = T + 1
row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
```

对每一行，从 `pos = 0` 开始填：

1. 缓冲区文档少于 `buffer_size` 时，继续读取并 token 化文档；
2. 在缓冲区中找出“**能完整放入剩余空间的最长文档**”；
3. 找到就完整放入，并从 buffer 移除；
4. 如果没有任何文档能完整放入，挑出**最短文档**，截取它的前 `remaining` 个 token，精确填满这一行，剩余后缀被丢弃。

这里的 best-fit 是局部启发式，不是全局最优装箱证明。选择最长可放文档会尽量缩小余量，从而比“随手取一个”更少触发裁剪。

假设 `T = 7`，所以行容量为 8；缓冲中有：

```text
A = [1, 10, 11]          长度 3
B = [1, 20]              长度 2
C = [1, 30, 31, 32]      长度 4
D = [1, 40, 41, 42, 43]  长度 5
```

开始余量为 8，最长可完整放入的是 D；余量变为 3。此时 A 恰好放入：

```text
[1, 40, 41, 42, 43, 1, 10, 11]
```

两个 `1` 分别是两篇文档的 BOS。若先放长度 4 的 C，余量为 4，算法会选择长度 3 的 A，余量只剩 1。没有长度 1 的完整文档可放，就会取当前最短的 B，只填其第一个 token `[1]`，并丢弃 `20`。

### 100% 利用率不等于 100% 保留数据

源码说的 **100% utilization** 是：每一行都恰好填满，没有 padding；右移后 `x/y` 中每个位置都有真实 token 并参与训练。它**不**表示读到的原始 token 全部用于更新。

当剩余空间放不下任一完整文档时，算法会裁剪一个文档后缀。`dataloader.py` 的注释给出当前经验量级：`T=2048` 时约 35% token 会因裁剪丢弃。这是当前数据分布和该长度下的近似观察，不是通用常数。

这是刻意的取舍：不 pad，计算位置不空耗；并让每个保留文档片段从自己的 BOS 开始，减少跨文档上下文的混杂。代价是数据覆盖下降，尤其当数据少、文档很长而 `T` 较短时更明显。源码注释也指出，这种情形可考虑旧的非 BOS-aligned loader；默认代码路径则选择当前方案。

```text
完整保留： [BOS d0 d1 d2] [BOS e0 e1]   → 所有 token 进入 row
裁剪填满： [BOS f0 f1 f2 f3 ...]          → 只保留 row 剩余容量所需的前缀
                                      └── 后缀被丢弃，不会移到下一行
padding：   无                               → 因此“位置利用率”可达 100%
```

所以要同时报告两件不同的事：**计算位置是否填满**，以及**读到的数据有多少被保留**。

---

## 5. 最小形状实验：从 `(B, T+1)` 到两个 `(B, T)`

为什么 packing 的容量是 `T+1` 而不是 `T`？因为同一条 token 行要同时产生“当前 token”与“下一个 token”。用一个最小例子直接看：

```python
import torch

rows = torch.tensor([
    [1, 40, 41, 42, 43, 1, 10, 11],
    [1, 50, 51, 1, 60, 61, 62, 63],
], dtype=torch.long)               # (B=2, T+1=8)

x = rows[:, :-1]
y = rows[:, 1:]

assert x.shape == (2, 7)
assert y.shape == (2, 7)
```

第一行结果为：

```text
x[0] = [ 1, 40, 41, 42, 43,  1, 10]
y[0] = [40, 41, 42, 43,  1, 10, 11]
```

也就是说，每个 `x` 位置的目标是右边紧邻的 `y` token。`43` 的目标是下一篇文档的 BOS；BOS 的目标才是 `10`。这已经足以理解数据接口，损失如何计算留给第 06 章。

真实 loader 为吞吐预先分配连续 buffer，而不是每次新建张量：

```text
row_buffer:  (B, T+1)      CPU 上组装的完整行
cpu_buffer:  (2*B*T,)      CPU staging 区（CUDA 时 pinned memory）
cpu_inputs:  (B, T)        cpu_buffer 的前半 view
cpu_targets: (B, T)        cpu_buffer 的后半 view
gpu_buffer:  (2*B*T,)      device 上连续空间
inputs:      (B, T)        gpu_buffer 的前半 view
targets:     (B, T)        gpu_buffer 的后半 view
```

先复制 `row_buffer[:, :-1]` 与 `row_buffer[:, 1:]` 到 CPU 的两个 view，再用一次 `gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)` 传输。CUDA 的 pinned memory 使 HtoD（host-to-device）复制可以非阻塞；这是性能安排，完全不改变上面的形状和语义。

---

## 6. Resume state：从哪里续读，为什么不是精确重放

训练 checkpoint 会保存 loader 交回的：

```python
{"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
```

分别表示 parquet 文件索引、最近读到的 row group 索引、循环轮次。`base_train.py` 恢复 checkpoint 后，把 `meta_data["dataloader_state_dict"]` 传回 train loader。

恢复到同一 parquet 时，`_document_batches()` 计算：

```python
base_idx = resume_rg_idx // ddp_world_size
base_idx += 1
rg_idx = base_idx * ddp_world_size + ddp_rank
```

直觉是：每个 rank 沿自己的 `rank, rank + world_size, ...` 序列前进一格，避免恢复后马上再读同一个分片 row group。

不过源码 docstring 明确称它为 **approximate resume**。状态没有保存 `doc_buffer` 中已经 token 化但尚未消费的文档，也没有保存正在装配的行位置或 buffer 内文档顺序；并且状态在补充文档批次时更新。因此中断后会跳过一部分已读数据，不能期待“不间断运行”和“中断再恢复”得到逐 token 完全相同的 batch 序列。它优先保证可继续训练并避开明显重复，而不是昂贵的精确数据重放。

## 常见误区

1. **“ClimbMix 已经是 GPT-2 token，所以 nanochat 不需要 tokenizer。”** 源数据确实有 token，但参考打包脚本将其 decode 后写为 `text`；运行时会重新编码。
2. **“一个 shard 就是一条训练样本。”** shard 是文件，文件含多个 row group，row group 含多篇文档；最终一行还可能拼多篇文档。
3. **“DDP 是每张卡随机抽几篇文本。”** 当前实现按 rank 间隔读取 row group，随机性主要来自预先 shuffle 与 packing 组合。
4. **“100% utilization 就没有任何 token 损失。”** 它只消除了 padding；裁剪仍会丢弃文档后缀。
5. **“BOS 只在整个 batch 的最左边。”** 每篇进入 buffer 的文档都 prepend BOS，一行中可出现多个 BOS。
6. **“resume 会精确接回中断时的下一 token。”** 不会；packing buffer 没有写进 state。
7. **“验证集是随机抽出的一部分。”** 当前是本地排序后的最后一个 parquet；正常下载时它固定为最后 shard。

## 小结

- 文本数据先用于训练 tokenizer；预训练时则由已经训练好的 tokenizer 将字符串转为 id。
- 当前 ClimbMix 被预打乱、重打包为 `text` 列的 Zstandard Parquet shard；正常下载约定最后 shard 为 val，其他为 train。
- row group 是可独立读取的 Parquet 块，也是 DDP 的分片边界：rank `r` 读取 `r, r + world_size, ...`。
- loader 批量编码文档、给每篇 prepend BOS，并用 best-fit 把它们填进 `(B, T+1)`；无 padding，但可能裁剪文档后缀。
- `x = rows[:, :-1]`、`y = rows[:, 1:]` 都是 `(B, T)`，这是训练代码看到的数据接口。
- checkpoint 只保存 parquet/row-group/epoch 位置，所以 loader 恢复是近似的。

## 练习

1. 执行 `python -m nanochat.dataset -n 3` 时，会请求哪些 shard 索引？为什么验证 shard 仍会被下载？
2. 一个 parquet 有 10 个 row group，`world_size=3`。分别列出 rank 0、1、2 读取的 index；哪几个 rank 少读一组？
3. 行容量为 10，缓冲中文档长度为 `6, 4, 3, 3`。按“最长完整可放”规则，前两次会选择什么？若已放长度 6、余量变成 2，又会发生什么？
4. 当 `B=4`、`T=128` 时，写出 `row_buffer`、`x`、`y`、`cpu_buffer` 的形状，并解释为什么需要 `T+1`。
5. 阅读 `_document_batches()` 的恢复分支。为什么 `resume_rg_idx // ddp_world_size` 后还要 `+1`？列出一个未被 state 保存、从而使恢复不精确的对象。
6. 如果数据很少、每篇文档普遍远长于 `T`，从数据覆盖、上下文边界和固定形状计算三个角度，评价 BOS-aligned 裁剪的取舍。

## 前后章导航

- **上一章**：[第 01 章：总览](./01-overview.md) 给出 base、SFT、RL 的全流程地图；本章定位了原始文本如何成为 base 训练 batch。
- **下一章**：[第 03 章：Tokenizer](./03-tokenizer.md) 讲解本章中作为黑箱调用的 tokenizer，解释字符串如何变成 token id。
- **目录与路线**：[`README.md`](./README.md)。
