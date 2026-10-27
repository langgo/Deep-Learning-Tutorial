# 第 02 章　数据集：从互联网文本到训练批次

> **本章定位**　第一部分·数据与表示，建立"自然语言 → 模型输入张量"的第一段桥梁。
>
> **中心问题**　一段原始文本是如何变成 GPT 真正吃进去的那两个固定形状的整数矩阵 `x`、`y` 的？
>
> **范围**　只讨论数据链路。交叉熵、反向传播、完整训练循环放到[第 06 章](./06-base-train-loop.md)。
>
> **先决条件**　已读[第 01 章](./01-overview.md)，建立全局地图。
>
> **本章产物**　对 ClimbMix shard、Parquet row group、BOS-aligned best-fit packing、近似恢复机制的完整视角，以及"`(B, T+1) → x/y`"这一关键形状转换的手推理解。
>
> **源码入口**　`nanochat/dataset.py`、`nanochat/dataloader.py`、`dev/repackage_data_reference.py`、`scripts/base_train.py`。

---

## 本章讲什么

初学者看一次 LLM 训练的第一反应，往往是把目光锁在 Transformer 架构、优化器或 GPU 上。但在这些模块启动之前，有一件更朴素的事：模型第一眼看到的不是一段网页，也不是一句自然语言，而是一个**形状固定的整数矩阵**。中间要走完一条很长的翻译链——下载文本、分段读取、分给多张卡、分词、给每篇文档打上开始标记、把长短不齐的文档塞进等长的行，最后错开一格得到“当前位置”和“下一位置的目标”。

**本章就把这条链路逐段拆开讲**。全链条可以先记成这一行：

```text
ClimbMix Parquet shards
  → row group（按卡切片）→ list[str]
  → tokenizer + 每篇文档前面加 BOS → list[list[int]]
  → BOS-aligned best-fit packing → rows: (B, T + 1)
  → 左右错开一格 → x, y: (B, T)
```

本章就是把这一行逐段拆开讲。读完你应该能：

1. 讲清"要先有数据才能训 tokenizer"和"预训练启动时先加载 tokenizer"为什么并不矛盾；
2. 定位 ClimbMix shard 的下载入口，说明当前 train / val 切分规则；
3. 说明 Parquet 文件、列、row group 分别是什么，以及多卡（DDP）为什么按 row group 分工；
4. 顺着真实 loader 追踪一批文档如何变成带 BOS 的 token 序列；
5. 手工跑一遍 BOS-aligned best-fit packing，分清"位置利用率"和"数据保留率"是两件事；
6. 写出 `(B, T+1)`、`x`、`y` 的形状，并理解"loader 的 resume 为什么只是近似恢复"。

为了避免每节重新举例，**全章使用同一组微型数据**：两篇短文本、每行容量 8 个 token（`T = 7`、`T + 1 = 8`），`B = 2`。所有"输入输出张量长什么样"的讨论都以它们为例。

## 预备词汇

正文里会反复出现几个词，这里一次说清，后文不再打断：

- **BPE / tokenizer**：把一段文本切成一串整数 id 的程序。它不是人为规定的词表，而是**从真实文本里学出来的**：先统计哪些字节片段常常相邻出现，再逐步合并成更大的片段。因此 tokenizer 的生命周期是先学再用，详见[第 03 章](./03-tokenizer.md)。
- **BOS**（beginning of sequence）：一个特殊 token id。本仓库在每篇文档的最前面插入它，表示"一篇新文档从这里开始"。
- **Parquet**：一种列式表格文件格式；一张"表"存成一个文件，内部按列、按块组织，适合流式读取。
- **row group**：Parquet 文件内部的一个"小块"，一次可以独立读出一批连续的行。本仓库的参考打包把每 1024 篇文档打成一个 row group。
- **DDP**（Distributed Data Parallel）：PyTorch 的多进程并行方式。多张卡 = 多个 Python 进程，每个进程有自己的 `rank`（编号）和共同的 `world_size`（总数）。每个 rank 要读到**不同的数据**，否则多卡只是在重复计算。
- **`B`、`T`**：一张卡**一次产出的行数**和**每行给模型预测的位置数**。对应命令行就是 `--device-batch-size` 和 `--max-seq-len`。本章例子里 `B = 2`，`T = 7`。

明确：所有 token id 都是整数，最终模型拿到的 `x` 和 `y` 都是 `(B, T)` 的 `LongTensor`。

## 一个贯穿全章的例子

想象我们只有两篇训练文本：

```text
文档 A: "The sky is blue."
文档 B: "Hi there."
```

假设 nanochat 的 tokenizer 把它们编码成：

```text
A: encode("The sky is blue.") = [314, 905, 17, 42]        （4 个 token）
B: encode("Hi there.")         = [71, 823]                 （2 个 token）
```

> 这里的 id 都是演示值，不是真实数值。真实 vocab 在[第 03 章](./03-tokenizer.md)里讲；本章只关心形状和位置。

记住这两串数字。本章后面每一步都会把它们再往前推一步。

---

## 1. 数据从哪里来？—— ClimbMix shard 与 train / val 切分

### 1.1 "先有数据，还是先有 tokenizer"？

**先回答两个常见问题**。

**为什么要先有数据？** 因为 BPE 不是写死的规则表，它要从真实文本里统计出来；没有文本就谈不上统计。所以当仓库从 FinewebEdu 升级到 ClimbMix 时，`nanochat/dataset.py` 的提示顺序是：先把 ClimbMix shard 下载下来，再执行 `python -m scripts.tok_train` 重新训练 tokenizer。

**那为什么预训练启动时又好像"先加载 tokenizer"？** 因为那时 tokenizer **已经训练完**了。`scripts/base_train.py` 的流程是：先 `get_tokenizer()` 把现成的 tokenizer 读进来，再创建 dataloader——因为 dataloader 的职责之一就是把字符串变成 id，没有 tokenizer 它干不了活。

两句话说的是 tokenizer 生命周期的**两个阶段**：训练之前 vs. 训练之后。它们不冲突。

### 1.2 ClimbMix shard 在哪里

当前预训练集是 ClimbMix 的一个重打包版本，托管在 Hugging Face 上。`nanochat/dataset.py` 里有这几行定义：

```python
BASE_URL  = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542
DATA_DIR  = os.path.join(get_base_dir(), "base_data_climbmix")
index_to_filename = lambda index: f"shard_{index:05d}.parquet"
```

所以文件名长这样：`shard_00000.parquet`、`shard_00001.parquet`、……、`shard_06542.parquet`。编号 `6542` 是上界。

下载时你写：

```bash
python -m nanochat.dataset -n 8
```

`-n 8` 的意思是"请求编号 0 到 7 的 8 个训练 shard"。此外，不管 `-n` 是多少，**代码都会额外下载编号 `6542` 的那个 shard**。这就是**验证集**（val）的那一份。想一次拉全部训练 shard，用 `-n -1`。

下载过程本身没什么神秘：`requests.get(..., stream=True)`，1 MiB 一块地写到 `.tmp` 临时文件，整体完成后 `os.rename` 为正式文件；失败最多重试 5 次、指数退避；多个文件用 `multiprocessing.Pool` 并行跑。

### 1.3 train / val 到底怎么切

读取端的做法只有一行：

```python
parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
```

换句话说：**把本地目录里所有 `.parquet` 文件按文件名字典序排好，最后一个当验证集，其余的全当训练集**。

因为上面那条"总是下载编号 6542"的规则，正常下载流程下，字典序最后一个就是 `shard_06542.parquet`。所以验证集本质上就是固定的那一份，不是随机采样的。

> **边界情形提醒**：如果你**手动**拷贝了一堆 shard 到本地，但其中并没有 `shard_06542.parquet`，代码不会去校验"最后一个是不是真的 6542"，它只按字典序取最后一个。所以你本地最大的那个编号会被当成 val。这不是 bug，是"读取端只按排序取最后一个"的简单约定。

如果你发现本地目录是旧的 `base_data`（FinewebEdu 时代的名字），代码还会 fallback 过去读。这只是旧数据升级到 ClimbMix 过程中的向后兼容，不是"把两套数据混合训练"的策略。

### 1.4 这些 Parquet 文件是怎么被做出来的

`dev/repackage_data_reference.py` 不是训练时会跑的脚本，它是**离线准备数据**用的，放在仓库里是为了让你知道 ClimbMix 的 shard 到底长什么样、是如何打乱和分块的。

参考脚本里关键的三件事：

1. **先打乱**：`ds = ds.shuffle(seed=42)`。整个数据集先做一次全局 shuffle。
2. **GPT-2 token 要被 decode 回文本**：ClimbMix 源数据存的是 GPT-2 tokenizer 编码出的 `tokens` 列。参考脚本先用 tiktoken 把它 decode 成**字符串**，再写出一个只有 `text` 列的 Parquet。所以运行时 nanochat 看到的是**纯文本**，会用自己训练出来的 tokenizer 重新编码，源数据里的 GPT-2 token 并不会直接进入训练。
3. **一个 shard 约 250M 字符、约 100 MB**：当累积到约 `250_000_000` 字符时写一个 shard，而且只在**文档数是 1024 的倍数**时才切——因为 1024 是下一节要讲的 row group 的大小，这样切出来的文件每个 row group 都是满的。写入用 Zstandard 压缩，等级 3。

**预先 shuffle 这件事很重要**。因为运行时 loader 的读取基本是按文件、按 row group 顺序读的，不额外做跨文件的随机采样；全局随机性早在打包这一步就固化进了磁盘。

---

## 2. 读文件 —— Parquet row group 与多卡分工

### 2.1 为什么要分 row group

一个 Parquet 文件可能有几 GB，训练过程中你不想一次性把整个文件读进内存，也不想**每读一篇文档都打开一次磁盘**。row group 正是为此设计的：一个 row group 就是文件里一批连续的行（本仓库是 1024 篇文档），它可以**独立被解压、读取和转换**。读一组文档的开销只是读一个 row group，而不是整个文件。

真正的读取逻辑非常短——打开文件、按索引取一个 row group、拿出 `text` 列：

```python
pf  = pq.ParquetFile(filepath)
rg  = pf.read_row_group(rg_idx)
texts = rg.column("text").to_pylist()
yield texts        # 一个 list[str]，长度 ~1024
```

一次性拿到 1024 篇原始字符串。这个粒度在"内存占用"和"磁盘往返次数"之间比较舒服。

### 2.2 多卡怎么分这些 row group

想象你有 4 张卡在训练，用 DDP：

```text
world_size = 4
rank 0, rank 1, rank 2, rank 3   —— 四个独立的 Python 进程
```

如果每张卡都读同样的 row group，四张卡只是在**并行计算同一笔梯度**，并没有看到更多数据。所以 `_document_batches()` 用的是最朴素的"按余数分工"：

```python
rg_idx = ddp_rank
while rg_idx < pf.num_row_groups:
    rg = pf.read_row_group(rg_idx)
    ...
    rg_idx += ddp_world_size
```

于是读取顺序就是：

```text
rank 0:  0,  4,  8, 12, ...
rank 1:  1,  5,  9, 13, ...
rank 2:  2,  6, 10, 14, ...
rank 3:  3,  7, 11, 15, ...
```

**选 row group 作为分片单位是经过取舍的**：

- 如果按**单篇文档**分，每一行都要跨进程协调，太琐碎；
- 如果按**整个文件**分，粒度又太粗——文件数本来就不多，可能有一张卡无事可做；
- row group 恰在中间：1024 篇一组，内存吃得下，分工又细到不浪费。

读到一个 row group 之后，代码还会按 `tokenizer_batch_size`（默认 128）把它切成更小的字符串批次，交给下一步的分词器。注意这些**只是分词批次**，还不是模型最终的 `(B, T)` 批次。

所有文件遍历完一轮后，生成器从头再来，并把 `epoch` 加一。所以数据流在逻辑上是**无限**的；这里的"epoch"只是说"loader 把本地文件列表走了几遍"，并不等同于"每个 token 严格看过几次"——因为后面还有裁剪。

---

## 3. 分词 —— 把 `list[str]` 变成每篇都带 BOS 的 `list[list[int]]`

训练入口在 `scripts/base_train.py` 里就一行：

```python
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len,
    split="train", device=device,
    resume_state_dict=dataloader_resume_state_dict,
)
x, y, dataloader_state_dict = next(train_loader)
```

进到 loader 内部，`refill_buffer()` 从上一节的 `_document_batches()` 取到一批 `list[str]`，然后批量编码：

```python
bos_token   = tokenizer.get_bos_token_id()
token_lists = tokenizer.encode(
    doc_batch, prepend=bos_token, num_threads=tokenizer_threads
)
```

默认 4 个 tokenizer 线程并行跑。注意 `prepend` 的含义：**不是"一行开头一个 BOS"，而是"每一篇文档"编码时都在最前面插入同一个 BOS**。

回到我们的两篇文档例子。假设 BOS 的 id 是 `1`，加 BOS 后：

```text
A: [1, 314, 905, 17, 42]          长度 5
B: [1,  71, 823]                   长度 3
```

这些长度不一的 `list[int]` 全部堆进一个叫 `doc_buffer` 的缓冲区，默认至少维持 1000 篇文档。它**不是**一个规则矩阵，就是一组“长短不齐的 token 序列”。这正是下一节要解决的问题。

> **`buffer_size = 1000` 不是随手选的量级**。下一节的 best-fit 每填一个位置就要在整个 buffer 里扫一遍找“能装下的最大文档”——buffer 越大，越可能找到一篇恰好塞满剩余空间的文档，裁剪率越低；但扫描成本和内存占用也越高。1000 是经验平衡点。

为什么要给每篇都加一个 BOS？一句话：**让模型总能看到"一篇新文档从这里开始"**。等到下一节把两篇文档拼在同一行里时，B 的开头如果没有 BOS，模型就会把 A 的末尾当成 B 的开头读——它不知道中间已经换了一篇文档。源码里 "BOS-aligned" 这个词说的就是这件事：**每篇文档的边界与一个 BOS 对齐**。

> 注意：当前这条代码路径只显式 prepend BOS，没有额外插 EOS（end of sequence）。文档结束靠的是"下一篇文档的 BOS 突然出现"，而不是一个显式的结束符。

---

## 4. 装箱 —— 把长短不齐的文档塞进 `(B, T+1)` 的方格

### 4.1 固定形状的代价

GPT 要吃**形状规则**的张量，但真实文档长短不齐。两个最朴素的办法都不划算：

- **逐文档 pad 到 T**：短文档后面全是 pad token，计算位置白白浪费。
- **按到达顺序硬拼**：填得满，但可能让某一篇文档被切成一半、另一半扔掉，而另一处又凑出不好的组合。

nanochat 选了一个折中：**best-fit packing**——每一行留出一个 token 的余量（容量 `T + 1`，稍后会解释为什么），然后按"**优先放完整文档，装不下才裁剪**"的规则填。

### 4.2 填一行的规则（走一遍小例子就懂）

容量预先分配好：

```python
row_capacity = T + 1               # 本章例子里是 8
row_buffer   = torch.empty((B, row_capacity), dtype=torch.long)
```

对一行从 `pos = 0` 开始，不断执行：

1. 如果 `doc_buffer` 里文档数不够，就回到上一节再读几批、分词，补满 buffer；
2. 在 buffer 里找出"**能完整放进剩余空间的、最长的那篇文档**"；
3. 找得到 → 完整放进去，把这篇从 buffer 里移除；
4. 找不到（没有任何一篇能完整放下）→ 挑 buffer 里**最短的那篇**，截取它的前 `remaining` 个 token，正好填满这一行；**截掉的后缀直接丢弃**，不留到下一行。

> **为什么挑“最短的”裁，而不是最长的**？因为被丢弃的是 `len(doc) - remaining` 个 token。挑最短的文档，这个差值就最小——浪费最少。举例：`remaining = 10`，buffer 里剩 `[50, 200, 800]` 的三篇文档。裁 50 的那篇只丢 40 个 token；裁 800 的那篇要丢 790 个。两种做法填进 row 的 10 个 token 是不同前缀，但真正影响数据保留率的是“丢多少”。

> **best-fit 是一个局部启发式**，不是全局最优的装箱算法。"每次优先选最长可放的"的直觉是：余量会尽量小，从而更不容易触发第 4 步的裁剪。但它不保证整体数据利用率最高。

### 4.3 用小例子手动填一行

延续我们那两篇文档的例子，再凑两篇让它更真实。`T = 7`，所以一行容量是 8：

```text
文档 A: [1, 314, 905, 17, 42]      长度 5
文档 B: [1,  71, 823]              长度 3
文档 C: [1, 500, 501, 502]         长度 4
文档 D: [1,  60]                   长度 2
```

开始填第一行，`remaining = 8`：

- 找最长的可完整放下的文档：A（长度 5）可以放，C（长度 4）也可以，B、D 更短。按规则选**最长**，放 A。
  - 行内容：`[1, 314, 905, 17, 42, _, _, _]`
  - `remaining = 3`
- 下一轮：现在剩 3 个空位。最长的可完整放下的文档是 B（长度 3）。放 B。
  - 行内容：`[1, 314, 905, 17, 42, 1, 71, 823]`
  - `remaining = 0`
- 这一行满了。

于是第一行的 token 序列是：

```text
row[0] = [1, 314, 905, 17, 42,   1,  71, 823]
         └────── 文档 A ──────┘ └── 文档 B ──┘
```

两个 `1` 分别是两篇文档各自的 BOS。

**对比"顺序硬拼"的问题就明显了**：如果不看长度直接按到达顺序填，可能先放了长度 4 的 C（`remaining = 4`），再想找长度 4 以内可完整放的最大文档——比如长度 3 的 B（`remaining = 1`），然后哪怕一个 token 都放不下的文档也得被裁剪成"只填一个 token"。本仓库源码注释里举过这种例子：余量 1 时取最短文档的前 1 个 token，剩下的几十、上百个 token 直接丢。

### 4.4 两个 100% 不是同一个 100%

源码里的措辞是 **100% utilization**，意思是：

> **每一行都正好被填满，没有一个 padding 位置**；最终 `x`、`y` 的每个计算位置都对应一个真实的 token，没有白跑的梯度。

这**不等于**"读进来的 token 都用上了"。当没有任何完整文档能放进剩余空间时，裁剪会丢掉一个文档的后缀。`nanochat/dataloader.py` 的注释给了一个经验量级：**在 `T = 2048` 时，大约 35% 的 token 会因裁剪而丢弃**。这是在当前数据分布和这个 `T` 下的近似观察，不是什么通用常数——改 `T` 或者换数据集，这个比例会变。

一张图直观看三种情况：

```text
完整保留:  [BOS d0 d1 d2]  [BOS e0 e1]       ─ 所有 token 都进入了 row
裁剪填满:  [BOS f0 f1 f2 f3 f4 ...]            ─ 只保留前缀；后缀被直接丢弃
padding:   无                                   ─ 因此"位置利用率"确实可以是 100%
```

这是**刻意的取舍**：不 pad，所以计算位置不空耗；每个保留下来的片段都从自己的 BOS 开始，减少"跨文档混读"的噪声；代价则是数据覆盖下降，尤其在数据少、文档远长于 `T` 的时候更明显。源码注释也指出：那种场景下可以考虑旧版本的非 BOS-aligned loader；默认路径就是现在这个。

**要同时报告两件事**：计算位置是否填满（utilization），以及读到的数据有多少被保留（retention）。它们不是同一个指标。

---

## 5. 形状翻转 —— 从一行 `(T+1)` 到两个 `(B, T)` 的 `x / y`

### 5.1 为什么是 `T + 1` 而不是 `T`

GPT 做自回归语言建模。给出前 `k` 个 token，预测第 `k + 1` 个。所以同一条 token 序列要同时提供两样东西：**当前位置的输入** 和 **下一位置的目标**。

一个长度 `T + 1` 的行刚好让它俩都有 `T` 个位置：

```python
rows = ...                       # (B, T+1)，上一节的 best-fit 产物
x = rows[:, :-1]                 # (B, T)   取每行除最后一个
y = rows[:, 1:]                  # (B, T)   取每行除第一个
```

这样每个 `x[i, t]` 的目标就是 `y[i, t] = rows[i, t + 1]`。

### 5.2 继续走小例子

接上一节填好的第一行，假设第二行也用类似办法凑出来：

```python
import torch

rows = torch.tensor([
    [1, 314, 905, 17,  42,   1,  71, 823],   # 行 0: A + B
    [1,  60,   1, 500, 501, 502,   1, 999],  # 行 1: D + C + 一个新 BOS 的小片段
], dtype=torch.long)                          # 形状 (B=2, T+1=8)

x = rows[:, :-1]      # (2, 7)
y = rows[:, 1:]       # (2, 7)
```

把第一行列出来看：

```text
位置       0    1    2    3    4    5    6    7
rows[0] =  1  314  905   17   42    1   71  823

x[0]    =  1  314  905   17   42    1   71          （去掉最后一个）
y[0]    =314  905   17   42    1   71  823          （去掉第一个）
```

每一对 `(x[0, t], y[0, t])` 的含义是："模型看到 `x[0, t]`，应该预测出 `y[0, t]`"。

有两个位置值得特别看：

- `x[0, 4] = 42`（文档 A 的最后一个真实 token）→ `y[0, 4] = 1`（文档 B 的 BOS）。模型要学会："A 这篇到此为止，下一步应该出现一个新文档的开头 BOS。"
- `x[0, 5] = 1`（B 的 BOS）→ `y[0, 5] = 71`（B 的第一个真实 token）。模型要学会："看到 BOS 之后，接下来该生成这篇文档的开头 token。"

这就是 best-fit packing + BOS 配合起来教给模型的两条隐式规则。**到这里，模型实际看到的接口已经完整定义了**。至于拿到 `x`、`y` 之后怎么计算 loss，不是本章的事情，看[第 06 章](./06-base-train-loop.md)。

### 5.3 真实 loader 做了一层性能包装

概念上 `x = rows[:, :-1]`、`y = rows[:, 1:]` 就够了。真实 loader 为了吞吐，把这几个张量预先一次性分配好，不每轮重建：

```text
row_buffer:   (B, T+1)        CPU 上组装的完整行
cpu_buffer:   (2*B*T,)        CPU 的 staging 区（CUDA 时是 pinned memory）
  ├─ cpu_inputs:   (B, T)     cpu_buffer 的前半 view
  └─ cpu_targets:  (B, T)     cpu_buffer 的后半 view
gpu_buffer:   (2*B*T,)        device 上连续空间
  ├─ inputs:   (B, T)         gpu_buffer 的前半 view
  └─ targets:  (B, T)         gpu_buffer 的后半 view
```

组装完 `row_buffer` 后，先把 `row_buffer[:, :-1]` 和 `row_buffer[:, 1:]` 复制到 CPU 的两个 view 里，再**用一次** `gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)` 做 host-to-device 传输。CUDA 下 pinned memory 让这一次复制不阻塞 Python 主线程；MPS / CPU 下就是同步 copy。这都是性能安排，**完全不改变形状和语义**。

---

## 6. 断点 —— 为什么 loader 的 resume 只是"近似"的

训练中保存 checkpoint 时，loader 会交出它自己的状态：

```python
{"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
```

三件事：当前读到哪个 parquet 文件、最近读到哪个 row group、第几遍扫数据集。`scripts/base_train.py` 恢复 checkpoint 后，把 `meta_data["dataloader_state_dict"]` 传回 loader。

回到同一个 parquet 文件时，`_document_batches()` 用这样的方式决定下一步从哪里开始：

```python
base_idx  = resume_rg_idx // ddp_world_size
base_idx += 1
rg_idx    = base_idx * ddp_world_size + ddp_rank
```

意思是：每张卡沿它自己的"每隔 world_size 一个"的轨道**往前跳一格**，避免从上次已经读过的 row group 原地重复。

但源码 docstring 很诚实地叫它 **approximate resume**。原因是：

- `doc_buffer` 里**已经分词但还没进 `(B, T+1)`** 的文档没有保存；
- 行装配到一半的位置信息（当前这一行填到 `pos` 多少、buffer 里现在还有哪些文档、顺序如何）也没保存；
- 加上 state 本身是在"**补充新一批文档时**"才更新的，更新粒度就不精细。

所以你**不能期待**“连续跑一次”和“中间断了再恢复”得到逐 token 完全一致的 batch 序列。这个设计的优先级是：**让训练能继续，并大体越过 checkpoint 记录的 row-group 进度**，而不是花代价保存整个预取和 packing 状态。

还要注意分布式边界：checkpoint 的 JSON metadata 只由 rank 0 写入，其中只有一份 `dataloader_state_dict`；恢复后各 rank 会用这份公共状态，再按自己的 rank 重新计算起点。因此不能仅凭 `+1` 就声称“最多跳过 `world_size` 个 row group”或给出固定的丢文档上限。实际偏移还取决于保存时 rank 0 已预取到哪里、`doc_buffer` 中尚未消费多少文档、其他 rank 当时各自的进度以及 row group 数量。对严格复现实验，最准确的结论只有：**恢复是近似的，训练可继续，但 batch 序列不会逐 token 重放。**

---

## 7. 回头看一张图

把全链路再拉通：

```text
磁盘上的 shard_*.parquet
       │
       │  pq.ParquetFile.read_row_group(rg_idx)
       ▼
一个 row group  = 1024 篇原始 text 字符串
       │
       │  按 rank 轮转：rank r 取 r, r+world_size, r+2*world_size, ...
       ▼
分发到各 rank 的 list[str]
       │
       │  tokenizer.encode(..., prepend=BOS)
       ▼
list[list[int]]：每篇文档前面挂着一个 BOS
       │
       │  BOS-aligned best-fit packing
       │  （最长可放的放满；放不下就裁剪最短文档的前缀）
       ▼
row_buffer: (B, T+1)
       │
       │  x = rows[:, :-1]
       │  y = rows[:, 1:]
       ▼
最终接口: x, y   都是 (B, T) 的 LongTensor
```

这张图就是本章要你带走的那一张脑图。到了[第 06 章](./06-base-train-loop.md)，训练循环拿到 `x, y` 之后要做的事，才是另一回故事。

---

## 常见误区

1. **"ClimbMix 已经是 GPT-2 token 了，所以 nanochat 不用自己 tokenize。"** 源数据的确是 token，但参考打包脚本已经把它们 decode 回 `text` 列再写出；运行时 nanochat 看到的是**纯文本**，用自己的 tokenizer 重新编码。
2. **"一个 shard 就是一条训练样本。"** 不是。shard 是一个 Parquet 文件，内含多个 row group，每个 row group 内含约 1024 篇文档；最终一行还可能把若干篇文档拼在一起。
3. **"DDP 就是每张卡随机抽几篇文本。"** 不是。本仓库是按 row group 轮转分片的；随机性大部分来自**预先 shuffle** 和 best-fit 的拼接组合，不是运行时的二次抽样。
4. **"100% utilization = 没有任何 token 损失。"** 100% utilization 只说明"没有 padding"，裁剪仍会丢掉文档后缀。`T = 2048` 时经验上 ~35% 的 token 被丢弃。
5. **"BOS 只出现在整个 batch 的最左边。"** 不是。每篇进入 buffer 的文档都 prepend BOS，一行里可以有多个 BOS。
6. **"resume 会精确接回中断时的下一 token。"** 不会。`doc_buffer` 和打包进度的状态没有写进 checkpoint，所以是近似恢复。
7. **"验证集是随机抽的一部分。"** 当前验证集是**本地文件按字典序排完后的最后一个**；正常下载流程下它固定是 `shard_06542.parquet`。

## 小结

- 原始文本先被用来训练 tokenizer；预训练启动时则由**训练好的** tokenizer 把字符串转为 id。两个阶段不冲突。
- 当前的 ClimbMix 已经预先 shuffle 过、被重打包成只有 `text` 列的 Zstandard Parquet shard；本地字典序最后一个文件是验证集。
- Parquet 的 row group 是独立可读的小块，也是多卡分工的边界：rank `r` 读取 `r, r + world_size, 2*world_size + r, ...`。
- loader 批量分词，并给**每篇文档**前面挂一个 BOS；再用 best-fit packing 把它们填进 `(B, T + 1)`。
- `x = rows[:, :-1]`、`y = rows[:, 1:]`，都是 `(B, T)`。这就是训练代码看到的最终数据接口。
- 100% utilization 是"没有 padding"的意思，不等于"所有 token 都用上了"。
- loader 的 resume 只保存文件 / row group / epoch 索引，所以中断续训是**近似**恢复，不是字节级重放。

## 练习

1. 跑 `python -m nanochat.dataset -n 3` 时，程序会向远端请求哪些 shard 索引？为什么验证 shard 还是会被下载？
2. 一个 parquet 文件有 10 个 row group，`world_size = 3`。分别写出 rank 0、1、2 读到的 row group index 序列；哪几个 rank 少读了一个？
3. 行容量为 10，buffer 里的文档长度依次为 `6, 4, 3, 3`。按 best-fit 规则，前两次会分别选谁？若已放长度 6、余量变为 2，接下来会发生什么？
4. 当 `B = 4`、`T = 128` 时，分别写出 `row_buffer`、`x`、`y`、`cpu_buffer` 的形状，并解释为什么必须用 `T + 1` 而不是 `T`。
5. 读 `_document_batches()` 的恢复分支。为什么 `resume_rg_idx // ddp_world_size` 之后要 `+1`？再列举一个**没有被 state 保存、因而使恢复不精确**的对象。
6. 假设你的数据非常少、而且几乎每篇文档都远长于 `T`。从"数据覆盖"、"上下文边界"和"固定形状计算"三个角度，评价 BOS-aligned 裁剪的取舍。

## 前后章导航

- **上一章**：[第 01 章：总览](./01-overview.md) 给出 base、SFT、RL 的全流程地图，本章把其中"原始文本 → 训练 batch"的那一段展开。
- **下一章**：[第 03 章：分词器：从字节到 Token](./03-tokenizer.md) 讲清本章当作黑箱使用的 tokenizer——字符串具体是怎么变成 token id 的。
- **目录与路线**：[`README.md`](./README.md)。
