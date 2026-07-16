#!/usr/bin/env python3
"""生成 word2vec 互动教学 notebook：word2vec_tutorial.ipynb。

用 nbformat 以编程方式拼装 notebook，比手写 .ipynb 的 JSON 更可维护。
运行：  uv run python build_notebook.py
"""
from __future__ import annotations

import os

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# ============================================================
# 标题
# ============================================================
md(r"""
# word2vec 从零实现 · 理论 + PyTorch + 可视化

> 一份**互动教学** notebook：既把 word2vec 的理论讲透（直觉 → 数学 → 图解），
> 又用 PyTorch **手写实现** Skip-gram 与 CBOW 两种模型，并配大量图表把每个关键机制画出来。

**你会学到什么**

1. 为什么需要词向量，它比 one-hot 好在哪里（分布式表示的直觉与历史）。
2. word2vec 的两种模型结构：**Skip-gram**（中心词 → 上下文）与 **CBOW**（上下文 → 中心词）。
3. 两种让训练可行的关键技巧：**负采样（Negative Sampling）** 与 **层次 Softmax**，以及**高频词下采样**。
4. 用 PyTorch 亲手实现 SGNS（Skip-gram + 负采样）和 CBOW，并训练出真实的词向量。
5. 如何评估与**可视化**词向量：近义词检索、`king - man + woman ≈ queen` 类比、2D 语义地图、相似度热力图。

**阅读方式**：从上往下依次运行每个代码单元（`Shift+Enter`）。理论单元与代码单元交替出现，
建议先读懂上面的 markdown，再运行下面的代码看结果。

---

**前置准备**（在终端里，项目根目录下执行）：

```bash
uv sync                                   # 安装依赖（torch/matplotlib/sklearn/...）
uv run python download_data.py --subset 5 # 下载 text8，并生成 5MB 小语料
uv run jupyter lab                        # 启动，然后打开本文件
```
""")

# ============================================================
# 0. 环境与数据
# ============================================================
md(r"""
## 0 · 环境准备与设备选择

先导入依赖、固定随机种子（保证结果可复现），并选择计算设备：
Apple Silicon 上优先用 **MPS**（Metal GPU 加速），有 NVIDIA 卡则用 **CUDA**，否则回退到 **CPU**。
""")

code(r"""
import os, math, random, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ---- 随机种子：让每次运行结果一致，便于教学复现 ----
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---- 设备选择：MPS(苹果) > CUDA(英伟达) > CPU ----
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("PyTorch 版本 :", torch.__version__)
print("计算设备     :", device)

# ---- 统一的浅色绘图风格 + 语义配色（贯穿全篇） ----
# 让 matplotlib 能显示中文：从系统常见 CJK 字体里挑一个可用的
import matplotlib.font_manager as fm
_cjk_candidates = ["PingFang SC", "Arial Unicode MS", "Heiti TC",
                   "Songti SC", "STHeiti", "Microsoft YaHei", "SimHei"]
_installed = {f.name for f in fm.fontManager.ttflist}
_cjk = [f for f in _cjk_candidates if f in _installed]
if _cjk:
    plt.rcParams["font.sans-serif"] = _cjk + plt.rcParams.get("font.sans-serif", [])
    print("中文字体已启用:", _cjk[0])
else:
    print("⚠️ 未找到中文字体，图中中文可能显示为方框（不影响计算与英文标签）。")

plt.rcParams.update({
    "figure.figsize": (8, 5), "figure.dpi": 110, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.facecolor": "#ffffff", "figure.facecolor": "#ffffff",
    "axes.unicode_minus": False,   # 正常显示负号
})
COLORS = {
    "blue":   "#2563eb",  # 主色 / 中心词
    "green":  "#10b981",  # 正样本 / 接受
    "red":    "#ef4444",  # 负样本 / 丢弃
    "amber":  "#f59e0b",  # 上下文 / 高亮
    "purple": "#8b5cf6",  # 其它类别
    "gray":   "#9ca3af",
}
print("绘图风格已就绪。")
""")

md(r"""
### 加载语料

我们用 **text8**：它是 Matt Mahoney 从 2006 年英文维基百科清洗出来的纯文本（全小写、只有字母和空格），
是 word2vec 官方 demo 采用的经典入门语料。

为了让 notebook 端到端能在几分钟内跑完，默认读取 `data/text8_small`（5MB，约 90 万词）。
想要更好的词向量质量时，把 `CORPUS_PATH` 换成完整的 `data/text8`（100MB）再跑一遍即可。
""")

code(r"""
# 优先用小语料（快），不存在则回退到完整 text8
_here = os.getcwd()
CANDIDATES = [
    os.path.join(_here, "data", "text8_small"),
    os.path.join(_here, "data", "text8"),
]
CORPUS_PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)
assert CORPUS_PATH is not None, (
    "找不到语料！请先在终端运行：uv run python download_data.py --subset 5"
)

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()

# text8 是空格分隔的单词流，直接 split 即可得到 token 列表
tokens = raw_text.split()
print("语料文件 :", CORPUS_PATH)
print("总词数   : {:,}".format(len(tokens)))
print("去重词数 : {:,}".format(len(set(tokens))))
print("前 30 个词 :", " ".join(tokens[:30]))
""")

# ============================================================
# 1. 前置知识 + 术语速查
# ============================================================
md(r"""
## 1 · 前置知识与关键术语速查

在开始之前，先用大白话过一遍会用到的概念。看不懂公式没关系，先建立直觉。

| 术语 | 一句话解释 |
|------|-----------|
| **词向量 / 词嵌入（word embedding）** | 用一串实数（如 100 个数）表示一个词，让语义相近的词，向量也相近。 |
| **one-hot 向量** | 词表有 V 个词，就用长度 V 的向量表示一个词：只有它自己那一位是 1，其余全 0。稀疏、且任意两词都“正交”（毫无关系）。 |
| **分布式表示（distributed representation）** | 词义被“分摊”到向量的每一维上，而不是集中在某一位。这正是 word2vec 产出的东西。 |
| **上下文窗口（context window）** | 以某个词为中心，前后各取若干个词作为它的“上下文”。窗口大小是超参数（常用 5）。 |
| **点积（dot product）** | 两个向量对应位相乘再相加：$a\cdot b=\sum_i a_i b_i$。它越大，通常代表两个向量越“同向”。 |
| **余弦相似度（cosine similarity）** | 把点积除以两个向量的长度：$\cos(a,b)=\dfrac{a\cdot b}{\lVert a\rVert\,\lVert b\rVert}$，取值 $[-1,1]$，衡量“方向”是否一致。判断词相似度就用它。 |
| **sigmoid 函数** | $\sigma(x)=\dfrac{1}{1+e^{-x}}$，把任意实数压到 $(0,1)$，可解释为“概率”。负采样里用它做二分类。 |
| **softmax** | 把一组实数变成一组和为 1 的概率：$\text{softmax}(z)_i=\dfrac{e^{z_i}}{\sum_j e^{z_j}}$，分母对**输入的每一项**求和。在 word2vec 里用它预测词时，输入是“词表里每个词的打分”，分母就要遍历整个词表——这正是 §5.1 要解决的慢点。 |
| **负采样（negative sampling）** | 一种“抄近路”训练技巧：不去算全词表的 softmax，而是随机抽几个“反例”词，把问题变成简单的二分类。 |

> 💡 **核心直觉一句话**：word2vec 的信念是——*“一个词的含义由它周围经常出现的词决定”*
> （分布式假设，J. R. Firth, 1957: *“You shall know a word by the company it keeps.”*）。
> 于是我们让模型反复做“完形填空”，填得好的副产品，就是一套好用的词向量。
""")

# ============================================================
# 2. 直觉与历史
# ============================================================
md(r"""
## 2 · 直觉理解与历史背景

**问题**：计算机只认数字，怎么把“词”喂给模型？最朴素的办法是 one-hot——但它有两个致命缺点：

1. **维度灾难**：词表几万个词，每个词就是几万维的稀疏向量。
2. **词与词之间没有关系**：`cat` 和 `dog` 的 one-hot 向量点积是 0，和 `cat` 与 `airplane` 一样“毫不相关”。one-hot 编码里没有任何语义。

**word2vec 的想法**（Mikolov 等人，2013）：与其手工设计特征，不如让模型**自己从大量文本里学**。
做法是设计一个极简的“完形填空”任务：

- **Skip-gram**：给你中心词 `fox`，预测它周围会出现哪些词（`quick`、`brown`、`jumps`…）。
- **CBOW**：反过来，给你周围的词，预测中间缺的那个词。

模型为了把这个任务做好，被迫把“经常出现在相似上下文里的词”安排到相近的向量位置。
训练完成后，我们**不关心完形填空本身**，而是把学到的那张 `词 → 向量` 的查找表（embedding 矩阵）拿出来用。

**一点历史**：

- 分布式假设可追溯到 1950 年代的语言学（Harris 1954、Firth 1957）。
- 2003 年 Bengio 等人提出神经语言模型，首次联合学习词向量。
- 2013 年 Mikolov 等人在 Google 提出 **word2vec**：结构极简、训练极快，第一次让“在十亿级语料上训练词向量”变得平民化，并展示了著名的 `king - man + woman ≈ queen` 向量运算，引爆了整个词嵌入时代。
- 之后有 GloVe（Stanford, 2014）、fastText（Facebook, 2016，引入子词）等；再往后就是 ELMo / BERT 等上下文相关的嵌入。word2vec 是这一切的起点。
""")

# ============================================================
# 3. one-hot vs 分布式表示（图）
# ============================================================
md(r"""
## 3 · 从 one-hot 到分布式表示

下面这张图直观对比两种表示（**注意两个面板的宽度 = 向量维度**）：

- **左边 one-hot**：维度 = **词表大小 V**。词表有多少词，向量就有多少维，且**只有 1 个 1、其余全是 0**（极度稀疏）。词表一大，维度就爆炸。
- **右边分布式（word2vec）**：维度是一个**固定的小常数**（这里用 4，真实中常用 100~300），**与词表大小无关**。稠密、且语义相近的词数值也相近——`cat` 与 `dog` 明显比 `cat` 与 `car` 更像。
""")

code(r"""
# 用一个稍大的玩具词表，才能看出 one-hot 的“维度爆炸”
vocab_demo = ["cat", "dog", "car", "the", "run", "fish",
              "tree", "book", "sun", "sea", "red", "two"]
V_demo = len(vocab_demo)      # = 12，one-hot 就是 12 维
DENSE_DIM = 4                 # 分布式维度：固定小常数，与词表大小无关

# 左：one-hot = V×V 单位阵（每行只有 1 个 1）
onehot = np.eye(V_demo)

# 右：分布式向量（此处为示意，真实向量后面会训练出来）
rng = np.random.default_rng(0)
dense = rng.uniform(0, 1, size=(V_demo, DENSE_DIM)).round(2)
# 手工设定前 3 个词，让 cat/dog 相近、car 不同（维度含义示意：毛茸茸/交通工具/宠物/抽象）
dense[0] = [0.92, 0.05, 0.88, 0.30]   # cat
dense[1] = [0.85, 0.08, 0.90, 0.42]   # dog
dense[2] = [0.10, 0.95, 0.05, 0.60]   # car

# 关键：让两个面板的宽度正比于“列数=维度”，宽窄对比才真实
fig = plt.figure(figsize=(13, 4.2), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[V_demo, DENSE_DIM], wspace=0.2)
ax0 = fig.add_subplot(gs[0]); ax1 = fig.add_subplot(gs[1])

# --- 左：one-hot（宽、稀疏，只标出那个 1）---
ax0.imshow(onehot, cmap="Blues", aspect="auto", vmin=0, vmax=1)
ax0.set_yticks(range(V_demo)); ax0.set_yticklabels(vocab_demo)
ax0.set_xticks(range(V_demo)); ax0.set_xticklabels([f"d{i}" for i in range(V_demo)], fontsize=8)
ax0.set_title(f"one-hot：{V_demo} 维（= 词表大小 V，随词表膨胀）")
for i in range(V_demo):
    ax0.text(i, i, "1", ha="center", va="center", color="white", fontsize=9)
ax0.grid(False)

# --- 右：distributed（窄、稠密，标出所有值）---
ax1.imshow(dense, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
ax1.set_yticks(range(V_demo)); ax1.set_yticklabels(vocab_demo)
ax1.set_xticks(range(DENSE_DIM)); ax1.set_xticklabels([f"d{i}" for i in range(DENSE_DIM)])
ax1.set_title(f"distributed：{DENSE_DIM} 维（固定，与 V 无关）")
for i in range(V_demo):
    for j in range(DENSE_DIM):
        ax1.text(j, i, f"{dense[i, j]:.2f}", ha="center", va="center",
                 color="black", fontsize=8)
ax1.grid(False)

plt.show()

# 维度对比：把“爆炸”说清楚
print("这个玩具词表 V={0}：one-hot = {0} 维（每个向量含 {1} 个 0）；分布式 = {2} 维。".format(
    V_demo, V_demo - 1, DENSE_DIM))
print("真实场景 V=50000：one-hot = 50000 维（仅 1 个 1，其余 49999 个全 0）；word2vec 只需 100~300 维。")
print("→ one-hot 维度随词表线性爆炸且极度稀疏；分布式维度固定，与词表大小解耦。\n")

# 用余弦相似度量化“稠密表示能体现语义”这件事（只看 cat/dog/car）
def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
ci, di, ri = 0, 1, 2   # cat, dog, car 的行号
print("one-hot 下： cos(cat, dog) = {:.3f}  |  cos(cat, car) = {:.3f}"
      .format(cos(onehot[ci], onehot[di]), cos(onehot[ci], onehot[ri])))
print("分布式下： cos(cat, dog) = {:.3f}  |  cos(cat, car) = {:.3f}"
      .format(cos(dense[ci], dense[di]), cos(dense[ci], dense[ri])))
print("→ one-hot 里任何两词相似度都是 0.000；分布式表示里 cat 和 dog 明显更相似。")
""")

# ============================================================
# 4. 两种模型结构（图）
# ============================================================
md(r"""
## 4 · 两种模型结构：Skip-gram 与 CBOW

word2vec 有两种“完形填空”的方向，用同一句例句 *“the quick brown fox jumps”*（中心词 = `fox`，窗口 = 2）来看：

- **Skip-gram**：输入**中心词** `fox`，分别预测它周围的每个**上下文词** `the / quick / brown / jumps`。
  → 一个中心词产生多个 (中心, 上下文) 训练样本。**对低频词更友好，小语料上通常效果更好。**
- **CBOW**（Continuous Bag-of-Words）：输入**上下文词集合** `{the, quick, brown, jumps}`（求平均），预测**中心词** `fox`。
  → 上下文一次性喂入。**训练更快，对高频词的表示更平滑。**

下图把两种数据流画出来：
""")

code(r"""
def draw_arch(ax, mode):
    ctx = ["the", "quick", "brown", "jumps"]
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, text, color, w=1.7, h=0.7):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h, boxstyle="round,pad=0.02",
            fc=color, ec="#374151", lw=1.2, alpha=0.9))
        ax.text(x, y, text, ha="center", va="center", fontsize=10, color="white")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=14, color="#6b7280", lw=1.3))

    ys = np.linspace(1, 5, len(ctx))
    if mode == "skipgram":
        ax.set_title("Skip-gram：中心词 → 上下文", fontsize=12)
        box(2, 3, "fox\n(center)", COLORS["blue"])              # 输入
        box(5, 3, "projection", COLORS["gray"], w=2.0)          # 投影(embedding)
        for y, w in zip(ys, ctx):                               # 多个输出
            box(8.3, y, w, COLORS["amber"])
            arrow(6.0, 3, 7.45, y)
        arrow(2.85, 3, 4.0, 3)
    else:
        ax.set_title("CBOW：上下文 → 中心词", fontsize=12)
        for y, w in zip(ys, ctx):                               # 多个输入
            box(1.7, y, w, COLORS["amber"])
            arrow(2.55, y, 4.0, 3)
        box(5, 3, "average", COLORS["gray"], w=2.0)             # 求平均
        box(8.3, 3, "fox\n(center)", COLORS["blue"])            # 输出
        arrow(6.0, 3, 7.45, 3)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
draw_arch(axes[0], "skipgram")
draw_arch(axes[1], "cbow")
plt.tight_layout(); plt.show()
""")

# ============================================================
# 5. 训练技巧（负采样 / 层次softmax）+ 图
# ============================================================
md(r"""
## 5 · 训练技巧：为什么需要负采样？

### 5.1 朴素做法为什么慢

> 💡 **先建立直觉**：模型看到中心词 `fox`，想说出"它周围最可能是哪个词"。
> 最自然的做法是给词表里**每一个词**打一个分，再用 softmax 把这些分变成"和为 1 的概率"。
> 麻烦就在这个"每一个词"——词表有几万个，每走一步都要把所有词过一遍。

以 Skip-gram 为例，给定中心词 $w_I$，预测上下文词 $w_O$ 的概率用 **softmax**：

$$
p(w_O \mid w_I) = \frac{\exp\!\big(v'_{w_O}{}^{\top} v_{w_I}\big)}{\sum_{w=1}^{V}\exp\!\big(v'_{w}{}^{\top} v_{w_I}\big)}
$$

其中 $v_w$ 是词 $w$ 作为**中心词**的向量，$v'_w$ 是它作为**上下文词**的向量（所以每个词有两套向量，下面 5.2 会解释为什么）。
训练目标是最大化语料上所有 (中心, 上下文) 对的对数似然。

**问题在分母**：它要对**整个词表 $V$**（几万甚至上百万个词）求和。

**慢在哪，算笔账**：假设词表 $V=5\text{万}$、语料有 $10\text{亿}$ 个训练样本、训练 5 轮：

$$
5 \times 10^9 \times 5\times10^4 = 2.5\times10^{14}\ \text{次 exp 运算}
$$

每个样本每一步都要算 5 万次指数——这就是朴素 softmax **$O(V)$** 慢到没法用的原因。
下面的负采样把每步的 5 万次，直接砍到 **$K+1 \approx 6$ 次**。

#### 把公式里每个符号讲清楚（配具体例子）

还是用例句 `the quick brown fox jumps`，取中心词 $w_I=$ `fox`、要预测的上下文词 $w_O=$ `quick`。
公式 $p(w_O\mid w_I)=\dfrac{\exp(v'_{w_O}{}^{\top}v_{w_I})}{\sum_{w=1}^{V}\exp(v'_{w}{}^{\top}v_{w_I})}$ 里的符号，逐个对号入座：

| 符号 | 在例子里指什么 | 是什么 / 怎么来的 |
|------|----------------|-------------------|
| $V$ | 词表大小，比如 5 万 | 语料里所有去重词的个数 |
| $w_I$ | 中心词 = `fox` | Input word，模型的输入 |
| $w_O$ | 目标上下文词 = `quick` | Output word，希望模型预测出的词 |
| $v_{w_I}$ = $v_{\text{fox}}$ | `fox` 作为**中心词**的向量 | 从 **`in_emb` 矩阵**里，按 `fox` 的行号取出的那一行（一个 $D$ 维向量，如 100 维） |
| $v'_{w_O}$ = $v'_{\text{quick}}$ | `quick` 作为**上下文词**的向量 | 从 **`out_emb` 矩阵**里，按 `quick` 的行号取出的那一行 |
| $v'_{w}$（分母里的 $w$） | 词表里**每一个词**的上下文向量 | 遍历 `out_emb` 的全部 $V$ 行——这就是慢的根源 |
| $v'_{w_O}{}^{\top}v_{w_I}$ | 两个向量的**点积**（一个标量） | 衡量 `quick` 和 `fox` 有多"合得来"，越大越可能是邻居 |

**这两套向量从哪来、怎么变好的？**

- 一开始 `in_emb` 和 `out_emb` 都是**随机初始化**的矩阵，形状都是 $V\times D$（$V$ 行词、每行 $D$ 维）。此时点积是乱的，预测也是乱的。
- 训练时，每见到一对真实共现的 (`fox`, `quick`)，就通过反向传播**微调** $v_{\text{fox}}$ 和 $v'_{\text{quick}}$，让它们点积变大（§5.5 会看到梯度长什么样）。
- 反复在海量语料上更新后，这两个矩阵的每一行就成了"学出来的词向量"。训练结束我们通常取 `in_emb`（即 $v_w$）当作最终词向量。

> 一句话：$v_{w_I}$ 是"**查 `in_emb` 表得到的中心词那一行**"，$v'_{w_O}$ 是"**查 `out_emb` 表得到的上下文词那一行**"，
> 它们不是预先算好的，而是**训练过程中被一步步学出来的参数**。下面用代码把这个"查表 + 点积 + softmax"跑一遍。
""")

code(r"""
# ---- 用一个玩具例子，把 softmax 公式里的每个量“查表 + 算出来” ----
np.random.seed(0)
toy_vocab = ["the", "quick", "brown", "fox", "jumps"]
Vt, Dt = len(toy_vocab), 4          # 玩具：词表 5 个词，向量 4 维
w2i = {w: i for i, w in enumerate(toy_vocab)}

# 两套向量矩阵（真实中是随机初始化、再由训练更新；这里直接随机造一组演示）
in_emb  = np.random.randn(Vt, Dt).round(2)   # 中心词矩阵：第 i 行 = 词 i 的 v_w
out_emb = np.random.randn(Vt, Dt).round(2)   # 上下文词矩阵：第 i 行 = 词 i 的 v'_w

wI, wO = "fox", "quick"              # 中心词 / 目标上下文词
v_wI  = in_emb[w2i[wI]]              # 查 in_emb 表：fox 作为中心词的向量  v_{w_I}
v_wO  = out_emb[w2i[wO]]             # 查 out_emb 表：quick 作为上下文的向量 v'_{w_O}
print("v_{{w_I}}  = v_fox   (查 in_emb 第{}行) = {}".format(w2i[wI], v_wI))
print("v'_{{w_O}} = v'_quick (查 out_emb 第{}行) = {}".format(w2i[wO], v_wO))

# 分子：目标词的点积再取 exp
score_O = v_wO @ v_wI
print("\n分子里的点积 v'_quick · v_fox = {:.4f}  ->  exp = {:.4f}".format(score_O, np.exp(score_O)))

# 分母：对词表里“每一个词”的上下文向量都算一遍点积->exp，再求和
print("\n分母：遍历词表全部 {} 个词（这就是 O(V) 慢的地方）".format(Vt))
scores = out_emb @ v_wI             # 一次算出 fox 与所有词的点积，形状 [V]
for w, s in zip(toy_vocab, scores):
    print("   exp(v'_{:<6s} · v_fox) = exp({:+.4f}) = {:.4f}".format(w, s, np.exp(s)))
denom = np.exp(scores).sum()

# softmax 概率
p = np.exp(score_O) / denom
print("\np(quick | fox) = 分子 / 分母 = {:.4f} / {:.4f} = {:.4f}".format(
    np.exp(score_O), denom, p))
print("（词表越大，分母要加的项越多——负采样就是来省掉这个全表求和的）")
""")

md(r"""
### 5.2 为什么每个词要“两套向量”？

这是初学者最容易卡住的地方。word2vec 给每个词准备了两个向量：
作为**中心词**时用 $v_w$（`in_emb`），作为**上下文词**时用 $v'_w$（`out_emb`）。

**为什么不共用一套？** 如果只有一套向量 $v_w$，那么"一个词预测它自己"的得分就是 $v_w \cdot v_w = \lVert v_w\rVert^2$，
这个值**恒为正且很大**。可是在真实文本里，一个词紧挨着自己出现其实很罕见（`the the` 很少见）。
共用一套向量会逼着模型给"自己和自己"打高分，与语料事实矛盾，训练目标就拧巴了。
用两套向量把"我作为中心"和"我作为邻居"解耦，这个矛盾就消失了。

> 训练完通常**只取中心词向量 $v_w$（`in_emb`）** 当作最终词向量；
> 另一种常见做法是把两套相加或拼接。本教程取 `in_emb`。

### 5.3 负采样（Negative Sampling）

**核心思想**：与其“在全词表里选对的那个”（多分类），不如把问题改成一堆**二分类**：

- 真实出现的 (中心词, 上下文词) 是**正样本**，标签 = 1，希望它们的向量点积大。
- 随机抽 $K$ 个词当**负样本**（“噪声词”），标签 = 0，希望它们和中心词的点积小。

于是每个正样本的损失变成（$\sigma$ 是 sigmoid）：

$$
L = -\log \sigma\!\big(v'_{w_O}{}^{\top} v_{w_I}\big)\;-\;\sum_{k=1}^{K}\mathbb{E}_{w_k \sim P_n(w)}\Big[\log \sigma\!\big(-\,v'_{w_k}{}^{\top} v_{w_I}\big)\Big]
$$

- 第一项：让正样本点积大 → $\sigma(\cdot)\to 1$ → $-\log\sigma \to 0$。
- 第二项：让负样本点积小（甚至为负）→ $\sigma(-\cdot)\to 1$ → 损失小。

复杂度从 $O(V)$ 降到 $O(K)$（$K$ 通常取 5~20），**这是 word2vec 能在海量语料上飞速训练的关键。**

### 5.4 手把手算一步（带真实数字）

光看公式容易发懵，我们用一句最小语料把负采样**算到底**：

> 语料： `the cat sat on the mat`　词表 = `[the, cat, sat, on, mat]`

取中心词 `sat`，它的一个真实邻居是 `cat`（**正样本**）。再随机抽 2 个词当**负样本**：`the`、`mat`。
于是这一步只需要处理 **1 个正样本 + 2 个负样本 = 3 个词**，而不是整个词表。

给每个词一个 3 维小向量（这里是编造的，只为演示），下面的代码把
点积 → sigmoid → 每一项损失 → 总损失 **逐个数字打印出来**，你可以对着公式核对。
""")

code(r"""
# ---- 一个玩具例子：手算负采样的一步 ----
# 中心词向量 v_c（sat 作为“中心词”）
v_c = np.array([0.5, -0.2,  0.8])

# 上下文词向量 v'（每个词作为“上下文词”时的向量）
v_pos = np.array([0.4,  0.1,  0.7])   # cat  —— 正样本(真实邻居)，希望点积大
v_neg1 = np.array([-0.3, 0.6, -0.2])  # the  —— 负样本(随机抽的)，希望点积小
v_neg2 = np.array([0.2, -0.5,  0.1])  # mat  —— 负样本

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# 1) 正样本：点积 -> sigmoid -> 损失项 -log σ(v_c·v_pos)
dot_pos = v_c @ v_pos
loss_pos = -np.log(sigmoid(dot_pos))
print("正样本 (sat, cat):")
print("  点积 v_c·v_pos      = {:+.4f}".format(dot_pos))
print("  σ(点积)             = {:.4f}   (越接近 1 越好)".format(sigmoid(dot_pos)))
print("  损失项 -log σ       = {:.4f}\n".format(loss_pos))

# 2) 负样本：点积 -> sigmoid(-点积) -> 损失项 -log σ(-v_c·v_neg)
loss_neg_total = 0.0
for name, v_neg in [("the", v_neg1), ("mat", v_neg2)]:
    dot_neg = v_c @ v_neg
    li = -np.log(sigmoid(-dot_neg))
    loss_neg_total += li
    print("负样本 (sat, {}):".format(name))
    print("  点积 v_c·v_neg      = {:+.4f}".format(dot_neg))
    print("  σ(-点积)            = {:.4f}   (越接近 1 越好，即点积越小越好)".format(sigmoid(-dot_neg)))
    print("  损失项 -log σ(-·)   = {:.4f}\n".format(li))

# 3) 总损失 = 正样本项 + 所有负样本项
L = loss_pos + loss_neg_total
print("=" * 40)
print("这一步的总损失 L = {:.4f} + {:.4f} = {:.4f}".format(
    loss_pos, loss_neg_total, L))
print("训练要做的：调整这 4 个向量，让 L 变小 —— 也就是把 (sat,cat) 拉近、把 (sat,the)/(sat,mat) 推远。")
""")

md(r"""
### 5.5 训练到底怎么“拉近 / 推远”？—— 梯度长什么样

上面算出了损失，但**向量是怎么被更新的**？这一步常被自动求导（autograd）藏起来，
其实 SGNS 的梯度形式非常漂亮。对损失 $L$ 关于中心词向量 $v_c$ 求导，可以证明：

$$
\frac{\partial L}{\partial v_c}
= \big(\sigma(v_c\cdot v_o) - 1\big)\,v_o \;+\; \sum_{k=1}^{K}\big(\sigma(v_c\cdot v_{n_k}) - 0\big)\,v_{n_k}
$$

统一成一句话：对每个样本，令**标签** $t$（正样本 $t=1$，负样本 $t=0$），则

$$
\text{梯度贡献} = \big(\sigma(v_c\cdot v_{\text{词}}) - t\big)\cdot v_{\text{词}}
$$

$\big(\sigma-t\big)$ 就是**“预测概率 − 真实标签”这个误差**：

- 正样本($t=1$)：若 $\sigma$ 还没到 1，$(\sigma-1)<0$，梯度下降会把 $v_c$ **朝 $v_o$ 方向拉**。
- 负样本($t=0$)：若 $\sigma$ 还没到 0，$(\sigma-0)>0$，梯度下降会把 $v_c$ **朝远离 $v_{n_k}$ 方向推**。

**误差越大，拉/推的力度越大**——这正是"拉近正样本、推远负样本"的数学来源。

#### 为什么"符号"就能决定拉近还是推远？—— 从更新公式看

上面只给了梯度，"拉/推"其实藏在**梯度下降的更新公式**里。更新规则是（$\eta$ 是学习率）：

$$
v_c \;\leftarrow\; v_c - \eta\,\frac{\partial L}{\partial v_c}
$$

**注意那个减号**：参数沿梯度的**反方向**移动。把上面的梯度贡献代入，并把减号乘进括号：

$$
v_c \;\leftarrow\; v_c - \eta\big(\sigma(v_c\cdot v_{\text{词}})-t\big)v_{\text{词}}
\;=\; v_c + \eta\underbrace{\big(t-\sigma(v_c\cdot v_{\text{词}})\big)}_{\text{系数 }\alpha}\,v_{\text{词}}
$$

于是每步更新，本质是在 $v_c$ 上**加一个 $v_{\text{词}}$ 方向的向量**，加多少由系数 $\alpha=t-\sigma$ 的**正负**决定：

**① 正样本**（$t=1$，词 = 真实上下文 $v_o$）：

$$
\alpha = 1-\sigma(v_c\cdot v_o) > 0 \quad(\text{因为 }\sigma<1)
\;\Rightarrow\; v_c \leftarrow v_c + \eta\,\alpha\,v_o
$$

加了一个**正倍**的 $v_o$ → $v_c$ 朝 $v_o$ 挪 → 下次点积更大、$\sigma$ 更接近 1。这就是**拉近**。

**② 负样本**（$t=0$，词 = 噪声词 $v_{n_k}$）：

$$
\alpha = 0-\sigma(v_c\cdot v_{n_k}) < 0 \quad(\text{因为 }\sigma>0)
\;\Rightarrow\; v_c \leftarrow v_c + \eta\,\alpha\,v_{n_k}
$$

加了一个**负倍**的 $v_{n_k}$（= 减去一个 $v_{n_k}$ 分量）→ $v_c$ 背离 $v_{n_k}$ → 下次点积更小、$\sigma$ 更接近 0。这就是**推远**。

| | 标签 $t$ | 系数 $\alpha=t-\sigma$ | 更新 $v_c \mathrel{+}= \eta\,\alpha\,v_{\text{词}}$ | 效果 |
|---|:---:|:---:|:---:|:---:|
| 正样本 $v_o$ | 1 | $1-\sigma > 0$ | 加**正**倍 $v_o$ | 朝 $v_o$ **拉近** |
| 负样本 $v_{n_k}$ | 0 | $-\sigma < 0$ | 加**负**倍 $v_{n_k}$ | 远离 $v_{n_k}$ **推开** |

> 更新方向永远是 $\pm v_{\text{词}}$，符号由"标签 − 预测"这个误差 $\alpha$ 决定：真实邻居误差为正→朝它走，噪声词误差为负→背它走。
> （对称地，同一步也在更新 $v_o$、$v_{n_k}$ 朝/背 $v_c$ 挪，所以是双方相向靠拢 / 相背远离。）

下面用 PyTorch autograd 验证手推的梯度公式和自动求导**完全一致**，
再**真的走一步**梯度下降：拆开看每一项都在拉近/推远，合起来则整体损失下降。
""")

code(r"""
import torch

# 用上面同一组向量，验证“手推梯度 == autograd 梯度”
t_vc = torch.tensor([0.5, -0.2, 0.8], requires_grad=True)
t_pos = torch.tensor([0.4, 0.1, 0.7])
t_negs = torch.stack([torch.tensor([-0.3, 0.6, -0.2]),
                      torch.tensor([0.2, -0.5, 0.1])])   # [2, 3]

# --- autograd 路线 ---
pos_term = -torch.log(torch.sigmoid(t_vc @ t_pos))
neg_term = -torch.log(torch.sigmoid(-(t_negs @ t_vc))).sum()
loss = pos_term + neg_term
loss.backward()
auto_grad = t_vc.grad.numpy()

# --- 手推公式路线：sum (σ(v_c·v) - t) * v ---
def sig(x): return 1.0 / (1.0 + np.exp(-x))
vc = np.array([0.5, -0.2, 0.8]); vp = np.array([0.4, 0.1, 0.7])
vn = np.array([[-0.3, 0.6, -0.2], [0.2, -0.5, 0.1]])
manual_grad = (sig(vc @ vp) - 1.0) * vp                      # 正样本 t=1
for v in vn:
    manual_grad += (sig(vc @ v) - 0.0) * v                   # 负样本 t=0

print("autograd 梯度 ∂L/∂v_c :", np.round(auto_grad, 6))
print("手推公式  梯度 ∂L/∂v_c :", np.round(manual_grad, 6))
print("两者是否一致          :", np.allclose(auto_grad, manual_grad, atol=1e-6))
print("\n结论：PyTorch 自动求导，做的就是 (σ - 标签)·v 这件事——只是替你算好了。")

# ---- 真的走一步梯度下降，验证“正样本点积↑、负样本点积↓” ----
# 只更新 v_c，让“拉近/推远”干净可见（真实训练里 v_o、v_n 也同步更新）
eta = 1.0                                  # 学习率（放大以便一眼看出变化）
grad_pos_only = (sig(vc @ vp) - 1.0) * vp  # 仅正样本项对 v_c 的梯度
vc_pos = vc - eta * grad_pos_only
print("\n" + "=" * 50)
print("只用正样本更新 v_c（η={}）：".format(eta))
print("  正样本 v_c·v_o :  {:+.4f}  →  {:+.4f}   （变大 = 拉近 ✓）".format(
    vc @ vp, vc_pos @ vp))

for name, v in zip(["the", "mat"], vn):
    grad_neg_only = (sig(vc @ v) - 0.0) * v   # 仅该负样本项对 v_c 的梯度
    vc_neg = vc - eta * grad_neg_only
    print("只用负样本 '{}' 更新 v_c：  v_c·v_{} :  {:+.4f}  →  {:+.4f}   （变小 = 推远 ✓）".format(
        name, name, vc @ v, vc_neg @ v))

# 全部样本一起更新（正 + 两个负）——真实的一步
vc_all = vc - eta * manual_grad
print("\n正+负一起更新（真实一步）后各点积：")
print("  正样本 v_c·v_o    :  {:+.4f}  →  {:+.4f}".format(vc @ vp, vc_all @ vp))
for name, v in zip(["the", "mat"], vn):
    print("  负样本 v_c·v_{:<4s}:  {:+.4f}  →  {:+.4f}".format(name, vc @ v, vc_all @ v))
print("→ 拆开看：每一项都朝正确方向（拉近/推远）；合在一起时，v_c 是各方向的‘合力’，")
print("  个别负样本点积可能因与正样本方向耦合而未必下降——但整体损失 L 一定下降。")

# 佐证：整体损失确实下降
def sgns_loss(vc_):
    l = -np.log(sig(vc_ @ vp))
    for v in vn: l += -np.log(sig(-(vc_ @ v)))
    return l
print("  验证：L  {:.4f}  →  {:.4f}   （下降 ✓）".format(sgns_loss(vc), sgns_loss(vc_all)))
""")

md(r"""
### 5.6 负样本从哪来？—— $3/4$ 次方分布

负样本不是均匀乱抽，而是按词频的 **$3/4$ 次方** 抽（Mikolov 的经验技巧）：

$$
P_n(w) = \frac{U(w)^{3/4}}{\sum_{w'} U(w')^{3/4}}
$$

其中 $U(w)$ 是词 $w$ 的频次。$3/4$ 次方的作用是**压高频、抬低频**：让 `the/of` 这种超高频词别被抽得太狠，也给低频词一点出场机会。

### 5.7 另一条路：层次 Softmax（Hierarchical Softmax）

负采样不是唯一的加速办法。**层次 softmax** 换了个思路：把所有词放到一棵**哈夫曼树**的叶子上
（高频词离根近、低频词离根远），预测一个词 = 预测“从根走到该叶子”的一串**左/右选择**。

- 每个**内部节点**是一个二分类器，用 sigmoid 判断"往左还是往右"。
- 预测某个词的概率 = 沿途每一步选择概率的**连乘**。
- 一个词的树深约 $\log_2 V$，所以每步只需算 $O(\log V)$ 个节点，而不是全词表 $O(V)$。

下图画一棵 4 个词的哈夫曼树，并算出走到某个叶子的概率：
""")

code(r"""
# ---- 层次 softmax：哈夫曼树 + 一条路径的概率算例 ----
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)

# 节点坐标：n0 根，n1/n2 内部节点；叶子 = 4 个词
nodes = {
    "n0": (5, 5), "n1": (3, 3.2), "n2": (7, 3.2),
    "the": (1.6, 1.3), "cat": (4.2, 1.3), "sat": (5.8, 1.3), "mat": (8.4, 1.3),
}
edges = [("n0", "n1", "左"), ("n0", "n2", "右"),
         ("n1", "the", "左"), ("n1", "cat", "右"),
         ("n2", "sat", "左"), ("n2", "mat", "右")]

for a, b, lab in edges:
    (x1, y1), (x2, y2) = nodes[a], nodes[b]
    on_path = (a, b) in [("n0", "n2"), ("n2", "sat")]   # 高亮 root→sat 这条路径
    ax.plot([x1, x2], [y1, y2],
            color=COLORS["red"] if on_path else COLORS["gray"],
            lw=2.4 if on_path else 1.2, zorder=1)
    ax.text((x1+x2)/2 + 0.15, (y1+y2)/2, lab, fontsize=9,
            color=COLORS["red"] if on_path else "#6b7280")

for name, (x, y) in nodes.items():
    is_leaf = name in ("the", "cat", "sat", "mat")
    ax.add_patch(mpatches.FancyBboxPatch(
        (x-0.55, y-0.28), 1.1, 0.56, boxstyle="round,pad=0.02",
        fc=COLORS["amber"] if is_leaf else COLORS["blue"],
        ec="#374151", alpha=0.9, zorder=2))
    ax.text(x, y, name, ha="center", va="center", color="white",
            fontsize=10, zorder=3)

ax.set_title("层次 Softmax：哈夫曼树（红色为 root → sat 的路径）")
plt.tight_layout(); plt.show()

# 路径概率算例：走到 sat = 在 n0 选“右”，再在 n2 选“左”
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
score_n0 = 0.8    # 内部节点 n0 打分（示意）
score_n2 = -0.4   # 内部节点 n2 打分（示意）
p_right_n0 = sigmoid(score_n0)         # 在 n0 往右的概率
p_left_n2 = 1 - sigmoid(score_n2)      # 在 n2 往左 = 1 - 往右
p_sat = p_right_n0 * p_left_n2
print("P(sat) = P(n0向右) × P(n2向左)")
print("       = σ({:.1f}) × (1-σ({:.1f}))".format(score_n0, score_n2))
print("       = {:.4f} × {:.4f} = {:.4f}".format(p_right_n0, p_left_n2, p_sat))
print("只算了 2 个内部节点（≈log2(4)），而不是遍历 4 个词——词表越大，省得越多。")
""")

md(r"""
两种技巧对比：**层次 softmax 对低频词更友好，但要建树、实现更复杂；负采样更直观、也是 gensim 默认。**
本教程主线用负采样。

下面两张图分别画出 **sigmoid** 和 **$3/4$ 次方分布** 的效果：
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

# ---- 左：sigmoid ----
x = np.linspace(-8, 8, 400)
sig = 1 / (1 + np.exp(-x))
axes[0].plot(x, sig, color=COLORS["blue"], lw=2.2)
axes[0].axhline(0.5, ls="--", color=COLORS["gray"], lw=1)
axes[0].axvline(0, ls="--", color=COLORS["gray"], lw=1)
axes[0].scatter([4, -4], [1/(1+np.exp(-4)), 1/(1+np.exp(4))],
                color=[COLORS["green"], COLORS["red"]], zorder=5, s=60)
axes[0].annotate("正样本希望点积大\nσ→1, 损失→0", (4, 0.98), (1.0, 0.6),
                 fontsize=9, color=COLORS["green"],
                 arrowprops=dict(arrowstyle="->", color=COLORS["green"]))
axes[0].annotate("负样本希望点积小\nσ(−·)→1", (-4, 0.02), (-7.8, 0.28),
                 fontsize=9, color=COLORS["red"],
                 arrowprops=dict(arrowstyle="->", color=COLORS["red"]))
axes[0].set_title(r"sigmoid  $\sigma(x)=1/(1+e^{-x})$")
axes[0].set_xlabel("dot product  v'·v"); axes[0].set_ylabel("σ")

# ---- 右：freq vs freq^0.75（示意频次） ----
demo_freq = np.array([1000, 500, 200, 80, 30, 10, 4, 2], dtype=float)
labels = [f"w{i}" for i in range(len(demo_freq))]
p_raw = demo_freq / demo_freq.sum()
p_075 = demo_freq**0.75 / (demo_freq**0.75).sum()
xi = np.arange(len(demo_freq)); wbar = 0.4
axes[1].bar(xi - wbar/2, p_raw, wbar, label=r"$U(w)$ 原始频率",
            color=COLORS["red"], alpha=0.85)
axes[1].bar(xi + wbar/2, p_075, wbar, label=r"$U(w)^{3/4}$ 归一化",
            color=COLORS["blue"], alpha=0.85)
axes[1].set_xticks(xi); axes[1].set_xticklabels(labels)
axes[1].set_title("负采样分布：3/4 次方压高频、抬低频")
axes[1].set_ylabel("采样概率"); axes[1].legend()

plt.tight_layout(); plt.show()
print("注意右图：高频词 w0 的采样概率被压低，低频词 w5~w7 被抬高——这正是 3/4 次方的效果。")
""")

# ============================================================
# 6. 数据预处理 + 图
# ============================================================
md(r"""
## 6 · 数据预处理（附 4 张图）

真正开始训练前要做四件事，每件都对应 word2vec 的一个设计点：

1. **建词表**：统计词频，丢掉出现次数 `< min_count` 的稀有词（噪声大、样本少，学不好）。
2. **高频词下采样（subsampling）**：`the/of/and` 这类词满篇都是，却几乎不携带语义。按下面的概率**随机丢弃**它们，
   既加速训练又提升质量。保留概率（gensim 风格，$t$ 为阈值，$f(w)$ 为词频比例）：
   $$P_{\text{keep}}(w) = \min\!\Big(1,\ \sqrt{\tfrac{t}{f(w)}}\Big)$$
   > 📌 **和原论文的差异**：Mikolov 原论文（2013b）的公式是 $P_{\text{keep}}(w)=\big(\sqrt{f(w)/t}+1\big)\cdot t/f(w)$。
   > 二者形状相近（都是"越高频越易丢"），但**不完全相同**。本教程采用更简洁、也被 gensim 使用的
   > $\sqrt{t/f(w)}$ 版本；对着原论文看时不必困惑。
3. **构建负采样分布**：即上一节的 $U(w)^{3/4}$，预先算好累积分布，训练时 $O(1)$ 抽负样本。
4. **滑动窗口生成样本**：用**动态窗口**（每个中心词随机取 $1\sim\text{window}$ 的半径，近的词权重天然更高）。
""")

code(r"""
# ---------- 超参数（可调）----------
MIN_COUNT   = 5       # 词频阈值：低于它的词直接丢弃
SUBSAMPLE_T = 1e-3    # 下采样阈值 t
WINDOW      = 5       # 上下文窗口最大半径

# ---------- 1) 建词表 ----------
counter = Counter(tokens)
vocab = [w for w, c in counter.most_common() if c >= MIN_COUNT]
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = vocab
V = len(vocab)
print("原始去重词数 : {:,}".format(len(counter)))
print("过滤后词表 V : {:,}  (min_count={})".format(V, MIN_COUNT))

# 把语料转成 id 序列（丢掉不在词表里的稀有词）
corpus_ids = np.array([word2idx[w] for w in tokens if w in word2idx], dtype=np.int64)
print("有效 token 数: {:,}".format(len(corpus_ids)))

# 各词频次（按 id 排列）
freqs = np.array([counter[w] for w in vocab], dtype=np.float64)
freq_ratio = freqs / freqs.sum()

# ---------- 2) 计算下采样保留概率 ----------
keep_prob = np.minimum(1.0, np.sqrt(SUBSAMPLE_T / freq_ratio))

# ---------- 3) 负采样分布（freq^0.75） ----------
neg_weights = freqs ** 0.75
neg_prob = neg_weights / neg_weights.sum()
neg_cumsum = np.cumsum(neg_prob)          # 累积分布，供 searchsorted O(log V) 抽样

def sample_negatives(n):
    '''按 3/4 次方分布抽 n 个负样本词 id（向量化，跨设备安全）。'''
    r = np.random.rand(n)
    return np.searchsorted(neg_cumsum, r)

print("\n最高频的 8 个词及其保留概率：")
for i in range(8):
    print("  {:<8s} freq={:>7d}  keep_prob={:.4f}".format(
        idx2word[i], int(freqs[i]), keep_prob[i]))
""")

md(r"""
### 先理解 Zipf 定律：为什么要用对数坐标

自然语言的词频极度不均：**少数词超高频，绝大多数词很罕见**。下面把同一组词频用两种坐标画出来，
体会为什么后面统一用 **log-log（双对数）** 坐标。
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
fs = np.sort(freqs)[::-1]                 # 频次从高到低排序
rk = np.arange(1, len(fs) + 1)            # 排名 1,2,3,...

# --- 左：线性坐标 —— 一个极端的 “L 形” ---
axes[0].plot(rk, fs, color=COLORS["blue"], lw=1.6)
axes[0].set_title("(1a) 线性坐标：freq vs rank（L 形，尾部糊成一条线）")
axes[0].set_xlabel("排名 rank"); axes[0].set_ylabel("频次 freq")
for i in range(4):                        # 标出最高频的几个词
    axes[0].annotate(idx2word[i], (rk[i], fs[i]),
                     xytext=(30, -6*i - 2), textcoords="offset points", fontsize=9,
                     arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=0.8))

# --- 右：log-log 坐标 —— 近似一条直线（幂律的标志）---
axes[1].loglog(rk, fs, color=COLORS["red"], lw=1.6)
axes[1].set_title("(1b) log-log 坐标：近似直线（头尾细节兼顾）")
axes[1].set_xlabel("排名 rank (log)"); axes[1].set_ylabel("频次 freq (log)")
axes[1].grid(alpha=0.3, which="both")

plt.tight_layout(); plt.show()

# 打印几个分位，佐证“头部极端、尾部漫长”
print("排名 → 频次（同一组数据的几个采样点）：")
for r in [1, 2, 5, 10, 100, 1000, 5000, len(fs)]:
    if r <= len(fs):
        print("  rank {:>6d}: freq {:>6d}  ({})".format(r, int(fs[r-1]), idx2word[r-1]))
print("\n线性图：只看得见 the/of 等几个巨头，rank>100 后全压在 0 附近、分辨不出。")
print("log-log 图：跨 4 个数量级的头尾都清晰，且近似直线 → 印证 Zipf 幂律 freq∝1/rank^s。")
""")

md(r"""
下面 4 张图把预处理的每一步“看得见”（其中第 (1) 张沿用上面的 log-log 坐标）：
""")

code(r"""
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# (1) Zipf：词频排名 vs 频次（log-log 近似一条直线，这是自然语言的普遍规律）
rank = np.arange(1, len(freqs) + 1)
axes[0, 0].loglog(rank, np.sort(freqs)[::-1], color=COLORS["blue"], lw=1.6)
axes[0, 0].set_title("(1) Zipf 定律（log-log）：少数词占据绝大多数出现次数")
axes[0, 0].set_xlabel("词频排名 (log)"); axes[0, 0].set_ylabel("频次 (log)")

# (2) 下采样保留概率 vs 词频比例：越高频越容易被丢
order = np.argsort(freq_ratio)
axes[0, 1].plot(freq_ratio[order], keep_prob[order], color=COLORS["red"], lw=2)
axes[0, 1].set_xscale("log")
axes[0, 1].set_title("(2) 下采样：词越高频，保留概率越低")
axes[0, 1].set_xlabel("词频比例 f(w) (log)"); axes[0, 1].set_ylabel("保留概率")

# (3) 负采样分布：原始 vs 3/4 次方（取最高频 15 个词看差异）
topn = 15
xi = np.arange(topn); wbar = 0.4
axes[1, 0].bar(xi - wbar/2, freq_ratio[:topn], wbar,
               label=r"$U(w)$", color=COLORS["red"], alpha=0.85)
axes[1, 0].bar(xi + wbar/2, neg_prob[:topn], wbar,
               label=r"$U(w)^{3/4}$", color=COLORS["blue"], alpha=0.85)
axes[1, 0].set_xticks(xi); axes[1, 0].set_xticklabels(
    [idx2word[i] for i in range(topn)], rotation=60, ha="right", fontsize=8)
axes[1, 0].set_title("(3) 负采样分布（Top-15 高频词）"); axes[1, 0].legend()

# (4) 滑动窗口示意
axes[1, 1].axis("off")
axes[1, 1].set_title("(4) 滑动窗口生成 (中心, 上下文) 对")
sent = ["the", "quick", "brown", "fox", "jumps", "over"]
center_i, win = 3, 2
for i, w in enumerate(sent):
    if i == center_i:
        c = COLORS["blue"]
    elif abs(i - center_i) <= win:
        c = COLORS["amber"]
    else:
        c = COLORS["gray"]
    axes[1, 1].add_patch(mpatches.FancyBboxPatch(
        (i * 1.5 + 0.3, 1.3), 1.3, 0.7, boxstyle="round,pad=0.02",
        fc=c, ec="#374151", alpha=0.9))
    axes[1, 1].text(i * 1.5 + 0.95, 1.65, w, ha="center", va="center",
                    color="white", fontsize=10)
axes[1, 1].text(center_i * 1.5 + 0.95, 2.3, "center", ha="center",
                color=COLORS["blue"], fontsize=9)
axes[1, 1].text(0.3, 0.7, "窗口内(amber)→与中心词配成正样本；蓝=中心词；灰=窗口外",
                fontsize=9, color="#374151")
axes[1, 1].set_xlim(0, 9.5); axes[1, 1].set_ylim(0.4, 2.6)

plt.tight_layout(); plt.show()
""")

md(r"""
### 应用下采样并生成训练样本

现在真正把上面的规则用到语料上：先按 `keep_prob` 随机丢弃高频词，再用**动态窗口**生成 (中心词, 上下文词) 对。
""")

code(r"""
# ---------- 下采样：按 keep_prob 决定每个 token 是否保留 ----------
rng = np.random.rand(len(corpus_ids))
kept_mask = rng < keep_prob[corpus_ids]
subsampled = corpus_ids[kept_mask]
print("下采样前 : {:,} tokens".format(len(corpus_ids)))
print("下采样后 : {:,} tokens  (丢弃了 {:.1%})".format(
    len(subsampled), 1 - len(subsampled) / len(corpus_ids)))

# ---------- 生成 Skip-gram 正样本对 (center, context) ----------
def build_skipgram_pairs(seq, window):
    centers, contexts = [], []
    n = len(seq)
    for i in range(n):
        # 动态窗口：近处的词更常被选中，等价于给近邻更高权重
        w = np.random.randint(1, window + 1)
        lo, hi = max(0, i - w), min(n, i + w + 1)
        for j in range(lo, hi):
            if j != i:
                centers.append(seq[i]); contexts.append(seq[j])
    return np.array(centers, dtype=np.int64), np.array(contexts, dtype=np.int64)

t0 = time.time()
sg_center, sg_context = build_skipgram_pairs(subsampled, WINDOW)
print("Skip-gram 训练对: {:,}  (耗时 {:.1f}s)".format(len(sg_center), time.time() - t0))
""")

# ============================================================
# 7. 手写 SGNS
# ============================================================
md(r"""
## 7 · 手写实现 ①：Skip-gram + 负采样（SGNS）

模型只有**两个 Embedding 矩阵**：

- `in_emb`：中心词向量 $v_{w_I}$（训练完通常拿它当最终词向量）。
- `out_emb`：上下文词向量 $v'_{w_O}$。

前向计算就是把第 5 节的负采样损失写成张量运算：

$$
L = -\log\sigma(v_c\cdot v_o)\;-\;\sum_{k=1}^{K}\log\sigma(-\,v_c\cdot v_{n_k})
$$
""")

code(r"""
class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, embed_dim)   # 中心词向量
        self.out_emb = nn.Embedding(vocab_size, embed_dim)  # 上下文词向量
        # 初始化：中心词均匀小随机，上下文置零（word2vec 的常见做法）
        nn.init.uniform_(self.in_emb.weight, -0.5 / embed_dim, 0.5 / embed_dim)
        nn.init.zeros_(self.out_emb.weight)

    def forward(self, center, context, negatives):
        # center:[B]  context:[B]  negatives:[B, K]
        v_c = self.in_emb(center)             # [B, D]
        v_o = self.out_emb(context)           # [B, D]
        v_n = self.out_emb(negatives)         # [B, K, D]

        # 正样本：点积 -> logsigmoid，希望大
        pos = torch.sum(v_c * v_o, dim=1)                     # [B]
        pos_loss = F.logsigmoid(pos)                          # [B]

        # 负样本：批量矩阵乘 [B,K,D] x [B,D,1] -> [B,K]，希望小
        neg = torch.bmm(v_n, v_c.unsqueeze(2)).squeeze(2)     # [B, K]
        neg_loss = F.logsigmoid(-neg).sum(dim=1)              # [B]

        # -(pos+neg) 是单个样本的损失；.mean() 对 batch 内 B 个样本取平均，
        # 让梯度尺度与 batch size 解耦（调 batch 不必重调学习率）
        return -(pos_loss + neg_loss).mean()

print(SkipGramNS(V, 100))
""")

md(r"""
### 通用训练循环

写一个通用的 `train_epochs`，SGNS 和后面的 CBOW 都能复用：每步抽 `NEG_K` 个负样本，前向算损失、反向更新。
""")

code(r"""
# ---------- 训练超参数（可调）----------
EMBED_DIM  = 100    # 词向量维度 D：每个词用多少个数表示（常用 100~300）
NEG_K      = 5      # 每个正样本配几个负样本 K（小语料可调大到 10~20）
BATCH_SIZE = 2048   # 每个 mini-batch 放多少个样本 B（越大越快但越吃内存）
EPOCHS     = 5      # 整个训练集过几遍
LR         = 5e-3   # 学习率：每步参数更新的步长

def iterate_minibatches(n, batch_size, shuffle=True):
    '''把 0..n-1 这 n 个样本下标切成一批批返回。
    shuffle=True 时每个 epoch 先打乱顺序——避免模型按固定次序看样本而学出偏差。'''
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for s in range(0, n, batch_size):
        yield idx[s:s + batch_size]          # 每次 yield 一批下标（最后一批可能不足 batch_size）

def train_skipgram(model, centers, contexts, epochs=EPOCHS,
                   batch_size=BATCH_SIZE, lr=LR, neg_k=NEG_K):
    model.to(device)                          # 把模型参数搬到计算设备（MPS/CUDA/CPU）
    opt = torch.optim.Adam(model.parameters(), lr=lr)   # Adam 优化器：自适应学习率，收敛稳
    n = len(centers)                          # 训练样本（中心-上下文对）总数
    history = []                              # 记录每个 epoch 的平均 loss，用于画曲线
    for ep in range(1, epochs + 1):
        t0, total, nb = time.time(), 0.0, 0   # 计时、累计 loss、累计 batch 数
        for bi in iterate_minibatches(n, batch_size):
            # bi 是本批样本的下标数组，len(bi) = 实际 batch 大小 B
            # ---- 取出这一批的中心词 / 正样本上下文词，并搬到设备上 ----
            c = torch.from_numpy(centers[bi]).to(device)   # [B]
            o = torch.from_numpy(contexts[bi]).to(device)  # [B]
            # ---- 为这一批的每个样本各抽 neg_k 个负样本，整形成 [B, K] ----
            neg = torch.from_numpy(
                sample_negatives(len(bi) * neg_k).reshape(len(bi), neg_k)
            ).to(device)                                    # [B, K]
            # ---- 前向：算出这一批的平均损失（标量）----
            loss = model(c, o, neg)
            # ---- 反向传播三步曲（PyTorch 固定套路）----
            opt.zero_grad()   # 1) 清空上一步残留的梯度（否则会累加）
            loss.backward()   # 2) 反向传播，自动算出每个参数的梯度
            opt.step()        # 3) 按梯度更新参数（拉近正样本、推远负样本）
            total += loss.item(); nb += 1      # loss.item() 取出标量值累加
        avg = total / nb                       # 本 epoch 所有 batch 的平均 loss
        history.append(avg)
        print("  epoch {:>2d}/{}  loss={:.4f}  ({:.1f}s)".format(
            ep, epochs, avg, time.time() - t0))
    return history

print("开始训练 SGNS ...")
sg_model = SkipGramNS(V, EMBED_DIM)            # 用词表大小 V 和维度 D 初始化模型
t0 = time.time()
sg_history = train_skipgram(sg_model, sg_center, sg_context)
print("SGNS 训练完成，总耗时 {:.1f}s".format(time.time() - t0))
""")

code(r"""
plt.figure(figsize=(7, 4))
plt.plot(range(1, len(sg_history) + 1), sg_history, "o-",
         color=COLORS["blue"], lw=2, label="SGNS train loss")
plt.title("SGNS 训练损失曲线"); plt.xlabel("epoch"); plt.ylabel("loss")
plt.legend(); plt.tight_layout(); plt.show()
print("损失稳定下降 → 模型正在学习：正样本点积变大、负样本点积变小。")
""")

# ============================================================
# 8. 手写 CBOW
# ============================================================
md(r"""
## 8 · 手写实现 ②：CBOW + 负采样

CBOW 反过来：把**上下文词向量求平均**当输入，去预测中心词。同样用负采样。

- 输入 `context`：形状 `[B, 2*window]`，不足处用 padding 补齐，并用 `mask` 标记哪些是真实词，做**带掩码的平均**。
- 其余损失结构与 SGNS 完全相同，因此可以复用负采样与训练思路。

**损失函数：和 SGNS 几乎一样，只换了点积的一方。** 先把上下文词向量求平均：

$$
\bar v_{\text{ctx}} = \frac{1}{|C|}\sum_{c\in C} v_c
\qquad(C\ \text{是窗口内的上下文词集合})
$$

然后套用和 SGNS **完全相同**的负采样损失（$w_O$ 是目标中心词，$w_k$ 是负样本）：

$$
L_{\text{CBOW}} = -\log\sigma\big(v'_{w_O}\cdot \bar v_{\text{ctx}}\big)
\;-\;\sum_{k=1}^{K}\log\sigma\big(-\,v'_{w_k}\cdot \bar v_{\text{ctx}}\big)
$$

**和 SGNS 的唯一区别**：点积里那个“查询向量”变了——

| | 损失里的查询向量（点积的一方） | 预测方向 |
|---|---|---|
| SGNS | 单个**中心词向量** $v_{w_I}$ | 中心词 → 上下文 |
| CBOW | 多个上下文词向量的**平均** $\bar v_{\text{ctx}}$ | 上下文（求平均）→ 中心词 |

> 把 SGNS 损失里的 $v_{w_I}$ 换成 $\bar v_{\text{ctx}}$，就得到 CBOW——负采样、sigmoid、取负对数、batch 平均全都不变。
> 这个“对上下文求平均”正是 **C**ontinuous **B**ag-**o**f-**W**ords（连续词袋）名字的由来：把上下文当成一个无序的“词袋”，平均成一个向量。
""")

code(r"""
# ---------- 为 CBOW 构建样本：(上下文集合, 中心词) ----------
def build_cbow_samples(seq, window):
    ctx_len = 2 * window
    contexts, targets, masks = [], [], []
    n = len(seq)
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        ctx = [seq[j] for j in range(lo, hi) if j != i]
        if not ctx:
            continue
        m = [1] * len(ctx) + [0] * (ctx_len - len(ctx))   # 掩码
        ctx = ctx + [0] * (ctx_len - len(ctx))            # padding 到定长
        contexts.append(ctx); targets.append(seq[i]); masks.append(m)
    return (np.array(contexts, dtype=np.int64),
            np.array(targets, dtype=np.int64),
            np.array(masks, dtype=np.float32))

t0 = time.time()
cb_context, cb_target, cb_mask = build_cbow_samples(subsampled, WINDOW)
print("CBOW 训练样本: {:,}  (耗时 {:.1f}s)".format(len(cb_target), time.time() - t0))


class CBOWNS(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, embed_dim)   # 上下文词向量(将被平均)
        self.out_emb = nn.Embedding(vocab_size, embed_dim)  # 中心(目标)词向量
        nn.init.uniform_(self.in_emb.weight, -0.5 / embed_dim, 0.5 / embed_dim)
        nn.init.zeros_(self.out_emb.weight)

    def forward(self, context, mask, target, negatives):
        # context:[B,C]  mask:[B,C]  target:[B]  negatives:[B,K]
        v_ctx = self.in_emb(context)                          # [B, C, D]
        m = mask.unsqueeze(2)                                 # [B, C, 1]
        # 带掩码的平均：只对真实上下文词求平均
        v_mean = (v_ctx * m).sum(1) / m.sum(1).clamp(min=1)   # [B, D]

        v_t = self.out_emb(target)                            # [B, D]
        v_n = self.out_emb(negatives)                         # [B, K, D]
        pos = torch.sum(v_mean * v_t, dim=1)                  # [B]
        neg = torch.bmm(v_n, v_mean.unsqueeze(2)).squeeze(2)  # [B, K]
        return -(F.logsigmoid(pos) + F.logsigmoid(-neg).sum(1)).mean()

print(CBOWNS(V, EMBED_DIM))
""")

code(r"""
def train_cbow(model, context, mask, target, epochs=EPOCHS,
               batch_size=BATCH_SIZE, lr=LR, neg_k=NEG_K):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(target)
    history = []
    for ep in range(1, epochs + 1):
        t0, total, nb = time.time(), 0.0, 0
        for bi in iterate_minibatches(n, batch_size):
            ctx = torch.from_numpy(context[bi]).to(device)
            m = torch.from_numpy(mask[bi]).to(device)
            tgt = torch.from_numpy(target[bi]).to(device)
            neg = torch.from_numpy(
                sample_negatives(len(bi) * neg_k).reshape(len(bi), neg_k)
            ).to(device)
            loss = model(ctx, m, tgt, neg)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); nb += 1
        avg = total / nb
        history.append(avg)
        print("  epoch {:>2d}/{}  loss={:.4f}  ({:.1f}s)".format(
            ep, epochs, avg, time.time() - t0))
    return history

print("开始训练 CBOW ...")
cb_model = CBOWNS(V, EMBED_DIM)
t0 = time.time()
cb_history = train_cbow(cb_model, cb_context, cb_mask, cb_target)
cbow_seconds = time.time() - t0
print("CBOW 训练完成，总耗时 {:.1f}s".format(cbow_seconds))
""")

code(r"""
plt.figure(figsize=(7, 4))
plt.plot(range(1, len(sg_history) + 1), sg_history, "o-",
         color=COLORS["blue"], lw=2, label="SGNS")
plt.plot(range(1, len(cb_history) + 1), cb_history, "s-",
         color=COLORS["purple"], lw=2, label="CBOW")
plt.title("SGNS vs CBOW 训练损失对比"); plt.xlabel("epoch"); plt.ylabel("loss")
plt.legend(); plt.tight_layout(); plt.show()
""")

# ============================================================
# 9. 评估与可视化
# ============================================================
md(r"""
## 9 · 评估与可视化

训练完成，我们取 `in_emb` 的权重作为最终词向量。下面做几件事：

1. **近义词检索** `most_similar`：用余弦相似度找与某个词最接近的词。
2. **词类比** `king - man + woman ≈ ?`：word2vec 最著名的“向量运算”。
3. **可视化**：把高维词向量降到 2D 画语义地图、相似度热力图、类比箭头图。

> ⚠️ **诚实提醒**：这里默认只用了 5MB 小语料训练几轮，向量质量有限——近义词通常已经不错，
> 但精确的类比（如严格得到 `queen`）往往需要**完整 text8（100MB）+ 更多轮次**。
> 想看更漂亮的结果，把第 6 节的 `CORPUS_PATH` 换成完整 `data/text8` 重跑即可。
""")

code(r"""
# 取中心词向量作为最终词向量，并做 L2 归一化（方便算余弦相似度）
def get_normalized_vectors(model):
    W = model.in_emb.weight.detach().cpu().numpy()
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    return W / np.clip(norms, 1e-8, None)

sg_vectors = get_normalized_vectors(sg_model)

def most_similar(word, vectors, topn=8):
    if word not in word2idx:
        return [(f"<'{word}' 不在词表中>", 0.0)]
    q = vectors[word2idx[word]]
    sims = vectors @ q                       # 已归一化，点积即余弦相似度
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i == word2idx[word]:
            continue
        out.append((idx2word[i], float(sims[i])))
        if len(out) >= topn:
            break
    return out

for w in ["one", "king", "city", "war", "water"]:
    print("most_similar('{}') →".format(w))
    for word, s in most_similar(w, sg_vectors, topn=6):
        print("    {:<15s} {:.3f}".format(word, s))
    print()
""")

code(r"""
def analogy(a, b, c, vectors, topn=5):
    '''求 a - b + c 最接近的词，例如 analogy('king','man','woman')。'''
    for w in (a, b, c):
        if w not in word2idx:
            return [(f"<'{w}' 不在词表中>", 0.0)]
    vec = vectors[word2idx[a]] - vectors[word2idx[b]] + vectors[word2idx[c]]
    vec = vec / np.clip(np.linalg.norm(vec), 1e-8, None)
    sims = vectors @ vec
    order = np.argsort(-sims)
    exclude = {word2idx[a], word2idx[b], word2idx[c]}
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((idx2word[i], float(sims[i])))
        if len(out) >= topn:
            break
    return out

print("类比：king - man + woman ≈ ?")
for word, s in analogy("king", "man", "woman", sg_vectors):
    print("    {:<15s} {:.3f}".format(word, s))

print("\n类比：paris - france + italy ≈ ?  (期望 rome 附近)")
for word, s in analogy("paris", "france", "italy", sg_vectors):
    print("    {:<15s} {:.3f}".format(word, s))
""")

md(r"""
### 可视化 ①：2D 语义地图（PCA）

把高频词的 100 维向量用 PCA 压到 2 维画散点。语义相近的词（如数字、方位、国家名）应当**聚在一起**。
""")

code(r"""
from sklearn.decomposition import PCA

# 选一批高频且有代表性的词来画图
show_words = [w for w in [
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "king", "queen", "man", "woman", "son", "father", "mother",
    "france", "england", "germany", "italy", "paris", "london", "war",
    "water", "river", "sea", "city", "town", "government", "president",
] if w in word2idx]

idxs = [word2idx[w] for w in show_words]
pts = PCA(n_components=2, random_state=SEED).fit_transform(sg_vectors[idxs])

plt.figure(figsize=(11, 8))
plt.scatter(pts[:, 0], pts[:, 1], color=COLORS["blue"], s=45, zorder=3)
for (x, y), w in zip(pts, show_words):
    plt.annotate(w, (x, y), fontsize=10, xytext=(4, 4),
                 textcoords="offset points")
plt.title("词向量 2D 语义地图（PCA 降维）")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout(); plt.show()
""")

md(r"""
### 可视化 ②：相似度热力图

选一组词，两两计算余弦相似度画成热力图。**对角线恒为 1**（自己和自己最像），
同类词（数字之间、家庭称谓之间）应当形成**更亮的方块**。
""")

code(r"""
heat_words = [w for w in ["one", "two", "three", "king", "queen",
                          "man", "woman", "france", "england"] if w in word2idx]
hi = [word2idx[w] for w in heat_words]
sub = sg_vectors[hi]
sim_mat = sub @ sub.T

fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(sim_mat, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(heat_words))); ax.set_xticklabels(heat_words, rotation=45, ha="right")
ax.set_yticks(range(len(heat_words))); ax.set_yticklabels(heat_words)
for i in range(len(heat_words)):
    for j in range(len(heat_words)):
        ax.text(j, i, f"{sim_mat[i, j]:.2f}", ha="center", va="center",
                color="black", fontsize=8)
fig.colorbar(im, ax=ax, label="cosine similarity")
ax.set_title("词对余弦相似度热力图"); ax.grid(False)
plt.tight_layout(); plt.show()
""")

md(r"""
### 可视化 ③：类比的“平行四边形”

word2vec 最惊艳的性质：语义关系表现为**向量方向**。
`man→woman` 的偏移方向，应当和 `king→queen` 的偏移方向大致平行。下图在 PCA 2D 空间里画出这些偏移箭头。
""")

code(r"""
pairs = [("man", "woman"), ("king", "queen"),
         ("father", "mother"), ("son", "daughter")]
pairs = [(a, b) for a, b in pairs if a in word2idx and b in word2idx]

flat = []
for a, b in pairs:
    flat += [a, b]
uniq = list(dict.fromkeys(flat))
uidx = [word2idx[w] for w in uniq]
coords2d = PCA(n_components=2, random_state=SEED).fit_transform(sg_vectors[uidx])
pos = {w: coords2d[i] for i, w in enumerate(uniq)}

plt.figure(figsize=(10, 7.5))
for w in uniq:
    x, y = pos[w]
    plt.scatter(x, y, color=COLORS["blue"], s=55, zorder=3)
    plt.annotate(w, (x, y), fontsize=11, xytext=(5, 5), textcoords="offset points")
for a, b in pairs:
    (x1, y1), (x2, y2) = pos[a], pos[b]
    plt.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle="-|>", color=COLORS["red"], lw=1.8))
plt.title("类比关系 = 平行的向量偏移（PCA 2D）\n理想情况下 man→woman 与 king→queen 方向相近")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout(); plt.show()
print("提示：小语料下方向未必完美平行；换成完整 text8 会明显更整齐。")
""")

# ============================================================
# 10. SGNS vs CBOW 对比
# ============================================================
md(r"""
## 10 · SGNS vs CBOW 横向对比

在**同一份语料、同样超参数**下对比两者。理论预期：

- **CBOW 训练更快**（一个样本预测一个中心词，样本数更少）。
- **Skip-gram 对低频词更友好，小语料上语义质量常更好**。

**为什么会这样？根源在“一个窗口产出几个样本”**（回顾 §7/§8）。设窗口内有 $2m$ 个上下文词：

- **Skip-gram**：中心词和**每个**上下文词单独配对 → 一个窗口产出 $2m$ 个样本（下面柱状图能看到样本数差距）。
- **CBOW**：把 $2m$ 个上下文词**平均成一个**再预测中心词 → 一个窗口只产出 **1** 个样本。

由此推出那两句建议：

| 场景 | 选谁 | 原因 |
|------|------|------|
| **小语料** | Skip-gram | 一个窗口拆成 $2m$ 条样本，训练信号多几倍，把有限语料**榨得更充分** |
| **低频词** | Skip-gram | 低频词**单独成对**参与、梯度不被稀释；而 CBOW 里它会和 `the/of` 等词**一起被平均掉、信号被淹没** |
| **求快** | CBOW | 一个窗口只 1 个样本，**总样本量小、训练快** |
| **高频词** | CBOW | 高频词样本充足，**平均能平滑噪声**，得到更稳的表示；且它不怕被稀释 |

> 一句话：Skip-gram“拆开、样本多、低频友好但慢”，CBOW“平均、样本少、快且对高频降噪”。
> 这也是 gensim 默认 `sg=0`（CBOW，求快）、想要低频词质量时改 `sg=1`（Skip-gram）的原因。

下面用两者各自的近义词结果 + 一张耗时柱状图做直观对比（数值会因语料规模波动，重点看趋势）。
""")

code(r"""
cb_vectors = get_normalized_vectors(cb_model)

print("同一个词，两种模型给出的近义词对比：\n")
for w in ["king", "one", "city"]:
    if w not in word2idx:
        continue
    sg_top = [x[0] for x in most_similar(w, sg_vectors, topn=5)]
    cb_top = [x[0] for x in most_similar(w, cb_vectors, topn=5)]
    print("  '{}'".format(w))
    print("     SGNS :", ", ".join(sg_top))
    print("     CBOW :", ", ".join(cb_top))
    print()

# 训练耗时 / 样本量对比
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(["SGNS", "CBOW"], [len(sg_center), len(cb_target)],
            color=[COLORS["blue"], COLORS["purple"]], alpha=0.85)
axes[0].set_title("训练样本数量对比"); axes[0].set_ylabel("样本数")
for i, v in enumerate([len(sg_center), len(cb_target)]):
    axes[0].text(i, v, "{:,}".format(v), ha="center", va="bottom", fontsize=9)

axes[1].plot(range(1, len(sg_history)+1), sg_history, "o-",
             color=COLORS["blue"], label="SGNS")
axes[1].plot(range(1, len(cb_history)+1), cb_history, "s-",
             color=COLORS["purple"], label="CBOW")
axes[1].set_title("收敛曲线对比"); axes[1].set_xlabel("epoch")
axes[1].set_ylabel("loss"); axes[1].legend()
plt.tight_layout(); plt.show()
""")

# ============================================================
# 11. 复杂度/参数/优缺点/陷阱/下一步/参考文献
# ============================================================
md(r"""
## 11 · 复杂度、参数、陷阱与延伸阅读

### 11.1 复杂度对比（每个训练样本）

| 方案 | 每步复杂度 | 说明 |
|------|-----------|------|
| 朴素 softmax | $O(V)$ | 分母要遍历整个词表，词表大时不可行 |
| **负采样** | $O(K)$ | 只更新 1 个正样本 + K 个负样本（K≈5~20） |
| 层次 softmax | $O(\log V)$ | 走哈夫曼树的一条根到叶路径 |

### 11.2 关键超参数怎么调

| 参数 | 作用 | 调大 / 调小 |
|------|------|------------|
| `EMBED_DIM` 维度 | 向量表达能力 | 常用 100~300；太大易过拟合、且慢 |
| `WINDOW` 窗口 | 上下文范围 | 大 → 偏“主题相关”；小 → 偏“句法/近义” |
| `NEG_K` 负样本数 | 每步反例个数 | 小语料用大些(10~20)，大语料 5 足矣 |
| `MIN_COUNT` | 词频下限 | 调大 → 词表更干净但丢词更多 |
| `SUBSAMPLE_T` | 下采样强度 | 调小 → 丢弃更多高频词 |
| `EPOCHS` / `LR` | 训练轮数 / 学习率 | 语料小需更多轮；lr 过大会震荡 |
| 模型选择 | Skip-gram vs CBOW | 小语料/低频词 → Skip-gram；求快/高频 → CBOW |

### 11.3 常见误区（初学者容易踩的坑）

1. **忘了下采样**：不丢高频词，`the/of` 会主导训练，向量质量和速度都变差。
2. **用错向量做相似度**：算余弦相似度前**一定要 L2 归一化**，否则高频词因向量模长大而“虚高”。
3. **在小语料上苛求类比效果**：`king-man+woman=queen` 需要足够大的语料才稳定，小 demo 上不成功很正常。
4. **混淆两套向量**：每个词有中心词向量和上下文词向量两套，通常取**中心词向量**当结果（也有做法是两者相加/拼接）。
5. **把 word2vec 当上下文相关模型**：它给每个词**一个固定向量**，无法区分“bank（银行）/bank（河岸）”。需要区分词义请用 BERT 等上下文模型。

### 11.4 下一步学什么

- **GloVe**（Pennington et al., 2014）：从全局共现矩阵出发的另一条词向量路线。
- **fastText**（Bojanowski et al., 2017）：引入**子词 n-gram**，能处理未登录词和形态丰富的语言。
- **上下文相关嵌入**：ELMo → **BERT / GPT**，同一个词在不同句子里向量不同，是现代 NLP 的基石。
- **想看工业级实现**：`gensim.models.Word2Vec`（本项目已把 `gensim` 列为可选依赖：`uv sync --extra gensim`），可与你手写的版本对照速度和效果。

### 11.5 参考文献

1. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space.* ICLR Workshop. arXiv:1301.3781. → 提出 CBOW 与 Skip-gram 结构。
2. Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS (NIPS) 2013. → 提出**负采样**、**高频词下采样**与短语学习。
3. Goldberg, Y., & Levy, O. (2014). *word2vec Explained: Deriving Mikolov et al.'s Negative-Sampling Word-Embedding Method.* arXiv:1402.3722. → 负采样目标函数的推导。
4. Rong, X. (2014). *word2vec Parameter Learning Explained.* arXiv:1411.2738. → 逐梯度手把手推导，适合入门。
5. Levy, O., & Goldberg, Y. (2014). *Neural Word Embedding as Implicit Matrix Factorization.* NeurIPS 2014. → 证明 SGNS 隐式地在做（偏移 PMI）矩阵分解。
6. Firth, J. R. (1957). *A Synopsis of Linguistic Theory.* → 分布式假设的经典出处（"You shall know a word by the company it keeps."）。

---

🎉 **恭喜！** 你已经从零实现并训练了 word2vec 的两种模型，还把每个关键机制都可视化了一遍。
把 `CORPUS_PATH` 换成完整 `data/text8`、`EPOCHS` 调大，再跑一遍，看看语义地图和类比会变得多整齐。
""")

# ============================================================
# 写出 notebook
# ============================================================
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "word2vec_tutorial.ipynb")
with open(OUT, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"已生成 notebook：{OUT}  （共 {len(cells)} 个单元）")
