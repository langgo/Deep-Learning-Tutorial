# word2vec 从零实现 · 互动教学

一套自包含的 **word2vec 教学材料**：把理论讲透（直觉 → 数学 → 图解），并用 **PyTorch 手写实现** Skip-gram 与 CBOW 两种模型，配大量 matplotlib 图表把每个关键机制画出来。

> 目标读者：想真正搞懂 word2vec 内部机制的学习者。全程手写实现，不把 gensim 当黑盒（gensim 仅作为可选对照）。

## 目录结构

```
word2vec/
├── pyproject.toml            # uv 项目与依赖定义（Python 3.12）
├── uv.lock                   # 锁定的依赖版本（uv sync 生成）
├── download_data.py          # text8 语料下载器（支持生成小语料）
├── build_notebook.py         # 用 nbformat 生成教学 notebook 的脚本
├── word2vec_tutorial.ipynb   # ★ 主教程：理论 + 代码 + 图表（已含运行结果）
├── README.md
└── data/                     # 运行 download_data.py 后生成
    ├── text8                 # 完整语料（100MB）
    └── text8_small           # 小语料（默认 5MB，跑得快）
```

## 快速开始（3 步）

前置：已安装 [uv](https://docs.astral.sh/uv/)（`brew install uv`）。

```bash
# 1) 创建虚拟环境并安装依赖（torch / matplotlib / scikit-learn / jupyter ...）
uv sync

# 2) 下载 text8 语料，并额外生成一个 5MB 小语料（便于快速跑通）
uv run python download_data.py --subset 5

# 3) 启动 JupyterLab，打开 word2vec_tutorial.ipynb，从上往下依次运行
uv run jupyter lab
```

在 macOS Apple Silicon 上，notebook 会自动使用 **MPS（Metal GPU）** 加速；有 NVIDIA 卡则用 CUDA，否则回退 CPU。

## 教程内容（notebook 章节）

| 章节 | 内容 |
|------|------|
| 0 | 环境准备与设备选择、加载语料 |
| 1 | 前置知识 + 关键术语速查表 |
| 2 | 直觉理解与历史背景（分布式假设、Mikolov 2013） |
| 3 | 从 one-hot 到分布式表示（对比图 + 余弦相似度） |
| 4 | 两种模型结构：Skip-gram vs CBOW（结构示意图） |
| 5 | 训练技巧：负采样、层次 softmax、3/4 次方分布（sigmoid 图 + 采样分布图） |
| 6 | 数据预处理：建词表、下采样、负采样表、滑动窗口（4 张图） |
| 7 | **手写 SGNS**（Skip-gram + 负采样）+ 训练损失曲线 |
| 8 | **手写 CBOW** + 与 SGNS 的收敛对比 |
| 9 | 评估与可视化：近义词、`king-man+woman` 类比、2D 语义地图、相似度热力图、类比箭头图 |
| 10 | SGNS vs CBOW 横向对比 |
| 11 | 复杂度、参数调优、常见误区、延伸阅读、参考文献 |

## 常用命令

```bash
# 下载完整语料 + 生成 10MB 小语料
uv run python download_data.py --subset 10

# 强制重新下载 / 解压
uv run python download_data.py --force

# 重新生成 notebook（修改 build_notebook.py 后）
uv run python build_notebook.py

# 命令行里执行整个 notebook（验证能否端到端跑通）
uv run jupyter nbconvert --to notebook --execute --inplace word2vec_tutorial.ipynb

# 可选：装上 gensim 做工业实现对照
uv sync --extra gensim
```

## 想要更好的结果？

默认用 5MB 小语料，几分钟就能跑完，近义词效果已经不错，但精确的词类比（严格得到 `queen`）通常需要更大语料。

想看更漂亮的语义地图和类比，把 notebook 第 6 节里的语料路径换成完整的 `data/text8`，并把 `EPOCHS` 调大（如 10~15）后重跑即可。

## 参考文献

- Mikolov et al. (2013). *Efficient Estimation of Word Representations in Vector Space.* ICLR Workshop. arXiv:1301.3781.
- Mikolov et al. (2013). *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS 2013.（负采样 + 下采样）
- Rong, X. (2014). *word2vec Parameter Learning Explained.* arXiv:1411.2738.（逐梯度推导，适合入门）
