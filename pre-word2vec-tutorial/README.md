# Word2Vec 之前的 NLP 核心技术 · 互动教学

一套自包含的中文教学材料，讲清 **Word2Vec 出现之前，NLP 最核心、最有代表性的技术路线**。

教程重点不是罗列历史名词，而是反复回答四个问题：

1. 这类方法为什么会诞生？
2. 它解决了前一种方案的什么问题？
3. 它的核心机制是怎么推导出来的？
4. 它为什么又被后来的 Word2Vec 推进一步？

## 目录结构

```text
pre-word2vec-tutorial/
├── pyproject.toml                 # uv 项目与依赖定义
├── build_notebook.py              # 用 nbformat 生成教学 notebook
├── pre_word2vec_tutorial.ipynb    # 主教程：背景 + 算法 + 示例 + 对比
└── README.md
```

## 快速开始

前置：已安装 [uv](https://docs.astral.sh/uv/)。

```bash
# 1) 安装依赖
uv sync

# 2) 生成 notebook
uv run python build_notebook.py

# 3) 启动 JupyterLab
uv run jupyter lab
```

打开 `pre_word2vec_tutorial.ipynb`，从上往下依次运行。

## 教程内容

| 章节 | 内容 |
|------|------|
| 0 | 环境准备、统一绘图风格、构造 toy corpus |
| 1 | 总览：Word2Vec 之前 NLP 在解决什么 |
| 2 | 规则、词典与模式匹配 |
| 3 | N-gram 语言模型、平滑、困惑度 |
| 4 | One-hot、Bag of Words、TF-IDF |
| 5 | 传统机器学习文本分类 |
| 6 | 序列标注：HMM、Viterbi、CRF 直觉 |
| 7 | 主题模型：LSA、PLSA、LDA |
| 8 | 共现矩阵、PMI、PPMI、SVD |
| 9 | 早期神经语言模型：tiny NNLM forward pass 与 full softmax 参数量 |
| 10 | 总对比：Word2Vec 输入、训练目标与评分规则 |
| 11 | 术语速查、下一步映射与参考文献 |

## 常用命令

```bash
# 重新生成 notebook
uv run python build_notebook.py

# 执行整个 notebook，验证能否端到端跑通
uv run jupyter nbconvert --to notebook --execute --inplace pre_word2vec_tutorial.ipynb
```

## 设计原则

- 每章都包含：诞生背景、解决的问题、核心思想、示例、优劣势、与 Word2Vec 的关系。
- 示例默认使用内置英文小语料，不需要下载数据。
- 图表只服务教学：时间线、矩阵热力图、Viterbi 路径、主题词、低维语义图、对比表。
- 数学尽量配手算过程，让初学者能看到中间步骤。

## 参考文献

- Harris, Z. S. (1954). *Distributional Structure.* Word.
- Firth, J. R. (1957). *A Synopsis of Linguistic Theory.*
- Salton, G., Wong, A., & Yang, C. S. (1975). *A Vector Space Model for Automatic Indexing.* Communications of the ACM.
- Viterbi, A. J. (1967). *Error Bounds for Convolutional Codes and an Asymptotically Optimum Decoding Algorithm.* IEEE Transactions on Information Theory.
- Rabiner, L. R. (1989). *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition.* Proceedings of the IEEE.
- Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harshman, R. (1990). *Indexing by Latent Semantic Analysis.* JASIS.
- Lafferty, J., McCallum, A., & Pereira, F. (2001). *Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data.* ICML.
- Bengio, Y., Ducharme, R., Vincent, P., & Janvin, C. (2003). *A Neural Probabilistic Language Model.* JMLR.
- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). *Latent Dirichlet Allocation.* JMLR.
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space.* ICLR Workshop.
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS.
