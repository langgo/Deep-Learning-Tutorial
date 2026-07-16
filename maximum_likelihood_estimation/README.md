# 极大似然估计 (Maximum Likelihood Estimation) · 互动教学

一份面向初学者的中文教学 Notebook，从"概率 vs 似然"到"深度学习损失函数背后的 MLE"，讲清极大似然估计到底在做什么，以及它为什么是现代机器学习最底层的推理原理之一。

## 目录结构

```text
maximum_likelihood_estimation/
├── pyproject.toml         # uv 项目与依赖定义
├── build_notebook.py      # 用 nbformat 生成教学 notebook
├── mle_tutorial.ipynb     # 生成的主教程
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

在 JupyterLab 里打开 `mle_tutorial.ipynb`，从上往下依次运行。

## 教程内容

| 章节 | 主题 |
|------|------|
| 0 | 环境准备、绘图风格、术语约定 |
| 1 | 概率 vs 似然：视角切换 |
| 2 | 似然函数 & 对数似然：为什么取 log |
| 3 | 手算 Bernoulli MLE：抛硬币的最优参数 |
| 4 | 手算 Gaussian MLE：`μ̂ = 均值`、`σ̂²` 的有偏性 |
| 5 | MLE 就是最小化 NLL：BCE / MSE / CE 的一致性 |
| 6 | 数值 MLE：`scipy.optimize`、坑与陷阱 |
| 7 | MLE vs MAP vs 贝叶斯后验：正则化的本质 |
| 8 | 术语速查、常见误区、参考文献 |

## 设计原则

- 每个方法都配 **背景 → 直觉 → 数学 → 手算例子 → 代码 → 可视化 → 陷阱**。
- 关键公式后立刻给一个带中间数值的手算示例。
- 交互式滑动条 (ipywidgets) 让读者拖动"观测样本 / 先验强度"，实时看到似然峰值移动。
- 图表只服务教学：概率视角 vs 似然视角、log-likelihood 曲面、优化轨迹、后验演化。

## 常用命令

```bash
# 重新生成 notebook
uv run python build_notebook.py

# 端到端执行整个 notebook，验证代码可跑通
uv run jupyter nbconvert --to notebook --execute --inplace mle_tutorial.ipynb
```

## 参考文献

- Fisher, R. A. (1922). *On the Mathematical Foundations of Theoretical Statistics.* Philosophical Transactions of the Royal Society A.
- Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury.
- Wasserman, L. (2004). *All of Statistics: A Concise Course in Statistical Inference.* Springer.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective.* MIT Press.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning.* Springer.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press. (Ch. 5.5 Maximum Likelihood Estimation)
