# PyTorch 从零讲透 · 互动教学

一套自包含的 **PyTorch 框架教学材料**：不讲某个具体模型，而是把你在别处当黑盒用的东西讲透——
**Tensor、autograd、nn.Module、损失与优化器、训练循环**——最后用一个 **FashionMNIST 图像分类**的端到端实战把所有零件串起来。

> 目标读者：会一点 Python / NumPy，想真正搞懂 PyTorch 内部机制、而不是背 API 的学习者。
> 全程「直觉 → 数学 → 代码对拍 → 可视化」，关键机制都用 matplotlib 画出来，并与手写实现互相验证。

## 目录结构

```
pytorch-tutorial/
├── pyproject.toml            # uv 项目与依赖定义（Python 3.12）
├── uv.lock                   # 锁定的依赖版本（uv sync 生成）
├── build_notebook.py         # 用 nbformat 以代码拼装教学 notebook 的脚本
├── pytorch_tutorial.ipynb    # ★ 主教程：理论 + 代码 + 图表（已含运行结果）
├── README.md
└── data/                     # 首次运行时 torchvision 自动下载 FashionMNIST 到这里
```

## 快速开始（3 步）

前置：已安装 [uv](https://docs.astral.sh/uv/)（`brew install uv`）。

```bash
# 1) 创建虚拟环境并安装依赖（torch / torchvision / matplotlib / jupyter ...）
uv sync

# 2) 启动 JupyterLab，打开 pytorch_tutorial.ipynb，从上往下依次运行
uv run jupyter lab

# 3)（可选）命令行里端到端执行整个 notebook，验证能否跑通
uv run jupyter nbconvert --to notebook --execute --inplace pytorch_tutorial.ipynb
```

在 macOS Apple Silicon 上，notebook 会自动使用 **MPS（Metal GPU）** 加速；有 NVIDIA 卡则用 CUDA，否则回退 CPU。
FashionMNIST 数据集（约 30MB）会在第 7 章首次运行时自动下载到 `data/`。

## 教程内容（notebook 章节）

| 章节 | 内容 | 关键实验 / 配图 |
|------|------|-----------------|
| 0 | 环境准备、设备选择、绘图风格 | — |
| 1 | **Tensor**：创建、dtype/device、索引切片、reshape/view、广播 | 广播规则图 |
| 2 | **autograd**：计算图、`backward()`、梯度累积、`no_grad`/`detach` | 手绘计算图 + 梯度验证 |
| 3 | **手写反向传播 vs autograd 对拍**：链式法则、数值梯度校验 | 三方梯度柱状图 + 决策边界 |
| 4 | **nn.Module**：参数管理、拆解 `nn.Linear`、手写 Linear 对拍 | 各层参数量 + 权重热力图 |
| 5 | **损失与优化器**：MSE/CrossEntropy、手写 SGD/Adam vs 内置对拍 | 优化轨迹对比图 |
| 6 | **训练循环骨架**：Dataset / DataLoader / mini-batch / train-eval | 拟合曲线 + 损失曲线 |
| 7 | **端到端实战**：FashionMNIST 分类 + `lr_scheduler` + 过拟合监控 | 样本图 + 损失/gap/lr 曲线 + 混淆矩阵 |
| 8 | **权重初始化与梯度消失**：为什么深网难训（small/Xavier/Kaiming） | 逐层梯度范数图 |
| 9 | **CNN**：`Conv2d`/`MaxPool2d`，在 FashionMNIST 上超过 MLP | CNN vs MLP + 卷积核可视化 |
| 10 | **模型保存/加载**：`state_dict`、checkpoint、续训 | 重建对拍验证 |
| 11 | **常见陷阱**：`zero_grad`、`.item()`、`train/eval`、设备不一致等 | 忘记 zero_grad 的发散曲线 |

## 常用命令

```bash
# 重新生成 notebook（修改 build_notebook.py 后）
uv run python build_notebook.py

# 命令行里执行整个 notebook（验证能否端到端跑通）
uv run jupyter nbconvert --to notebook --execute --inplace pytorch_tutorial.ipynb
```

## 设计原则

- **不把框架当黑盒**：`nn.Linear`、SGD、Adam、反向传播都先手写一遍，再与 PyTorch 内置实现**对拍**（数值对齐到小数点后多位），让你确信「内置的没有魔法」。
- **理论与代码交替**：每个机制先讲直觉与数学，再用最小代码验证。
- **端到端闭环**：最后一章把前面所有零件（Tensor / autograd / Module / 优化器 / DataLoader）组装成一个真实可训练、可评估的图像分类器。

## 参考资料

- PyTorch 官方文档：<https://pytorch.org/docs/stable/index.html>
- 官方教程 *Deep Learning with PyTorch: A 60 Minute Blitz*
- *Learn the Basics* 系列：<https://pytorch.org/tutorials/beginner/basics/intro.html>
- Xiao, Rasul, Vollgraf (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.* arXiv:1708.07747.
