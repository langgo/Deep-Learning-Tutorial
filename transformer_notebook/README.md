# Transformer 从零到工程实践 · 互动教学

一份面向普通软件开发工程师的中文 Transformer 教学 notebook。教程从序列建模问题出发，逐步讲清原版 Encoder-Decoder Transformer，再把同一套组件映射到 GPT 的 decoder-only 路线和 BERT 的 encoder-only 路线。

这不是只介绍公式的材料。每个关键机制都会按“要解决什么问题 → 直觉 → 符号解释 → 手算例子 → 可视化 → PyTorch 实现 → 工程坑”的节奏展开。

## 目录结构

```text
transformer_notebook/
├── pyproject.toml              # uv 项目与依赖定义
├── build_notebook.py           # 用 nbformat 生成教学 notebook
├── transformer_tutorial.ipynb  # 生成的主教程
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

打开 `transformer_tutorial.ipynb` 后，从上往下依次运行。教程只使用内置 toy corpus，不需要下载外部数据。

## 教程内容

| 章节 | 主题 |
|------|------|
| 0 | 环境准备、学习路线、术语约定 |
| 1 | 为什么需要 Transformer：RNN/CNN 的局限与序列建模任务 |
| 2 | 输入表示：token id、embedding、位置编码（含 RoPE/ALiBi、Tokenizer/BPE 进阶） |
| 3 | Dot-product attention：Q/K/V 的来源和 3-token 手算 |
| 4 | Scaled attention：为什么除以 `sqrt(d_k)` |
| 5 | Multi-head attention：多头如何看不同关系（含 MHA/GQA/MQA 进阶） |
| 6 | Mask：padding mask、causal mask、cross-attention mask |
| 7 | Transformer Block：Residual、LayerNorm、FFN（含 RMSNorm、SwiGLU 进阶） |
| 8 | Encoder、Decoder 与原版 Encoder-Decoder Transformer |
| 9 | PyTorch 实战：toy copy/translation 训练和 attention 可视化 |
| 10 | GPT 路线：decoder-only、causal LM、采样生成 |
| 11 | BERT 路线：encoder-only、MLM、CLS 表征 |
| 12 | 训练/推理工程实践：shape、mask、teacher forcing、KV cache 最小实现 |
| 13 | LLM 推理工程：prefill/decode、KV cache 显存、采样、约束解码、vLLM 生态 |
| 14 | 总结、现代 LLM 组件替换速查表、后续路线 |

## 常用命令

```bash
# 重新生成 notebook
uv run python build_notebook.py

# 端到端执行整个 notebook，验证代码可跑通
uv run jupyter nbconvert --to notebook --execute --inplace transformer_tutorial.ipynb
```

## 设计原则

- 公式必须解释每个符号的含义，并说明它解决的实际问题。
- 不直接跳到最终结论；先讲依赖概念，再组合成完整结构。
- 每章至少有一个图、表、手算或可执行代码片段。
- 训练任务保持 toy 规模，CPU 也能跑完；工程结构尽量贴近真实 PyTorch 写法。
- 交互式可视化优先使用 `ipywidgets`，不可用时降级为静态图。

## 参考文献

- Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL.
- Radford et al. (2018). *Improving Language Understanding by Generative Pre-Training.*
- Radford et al. (2019). *Language Models are Unsupervised Multitask Learners.*
- Alammar, J. *The Illustrated Transformer.*
- Harvard NLP. *The Annotated Transformer.*
