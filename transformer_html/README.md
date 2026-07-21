# Transformer 从 0 到 1 · 单页互动教程

一个面向普通软件开发工程师的中文 Transformer 教程。教程以纯静态 `index.html` 交付，内联 CSS 和 Vanilla JS，无需构建、无需网络依赖，可直接用浏览器打开。

## 打开方式

```bash
open index.html
```

如果不使用 macOS，也可以在文件管理器或浏览器中直接打开 `index.html`。

## 内容结构

| 章节 | 主题 |
|------|------|
| 0 | 为什么需要 Transformer：RNN/CNN 与 self-attention 的差异 |
| 1 | token、embedding、position encoding |
| 2 | scaled dot-product attention 的完整计算过程 |
| 3 | multi-head attention 的并行关系视角 |
| 4 | Transformer block：残差、LayerNorm、FFN |
| 5 | 原始 encoder-decoder 架构与 masked / cross attention |
| 6 | logits、softmax、cross entropy 与训练目标 |
| 7 | BERT、GPT、T5 的架构演化 |
| 8 | KV cache、上下文窗口、采样参数等 LLM 工程问题 |
| 9 | 实践意义与常见误区 |
| 10 | 端到端复盘、术语表和继续学习建议 |

## 设计原则

- 每个关键公式都说明变量来源、符号含义和实际工程语义。
- 先定义前置概念，再引出依赖它的公式或结论。
- 使用小规模 toy 矩阵做浏览器内推演，展示点积、缩放、softmax、加权求和、loss 等中间值。
- 交互控件默认有可用数据，不要求读者先配置环境或输入复杂内容。
- 页面保持单文件可移植，便于后续发布到 GitHub Pages 或直接分享。
