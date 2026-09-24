# 附录 A：零数据 Toy Checks

这些检查不下载数据、不加载 tokenizer 或 checkpoint，也不需要 PyTorch。它们只用 Python 标准库，把教程里最容易错的六个不变量变成可运行的 `assert`。

从仓库根目录运行：

```bash
python3 tutorial/toy_checks.py
```

预期看到六行 `PASS`，最后是：

```text
All 6 checks passed.
```

## 检查覆盖什么？

| 检查 | 对应概念 | 建议搭配章节 |
|---|---|---|
| `check_shift_and_sft_mask` | `x/y` 右移，SFT 必须用 `mask[1:]` | 第 06、09 章 |
| `check_cross_entropy` | 一个位置的稳定 softmax 与交叉熵 | 第 06 章 |
| `check_attention_shapes` | `(B,T,C)` 拆成 `(B,T,H,D)` | 第 04 章 |
| `check_bpb` | `nats / (ln2 × bytes)` | 第 08 章 |
| `check_group_advantages` | 同题奖励减均值 | 第 12 章 |
| `check_pass_at_k` | 前 `k` 个候选任一成功 | 第 11、12 章 |

这些代码是**教学化简**，不是 nanochat 生产实现。例如 attention 检查只验证形状，没有计算 QK、mask 或 softmax；交叉熵只处理一个位置。它们的作用是让关键索引和公式先在小输入上变得可见。

## 推荐玩法

1. 原样运行，确认环境至少有 Python 3。
2. 每次只故意破坏一个条件，例如把 `mask[1:]` 改成 `mask[:-1]`，观察哪个断言失败。
3. 回到对应章节和真实源码，寻找同一不变量在张量版本中的实现。

不要把全部断言删掉后只看打印值：检查失败正是实验最有价值的反馈。

---

- 返回[教程目录](./README.md)
- 查术语：[附录 B：术语表](./glossary.md)
