#!/usr/bin/env python3
"""下载 word2vec 教学用的 text8 语料。

text8 是什么？
----------------
- 它是 Matt Mahoney 从 2006-03-03 的英文维基百科 dump 清洗出来的纯文本：
  取维基百科前 10^9 字节（enwik9），去掉 XML/markup、转小写、只保留 26 个字母和空格，
  得到 fil9；再取 fil9 的前 10^8 字节（约 100MB），就是 text8。
- word2vec 的官方 demo（Google/Mikolov 团队的 C 代码 demo-word.sh）默认下载并使用它，
  所以它是最经典的入门语料：小、干净、单个大文件，直接就能喂给模型。

用法
----
    # 下载完整 text8（约 31MB 压缩 -> 100MB 解压），保存到 ./data/text8
    python download_data.py

    # 额外生成一个只取前 5MB 的小语料 ./data/text8_small，几十秒就能训完，适合边学边跑
    python download_data.py --subset 5

    # 强制重新下载
    python download_data.py --force

备选方案
--------
如果你已经安装了 gensim，也可以一行下载（它内部同样是 text8）：

    import gensim.downloader as api
    corpus = api.load("text8")   # 返回可迭代的分词句子

本脚本只依赖 Python 标准库（urllib + zipfile），不需要任何第三方包。
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile

# text8 的官方镜像地址。若失效可换成常见镜像，例如：
#   https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8.zip
TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ZIP_PATH = os.path.join(DATA_DIR, "text8.zip")
TEXT8_PATH = os.path.join(DATA_DIR, "text8")

# 完整 text8 解压后正好是 100,000,000 字节，用于下载完整性校验
EXPECTED_TEXT8_BYTES = 100_000_000


def _human(nbytes: float) -> str:
    """把字节数格式化成人类可读字符串。"""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}GB"


_last_pct = [-1]  # 记录上次打印的整数百分比，用于非终端环境降频


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """urlretrieve 的下载进度回调。

    终端(TTY)下用 `\\r` 刷新出进度条；非终端(如日志/管道)下则每 10% 换行打印一次，
    避免把成千上万行进度刷进日志。
    """
    downloaded = block_num * block_size
    is_tty = sys.stdout.isatty()
    if total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        if is_tty:
            bar_len = 30
            filled = int(bar_len * pct / 100)
            bar = "#" * filled + "-" * (bar_len - filled)
            sys.stdout.write(
                f"\r  [{bar}] {pct:5.1f}%  {_human(downloaded)}/{_human(total_size)}"
            )
            sys.stdout.flush()
        else:
            step = int(pct // 10) * 10
            if step != _last_pct[0]:
                _last_pct[0] = step
                print(f"  下载进度 {step:3d}%  {_human(downloaded)}/{_human(total_size)}")
    elif is_tty:
        sys.stdout.write(f"\r  下载中… {_human(downloaded)}")
        sys.stdout.flush()


def download_zip(force: bool = False) -> None:
    """下载 text8.zip（若已存在则跳过）。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(ZIP_PATH) and not force:
        print(f"✓ 压缩包已存在，跳过下载：{ZIP_PATH} ({_human(os.path.getsize(ZIP_PATH))})")
        return
    print(f"↓ 正在下载 text8：{TEXT8_URL}")
    try:
        urllib.request.urlretrieve(TEXT8_URL, ZIP_PATH, _progress_hook)
        sys.stdout.write("\n")
    except Exception as exc:  # noqa: BLE001 - 给用户一个可操作的提示
        sys.stdout.write("\n")
        raise SystemExit(
            f"下载失败：{exc}\n"
            f"可以手动下载 {TEXT8_URL} 并放到 {ZIP_PATH}，或改用镜像地址后重试。"
        )
    print(f"✓ 下载完成：{ZIP_PATH} ({_human(os.path.getsize(ZIP_PATH))})")


def extract_zip(force: bool = False) -> None:
    """解压出 text8 纯文本文件（若已存在且大小正确则跳过）。"""
    if (
        os.path.exists(TEXT8_PATH)
        and os.path.getsize(TEXT8_PATH) == EXPECTED_TEXT8_BYTES
        and not force
    ):
        print(f"✓ 语料已存在，跳过解压：{TEXT8_PATH} ({_human(os.path.getsize(TEXT8_PATH))})")
        return
    print(f"⇲ 正在解压：{ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_DIR)
    size = os.path.getsize(TEXT8_PATH)
    print(f"✓ 解压完成：{TEXT8_PATH} ({_human(size)})")
    if size != EXPECTED_TEXT8_BYTES:
        print(
            f"⚠ 警告：解压后大小为 {size} 字节，与预期的 {EXPECTED_TEXT8_BYTES} 不一致，"
            "文件可能不完整。"
        )


def make_subset(mb: int, force: bool = False) -> str:
    """从 text8 截取前 `mb` MB，生成一个小语料，便于快速训练与调试。

    text8 全文是空格分隔的单词、且没有换行，所以按字节截断后
    需要回退到最后一个完整单词，避免把词切成两半。
    """
    subset_path = os.path.join(DATA_DIR, "text8_small")
    n_bytes = mb * 1024 * 1024
    if os.path.exists(subset_path) and not force:
        print(f"✓ 小语料已存在，跳过：{subset_path} ({_human(os.path.getsize(subset_path))})")
        return subset_path

    with open(TEXT8_PATH, "r", encoding="utf-8") as f:
        chunk = f.read(n_bytes)

    # 回退到最后一个空格，保证不截断单词（除非整块没有空格）
    last_space = chunk.rfind(" ")
    if last_space > 0:
        chunk = chunk[:last_space]

    with open(subset_path, "w", encoding="utf-8") as f:
        f.write(chunk)
    print(
        f"✓ 已生成小语料：{subset_path} "
        f"({_human(os.path.getsize(subset_path))}, 约 {len(chunk.split()):,} 个词)"
    )
    return subset_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="下载 word2vec 教学用的 text8 语料。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        metavar="MB",
        help="额外生成一个只取前 MB 兆字节的小语料（data/text8_small），便于快速训练。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载 / 解压 / 重建小语料。",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("word2vec 数据集下载器 · text8")
    print("=" * 60)

    download_zip(force=args.force)
    extract_zip(force=args.force)

    if args.subset is not None:
        if args.subset <= 0:
            raise SystemExit("--subset 必须是正整数（单位 MB）。")
        make_subset(args.subset, force=args.force)

    print("-" * 60)
    print("完成。数据位于：", DATA_DIR)
    print("在 notebook 里可直接读取：")
    print(f'    with open(r"{TEXT8_PATH}") as f: text = f.read()')
    if args.subset is not None:
        print("小语料（推荐先用它跑通）：")
        print(f'    with open(r"{os.path.join(DATA_DIR, "text8_small")}") as f: text = f.read()')


if __name__ == "__main__":
    main()
