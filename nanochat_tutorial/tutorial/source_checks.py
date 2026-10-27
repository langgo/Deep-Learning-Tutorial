"""Guard tutorial statements that are easy to invalidate when nanochat changes.

Run from nanochat_tutorial/:
    python3 tutorial/source_checks.py

The checks only read source files and use Python's standard library.
They intentionally fail loudly after an upstream change so the prose is reviewed.
"""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "nanochat"


def text(path):
    return (UPSTREAM / path).read_text(encoding="utf-8")


def require(path, *needles):
    source = text(path)
    missing = [needle for needle in needles if needle not in source]
    assert not missing, f"{path} missing expected source facts: {missing}"


def check_pipeline():
    speedrun = text("runs/speedrun.sh")
    for stage in ("scripts.tok_train", "scripts.base_train", "scripts.base_eval", "scripts.chat_sft", "scripts.chat_eval"):
        assert stage in speedrun, f"speedrun no longer contains {stage}"
    assert "--depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8" in speedrun
    assert "scripts.chat_rl" not in speedrun, "speedrun now includes RL; update chapters 01/13"


def check_model_contract():
    require(
        "nanochat/gpt.py",
        'window_pattern: str = "SSSL"',
        "class GPT",
        "softcap = 15",
        "self.smear_gate = Linear(24, 1, bias=False)",
        "self.backout_lambda",
        "self.resid_lambdas",
        "self.x0_lambdas",
    )
    require(
        "scripts/base_train.py",
        "n_kv_head=num_heads",
        'parser.add_argument("--head-dim", type=int, default=128',
        'default=12, help="calculate num_iterations to maintain data:param ratio',
        'default="SSSL"',
    )
    require(
        "nanochat/optim.py",
        "class MuonAdamW",
        "elif p.numel() < 1024",
        "dist.reduce_scatter_tensor",
    )


def check_data_contract():
    require(
        "nanochat/dataloader.py",
        "Handles DDP sharding and approximate resume",
        "Every row starts with BOS",
        "Approximately 35% of all tokens are discarded due to cropping",
    )
    require("nanochat/dataset.py", "MAX_SHARD = 6542", "base_data_climbmix")


def check_sft_and_eval_contract():
    sft = text("scripts/chat_sft.py")
    for task in ("SmolTalk", "MMLU", "GSM8K"):
        assert task in sft
    assert "SpellingBee" not in sft, "SFT mixture changed; update chapter 09"
    require(
        "scripts/chat_eval.py",
        "['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval']",
        "all_tasks_were_evaluated",
    )


def check_rl_contract():
    require(
        "scripts/chat_rl.py",
        "advantages = rewards - mu",
        "there is no need to add PPO ratio+clip because we are on policy",
        "None, # note: we don't bother to save the optimizer state",
    )


def check_inference_benchmark():
    require(
        "scripts/infer_bench.py",
        "TTFT",
        "TPOT",
        "MBU",
        "infer_bench currently assumes a CUDA GPU",
        "ddp_world_size == 1",
    )


def _heading_slug(title):
    """Approximate GitHub's Markdown heading slug for the headings used here."""
    title = title.strip().lower()
    title = re.sub(r"[`*~]", "", title)
    title = re.sub(r"[^\w\-\u4e00-\u9fff ]", "", title)
    return re.sub(r"-+", "-", title.replace(" ", "-")).strip("-")


def _markdown_anchors(path):
    anchors = set()
    counts = {}
    in_code = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            in_code = not in_code
            continue
        match = None if in_code else re.match(r"^#{1,6}\s+(.+)$", line)
        if not match:
            continue
        base = _heading_slug(match.group(1))
        count = counts.get(base, 0)
        counts[base] = count + 1
        anchors.add(base if count == 0 else f"{base}-{count}")
    return anchors


def check_local_links():
    anchor_cache = {}
    for markdown in (ROOT / "tutorial").glob("*.md"):
        source = markdown.read_text(encoding="utf-8")
        for target in source.split("](")[1:]:
            target = target.split(")", 1)[0]
            link, _, fragment = target.partition("#")
            if link.startswith("./"):
                destination = (markdown.parent / link).resolve()
                assert destination.exists(), f"broken link in {markdown}: {link}"
                if fragment and destination.suffix == ".md":
                    anchors = anchor_cache.setdefault(destination, _markdown_anchors(destination))
                    assert fragment in anchors, f"broken anchor in {markdown}: {target}"


def check_no_external_reference_mentions():
    # The book must not advertise a specific third-party aggregator by name.
    # Build the forbidden term dynamically so this file itself does not contain it.
    forbidden = ("deep" + "wiki").lower()
    for path in (ROOT / "tutorial").rglob("*"):
        if not path.is_file() or path.name == "source_checks.py":
            continue
        if forbidden in path.name.lower():
            raise AssertionError(f"forbidden term appears in filename: {path}")
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if forbidden in text.lower():
            raise AssertionError(f"forbidden term appears in file contents: {path}")


def main():
    checks = [
        check_pipeline,
        check_model_contract,
        check_data_contract,
        check_sft_and_eval_contract,
        check_rl_contract,
        check_inference_benchmark,
        check_local_links,
        check_no_external_reference_mentions,
    ]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print(f"\nAll {len(checks)} source checks passed.")


if __name__ == "__main__":
    main()
