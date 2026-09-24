"""nanochat tutorial: zero-data, zero-checkpoint checks.

Run from the repository root:
    python3 tutorial/toy_checks.py

Only Python's standard library is required.
"""

from math import exp, log


def check_shift_and_sft_mask():
    row = [10, 11, 12, 13, 14, 15]
    mask = [0, 0, 0, 1, 1, 1]
    x, y = row[:-1], row[1:]
    targets = [token if keep else -1 for token, keep in zip(y, mask[1:])]
    assert x == [10, 11, 12, 13, 14]
    assert y == [11, 12, 13, 14, 15]
    assert targets == [-1, -1, 13, 14, 15]


def cross_entropy(logits, target):
    """Cross-entropy for one position, using a stable log-sum-exp."""
    peak = max(logits)
    return -(logits[target] - peak - log(sum(exp(x - peak) for x in logits)))


def check_cross_entropy():
    logits = [0.0, log(2), 0.0]
    loss = cross_entropy(logits, target=1)
    assert abs(loss - log(2)) < 1e-12  # probabilities are [1/4, 1/2, 1/4]


def check_attention_shapes():
    batch, time, channels, heads = 2, 5, 12, 3
    head_dim = channels // heads
    assert channels % heads == 0
    assert (batch, time, channels) == (2, 5, 12)
    assert (batch, time, heads, head_dim) == (2, 5, 3, 4)  # q/k/v
    assert (batch, heads, time, time) == (2, 3, 5, 5)      # attention scores


def check_bpb():
    total_nats = log(2) * 12
    total_bytes = 8
    bpb = total_nats / (log(2) * total_bytes)
    assert bpb == 1.5


def check_group_advantages():
    rewards = [1.0, 0.0, 1.0, 0.0]
    mean = sum(rewards) / len(rewards)
    advantages = [reward - mean for reward in rewards]
    assert advantages == [0.5, -0.5, 0.5, -0.5]
    assert abs(sum(advantages)) < 1e-12


def pass_at_k(outcomes, k):
    return any(outcomes[:k])


def check_pass_at_k():
    outcomes = [False, False, True, False]
    assert not pass_at_k(outcomes, 1)
    assert not pass_at_k(outcomes, 2)
    assert pass_at_k(outcomes, 3)
    assert pass_at_k(outcomes, 4)


def main():
    checks = [
        check_shift_and_sft_mask,
        check_cross_entropy,
        check_attention_shapes,
        check_bpb,
        check_group_advantages,
        check_pass_at_k,
    ]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print(f"\nAll {len(checks)} checks passed.")


if __name__ == "__main__":
    main()
