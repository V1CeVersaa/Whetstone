from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any

from whetstone.train.types import MathRolloutSample
from whetstone.utils.jsonl import append_jsonl


class MetricsLogger:
    """Appends one JSON object per logging event to ``metrics.jsonl``.

    Training metrics are a stream, not a single final dict: every row carries
    its ``step`` so loss curves can be reconstructed from the artifact alone.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def log(self, step: int, values: dict[str, Any]) -> dict[str, Any]:
        row = {"step": step, **values}
        append_jsonl(row, self.path)
        return row


def summarize_rollout_step(
    samples: list[MathRolloutSample],
    advantages: list[float],
    *,
    group_size: int,
) -> dict[str, Any]:
    """Aggregate one RL step's rollout samples into loggable metrics.

    The single most important signal is ``nonzero_group_variance_rate``: groups
    whose rewards all agree produce zero advantages and therefore no learning
    signal. If this rate sits near zero, the run is not learning anything.
    """
    rewards = [sample.reward for sample in samples]
    completion_tokens = [sample.num_completion_tokens for sample in samples]
    reasons = [sample.verifier_reason for sample in samples]
    parse_failures = {
        "empty_completion",
        "too_long",
        "no_answer_found",
        "parse_error",
        "unsupported_expression",
    }

    groups = [rewards[i : i + group_size] for i in range(0, len(rewards), group_size)]
    nonzero_variance_groups = sum(1 for group in groups if len(set(group)) > 1)
    all_correct_groups = sum(1 for group in groups if min(group) == 1.0)
    all_wrong_groups = sum(1 for group in groups if max(group) == 0.0)

    return {
        "num_samples": len(samples),
        "num_groups": len(groups),
        "mean_reward": fmean(rewards) if rewards else 0.0,
        "reward_std": pstdev(rewards) if len(rewards) > 1 else 0.0,
        "pass_rate": sum(1 for sample in samples if sample.passed) / max(1, len(samples)),
        "parse_success_rate": sum(1 for reason in reasons if reason not in parse_failures)
        / max(1, len(reasons)),
        "no_answer_rate": reasons.count("no_answer_found") / max(1, len(reasons)),
        "wrong_answer_rate": reasons.count("wrong_answer") / max(1, len(reasons)),
        "avg_completion_tokens": fmean(completion_tokens) if completion_tokens else 0.0,
        "median_completion_tokens": median(completion_tokens) if completion_tokens else 0.0,
        "group_reward_std": fmean(pstdev(group) if len(group) > 1 else 0.0 for group in groups)
        if groups
        else 0.0,
        "nonzero_group_variance_rate": nonzero_variance_groups / max(1, len(groups)),
        "all_correct_group_rate": all_correct_groups / max(1, len(groups)),
        "all_wrong_group_rate": all_wrong_groups / max(1, len(groups)),
        "mean_advantage": fmean(advantages) if advantages else 0.0,
        "mean_abs_advantage": fmean(abs(adv) for adv in advantages) if advantages else 0.0,
    }
