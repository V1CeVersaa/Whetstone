from collections import Counter
from statistics import fmean, median
from typing import Any

# Extraction failures under math_verify: nothing judgeable was produced.
FAILURE_REASONS = {
    "empty_completion",
    "too_long",
    "no_answer_found",
    "verifier_error",
}


def compute_metrics(
    rows: list[dict[str, Any]], *, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Aggregate prediction rows into a metrics dict.

    Selects domain-specific metrics (math, code, or mixed), merges in common
    token/reward stats, and appends any ``extra`` run-level values. Computed from
    saved rows so analysis can be rerun without regenerating completions.
    """
    if not rows:
        return {"num_examples": 0, **(extra or {})}
    domains = {row.get("domain") for row in rows}

    if domains == {"math"}:
        metrics = compute_math_metrics(rows)
    elif domains == {"code"}:
        metrics = compute_code_metrics(rows)
    else:
        metrics = compute_mixed_metrics(rows)
    metrics.update(compute_common_metrics(rows))

    if extra:
        metrics.update(extra)
    return metrics


def compute_common_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute domain-agnostic stats: counts, mean reward, token stats, reason counts."""
    completion_tokens = [int(row.get("num_completion_tokens") or 0) for row in rows]
    prompt_tokens = [int(row.get("num_prompt_tokens") or 0) for row in rows]
    rewards = [float(row.get("reward") or 0.0) for row in rows]
    return {
        "num_examples": len(rows),
        "mean_reward": sum(rewards) / len(rewards),
        "avg_prompt_tokens": sum(prompt_tokens) / len(prompt_tokens),
        "avg_completion_tokens": sum(completion_tokens) / len(completion_tokens),
        "median_completion_tokens": median(completion_tokens),
        "reason_counts": dict(Counter(str(row.get("reason")) for row in rows)),
    }


def compute_math_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute math-specific rates: accuracy, parse success, and failure-reason rates."""
    num_examples = len(rows)
    reasons = Counter(str(row.get("reason")) for row in rows)
    parsed = [row for row in rows if row.get("reason") not in FAILURE_REASONS]
    return {
        "accuracy": count_passed(rows) / num_examples,
        "parse_success_rate": len(parsed) / num_examples,
        "boxed_completion_rate": sum(
            1 for row in rows if "\\boxed{" in str(row.get("completion") or "")
        )
        / num_examples,
        "no_answer_rate": reasons["no_answer_found"] / num_examples,
        "wrong_answer_rate": reasons["wrong_answer"] / num_examples,
        "conflicting_answer_rate": count_diagnostic_flag(rows, "had_conflict") / num_examples,
    }


def compute_code_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute code-specific rates: pass@1, average public-test pass rate, error rates."""
    num_examples = len(rows)
    reasons = Counter(str(row.get("reason")) for row in rows)
    rewards = (float(row.get("reward") or 0.0) for row in rows)
    return {
        "pass_at_1": count_passed(rows) / num_examples,
        "avg_public_test_pass_rate": fmean(rewards),
        "compile_error_rate": reasons["compile_error"] / num_examples,
        "runtime_error_rate": reasons["runtime_error"] / num_examples,
        "timeout_rate": reasons["timeout"] / num_examples,
        "wrong_answer_rate": reasons["wrong_answer"] / num_examples,
        "empty_code_rate": reasons["empty_code"] / num_examples,
        "forbidden_import_rate": reasons["forbidden_import"] / num_examples,
    }


def compute_mixed_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute a minimal accuracy metric when rows span multiple domains."""
    return {"accuracy": count_passed(rows) / len(rows)}


def count_passed(rows: list[dict[str, Any]]) -> int:
    """Count rows whose ``passed`` flag is truthy."""
    return sum(1 for row in rows if bool(row.get("passed")))


def count_diagnostic_flag(rows: list[dict[str, Any]], key: str) -> int:
    """Count rows whose diagnostics mapping contains a truthy flag named ``key``."""
    total = 0
    for row in rows:
        diagnostics = row.get("diagnostics")
        if isinstance(diagnostics, dict) and bool(diagnostics.get(key)):
            total += 1
    return total
