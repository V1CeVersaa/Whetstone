"""Offline validation for completed SFT and Math-RL run directories."""

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml

from whetstone.core.config import load_yaml
from whetstone.utils.hash import stable_hash
from whetstone.utils.jsonl import read_jsonl_list

PASSING_REASONS = {"correct", "passed"}
MODEL_WEIGHT_FILES = {
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
}


def validate_training_run(
    run_dir: str | Path,
    *,
    require_checkpoint: bool = True,
    require_clean_source: bool = True,
    require_provenance: bool = True,
    require_loss_decrease: bool = False,
) -> dict[str, Any]:
    """Validate one completed training run and return a machine-readable report."""
    path = Path(run_dir)
    errors: list[str] = []
    config = _load_yaml(path / "train_config.yaml", errors)
    env = _load_json(path / "env.json", errors)
    status = _load_json(path / "status.json", errors)
    final_metrics = _load_json(path / "final_metrics.json", errors)
    metric_rows = _load_jsonl(path / "metrics.jsonl", errors)
    kind = _run_kind(config, errors)
    summary: dict[str, Any] = {"run_dir": str(path), "kind": kind}

    _validate_common(
        path,
        config=config,
        env=env,
        status=status,
        final_metrics=final_metrics,
        metric_rows=metric_rows,
        errors=errors,
        summary=summary,
        require_checkpoint=require_checkpoint,
        require_clean_source=require_clean_source,
        require_provenance=require_provenance,
    )
    if kind == "sft":
        _validate_sft(
            path,
            config=config,
            final_metrics=final_metrics,
            metric_rows=metric_rows,
            errors=errors,
            summary=summary,
            require_loss_decrease=require_loss_decrease,
        )
    elif kind == "math_rl":
        _validate_math_rl(
            path,
            config=config,
            final_metrics=final_metrics,
            metric_rows=metric_rows,
            errors=errors,
            summary=summary,
        )

    return {"valid": not errors, "errors": errors, "summary": summary}


def _validate_common(
    run_dir: Path,
    *,
    config: dict[str, Any],
    env: dict[str, Any],
    status: dict[str, Any],
    final_metrics: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    errors: list[str],
    summary: dict[str, Any],
    require_checkpoint: bool,
    require_clean_source: bool,
    require_provenance: bool,
) -> None:
    if status.get("status") != "completed":
        errors.append(f"status.json status is {status.get('status')!r}, expected 'completed'")
    if not metric_rows:
        errors.append("metrics.jsonl contains no rows")
    _append_nonfinite_errors(final_metrics, "final_metrics.json", errors)
    _append_nonfinite_errors(metric_rows, "metrics.jsonl", errors)

    if require_checkpoint:
        _validate_checkpoint(run_dir / "checkpoints" / "last", errors)
    _validate_provenance(
        env,
        config,
        errors,
        summary,
        require_clean_source=require_clean_source,
        require_provenance=require_provenance,
    )
    summary["status"] = status.get("status")
    summary["metric_rows"] = len(metric_rows)
    summary["num_steps"] = final_metrics.get("num_steps")
    summary["peak_gpu_memory_mb"] = final_metrics.get("peak_gpu_memory_mb")


def _validate_provenance(
    env: dict[str, Any],
    config: dict[str, Any],
    errors: list[str],
    summary: dict[str, Any],
    *,
    require_clean_source: bool,
    require_provenance: bool,
) -> None:
    source_state = env.get("source_state")
    packages = env.get("packages")
    config_sha256 = env.get("config_sha256")
    expected_config_sha256 = stable_hash(config)
    if config_sha256 != expected_config_sha256:
        errors.append(
            f"env.json config_sha256={config_sha256!r} does not match "
            f"train_config.yaml={expected_config_sha256!r}"
        )
    if not isinstance(source_state, dict):
        if require_provenance:
            errors.append("env.json is missing source_state provenance")
        return

    required_fields = {"git_commit", "source_dirty", "source_tree_sha256", "source_file_count"}
    missing_fields = sorted(required_fields - source_state.keys())
    if missing_fields and require_provenance:
        errors.append(f"env.json source_state is missing fields: {missing_fields}")
    if require_provenance and not source_state.get("git_commit"):
        errors.append("env.json source_state has no git_commit")
    source_hash = source_state.get("source_tree_sha256")
    if require_provenance and (not isinstance(source_hash, str) or len(source_hash) != 64):
        errors.append("env.json source_state has an invalid source_tree_sha256")
    if require_provenance and not isinstance(source_state.get("source_file_count"), int):
        errors.append("env.json source_state has an invalid source_file_count")
    if require_clean_source and source_state.get("source_dirty") is not False:
        errors.append(
            f"runtime source was dirty: {source_state.get('source_status', '<missing status>')}"
        )
    if require_provenance and (not isinstance(packages, dict) or not packages.get("math-verify")):
        errors.append("env.json packages is missing the math-verify version")

    summary["git_commit"] = source_state.get("git_commit")
    summary["git_dirty"] = source_state.get("git_dirty")
    summary["source_dirty"] = source_state.get("source_dirty")
    summary["source_tree_sha256"] = source_state.get("source_tree_sha256")
    summary["config_sha256"] = config_sha256
    if isinstance(packages, dict):
        summary["math_verify_package"] = packages.get("math-verify")


def _validate_checkpoint(checkpoint_dir: Path, errors: list[str]) -> None:
    if not checkpoint_dir.is_dir():
        errors.append(f"missing checkpoint directory: {checkpoint_dir}")
        return
    names = {path.name for path in checkpoint_dir.iterdir() if path.is_file()}
    if "config.json" not in names:
        errors.append(f"checkpoint has no config.json: {checkpoint_dir}")
    if not names.intersection(MODEL_WEIGHT_FILES):
        errors.append(f"checkpoint has no recognized model weights: {checkpoint_dir}")
    if "tokenizer_config.json" not in names:
        errors.append(f"checkpoint has no tokenizer_config.json: {checkpoint_dir}")


def _validate_sft(
    run_dir: Path,
    *,
    config: dict[str, Any],
    final_metrics: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    errors: list[str],
    summary: dict[str, Any],
    require_loss_decrease: bool,
) -> None:
    preprocessing = _load_json(run_dir / "preprocessing.json", errors)
    tokenization = _load_json(run_dir / "tokenization_audit.json", errors)
    configured_threshold = float(
        config.get("preprocessing", {}).get("max_decode_mismatch_rate", 0.0)
    )
    decode_mismatch_rate = _as_float(tokenization.get("decode_mismatch_rate"))
    if decode_mismatch_rate is None:
        errors.append("tokenization_audit.json is missing decode_mismatch_rate")
    elif decode_mismatch_rate > configured_threshold:
        errors.append(f"decode_mismatch_rate={decode_mismatch_rate} exceeds {configured_threshold}")

    candidate_rate = _required_rate(preprocessing, "target_candidate_verifiable_rate", errors)
    emitted_rate = _required_rate(preprocessing, "emitted_target_verifier_pass_rate", errors)
    initial_parse_failure_rate = _required_rate(
        preprocessing, "target_initial_parse_failure_rate", errors
    )
    initial_answer_mismatch_rate = _required_rate(
        preprocessing, "target_initial_answer_mismatch_rate", errors
    )
    declared_conflict_rate = _required_rate(preprocessing, "target_declared_conflict_rate", errors)
    ensure_verifiable = bool(config.get("preprocessing", {}).get("ensure_verifiable_target", True))
    if ensure_verifiable and emitted_rate is not None and not math.isclose(emitted_rate, 1.0):
        errors.append(f"emitted_target_verifier_pass_rate={emitted_rate}, expected 1.0")

    first_loss = _as_float(final_metrics.get("first_train_loss"))
    final_loss = _as_float(final_metrics.get("final_train_loss"))
    loss_decreased = first_loss is not None and final_loss is not None and final_loss < first_loss
    if require_loss_decrease and not loss_decreased:
        errors.append(f"SFT loss did not decrease: first={first_loss}, final={final_loss}")
    _validate_expected_steps(config.get("training", {}), final_metrics, errors)
    grad_norms = _finite_metric_values(metric_rows, "grad_norm")
    if len(grad_norms) != len(metric_rows):
        errors.append("SFT metrics rows must all contain a finite grad_norm")

    summary.update(
        {
            "num_loaded_examples": preprocessing.get("num_loaded_examples"),
            "num_sft_examples": preprocessing.get("num_sft_examples"),
            "num_training_examples": final_metrics.get("num_training_examples"),
            "num_overlong_dropped": final_metrics.get("num_overlong_dropped"),
            "target_candidate_verifiable_rate": candidate_rate,
            "emitted_target_verifier_pass_rate": emitted_rate,
            "target_initial_parse_failure_rate": initial_parse_failure_rate,
            "target_initial_answer_mismatch_rate": initial_answer_mismatch_rate,
            "target_declared_conflict_rate": declared_conflict_rate,
            "decode_mismatch_rate": decode_mismatch_rate,
            "separate_joint_mismatch_rate": tokenization.get("separate_joint_mismatch_rate"),
            "overlong_rate": tokenization.get("overlong_rate"),
            "first_train_loss": first_loss,
            "final_train_loss": final_loss,
            "loss_decreased": loss_decreased,
            "grad_norm_min": min(grad_norms) if grad_norms else None,
            "grad_norm_max": max(grad_norms) if grad_norms else None,
        }
    )


def _validate_math_rl(
    run_dir: Path,
    *,
    config: dict[str, Any],
    final_metrics: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    errors: list[str],
    summary: dict[str, Any],
) -> None:
    rollout_rows = _load_jsonl(run_dir / "rollout_samples.jsonl", errors)
    rl_config = config.get("rl", {})
    group_size = int(rl_config.get("group_size", 0) or 0)
    prompts_per_step = int(rl_config.get("prompts_per_step", 0) or 0)
    max_steps = int(rl_config.get("max_steps", 0) or 0)
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rollout_rows):
        _validate_rollout_row(row, index=index, errors=errors)
        step = row.get("step")
        group_id = row.get("group_id")
        if isinstance(step, int) and isinstance(group_id, str) and group_id:
            groups[(step, group_id)].append(row)

    max_advantage_sum_abs = 0.0
    for key, rows in groups.items():
        if len(rows) != group_size:
            errors.append(f"group {key!r} has {len(rows)} rows, expected {group_size}")
        advantage_sum = sum(_as_float(row.get("advantage")) or 0.0 for row in rows)
        max_advantage_sum_abs = max(max_advantage_sum_abs, abs(advantage_sum))
        if not math.isclose(advantage_sum, 0.0, abs_tol=1.0e-8):
            errors.append(f"group {key!r} advantage sum is {advantage_sum}")

    expected_rows = max_steps * prompts_per_step * group_size
    expected_groups = max_steps * prompts_per_step
    if len(rollout_rows) != expected_rows:
        errors.append(f"rollout row count is {len(rollout_rows)}, expected {expected_rows}")
    if len(groups) != expected_groups:
        errors.append(f"rollout group count is {len(groups)}, expected {expected_groups}")
    if final_metrics.get("num_rollout_samples") != len(rollout_rows):
        errors.append("final_metrics num_rollout_samples does not match rollout_samples.jsonl")
    _validate_expected_steps(rl_config, final_metrics, errors)

    reward_values = [
        value for row in rollout_rows if (value := _as_float(row.get("reward"))) is not None
    ]
    reward_mean = fmean(reward_values) if reward_values else None
    reported_reward = _as_float(final_metrics.get("mean_reward_overall"))
    if (
        reward_mean is not None
        and reported_reward is not None
        and not math.isclose(reward_mean, reported_reward, abs_tol=1.0e-12)
    ):
        errors.append(
            f"mean_reward_overall={reported_reward} does not match rollouts={reward_mean}"
        )
    nonzero_rates = _finite_metric_values(metric_rows, "nonzero_group_variance_rate")
    mean_abs_advantages = _finite_metric_values(metric_rows, "mean_abs_advantage")
    grad_norms = _finite_metric_values(metric_rows, "grad_norm")
    completion_lengths = _finite_metric_values(metric_rows, "avg_completion_tokens")
    boxed_completion_rates = _finite_metric_values(metric_rows, "boxed_completion_rate")
    required_metric_values = {
        "nonzero_group_variance_rate": nonzero_rates,
        "mean_abs_advantage": mean_abs_advantages,
        "grad_norm": grad_norms,
        "avg_completion_tokens": completion_lengths,
        "boxed_completion_rate": boxed_completion_rates,
    }
    for key, values in required_metric_values.items():
        if len(values) != len(metric_rows):
            errors.append(f"Math-RL metrics rows must all contain a finite {key}")
    summary.update(
        {
            "num_rollout_samples": len(rollout_rows),
            "num_groups": len(groups),
            "group_size": group_size,
            "max_advantage_group_sum_abs": max_advantage_sum_abs,
            "mean_reward_overall": reward_mean,
            "mean_nonzero_group_variance_rate": (fmean(nonzero_rates) if nonzero_rates else None),
            "mean_abs_advantage": (fmean(mean_abs_advantages) if mean_abs_advantages else None),
            "grad_norm_min": min(grad_norms) if grad_norms else None,
            "grad_norm_max": max(grad_norms) if grad_norms else None,
            "mean_completion_tokens": (fmean(completion_lengths) if completion_lengths else None),
            "mean_boxed_completion_rate": (
                fmean(boxed_completion_rates) if boxed_completion_rates else None
            ),
        }
    )


def _validate_rollout_row(row: dict[str, Any], *, index: int, errors: list[str]) -> None:
    reward = _as_float(row.get("reward"))
    passed = row.get("passed")
    reason = row.get("verifier_reason")
    advantage = _as_float(row.get("advantage"))
    if reward not in {0.0, 1.0}:
        errors.append(f"rollout row {index} has non-binary reward {reward!r}")
    if not isinstance(passed, bool):
        errors.append(f"rollout row {index} has non-boolean passed={passed!r}")
    elif reward is not None and reward != float(passed):
        errors.append(f"rollout row {index} reward={reward} disagrees with passed={passed}")
    if not isinstance(reason, str) or not reason:
        errors.append(f"rollout row {index} has no verifier_reason")
    elif isinstance(passed, bool) and ((reason in PASSING_REASONS) != passed):
        errors.append(
            f"rollout row {index} verifier_reason={reason!r} disagrees with passed={passed}"
        )
    if advantage is None or not math.isfinite(advantage):
        errors.append(f"rollout row {index} has invalid advantage={row.get('advantage')!r}")


def _validate_expected_steps(
    config_block: dict[str, Any], final_metrics: dict[str, Any], errors: list[str]
) -> None:
    expected = config_block.get("max_steps")
    actual = final_metrics.get("num_steps")
    if expected is not None and actual != expected:
        errors.append(f"final num_steps={actual!r}, expected configured max_steps={expected!r}")


def _run_kind(config: dict[str, Any], errors: list[str]) -> str:
    has_sft = "training" in config
    has_rl = "rl" in config
    if has_sft == has_rl:
        errors.append("train_config.yaml must contain exactly one of training or rl")
        return "unknown"
    return "sft" if has_sft else "math_rl"


def _required_rate(payload: dict[str, Any], key: str, errors: list[str]) -> float | None:
    value = _as_float(payload.get(key))
    if value is None:
        errors.append(f"preprocessing.json is missing {key}")
    elif not 0.0 <= value <= 1.0:
        errors.append(f"preprocessing.json {key} is outside [0, 1]: {value}")
    return value


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _finite_metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _as_float(row.get(key))
        if value is not None and math.isfinite(value):
            values.append(value)
    return values


def _append_nonfinite_errors(value: Any, label: str, errors: list[str]) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(f"{label} contains non-finite value")
    elif isinstance(value, dict):
        for key, child in value.items():
            _append_nonfinite_errors(child, f"{label}.{key}", errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _append_nonfinite_errors(child, f"{label}[{index}]", errors)


def _load_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"missing required artifact: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"could not read {path}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"expected JSON object in {path}")
        return {}
    return value


def _load_yaml(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"missing required artifact: {path}")
        return {}
    try:
        return load_yaml(path)
    except (OSError, TypeError, yaml.YAMLError) as exc:
        errors.append(f"could not read {path}: {exc}")
        return {}


def _load_jsonl(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        errors.append(f"missing required artifact: {path}")
        return []
    try:
        return read_jsonl_list(path)
    except (OSError, TypeError, ValueError) as exc:
        errors.append(f"could not read {path}: {exc}")
        return []
