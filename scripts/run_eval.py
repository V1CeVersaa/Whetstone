"""Run one evaluation from YAML config file(s), with CLI overrides.

The general evaluation entry point: smoke tests, checkpoint evaluation, and
benchmark comparisons all go through here. Multiple config paths deep-merge in
order; CLI flags merge on top of the files, so the saved ``run_config.yaml``
always reflects exactly what ran. Works both as a plain Python process and
under torchrun (distributed state is auto-detected).

Examples:
    # Smoke test as-is
    python scripts/run_eval.py --config configs/eval/gsm8k_smoke.yaml

    # Same benchmark, different checkpoints -- no overlay file needed
    python scripts/run_eval.py --config configs/eval/gsm8k_benchmark.yaml
    python scripts/run_eval.py --config configs/eval/gsm8k_benchmark.yaml \
        --model runs/<sft_run>/checkpoints/last --run-name gsm8k_benchmark_sft

    # Arbitrary dotted overrides for anything without a dedicated flag
    python scripts/run_eval.py --config configs/eval/gsm8k_smoke.yaml \
        --set generation.max_new_tokens=256 --set dataset.limit=null

    torchrun --standalone --nproc_per_node=2 scripts/run_eval.py \
        --config configs/eval/openr1_math_smoke.yaml
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from whetstone.eval.runner import run_evaluation

SUMMARY_KEYS = [
    "num_examples",
    "accuracy",
    "pass_at_1",
    "mean_reward",
    "parse_success_rate",
    "no_answer_rate",
    "wrong_answer_rate",
    "avg_completion_tokens",
    "tokens_per_second",
    "wall_clock_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one evaluation from YAML config file(s), with CLI overrides."
    )
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="One or more YAML config paths, deep-merged in order",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model.name_or_path (hub id or local checkpoint directory)",
    )
    parser.add_argument("--dataset", default=None, help="Override dataset.name")
    parser.add_argument("--split", default=None, help="Override dataset.split")
    parser.add_argument(
        "--limit",
        default=None,
        help="Override dataset.limit (an integer, or 'null' for the full split)",
    )
    parser.add_argument("--run-name", default=None, help="Override run.name")
    parser.add_argument("--output-dir", default=None, help="Override run.output_dir")
    parser.add_argument("--seed", type=int, default=None, help="Override run.seed")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Generic dotted override, e.g. generation.max_new_tokens=256; "
        "values are parsed as YAML (ints, floats, bools, null). Repeatable.",
    )
    return parser.parse_args()


def put_dotted(overrides: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``overrides['a']['b'] = value`` for a dotted key ``"a.b"``."""
    keys = dotted_key.split(".")
    node = overrides
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Translate CLI flags into a nested config-override dict."""
    overrides: dict[str, Any] = {}
    if args.model is not None:
        put_dotted(overrides, "model.name_or_path", args.model)
    if args.dataset is not None:
        put_dotted(overrides, "dataset.name", args.dataset)
    if args.split is not None:
        put_dotted(overrides, "dataset.split", args.split)
    if args.limit is not None:
        put_dotted(overrides, "dataset.limit", yaml.safe_load(args.limit))
    if args.run_name is not None:
        put_dotted(overrides, "run.name", args.run_name)
    if args.output_dir is not None:
        put_dotted(overrides, "run.output_dir", args.output_dir)
    if args.seed is not None:
        put_dotted(overrides, "run.seed", args.seed)
    for item in args.set:
        dotted_key, separator, raw_value = item.partition("=")
        if not separator or not dotted_key:
            msg = f"--set expects SECTION.KEY=VALUE, got {item!r}"
            raise SystemExit(msg)
        put_dotted(overrides, dotted_key.strip(), yaml.safe_load(raw_value))
    return overrides


def print_summary(run_dir: Path) -> None:
    """Print the run's headline metrics (rank 0 only; metrics.json is its artifact)."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    print(f"\n=== {run_dir} ===")
    for key in SUMMARY_KEYS:
        value = metrics.get(key)
        if value is None:
            continue
        formatted = f"{value:.4f}" if isinstance(value, float) else value
        print(f"{key}: {formatted}")


def main() -> None:
    args = parse_args()
    run_dir = run_evaluation(args.config, overrides=build_overrides(args))
    if int(os.environ.get("RANK", "0")) == 0:
        print_summary(Path(run_dir))
        print(run_dir)


if __name__ == "__main__":
    main()
