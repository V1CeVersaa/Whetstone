"""Run one supervised fine-tuning run from YAML config file(s), with CLI overrides.

Thin CLI over :func:`whetstone.algorithms.sft.run_sft`. Multiple config paths
deep-merge in order; CLI flags merge on top of the files, so the saved
``train_config.yaml`` always reflects exactly what ran. Invalid configs fail
before any model or dataset is loaded.

Examples:
    python scripts/train_sft.py --config configs/train/sft_math_overfit.yaml

    # Same config, different base model or budget -- no config edit needed
    python scripts/train_sft.py --config configs/train/sft_math_smoke.yaml \
        --model Qwen/Qwen3-1.7B-Base --run-name sft_math_smoke_1p7B \
        --set training.max_steps=100
"""

import argparse
from typing import Any

from whetstone.algorithms.sft import run_sft
from whetstone.core.config import parse_set_overrides, put_dotted_override


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one supervised fine-tuning run from YAML config file(s), with CLI overrides."
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
    parser.add_argument("--run-name", default=None, help="Override run.name")
    parser.add_argument("--seed", type=int, default=None, help="Override run.seed")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Generic dotted override, e.g. training.max_steps=100; values are "
        "parsed as YAML (ints, floats, bools, null). Repeatable; later items win.",
    )
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Translate CLI flags into a nested config-override dict (--set wins)."""
    overrides: dict[str, Any] = {}
    if args.model is not None:
        put_dotted_override(overrides, "model.name_or_path", args.model)
    if args.run_name is not None:
        put_dotted_override(overrides, "run.name", args.run_name)
    if args.seed is not None:
        put_dotted_override(overrides, "run.seed", args.seed)
    return parse_set_overrides(args.set, into=overrides)


def main() -> None:
    args = parse_args()
    try:
        overrides = build_overrides(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    run_dir = run_sft(args.config, overrides=overrides)
    print(run_dir)


if __name__ == "__main__":
    main()
