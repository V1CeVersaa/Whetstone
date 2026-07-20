"""Run one Math-RL v0 training run from YAML config file(s), with CLI overrides.

Thin CLI over :func:`whetstone.algorithms.math_rl.run_math_rl`. Multiple config
paths deep-merge in order; CLI flags merge on top of the files, so the saved
``train_config.yaml`` always reflects exactly what ran. The policy should
initialize from an SFT checkpoint.

Examples:
    python scripts/train_math_rl.py --config configs/train/math_rl_smoke.yaml

    # Same config, different SFT checkpoint -- no config edit needed
    python scripts/train_math_rl.py --config configs/train/math_rl_smoke.yaml \
        --policy runs/<sft_run>/checkpoints/last --run-name math_rl_smoke_v2

    # Arbitrary dotted overrides for anything without a dedicated flag
    python scripts/train_math_rl.py --config configs/train/math_rl_smoke.yaml \
        --set rl.max_steps=5 --set rl.eval_every=null
"""

import argparse
from typing import Any

from whetstone.algorithms.math_rl import run_math_rl
from whetstone.core.config import parse_set_overrides, put_dotted_override


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Math-RL v0 training run from YAML config file(s), with CLI overrides."
    )
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="One or more YAML config paths, deep-merged in order",
    )
    parser.add_argument(
        "--policy",
        default=None,
        help="Override model.policy_name_or_path (SFT checkpoint directory or hub id)",
    )
    parser.add_argument("--run-name", default=None, help="Override run.name")
    parser.add_argument("--seed", type=int, default=None, help="Override run.seed")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Generic dotted override, e.g. rl.max_steps=5; values are parsed "
        "as YAML (ints, floats, bools, null). Repeatable; later items win.",
    )
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Translate CLI flags into a nested config-override dict (--set wins)."""
    overrides: dict[str, Any] = {}
    if args.policy is not None:
        put_dotted_override(overrides, "model.policy_name_or_path", args.policy)
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
    run_dir = run_math_rl(args.config, overrides=overrides)
    print(run_dir)


if __name__ == "__main__":
    main()
