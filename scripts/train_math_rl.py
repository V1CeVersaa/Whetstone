"""Run one Math-RL v0 training run from YAML config file(s).

Thin CLI over :func:`whetstone.algorithms.math_rl.run_math_rl`. The policy
should initialize from an SFT checkpoint (set ``model.policy_name_or_path`` to
the checkpoint directory). With ``rl.kl_beta: 0.0`` no reference model is
loaded.

Example:
    python scripts/train_math_rl.py --config configs/train/math_rl_smoke.yaml
"""

import argparse

from whetstone.algorithms.math_rl import run_math_rl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Math-RL v0 training run from YAML config file(s)."
    )
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="One or more YAML config paths, deep-merged in order",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_math_rl(args.config)
    print(run_dir)


if __name__ == "__main__":
    main()
