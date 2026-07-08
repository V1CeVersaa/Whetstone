"""Run one supervised fine-tuning run from YAML config file(s).

Thin CLI over :func:`whetstone.algorithms.sft.run_sft`. Multiple config paths
are deep-merged in order (later files override earlier ones). Invalid configs
fail before any model or dataset is loaded.

Examples:
    python scripts/train_sft.py --config configs/train/sft_math_overfit.yaml
    python scripts/train_sft.py --config configs/train/sft_math_smoke.yaml
"""

import argparse

from whetstone.algorithms.sft import run_sft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one supervised fine-tuning run from YAML config file(s)."
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
    run_dir = run_sft(args.config)
    print(run_dir)


if __name__ == "__main__":
    main()
