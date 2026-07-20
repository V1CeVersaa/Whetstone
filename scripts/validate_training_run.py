"""Validate a completed SFT or Math-RL run directory without loading a model."""

import argparse
import json

from whetstone.train.validation import validate_training_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate training artifacts, provenance, and SFT/RL invariants."
    )
    parser.add_argument("--run-dir", required=True, help="Completed training run directory")
    parser.add_argument(
        "--allow-missing-checkpoint",
        action="store_true",
        help="Allow summaries copied with checkpoints excluded",
    )
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="Report but do not reject source_dirty=true",
    )
    parser.add_argument(
        "--allow-legacy-provenance",
        action="store_true",
        help="Allow runs created before source_state and math-verify version capture",
    )
    parser.add_argument(
        "--require-loss-decrease",
        action="store_true",
        help="Require final SFT loss to be lower than the first measured loss",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_training_run(
        args.run_dir,
        require_checkpoint=not args.allow_missing_checkpoint,
        require_clean_source=not args.allow_dirty_source,
        require_provenance=not args.allow_legacy_provenance,
        require_loss_decrease=args.require_loss_decrease,
    )
    print(json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
