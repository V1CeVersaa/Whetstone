"""Print a run's aggregate metrics and representative failures.

Reads metrics.json and predictions.jsonl from a run directory and shows the
metrics followed by a few sample rows per non-passing verifier reason. Looking
at concrete failures grouped by reason is much faster than staring at
aggregate accuracy.

Example:
    python scripts/summarize_run.py --run_dir runs/<run_dir>
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from whetstone.utils.jsonl import read_jsonl_list

PASSING_REASONS = {"correct", "passed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a run's aggregate metrics and representative failures."
    )
    parser.add_argument("--run_dir", required=True, help="Run directory to summarize")
    parser.add_argument(
        "--per-reason", type=int, default=3, help="Max sample rows shown per failure reason"
    )
    parser.add_argument(
        "--preview-chars", type=int, default=240, help="Max characters of completion tail shown"
    )
    return parser.parse_args()


def tail_preview(text: str, max_chars: int) -> str:
    """Collapse whitespace and keep the tail of ``text``, where final answers live."""
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return f"...{compact[-max_chars:]}"


def print_sample(row: dict[str, Any], preview_chars: int) -> None:
    print(f"  uid: {row.get('uid')}")
    print(f"    passed={row.get('passed')} reward={row.get('reward')}")
    if row.get("domain") == "math":
        print(f"    gold_answer: {row.get('gold_answer')}")
        print(f"    extracted_answer: {row.get('extracted_answer')}")
    else:
        diagnostics = row.get("diagnostics") or {}
        print(
            f"    tests: {diagnostics.get('num_passed')}/{diagnostics.get('num_tests')} passed, "
            f"first_failed_test_index={diagnostics.get('first_failed_test_index')}"
        )
        stderr_preview = str(diagnostics.get("stderr_preview") or "").strip()
        if stderr_preview:
            print(f"    stderr: {tail_preview(stderr_preview, preview_chars)}")
    print(f"    completion_tail: {tail_preview(str(row.get('completion') or ''), preview_chars)}")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "predictions.jsonl"

    if metrics_path.exists():
        print(f"# metrics ({metrics_path})")
        print(json.dumps(json.loads(metrics_path.read_text(encoding="utf-8")), indent=2))
    else:
        print(f"missing {metrics_path}")

    if not predictions_path.exists():
        print(f"missing {predictions_path}")
        return

    rows = read_jsonl_list(predictions_path)
    by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_reason[str(row.get("reason") or "unknown")].append(row)

    print(f"\n# outcomes ({len(rows)} prediction(s))")
    for reason in sorted(by_reason, key=lambda key: -len(by_reason[key])):
        print(f"  {reason}: {len(by_reason[reason])}")

    failure_reasons = [reason for reason in sorted(by_reason) if reason not in PASSING_REASONS]
    for reason in failure_reasons:
        print(f"\n# {reason} ({len(by_reason[reason])} row(s), showing up to {args.per_reason})")
        for row in by_reason[reason][: args.per_reason]:
            print_sample(row, args.preview_chars)


if __name__ == "__main__":
    main()
