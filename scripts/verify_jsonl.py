"""Re-run a verifier over saved predictions.jsonl completions.

Reconstructs each row's example and completion from the flattened prediction
schema, re-verifies without regenerating anything, and reports how verdicts
changed. This is how verifier versions are compared on identical raw outputs:
raw completions are the source of truth, so improving a verifier must never
require rerunning the model.

Example:
    python scripts/verify_jsonl.py \
        --input runs/<run_dir>/predictions.jsonl \
        --verifier math_answer \
        --output runs/<run_dir>/predictions_reverified.jsonl
"""

import argparse
import json
from collections import Counter
from typing import Any

from whetstone.core.config import VerifierConfig
from whetstone.core.types import ModelCompletion, VerificationResult, WhetstoneExample
from whetstone.eval.metrics import compute_metrics
from whetstone.utils.jsonl import read_jsonl_list, write_jsonl
from whetstone.verify import VERIFIER_REGISTRY, build_verifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run a verifier over saved predictions.jsonl completions."
    )
    parser.add_argument("--input", required=True, help="Path to a predictions.jsonl file")
    parser.add_argument(
        "--verifier",
        required=True,
        help=f"Verifier name; known: {', '.join(VERIFIER_REGISTRY.names())}",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write re-verified rows (never overwrites the input in place)",
    )
    parser.add_argument("--max-chars", type=int, default=20000, help="[math_answer] length cap")
    parser.add_argument("--tests", default="public", help="[code_exec] test group to grade")
    parser.add_argument(
        "--timeout-seconds", type=float, default=3.0, help="[code_exec] per-test timeout"
    )
    parser.add_argument(
        "--max-output-bytes", type=int, default=20000, help="[code_exec] stdout/stderr cap"
    )
    parser.add_argument(
        "--sandbox-backend", default="subprocess", help="[code_exec] execution backend"
    )
    return parser.parse_args()


def row_to_example(row: dict[str, Any]) -> WhetstoneExample:
    """Rebuild the WhetstoneExample embedded in a flattened prediction row."""
    domain = row.get("domain")
    if domain not in ("math", "code"):
        msg = f"row {row.get('uid')!r} has unsupported domain {domain!r}"
        raise ValueError(msg)
    return WhetstoneExample(
        uid=str(row.get("uid")),
        domain=domain,
        source=str(row.get("source")),
        split=str(row.get("split")),
        prompt_raw=str(row.get("prompt_raw") or ""),
        reference_solution=row.get("reference_solution"),
        final_answer=row.get("gold_answer"),
        tests=row.get("tests"),
        metadata=row.get("example_metadata") or {},
    )


def row_to_completion(row: dict[str, Any]) -> ModelCompletion:
    """Rebuild the ModelCompletion embedded in a flattened prediction row."""
    return ModelCompletion(
        uid=str(row.get("uid")),
        completion=str(row.get("completion") or ""),
        full_text=str(row.get("full_text") or ""),
        num_prompt_tokens=int(row.get("num_prompt_tokens") or 0),
        num_completion_tokens=int(row.get("num_completion_tokens") or 0),
        finish_reason=row.get("finish_reason"),
        generation_metadata=row.get("generation_metadata") or {},
    )


def apply_verification(row: dict[str, Any], result: VerificationResult) -> dict[str, Any]:
    """Return a copy of ``row`` with the verification fields replaced."""
    return {
        **row,
        "extracted_answer": result.extracted_answer,
        "passed": result.passed,
        "reward": result.reward,
        "score": result.score,
        "reason": result.reason,
        "diagnostics": result.diagnostics,
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl_list(args.input)
    verifier = build_verifier(
        VerifierConfig(
            name=args.verifier,
            max_chars=args.max_chars,
            tests=args.tests,
            timeout_seconds=args.timeout_seconds,
            max_output_bytes=args.max_output_bytes,
            sandbox_backend=args.sandbox_backend,
        )
    )

    reverified_rows: list[dict[str, Any]] = []
    transitions: Counter[tuple[str, str]] = Counter()
    for row in rows:
        result = verifier.verify(row_to_example(row), row_to_completion(row))
        if str(row.get("reason")) != result.reason or bool(row.get("passed")) != result.passed:
            transitions[(str(row.get("reason")), result.reason)] += 1
        reverified_rows.append(apply_verification(row, result))

    old_passed = sum(1 for row in rows if bool(row.get("passed")))
    new_passed = sum(1 for row in reverified_rows if bool(row.get("passed")))
    print(f"reverified {len(rows)} row(s) with {args.verifier}")
    print(f"passed: {old_passed} -> {new_passed} ({new_passed - old_passed:+d})")
    if transitions:
        print("changed verdicts (old reason -> new reason):")
        for (old_reason, new_reason), count in sorted(transitions.items()):
            print(f"  {old_reason} -> {new_reason}: {count}")
    else:
        print("no verdicts changed")

    print()
    print(json.dumps(compute_metrics(reverified_rows), indent=2, ensure_ascii=True))

    if args.output:
        write_jsonl(reverified_rows, args.output)
        print(f"\nwrote {len(reverified_rows)} re-verified row(s) to {args.output}")


if __name__ == "__main__":
    main()
