"""Show normalized WhetstoneExample summaries for a dataset split.

Loads a few rows through a registered dataset adapter and prints a compact,
non-dumping summary of each normalized example. This is the first debugging
step when wiring a new dataset: it proves the adapter emits valid examples
before any prompt rendering or model inference happens.

Example:
    python scripts/inspect_dataset.py --dataset openr1_math --split train --limit 3
"""

import argparse

from whetstone.core.types import WhetstoneExample
from whetstone.data import DATASET_REGISTRY, get_dataset_adapter
from whetstone.data.base import preview_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show normalized WhetstoneExample summaries for a dataset split."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help=f"Dataset adapter name; known: {', '.join(DATASET_REGISTRY.names())}",
    )
    parser.add_argument("--split", default="train", help="Dataset split to load")
    parser.add_argument("--limit", type=int, default=3, help="Max examples to load and show")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream rows lazily instead of downloading the full split",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=160,
        help="Max characters for text field previews",
    )
    return parser.parse_args()


def print_example(example: WhetstoneExample, preview_chars: int) -> None:
    print(f"uid: {example.uid}")
    print(f"domain: {example.domain}")
    print(f"source: {example.source}")
    print(f"split: {example.split}")
    print(f"prompt_raw_preview: {preview_text(example.prompt_raw, preview_chars)}")
    print(f"reference_solution_preview: {preview_text(example.reference_solution, preview_chars)}")
    print(f"final_answer: {example.final_answer}")
    if example.tests is not None:
        test_counts = {
            key: len(value) for key, value in example.tests.items() if isinstance(value, list)
        }
        print(f"test_counts: {test_counts}")
    print(f"metadata_keys: {sorted(example.metadata)}")


def main() -> None:
    args = parse_args()
    adapter_kwargs = {"streaming": True} if args.streaming else {}
    adapter = get_dataset_adapter(args.dataset, **adapter_kwargs)
    examples = adapter.load(split=args.split, limit=args.limit)

    print(f"dataset={args.dataset} split={args.split} loaded={len(examples)} example(s)")
    for example in examples:
        print()
        print_example(example, args.preview_chars)


if __name__ == "__main__":
    main()
