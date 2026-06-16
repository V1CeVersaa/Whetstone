import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield each JSON object from a JSONL file, skipping blank lines.

    Args:
        path: Path to the ``.jsonl`` file.

    Yields:
        One decoded dict per non-empty line.

    Raises:
        ValueError: On a line that is not valid JSON (message includes the line number).
        TypeError: On a line whose JSON value is not an object.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                value = json.loads(stripped)
                if not isinstance(value, dict):
                    raise TypeError(f"Expected dict, got {type(value)} on line {line_num}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Error parsing JSONL line {line_num}: {e}") from e

            yield value


def write_jsonl(records: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    """Write records as one JSON object per line, creating parent dirs as needed.

    Accepts any iterable (including generators) so large prediction streams need
    not be materialized in memory.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True, sort_keys=False) + "\n")


def append_jsonl(record: Mapping[str, Any], path: str | Path) -> None:
    """Append a single record as one JSON line, creating parent dirs as needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True, sort_keys=False) + "\n")


def read_jsonl_list(path: str | Path) -> list[dict[str, Any]]:
    """Eagerly read a JSONL file into a list of dicts."""
    return list(read_jsonl(path))
