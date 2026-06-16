from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from whetstone.utils.jsonl import read_jsonl, write_jsonl

T = TypeVar("T")


def shard_sequence(items: Sequence[T], *, rank: int, world_size: int) -> list[T]:
    """Return this rank's strided slice of ``items``.

    Strided (``items[rank::world_size]``) sharding partitions the sequence with
    no duplicates or gaps across ranks, and is easy to verify.

    Raises:
        ValueError: If ``world_size`` is not positive.
    """
    if world_size < 1:
        msg = f"world_size must be positive, got {world_size}"
        raise ValueError(msg)
    return list(items[rank::world_size])


def merge_jsonl_files(
    inputs: Sequence[str | Path],
    output: str | Path,
    *,
    sort_key: Callable[[dict[str, Any]], Any] | None = None,
) -> None:
    """Concatenate several JSONL files into one, optionally restoring row order.

    Used on rank 0 to merge per-rank prediction files, avoiding write races.
    Strided sharding writes rank-local files out of original sequence order, so
    callers can pass ``sort_key`` when a stable row index is available.
    """
    rows = []
    for path in inputs:
        rows.extend(read_jsonl(path))
    if sort_key is not None:
        rows.sort(key=sort_key)
    write_jsonl(rows, output)
