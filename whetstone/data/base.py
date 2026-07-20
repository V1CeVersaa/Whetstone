from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, Protocol

from datasets import load_dataset

from whetstone.core.registry import Registry
from whetstone.core.types import WhetstoneExample
from whetstone.utils.logging import get_logger

logger = get_logger(__name__)


class DatasetAdapter(Protocol):
    name: str
    domain: Literal["math", "code"]

    def load(self, split: str, limit: int | None = None) -> list[WhetstoneExample]: ...


DATASET_REGISTRY: Registry[Callable[..., DatasetAdapter]] = Registry()


def take_rows(rows: Iterable[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """Materialize up to ``limit`` rows into a list of plain dicts."""
    selected: list[dict[str, Any]] = []
    for row in rows:
        selected.append(dict(row))
        if limit is not None and len(selected) >= limit:
            break
    return selected


def load_hf_rows(
    dataset_name: str,
    split: str,
    limit: int | None,
    *,
    data_files: str | Sequence[str] | dict[str, str] | None = None,
    streaming: bool = False,
    name: str | None = None,
) -> list[dict[str, Any]]:
    """Load rows from a Hugging Face dataset and return up to ``limit`` of them.

    Args:
        dataset_name: Hub dataset id or local loader name.
        split: Split to load, e.g. ``"train"``.
        limit: Max rows to return; ``None`` for all.
        data_files: Optional explicit data files passed to ``load_dataset``.
        streaming: If True, iterate lazily instead of downloading the full split.
        name: Optional dataset config/subset name.

    Returns:
        A list of plain row dicts.
    """
    load_split = split
    take_limit = limit
    if not streaming and limit is not None and "[" not in split:
        load_split = f"{split}[:{limit}]"
        take_limit = None

    # Hub loads can involve slow downloads; say what is being fetched and how.
    logger.info(f"Loading dataset {dataset_name} split={split} limit={limit} streaming={streaming}")
    dataset = load_dataset(
        dataset_name,
        name=name,
        split=load_split,
        data_files=data_files,
        streaming=streaming,
    )
    return take_rows(dataset, take_limit)


def preview_text(value: str | None, max_chars: int = 120) -> str:
    """Collapse whitespace and truncate text to ``max_chars`` for compact display."""
    if not value:
        return ""
    compact = " ".join(str(value).split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 3]}..."


def get_dataset_adapter(name: str, **kwargs: Any) -> DatasetAdapter:
    """Resolve a dataset name to its adapter instance."""
    factory = DATASET_REGISTRY.get(name.strip().lower())
    return factory(**kwargs)


def get_dataset_domain(name: str) -> Literal["math", "code"]:
    """Return the declared domain of a registered dataset without loading rows."""
    factory = DATASET_REGISTRY.get(name.strip().lower())
    domain = getattr(factory, "domain", None)
    if domain not in {"math", "code"}:
        msg = f"Dataset adapter {name!r} does not declare a supported domain"
        raise ValueError(msg)
    return domain
