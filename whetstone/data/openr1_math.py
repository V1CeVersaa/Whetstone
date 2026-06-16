from collections.abc import Sequence
from typing import Any

from whetstone.core.types import WhetstoneExample
from whetstone.data.base import load_hf_rows


class OpenR1MathAdapter:
    """Adapter for OpenR1-Math-220k.

    Preserves reasoning-dataset provenance (``problem_type``, ``question_type``,
    ``source``, ``uuid``, correctness metadata, ...) under ``metadata`` rather
    than flattening it away. Pass ``rows=`` for in-memory fixtures.
    """

    name = "openr1_math"

    def __init__(
        self,
        *,
        rows: Sequence[dict[str, Any]] | None = None,
        dataset_name: str = "open-r1/OpenR1-Math-220k",
        streaming: bool = False,
    ) -> None:
        self.rows = list(rows) if rows is not None else None
        self.dataset_name = dataset_name
        self.streaming = streaming

    def load(self, split: str, limit: int | None = None) -> list[WhetstoneExample]:
        rows = (
            list(self.rows[:limit] if limit is not None else self.rows)
            if self.rows is not None
            else load_hf_rows(self.dataset_name, split, limit, streaming=self.streaming)
        )
        return [
            self.row_to_example(row, split=split, index=index) for index, row in enumerate(rows)
        ]

    def row_to_example(
        self, row: dict[str, Any], *, split: str, index: int = 0
    ) -> WhetstoneExample:
        """Normalize one OpenR1-Math row, keeping reasoning metadata in ``metadata``."""
        metadata_keys = [
            "problem_type",
            "question_type",
            "source",
            "uuid",
            "correctness_math_verify",
            "correctness_llama",
            "finish_reasons",
        ]
        metadata = {key: row[key] for key in metadata_keys if key in row}
        metadata["row_index"] = index

        return WhetstoneExample(
            uid=f"openr1_math:{split}:{row.get('uuid')}",
            domain="math",
            source="openr1_math",
            split=split,
            prompt_raw=str(row.get("problem")),
            reference_solution=str(row.get("solution")),
            final_answer=str(row.get("answer")),
            metadata=metadata,
        )
