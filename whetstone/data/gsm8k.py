from collections.abc import Sequence
from typing import Any

from whetstone.core.types import WhetstoneExample
from whetstone.data.base import load_hf_rows


class GSM8KAdapter:
    """Adapter for GSM8K.

    Parses the ``#### answer`` convention into ``final_answer`` so the math
    verifier's exact-match path can be exercised without a large download. Pass
    ``rows=`` to run against in-memory fixtures instead of the Hub.
    """

    name = "gsm8k"

    def __init__(
        self,
        *,
        rows: Sequence[dict[str, Any]] | None = None,
        # The Hub retired bare canonical ids; "gsm8k" alone is rejected by
        # datasets >= 5.x, so the namespaced repo id is required.
        dataset_name: str = "openai/gsm8k",
        config_name: str = "main",
        streaming: bool = False,
    ) -> None:
        self.rows = list(rows) if rows is not None else None
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.streaming = streaming

    def load(self, split: str, limit: int | None = None) -> list[WhetstoneExample]:
        """Load up to ``limit`` GSM8K rows from fixtures or the Hub as examples."""
        rows = (
            list(self.rows[:limit] if limit is not None else self.rows)
            if self.rows is not None
            else load_hf_rows(
                self.dataset_name,
                split,
                limit,
                streaming=self.streaming,
                name=self.config_name,
            )
        )
        return [
            self.row_to_example(row, split=split, index=index) for index, row in enumerate(rows)
        ]

    def row_to_example(
        self, row: dict[str, Any], *, split: str, index: int = 0
    ) -> WhetstoneExample:
        """Normalize one GSM8K row, extracting the ``#### answer`` final answer."""
        question = str(row.get("question"))
        answer = str(row.get("answer"))
        final_answer = extract_gsm8k_final_answer(answer)
        uid = f"gsm8k:{split}:{index}"
        return WhetstoneExample(
            uid=uid,
            domain="math",
            source="gsm8k",
            split=split,
            prompt_raw=question,
            reference_solution=answer,
            final_answer=final_answer,
            metadata={"dataset": "gsm8k", "row_index": index},
        )


def extract_gsm8k_final_answer(answer: str) -> str | None:
    """Return the text after the last ``####`` marker, or ``None`` if absent."""
    if "####" not in answer:
        return None
    tail = answer.rsplit("####", maxsplit=1)[-1]
    return tail.strip().rstrip(".") or None
