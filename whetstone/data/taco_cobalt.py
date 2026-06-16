import json
from collections.abc import Sequence
from typing import Any

from whetstone.core.types import WhetstoneExample
from whetstone.data.base import load_hf_rows


class TacoCobaltAdapter:
    """Adapter for TACO-Cobalt.

    Splits test cases into ``public`` and ``hidden`` lists of ``{"input",
    "output"}`` dicts. When only a combined ``test_cases`` field exists, the
    first four are treated as public and the rest as hidden, matching the
    dataset's documented layout. Pass ``rows=`` for in-memory fixtures.
    """

    name = "taco_cobalt"

    def __init__(
        self,
        *,
        rows: Sequence[dict[str, Any]] | None = None,
        dataset_name: str = "osunlp/TACO-Cobalt",
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
        """Normalize one TACO-Cobalt row into a code example."""
        public_tests = parse_test_cases(row.get("public_test_cases"))
        hidden_tests = parse_test_cases(row.get("hidden_test_cases"))

        return WhetstoneExample(
            uid=f"taco_cobalt:{split}:{row.get('id')}",
            domain="code",
            source="taco_cobalt",
            split=split,
            prompt_raw=str(row.get("question")),
            reference_solution=None,
            tests={"public": public_tests, "hidden": hidden_tests},
            metadata={
                "difficulty": row.get("difficulty"),
                "row_index": index,
                "has_hidden_tests": bool(hidden_tests),
                "num_public_tests": len(public_tests),
                "num_hidden_tests": len(hidden_tests),
            },
        )


def parse_test_cases(value: Any) -> list[dict[str, str]]:
    """Normalize varied test-case encodings into ``[{"input", "output"}, ...]``.

    Handles JSON-encoded strings, parallel ``inputs``/``outputs`` arrays, single
    ``input``/``output`` (or ``stdin``/``stdout``) dicts, and nested lists.
    Unrecognized shapes yield an empty list.
    """
    value = parse_json_if_needed(value)

    if value is None:
        return []

    if isinstance(value, dict):
        if "inputs" in value and "outputs" in value:
            return [
                {"input": str(test_input), "output": str(test_output)}
                for test_input, test_output in zip(value["inputs"], value["outputs"], strict=False)
            ]
        if "input" in value and "output" in value:
            return [{"input": str(value["input"]), "output": str(value["output"])}]
        if "stdin" in value and "stdout" in value:
            return [{"input": str(value["stdin"]), "output": str(value["stdout"])}]

    if isinstance(value, list):
        cases: list[dict[str, str]] = []
        for item in value:
            parsed = parse_test_cases(item)
            cases.extend(parsed)
        return cases

    return []


def parse_json_if_needed(value: Any) -> Any:
    """Parse ``value`` as JSON only if it looks like a JSON array/object string.

    Non-strings pass through unchanged; strings that do not start with ``[``/``{``
    or fail to parse are returned as-is.
    """
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    if stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value
