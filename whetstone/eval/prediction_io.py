from pathlib import Path

from whetstone.core.types import PredictionRecord
from whetstone.utils.jsonl import read_jsonl, write_jsonl


def prediction_to_row(record: PredictionRecord) -> dict:
    """Flatten a :class:`PredictionRecord` into its JSONL row form."""
    return record.to_flat_dict()


def write_predictions(records: list[PredictionRecord], path: str | Path) -> None:
    """Write prediction records to ``path`` as one flattened JSON object per line."""
    write_jsonl((prediction_to_row(record) for record in records), path)


def read_prediction_rows(path: str | Path) -> list[dict]:
    """Read a ``predictions.jsonl`` file into a list of row dicts."""
    return list(read_jsonl(path))
