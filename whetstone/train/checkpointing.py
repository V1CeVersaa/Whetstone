import json
from pathlib import Path
from typing import Any

from whetstone.core.paths import ensure_dir
from whetstone.utils.logging import get_logger

logger = get_logger(__name__)

TRAINING_STATE_FILENAME = "training_state.json"


def checkpoint_dir_for_step(run_dir: str | Path, step: int) -> Path:
    """Return ``<run_dir>/checkpoints/step_XXXXXX`` for ``step``."""
    return Path(run_dir) / "checkpoints" / f"step_{step:06d}"


def last_checkpoint_dir(run_dir: str | Path) -> Path:
    """Return ``<run_dir>/checkpoints/last``."""
    return Path(run_dir) / "checkpoints" / "last"


def save_checkpoint(
    *,
    model: Any,
    tokenizer: Any,
    checkpoint_dir: str | Path,
    training_state: dict[str, Any],
) -> Path:
    """Write a plain ``save_pretrained`` checkpoint plus ``training_state.json``.

    Full (unsharded) Hugging Face checkpoints are intentional for this phase:
    they are directly loadable by the Foundation eval runner via
    ``model.name_or_path``. Tokenizers without ``save_pretrained`` (test
    doubles) are skipped.
    """
    target = ensure_dir(checkpoint_dir)
    model.save_pretrained(target)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(target)
    state_path = target / TRAINING_STATE_FILENAME
    state_path.write_text(
        json.dumps(training_state, indent=4, ensure_ascii=True), encoding="utf-8"
    )
    logger.info(f"Saved checkpoint to {target}")
    return target


def load_training_state(checkpoint_dir: str | Path) -> dict[str, Any] | None:
    """Read ``training_state.json`` from a checkpoint dir, or ``None`` if absent."""
    state_path = Path(checkpoint_dir) / TRAINING_STATE_FILENAME
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))
