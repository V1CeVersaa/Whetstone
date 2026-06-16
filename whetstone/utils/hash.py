import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def stable_hash(data: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 hex digest of a mapping."""
    payload = json.dumps(dict(data), sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_commit(root: str | Path) -> str | None:
    """Return the current ``HEAD`` commit hash, or ``None`` if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None
