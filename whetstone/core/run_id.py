import re
from datetime import datetime


def slugify(value: str) -> str:
    """Lowercase ``value`` and collapse non-alphanumerics to underscores. Returns ``"run"`` if ``value`` is empty."""
    lowered = value.lower()
    collapsed = re.sub(r"[^a-z0-9]", "_", lowered)
    return collapsed.strip("_") or "run"


def make_run_id(name: str, timestamp: datetime | None = None) -> str:
    """Build a sortable run id like ``"2026-06-12_153000_my_run"`` from a name and optional timestamp."""
    timestamp = (timestamp or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    return f"{timestamp}_{slugify(name)}"
