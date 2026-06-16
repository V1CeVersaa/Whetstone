from pathlib import Path


def project_root() -> Path:
    """Return the project root ``.../Whetstone``."""
    return Path(__file__).resolve().parents[2]  # Whetstone/whetstone/core/paths.py -> Whetstone/


def resolve_project_path(path: str | Path) -> Path:
    """Resolve ``path`` against the project root, leaving absolute paths as-is."""
    return Path(path).resolve() if Path(path).is_absolute() else (project_root() / path).resolve()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
