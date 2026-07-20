import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SOURCE_FINGERPRINT_PATHS = (
    "whetstone",
    "scripts",
    "configs",
    "pyproject.toml",
    "uv.lock",
)
IGNORED_SOURCE_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}


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


def git_source_state(root: str | Path) -> dict[str, Any]:
    """Capture commit, dirty state, and a content hash for executable project sources.

    Repository dirtiness and runtime-source dirtiness are separate. Untracked
    research notes or result documents can leave the repository dirty without
    changing the code/config that produced a run. ``source_dirty`` only covers
    paths in ``SOURCE_FINGERPRINT_PATHS`` and is the reproducibility gate.
    """
    root_path = Path(root)
    repo_status = _git_status(root_path)
    source_status = _git_status(root_path, SOURCE_FINGERPRINT_PATHS)
    source_hash, source_file_count = source_tree_hash(root_path)
    return {
        "git_commit": git_commit(root_path),
        "git_dirty": bool(repo_status),
        "git_status": repo_status,
        "source_dirty": bool(source_status),
        "source_status": source_status,
        "source_tree_sha256": source_hash,
        "source_file_count": source_file_count,
        "source_scope": list(SOURCE_FINGERPRINT_PATHS),
    }


def source_tree_hash(root: str | Path) -> tuple[str, int]:
    """Hash tracked and untracked runtime source files in stable path order."""
    root_path = Path(root)
    relative_paths = _git_source_files(root_path)
    if relative_paths is None:
        relative_paths = _walk_source_files(root_path)

    digest = hashlib.sha256()
    count = 0
    for relative in sorted(set(relative_paths), key=lambda path: path.as_posix()):
        path = root_path / relative
        if not path.is_file() or any(part in IGNORED_SOURCE_PARTS for part in relative.parts):
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
        count += 1
    return digest.hexdigest(), count


def _git_status(root: Path, paths: tuple[str, ...] = ()) -> list[str]:
    command = ["git", "status", "--porcelain=v1", "--untracked-files=all"]
    if paths:
        command.extend(["--", *paths])
    try:
        result = subprocess.run(
            command,
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def _git_source_files(root: Path) -> list[Path] | None:
    try:
        result = subprocess.run(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                *SOURCE_FINGERPRINT_PATHS,
            ],
            cwd=root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [
        Path(item.decode("utf-8", errors="surrogateescape"))
        for item in result.stdout.split(b"\0")
        if item
    ]


def _walk_source_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for scope in SOURCE_FINGERPRINT_PATHS:
        target = root / scope
        if target.is_file():
            paths.append(target.relative_to(root))
        elif target.is_dir():
            paths.extend(path.relative_to(root) for path in target.rglob("*") if path.is_file())
    return paths
