"""Package the repo's code, tests, configs, and docs into one zip for review.

Uses ``git ls-files --cached --others --exclude-standard`` so the file list is
exactly what the repository considers source: tracked files plus untracked
ones, with everything in ``.gitignore`` (.venv, __pycache__, .pytest_cache,
runs/, ...) excluded automatically. Falls back to a filtered directory walk if
git is unavailable.

Example:
    python scripts/package_repo.py                # -> whetstone_fable.zip
    python scripts/package_repo.py --output /tmp/whetstone_fable.zip
"""

import argparse
import subprocess
import zipfile
from pathlib import Path

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
    "runs",
}
EXCLUDED_SUFFIXES = {".zip", ".pyc", ".pyo", ".so"}
EXCLUDED_NAMES = {".DS_Store"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def git_file_list(root: Path) -> list[Path] | None:
    """Tracked + untracked-but-not-ignored files, or None when git is unusable."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [root / line for line in result.stdout.splitlines() if line.strip()]


def walked_file_list(root: Path) -> list[Path]:
    """Fallback: walk the tree, skipping the well-known junk directories."""
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        files.append(path)
    return files


def should_include(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part in EXCLUDED_DIRS for part in relative.parts):
        return False
    if path.suffix in EXCLUDED_SUFFIXES or path.name in EXCLUDED_NAMES:
        return False
    return path.is_file()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package code, tests, configs, and docs into one zip."
    )
    parser.add_argument(
        "--output",
        default="whetstone_fable.zip",
        help="Output zip path (default: whetstone_fable.zip in the repo root)",
    )
    args = parser.parse_args()

    root = repo_root()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output

    files = git_file_list(root)
    source = "git ls-files"
    if files is None:
        files = walked_file_list(root)
        source = "directory walk"
    files = sorted(path for path in files if should_include(path, root) and path != output)

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=f"whetstone_fable/{path.relative_to(root)}")

    size_kib = output.stat().st_size / 1024
    print(f"wrote {output} ({len(files)} files, {size_kib:.0f} KiB, file list via {source})")


if __name__ == "__main__":
    main()
