import ast
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:
    import resource
except ImportError:
    resource = None

FORBIDDEN_IMPORT_ROOTS = {
    "multiprocessing",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "threading",
}

type StaticValidationFailure = Literal["empty_code", "compile_error", "forbidden_import"]


@dataclass(frozen=True)
class ProgramRunResult:
    """Outcome of running a candidate program once.

    ``returncode`` is ``None`` when the run was killed for timing out. ``stdout``
    and ``stderr`` are already truncated to the configured preview size.
    """

    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    excessive_output: bool


def validate_python_program(code: str) -> StaticValidationFailure | None:
    """Statically screen generated code before executing it.

    Returns a failure reason (``empty_code``, ``compile_error``, or
    ``forbidden_import``) or ``None`` if the code passes screening. This is a
    cheap filter for accidental bad cases, not a security boundary.
    """
    if not code.strip():
        return "empty_code"
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "compile_error"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", maxsplit=1)[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    return "forbidden_import"

        if isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", maxsplit=1)[0]
            if root in FORBIDDEN_IMPORT_ROOTS:
                return "forbidden_import"

    return None


def run_python_subprocess(
    code: str,
    *,
    stdin: str,
    timeout_seconds: float,
    max_output_bytes: int,
) -> ProgramRunResult:
    """Run ``code`` in a subprocess with one test's stdin and capture the result.

    Executes in a fresh temporary directory with a wall-clock timeout and (on
    Unix) a file-size rlimit, returning a :class:`ProgramRunResult`. This is
    the Foundation development backend, not a complete sandbox.
    """
    with tempfile.TemporaryDirectory(prefix="whetstone_code_") as temp_dir:
        program_path = Path(temp_dir) / "solution.py"
        program_path.write_text(code, encoding="utf-8")
        # The subprocess backend is for Foundation smoke tests; it is not a full sandbox.
        process = subprocess.Popen(
            [sys.executable, str(program_path)],
            cwd=temp_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _apply_subprocess_resource_limits(process.pid)
        try:
            stdout, stderr = process.communicate(stdin, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            return ProgramRunResult(
                returncode=None,
                stdout=preview(stdout, max_output_bytes),
                stderr=preview(stderr, max_output_bytes),
                timed_out=True,
                excessive_output=False,
            )
        excessive = (
            len(stdout.encode("utf-8")) > max_output_bytes
            or len(stderr.encode("utf-8")) > max_output_bytes
        )
        return ProgramRunResult(
            returncode=process.returncode,
            stdout=preview(stdout, max_output_bytes),
            stderr=preview(stderr, max_output_bytes),
            timed_out=False,
            excessive_output=excessive,
        )


def _apply_subprocess_resource_limits(pid: int) -> None:
    """Apply best-effort post-spawn resource limits to a subprocess.

    Avoids ``preexec_fn``, which is unsafe in multithreaded parent processes.
    Linux provides ``resource.prlimit`` for setting limits on an existing child;
    platforms without it silently run without this extra guardrail.
    """
    if resource is None:
        return

    prlimit = getattr(resource, "prlimit", None)
    if prlimit is None:
        return
    try:
        prlimit(pid, resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    except OSError:
        return


def preview(text: str, max_bytes: int) -> str:
    """Truncate ``text`` to at most ``max_bytes`` UTF-8 bytes, decoding leniently."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="replace")


def normalize_stdout(text: str) -> str:
    """Canonicalize program output for comparison.

    Normalizes line endings, strips trailing whitespace per line, and drops
    trailing blank lines so cosmetic differences do not count as wrong answers.
    """
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines).strip()
