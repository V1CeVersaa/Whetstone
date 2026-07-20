import logging
import os
from pathlib import Path

# All Whetstone loggers live under this namespace so configuration can target
# the whole tree at once without touching the root logger (and other libraries).
NAMESPACE = "whetstone"

# Third-party libraries that install their own stream handlers and formats.
# We re-home them onto our handler so their messages read like ours instead of
# a second style interleaved with it (e.g. transformers' pad_token / generation
# advisories, datasets download notices, hub auth warnings).
LIBRARY_LOGGERS = ("transformers", "datasets", "huggingface_hub", "accelerate")


def _is_configured() -> bool:
    """Whether the namespace logger already carries a console handler.

    A run-dir file handler alone does not count: levels and console output
    are configure_logging's job, and attaching a log file must not suppress it.
    """
    return any(
        not isinstance(handler, logging.FileHandler)
        for handler in logging.getLogger(NAMESPACE).handlers
    )


def _make_formatter(rank: int) -> logging.Formatter:
    """The one log format, stamping ``rank`` into every line."""
    return logging.Formatter(
        fmt=f"%(asctime)s %(levelname)s [rank{rank}] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _make_handler(rank: int) -> logging.StreamHandler:
    """Build the shared stream handler."""
    handler = logging.StreamHandler()
    handler.setFormatter(_make_formatter(rank))
    return handler


def _remove_stream_handlers(logger: logging.Logger) -> None:
    """Drop console handlers, keeping attached run-file handlers.

    Reconfiguration (e.g. ``configure_logging(force=True)`` once the real rank
    is known) must not silently detach a run directory's log file.
    """
    for handler in list(logger.handlers):
        if not isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)


def configure_logging(
    level: int | str = logging.INFO,
    *,
    rank: int | None = None,
    force: bool = False,
) -> None:
    """Install one stream handler on the ``whetstone`` logger tree.

    Idempotent: repeated calls are no-ops unless ``force`` is set, so importing
    a module that logs does not stack duplicate handlers. The rank is stamped
    into every line, which is what makes multi-process ``torchrun`` output
    readable; it defaults to the ``RANK`` env var (``0`` when unset). Also
    re-homes third-party library logs onto the same handler so all output
    follows one format (see :func:`route_library_logging`).

    Args:
        level: Log level for the namespace logger (name or numeric).
        rank: Process rank to stamp; defaults to ``$RANK`` or ``0``.
        force: Reconfigure even if already configured (e.g. once the real rank
            is known after distributed init).
    """
    if _is_configured() and not force:
        return
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))
    logger = logging.getLogger(NAMESPACE)
    _remove_stream_handlers(logger)
    logger.addHandler(_make_handler(rank))
    logger.setLevel(level)
    # Don't bubble up to the root logger; we own this subtree's formatting.
    logger.propagate = False
    route_library_logging(rank=rank)


def route_library_logging(*, rank: int | None = None, level: int = logging.WARNING) -> None:
    """Re-home third-party library logs onto Whetstone's handler and format.

    Libraries like ``transformers`` install their own root handler with a
    different format; left alone their messages interleave with ours in a
    second style. This clears each library logger's handlers, attaches our
    rank-stamped handler, and stops propagation so every line reads
    ``HH:MM:SS LEVEL [rankN] transformers: ...``. Non-main ranks are muted to
    errors only, so ``torchrun`` runs are not N times as noisy.

    Call after the libraries are imported (the run entry points do, via the
    ``force=True`` reconfigure); importing a library later re-adds its own
    handler, so a subsequent forced reconfigure re-homes it again.
    """
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))
    effective_level = level if rank == 0 else logging.ERROR
    handler = _make_handler(rank)
    for name in LIBRARY_LOGGERS:
        lib_logger = logging.getLogger(name)
        _remove_stream_handlers(lib_logger)
        lib_logger.addHandler(handler)
        lib_logger.setLevel(effective_level)
        lib_logger.propagate = False


def attach_run_dir_logging(run_dir: str | Path, *, rank: int | None = None) -> Path:
    """Mirror every Whetstone and library log line into a file in ``run_dir``.

    Queued or detached runs (pueue, nohup, ...) lose stdout; the run.log file
    keeps the full log beside the artifacts it describes. Rank 0 writes
    ``run.log``; other ranks write ``run_rank{N}.log`` so torchrun processes
    never interleave one file. Idempotent per target file, and reconfiguration
    via ``configure_logging(force=True)`` preserves file handlers, so call
    order does not matter. Returns the log file path.
    """
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))
    target = Path(run_dir) / ("run.log" if rank == 0 else f"run_rank{rank}.log")
    target.parent.mkdir(parents=True, exist_ok=True)

    target_resolved = target.resolve()
    loggers = [logging.getLogger(NAMESPACE)] + [logging.getLogger(n) for n in LIBRARY_LOGGERS]
    file_handler: logging.FileHandler | None = None
    for logger in loggers:
        if any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename).resolve() == target_resolved
            for handler in logger.handlers
        ):
            continue
        if file_handler is None:
            file_handler = logging.FileHandler(target, encoding="utf-8")
            file_handler.setFormatter(_make_formatter(rank))
        logger.addHandler(file_handler)
    return target


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``whetstone`` namespace, configuring once.

    Pass ``__name__`` from a module inside the package (already
    ``whetstone....``) and it is used as-is; any other name is nested under the
    namespace so it inherits the shared handler and level.
    """
    if not _is_configured():
        configure_logging()
    if name != NAMESPACE and not name.startswith(f"{NAMESPACE}."):
        name = f"{NAMESPACE}.{name}"
    return logging.getLogger(name)
