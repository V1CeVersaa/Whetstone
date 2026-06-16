import logging
import os

# All Whetstone loggers live under this namespace so configuration can target
# the whole tree at once without touching the root logger (and other libraries).
NAMESPACE = "whetstone"

_configured = False


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
    readable; it defaults to the ``RANK`` env var (``0`` when unset).

    Args:
        level: Log level for the namespace logger (name or numeric).
        rank: Process rank to stamp; defaults to ``$RANK`` or ``0``.
        force: Reconfigure even if already configured (e.g. once the real rank
            is known after distributed init).
    """
    global _configured
    if _configured and not force:
        return
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt=f"%(asctime)s %(levelname)s [rank{rank}] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger = logging.getLogger(NAMESPACE)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)
    # Don't bubble up to the root logger; we own this subtree's formatting.
    logger.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``whetstone`` namespace, configuring once.

    Pass ``__name__`` from a module inside the package (already
    ``whetstone....``) and it is used as-is; any other name is nested under the
    namespace so it inherits the shared handler and level.
    """
    if not _configured:
        configure_logging()
    if name != NAMESPACE and not name.startswith(f"{NAMESPACE}."):
        name = f"{NAMESPACE}.{name}"
    return logging.getLogger(name)
