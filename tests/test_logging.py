"""Run-directory file logging: every line lands beside the artifacts.

Queued/detached runs (pueue, nohup) lose stdout, so ``attach_run_dir_logging``
mirrors the whetstone tree and re-homed library loggers into
``<run_dir>/run.log``. These tests pin the mirroring, idempotence, rank
naming, and survival across a forced reconfigure.
"""

import logging

import pytest

from whetstone.utils.logging import (
    LIBRARY_LOGGERS,
    NAMESPACE,
    attach_run_dir_logging,
    configure_logging,
    get_logger,
)


@pytest.fixture(autouse=True)
def _isolated_handlers():
    """Snapshot and restore global logging state around each test."""
    tracked = [NAMESPACE, *LIBRARY_LOGGERS]
    saved = {name: list(logging.getLogger(name).handlers) for name in tracked}
    yield
    for name in tracked:
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            if handler not in saved[name]:
                logger.removeHandler(handler)
                handler.close()
        for handler in saved[name]:
            if handler not in logger.handlers:
                logger.addHandler(handler)


def test_whetstone_and_library_lines_are_mirrored_to_file(tmp_path) -> None:
    log_path = attach_run_dir_logging(tmp_path, rank=0)
    get_logger("whetstone.test_logging").info("training started")
    logging.getLogger("transformers").warning("library advisory")

    content = log_path.read_text(encoding="utf-8")
    assert log_path == tmp_path / "run.log"
    assert "training started" in content
    assert "library advisory" in content
    assert "[rank0]" in content


def test_attach_is_idempotent_per_file(tmp_path) -> None:
    attach_run_dir_logging(tmp_path, rank=0)
    attach_run_dir_logging(tmp_path, rank=0)
    get_logger("whetstone.test_logging").info("once only")

    content = (tmp_path / "run.log").read_text(encoding="utf-8")
    assert content.count("once only") == 1


def test_nonzero_rank_gets_its_own_file(tmp_path) -> None:
    log_path = attach_run_dir_logging(tmp_path, rank=1)
    assert log_path == tmp_path / "run_rank1.log"


def test_forced_reconfigure_keeps_the_file_handler(tmp_path) -> None:
    attach_run_dir_logging(tmp_path, rank=0)
    # Entry points reconfigure once the real rank is known; the run's file
    # handler must survive it (only console handlers are replaced).
    configure_logging(force=True)
    get_logger("whetstone.test_logging").info("after reconfigure")
    logging.getLogger("datasets").warning("library after reconfigure")

    content = (tmp_path / "run.log").read_text(encoding="utf-8")
    assert "after reconfigure" in content
    assert "library after reconfigure" in content
