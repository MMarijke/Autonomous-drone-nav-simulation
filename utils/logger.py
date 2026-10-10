"""Console + file logging and run-directory management."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logger(name: str = "dronenav", log_dir: str | Path | None = None,
                 level: int = logging.INFO) -> logging.Logger:
    """Return a logger writing to stderr (and ``log_dir/<name>.log`` if given).

    Safe to call repeatedly: handlers are replaced rather than stacked.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    formatter = logging.Formatter(_FMT, datefmt="%H:%M:%S")
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def make_run_dir(base: str | Path, name: str, stamp: bool = True) -> Path:
    """Create ``base/name[_YYYYmmdd-HHMMSS]`` and return it."""
    suffix = time.strftime("_%Y%m%d-%H%M%S") if stamp else ""
    path = Path(base) / f"{name}{suffix}"
    path.mkdir(parents=True, exist_ok=True)
    return path
