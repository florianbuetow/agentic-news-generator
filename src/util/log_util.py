"""Centralized logging setup for the Agentic News Generator."""

import logging
import sys
from pathlib import Path

_FMT = "[%(asctime)s] [%(levelname)s] | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)


def configure_root_logger(log_dir: Path) -> None:
    """Configure root logger with stdout, app.log, and error.log handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    app_handler = logging.FileHandler(log_dir / "app.log")
    app_handler.setLevel(logging.DEBUG)
    app_handler.setFormatter(formatter)

    error_handler = logging.FileHandler(log_dir / "error.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(stdout_handler)
    root.addHandler(app_handler)
    root.addHandler(error_handler)
