"""Rich-backed logging. One place so every entry point looks the same."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console(stderr=True)
_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    """Install a Rich handler on the root logger. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    from backend.config import get_settings

    lvl = (level or get_settings().log_level).upper()
    logging.basicConfig(
        level=lvl,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    # These libraries are chatty at DEBUG and add nothing.
    for noisy in ("httpx", "httpcore", "urllib3", "nflreadpy"):
        logging.getLogger(noisy).setLevel(max(logging.INFO, logging.getLevelName(lvl)))
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
