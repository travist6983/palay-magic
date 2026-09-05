"""DuckDB connection management.

One database file, ``data/proplab.duckdb``. Read/write connections are opened per call site and
closed by the context manager; a module-level singleton is available for the FastAPI app, which
only ever reads.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import duckdb
import polars as pl

from backend.config import get_settings
from backend.logging_setup import get_logger

log = get_logger(__name__)

_singleton: duckdb.DuckDBPyConnection | None = None
_lock = threading.Lock()


@contextmanager
def connect(read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open a DuckDB connection to the PropLab database and close it afterwards.

    Args:
        read_only: open without acquiring the write lock. Fails if the file does not exist yet.
    """
    settings = get_settings()
    settings.ensure_dirs()
    con = duckdb.connect(str(settings.db_path), read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return a process-wide connection. Used by the read-only API layer.

    DuckDB connections are not thread-safe; call ``.cursor()`` on this for per-request use.
    """
    global _singleton
    with _lock:
        if _singleton is None:
            settings = get_settings()
            settings.ensure_dirs()
            _singleton = duckdb.connect(str(settings.db_path))
        return _singleton


def close_connection() -> None:
    """Close the singleton connection, if one is open."""
    global _singleton
    with _lock:
        if _singleton is not None:
            _singleton.close()
            _singleton = None


def query(sql: str, params: Sequence[Any] | None = None) -> list[tuple]:
    """Run a read-only query against the singleton connection and return raw rows."""
    cur = get_connection().cursor()
    try:
        return cur.execute(sql, params or []).fetchall()
    finally:
        cur.close()


def query_df(sql: str, params: Sequence[Any] | None = None) -> pl.DataFrame:
    """Run a read-only query and return a polars DataFrame."""
    cur = get_connection().cursor()
    try:
        return cur.execute(sql, params or []).pl()
    finally:
        cur.close()
