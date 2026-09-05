"""DuckDB connection management.

One database file, ``data/proplab.duckdb``. Read/write connections are opened per call site and
closed by the context manager; a module-level singleton is available for the FastAPI app, which
only ever reads.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from backend.config import get_settings
from backend.logging_setup import get_logger

log = get_logger(__name__)

_singleton: duckdb.DuckDBPyConnection | None = None
_lock = threading.Lock()
_read_only = False
_serve_snapshot = False


@contextmanager
def connect(read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open a DuckDB connection to the PropLab database and close it afterwards.

    In a process that has called :func:`set_read_only` -- the API -- every connection is forced
    read-only, including ones a shared helper opens for itself. DuckDB refuses to hold two
    connections to one file with different configurations, so honouring the process-wide flag here
    is what lets read-only query helpers live in modules that also write.

    Args:
        read_only: open without acquiring the write lock. Fails if the file does not exist yet.
    """
    settings = get_settings()
    settings.ensure_dirs()
    con = duckdb.connect(str(active_db_path()), read_only=read_only or _read_only)
    try:
        yield con
    finally:
        con.close()


def use_serve_snapshot(value: bool = True) -> None:
    """Read from the published snapshot instead of the live database.

    The API calls this at startup so a refresh can run underneath it (see
    ``Settings.serve_db_path``). Falls back to the live file when no snapshot exists yet.
    """
    global _serve_snapshot
    close_connection()
    _serve_snapshot = value


def active_db_path():
    """Which database file this process is using."""
    settings = get_settings()
    if _serve_snapshot and settings.serve_db_path.exists():
        return settings.serve_db_path
    return settings.db_path


def publish_snapshot() -> Path | None:
    """Copy the live database to the serve path, atomically.

    Called at the end of a refresh. The API picks the new file up on its next connection, so a
    running app updates without a restart and never observes a partially written week.
    """
    import shutil

    settings = get_settings()
    if not settings.db_path.exists():
        return None
    tmp = settings.serve_db_path.with_suffix(".tmp")
    try:
        shutil.copy2(settings.db_path, tmp)
        os.replace(tmp, settings.serve_db_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        log.exception("could not publish the read-only snapshot")
        return None
    log.info("published read-only snapshot to %s", settings.serve_db_path.name)
    return settings.serve_db_path


def set_read_only(value: bool = True) -> None:
    """Open the singleton read-only from here on.

    DuckDB allows a single writer per file, so an API process holding a read/write handle blocks
    ``make refresh`` entirely. The API only ever reads, so it declares that and the two can run
    side by side -- which is exactly what ``make dev`` does.
    """
    global _read_only
    close_connection()
    _read_only = value


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return a process-wide connection. Used by the read-only API layer.

    DuckDB connections are not thread-safe; call ``.cursor()`` on this for per-request use.
    """
    global _singleton
    with _lock:
        if _singleton is None:
            get_settings().ensure_dirs()
            _singleton = duckdb.connect(str(active_db_path()), read_only=_read_only)
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
