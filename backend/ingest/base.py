"""Shared plumbing for every ingest module.

Three guarantees the rest of the app relies on (docs/DECISIONS.md D8):

1. **Nothing fatal.** A source that is down logs, records a red freshness badge, and returns an
   empty result. The app still loads with stale data.
2. **Atomic writes.** A Parquet file is either the previous good version or the new one, never a
   half-written file that breaks every DuckDB view reading the glob.
3. **Auditable freshness.** Every attempt writes to ``source_freshness``, which drives the
   green/yellow/red badges in the UI header (§8).
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import polars as pl
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.db.connection import connect
from backend.logging_setup import get_logger

log = get_logger(__name__)

T = TypeVar("T")

# Freshness thresholds, in hours since the last successful pull.
FRESHNESS_THRESHOLDS: dict[str, tuple[float, float]] = {
    # source: (green_below, yellow_below)  -- above the second value is red
    "nflverse": (36.0, 96.0),
    "sleeper": (30.0, 72.0),
    "espn": (36.0, 96.0),
    "the-odds-api": (60.0, 200.0),
    "anthropic": (200.0, 800.0),
}


def utcnow() -> datetime:
    """Timezone-aware current UTC time. One place so tests can monkeypatch it."""
    return datetime.now(UTC)


@dataclass
class IngestResult:
    """What one ingest step did. Truthy when it wrote something."""

    source: str
    dataset: str
    n_rows: int = 0
    path: Path | None = None
    ok: bool = True
    skipped: bool = False
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.ok and not self.skipped

    def __str__(self) -> str:
        if self.skipped:
            return f"{self.dataset}: skipped ({self.detail})"
        if not self.ok:
            return f"{self.dataset}: FAILED ({self.detail})"
        return f"{self.dataset}: {self.n_rows:,} rows"


def resilient(source: str, dataset: str = "") -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorate an ingest function so a failure degrades instead of raising.

    On exception: logs with traceback, records a red badge for ``source``, and returns a failed
    :class:`IngestResult` rather than propagating. Functions that return something other than an
    ``IngestResult`` get ``None`` on failure.
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            name = dataset or fn.__name__
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - degrading is the point
                log.exception("%s/%s failed: %s", source, name, exc)
                record_freshness(source, ok=False, detail=f"{type(exc).__name__}: {exc}")
                return IngestResult(
                    source=source, dataset=name, ok=False, detail=f"{type(exc).__name__}: {exc}"
                )
            return result

        return wrapper

    return decorator


# Network calls: 4 attempts, exponential backoff capped at 20s.
network_retry = retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=1, max=20),
    retry=retry_if_exception_type((OSError, TimeoutError, ConnectionError)),
    reraise=True,
)


def write_parquet_atomic(df: pl.DataFrame, path: Path) -> Path:
    """Write ``df`` to ``path`` via a temp file and an atomic rename.

    A DuckDB view globbing this directory must never observe a truncated file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    try:
        df.write_parquet(tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    log.debug("wrote %s (%d rows, %.1f MB)", path.name, df.height, path.stat().st_size / 1e6)
    return path


def record_freshness(
    source: str,
    ok: bool,
    detail: str = "",
    n_rows: int | None = None,
) -> None:
    """Upsert this source's row in ``source_freshness``.

    A successful attempt stamps ``last_success_at`` green. A failure keeps the previous success
    timestamp and downgrades the badge according to how stale that success now is, so the UI can
    distinguish "the source broke but the data is an hour old" from "we have nothing recent".
    """
    now = utcnow()
    try:
        with connect() as con:
            prev = con.execute(
                "SELECT last_success_at, n_rows FROM source_freshness WHERE source = ?", [source]
            ).fetchone()
            last_success = now if ok else (prev[0] if prev else None)
            rows = n_rows if n_rows is not None else (prev[1] if prev else None)
            status = _status_for(source, last_success, ok=ok)

            con.execute(
                """
                INSERT INTO source_freshness
                    (source, last_attempt_at, last_success_at, status, detail, n_rows)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (source) DO UPDATE SET
                    last_attempt_at = excluded.last_attempt_at,
                    last_success_at = excluded.last_success_at,
                    status          = excluded.status,
                    detail          = excluded.detail,
                    n_rows          = excluded.n_rows
                """,
                [source, now, last_success, status, detail[:500], rows],
            )
    except Exception:  # noqa: BLE001 - bookkeeping must never break ingest
        log.exception("could not record freshness for %s", source)


def _status_for(source: str, last_success: datetime | None, ok: bool) -> str:
    """Green while the last success is recent, yellow when aging, red when stale or never."""
    if last_success is None:
        return "red"
    green_h, yellow_h = FRESHNESS_THRESHOLDS.get(source, (36.0, 96.0))
    if last_success.tzinfo is None:
        last_success = last_success.replace(tzinfo=UTC)
    age_h = (utcnow() - last_success).total_seconds() / 3600.0
    if age_h < green_h:
        return "green" if ok else "yellow"
    if age_h < yellow_h:
        return "yellow"
    return "red"
