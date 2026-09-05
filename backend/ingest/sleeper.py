"""Sleeper API ingest: live injury designations, depth chart order, and the ID crosswalk.

Source
------
Sleeper's read-only public API. **No authentication, no key, no registration.**

* ``GET https://api.sleeper.app/v1/players/nfl`` — the full player dump (~15 MB of JSON,
  ~12k records). Sleeper explicitly asks that this be called **no more than once per day**.
* ``GET https://api.sleeper.app/v1/state/nfl`` — current ``week`` / ``season`` /
  ``season_type``. Cheap, but still cached for an hour so a refresh loop cannot hammer it.

Rate limiting (docs/DECISIONS.md D8)
-----------------------------------
The once-per-day request is enforced in code, not by convention: :func:`ingest_players` refuses
to re-fetch while ``data/raw/sleeper_players.parquet`` is younger than
``get_settings().sleeper_min_cache_age_hours`` (default 20.0) and returns a *skipped*
:class:`~backend.ingest.base.IngestResult`. ``force=True`` is the only override.

What this module produces
-------------------------
1. ``data/raw/sleeper_players.parquet`` — the flattened dump (view ``raw_sleeper_players``).
2. ``data/cache/sleeper/{timestamp}.json.gz`` — the untouched JSON, archived for auditing and so
   a lost diff can be reconstructed by hand.
3. ``injury_events`` rows — one per observed change in ``injury_status``,
   ``practice_participation``, ``depth_chart_order`` or ``status`` versus the previous snapshot.
   Depth chart movement routinely precedes the official designation, which is the whole point.
4. ``injury_status`` rows — the current live state per player, ``source = 'sleeper'``.

Identity (docs/DECISIONS.md D6)
------------------------------
Sleeper omits ``gsis_id`` on **two thirds** of the dump, including ~80% of the players who
currently carry an injury designation, so a record with no ``gsis_id`` of its own is resolved
through the ``players`` crosswalk (``sleeper_id`` first, then ``espn_id``) before anything is
dropped. Without that step ``injury_status`` silently loses about half of every live designation
Sleeper publishes. See :func:`_resolve_gsis`.

Every player-dump failure degrades: it logs, downgrades the ``sleeper`` badge in
``source_freshness``, and returns a failed result. Nothing here raises into the caller. The cheap
``/v1/state/nfl`` poll deliberately leaves that badge alone — it tracks the daily dump, which is
the data the UI actually renders.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from backend.config import get_settings
from backend.db.connection import connect
from backend.db.views import raw_path
from backend.ingest.base import (
    IngestResult,
    network_retry,
    record_freshness,
    resilient,
    utcnow,
    write_parquet_atomic,
)
from backend.logging_setup import get_logger

log = get_logger(__name__)

SOURCE = "sleeper"
DATASET = "players"

PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"
STATE_URL = "https://api.sleeper.app/v1/state/nfl"

PLAYERS_FILE = "sleeper_players.parquet"
STATE_CACHE_FILE = "sleeper_state.json"
ARCHIVE_SUBDIR = "sleeper"

STATE_CACHE_TTL_HOURS = 1.0
"""``/v1/state/nfl`` is cheap, but there is no reason to ask more than hourly."""

PLAYERS_TIMEOUT = 120.0
STATE_TIMEOUT = 20.0

_HEADERS = {"User-Agent": "PropLab/0.1 (local research app)", "Accept": "application/json"}

#: Fields copied straight across as Utf8. Sleeper omits keys entirely on some records and returns
#: ints for a few ids, so every one of these goes through :func:`_as_str`.
STRING_FIELDS: tuple[str, ...] = (
    "gsis_id",
    "espn_id",
    "sportradar_id",
    "pfr_id",
    "rotowire_id",
    "full_name",
    "first_name",
    "last_name",
    "team",
    "position",
    "depth_chart_position",
    "status",
    "injury_status",
    "injury_body_part",
    "injury_start_date",
    "injury_notes",
    "practice_participation",
    "practice_description",
)

#: The only genuinely numeric fields on the dump.
INT_FIELDS: tuple[str, ...] = ("depth_chart_order", "age", "years_exp")

#: Working column holding the ``gsis_id`` after the crosswalk lookup in :func:`_resolve_gsis`.
#: Never written to Parquet -- ``raw_sleeper_players`` stays a faithful copy of the dump, and
#: ``crosswalk.py`` reads that view to *build* the crosswalk we are borrowing here.
RESOLVED_GSIS = "_resolved_gsis_id"

#: Snapshot-to-snapshot changes we care about (§3 / docs/reference "Tier 2").
DIFF_FIELDS: tuple[str, ...] = (
    "injury_status",
    "practice_participation",
    "depth_chart_order",
    "status",
)

SCHEMA: dict[str, pl.DataType] = {
    "sleeper_id": pl.Utf8,
    **{f: pl.Utf8 for f in STRING_FIELDS},
    **{f: pl.Int32 for f in INT_FIELDS},
    "active": pl.Boolean,
    "fetched_at": pl.Datetime(time_unit="us", time_zone="UTC"),
}

#: Column order written to Parquet.
COLUMNS: tuple[str, ...] = (
    "sleeper_id",
    "gsis_id",
    "espn_id",
    "sportradar_id",
    "pfr_id",
    "rotowire_id",
    "full_name",
    "first_name",
    "last_name",
    "team",
    "position",
    "depth_chart_position",
    "depth_chart_order",
    "status",
    "injury_status",
    "injury_body_part",
    "injury_start_date",
    "injury_notes",
    "practice_participation",
    "practice_description",
    "age",
    "years_exp",
    "active",
    "fetched_at",
)


class SleeperError(RuntimeError):
    """A non-retryable Sleeper response (4xx other than 429, or malformed JSON)."""


# ---------------------------------------------------------------------------
# value coercion -- Sleeper is loose about types and frequently omits keys
# ---------------------------------------------------------------------------


def _as_str(value: object) -> str | None:
    """Coerce a Sleeper value to a trimmed string, or ``None`` when absent/empty.

    Sleeper returns ``espn_id``/``rotowire_id`` as integers and ships ``gsis_id`` with a leading
    space on 866 of the 3,893 records that have one -- 22% (verified 2026-09-05) -- which would
    silently break every join on ``gsis_id`` (D6). Both are normalised here.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip()
    return text or None


def _as_int(value: object) -> int | None:
    """Coerce a Sleeper value to an int, or ``None`` when absent/blank/non-numeric."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _as_bool(value: object) -> bool | None:
    """Coerce a Sleeper value to a bool, or ``None`` when absent/unrecognised."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


@network_retry
def _get_json(url: str, timeout: float) -> Any:
    """GET ``url`` and parse JSON, with retry on anything transient.

    ``network_retry`` only retries ``OSError``/``TimeoutError``/``ConnectionError``, and httpx's
    exceptions descend from neither, so transient failures (connection errors, timeouts, 429,
    5xx) are re-raised as ``OSError`` to make them retryable. A 4xx is raised as
    :class:`SleeperError` so we fail fast instead of hammering a URL that will not work.
    """
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=_HEADERS) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status == 429 or status >= 500:
            raise OSError(f"{url}: HTTP {status} (transient)") from exc
        raise SleeperError(f"{url}: HTTP {status}") from exc
    except httpx.HTTPError as exc:
        raise OSError(f"{url}: {type(exc).__name__}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SleeperError(f"{url}: response was not JSON: {exc}") from exc


# ---------------------------------------------------------------------------
# /v1/state/nfl
# ---------------------------------------------------------------------------


def _state_cache_path() -> Path:
    return get_settings().cache_dir / STATE_CACHE_FILE


def _read_state_cache() -> tuple[dict[str, Any] | None, float]:
    """Return the cached state payload and its age in hours (``inf`` when there is none)."""
    path = _state_cache_path()
    if not path.exists():
        return None, float("inf")
    try:
        blob = json.loads(path.read_text())
        payload = blob.get("state")
        if not isinstance(payload, dict):
            return None, float("inf")
        fetched_at = datetime.fromisoformat(blob["fetched_at"])
        age_h = (utcnow() - fetched_at).total_seconds() / 3600.0
        return payload, age_h
    except Exception:  # noqa: BLE001 - a corrupt cache must not break the app
        log.warning("unreadable Sleeper state cache at %s; ignoring it", path)
        return None, float("inf")


def _write_state_cache(payload: dict[str, Any]) -> None:
    """Persist the state payload with its fetch timestamp. Best effort."""
    path = _state_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"fetched_at": utcnow().isoformat(), "state": payload}, indent=2))
        tmp.replace(path)
    except OSError:
        log.warning("could not write Sleeper state cache to %s", path, exc_info=True)


def _fetch_state_remote() -> dict[str, Any] | None:
    """GET ``/v1/state/nfl``, returning ``None`` on any failure. Never raises.

    Deliberately does **not** touch ``source_freshness``. That badge is the UI's only signal for
    how stale the injury data is (§8, D8), and it tracks ``/v1/players/nfl`` -- the once-a-day
    dump the app actually renders. Letting this hourly, near-free poll write it breaks the badge
    in both directions: a success would stamp ``last_success_at`` fresh often enough that the
    30h/72h thresholds could never fire while the dump was failing, and a transient blip here
    would flag the source red even though the dump is fine and this call recovered from cache.
    """
    try:
        payload = _get_json(STATE_URL, STATE_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - the caller falls back to the cached copy
        log.warning("Sleeper /v1/state/nfl failed: %s: %s", type(exc).__name__, exc)
        return None
    if not isinstance(payload, dict) or "season" not in payload:
        log.warning("unexpected /v1/state/nfl payload: %s", type(payload).__name__)
        return None
    return payload


def fetch_state(force: bool = False) -> dict[str, Any] | None:
    """Return Sleeper's current league state (``week``, ``season``, ``season_type``, ...).

    Served from ``data/cache/sleeper_state.json`` while that copy is younger than
    :data:`STATE_CACHE_TTL_HOURS`. On a network failure the stale cached copy is returned instead,
    and only when there is no cache at all does this return ``None``.

    Args:
        force: ignore the one-hour cache and re-fetch.

    Returns:
        The parsed payload, the cached payload, or ``None`` if both are unavailable.
    """
    cached, age_h = _read_state_cache()
    if cached is not None and not force and age_h < STATE_CACHE_TTL_HOURS:
        log.debug("Sleeper state served from cache (%.2fh old)", age_h)
        return cached

    payload = _fetch_state_remote()
    if payload is not None:
        _write_state_cache(payload)
        log.info(
            "Sleeper state: season=%s week=%s type=%s",
            payload.get("season"),
            payload.get("week"),
            payload.get("season_type"),
        )
        return payload

    if cached is not None:
        log.warning("Sleeper state unreachable; using cached copy (%.1fh old)", age_h)
        return cached

    log.warning("Sleeper state unreachable and nothing cached")
    return None


# ---------------------------------------------------------------------------
# /v1/players/nfl
# ---------------------------------------------------------------------------


def cache_age_hours() -> float:
    """Age in hours of ``data/raw/sleeper_players.parquet`` (``inf`` when it does not exist)."""
    path = raw_path(PLAYERS_FILE)
    if not path.exists():
        return float("inf")
    return (utcnow().timestamp() - path.stat().st_mtime) / 3600.0


def _flatten(payload: dict[str, Any], fetched_at: datetime) -> pl.DataFrame:
    """Flatten the ``{sleeper_id: {...}}`` dump into a typed DataFrame.

    Records with a non-dict body are dropped. The 32 team-defense pseudo-players Sleeper includes
    (keys ``"HOU"``, ``"NE"``, ...) are kept: they carry no ``gsis_id``, so they fall out of every
    downstream join on their own.
    """
    rows: list[dict[str, Any]] = []
    for sleeper_id, record in payload.items():
        if not isinstance(record, dict):
            continue
        row: dict[str, Any] = {"sleeper_id": _as_str(sleeper_id)}
        for field in STRING_FIELDS:
            row[field] = _as_str(record.get(field))
        for field in INT_FIELDS:
            row[field] = _as_int(record.get(field))
        row["active"] = _as_bool(record.get("active"))
        row["fetched_at"] = fetched_at
        rows.append(row)

    if not rows:
        return pl.DataFrame(schema=SCHEMA).select(COLUMNS)
    return pl.DataFrame(rows, schema=SCHEMA).select(COLUMNS)


def _crosswalk_lookup() -> tuple[pl.DataFrame, pl.DataFrame] | None:
    """``(sleeper_id -> gsis_id, espn_id -> gsis_id)`` from the ``players`` crosswalk table.

    Returns ``None`` when the table is missing or empty -- the very first backfill run writes the
    Sleeper dump *before* ``crosswalk.build_players`` has ever run, and a locked database is not
    worth failing an ingest over. Ambiguous keys (the same id mapped to two ``gsis_id`` values)
    are dropped rather than guessed, so a lookup can never fan a row out or attach a wrong id.
    """
    try:
        with connect() as con:
            frame = con.execute(
                "SELECT gsis_id, sleeper_id, espn_id FROM players WHERE gsis_id IS NOT NULL"
            ).pl()
    except Exception as exc:  # noqa: BLE001 - no crosswalk just means no recovery this run
        log.warning("players crosswalk unavailable (%s); gsis_id recovery skipped", exc)
        return None
    if frame.is_empty():
        log.warning("players crosswalk is empty; gsis_id recovery skipped")
        return None

    frame = frame.with_columns(pl.col("gsis_id", "sleeper_id", "espn_id").cast(pl.Utf8))

    def _unique_map(key: str, out: str) -> pl.DataFrame:
        return (
            frame.select(pl.col(key), pl.col("gsis_id").alias(out))
            .filter(pl.col(key).is_not_null())
            .unique(subset=[key], keep="none")
        )

    return _unique_map("sleeper_id", "_gsis_by_sleeper"), _unique_map("espn_id", "_gsis_by_espn")


def _resolve_gsis(current: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Add :data:`RESOLVED_GSIS` to ``current``, filling gaps from the ``players`` crosswalk.

    Sleeper ships ``gsis_id`` on only 3,893 of 12,226 records, and the omissions are *not* the
    inactive tail: of the 786 players carrying a live ``injury_status``, 630 have no ``gsis_id``
    and 386 of those are on an active roster (verified 2026-09-05). Dropping them -- which is what
    keying ``injury_status`` on Sleeper's own ``gsis_id`` does -- throws away half of the exact
    signal this source exists to provide.

    ``players`` already knows those ids: ``crosswalk.py`` recovers them from the rosters and by
    name, and the schema carries ``players_sleeper_idx`` for precisely this join (D6). Sleeper's
    own ``gsis_id`` always wins; the crosswalk only fills in what is missing.

    Returns the frame plus the number of ids recovered.
    """
    maps = _crosswalk_lookup()
    if maps is None:
        return current.with_columns(pl.col("gsis_id").alias(RESOLVED_GSIS)), 0

    by_sleeper, by_espn = maps
    resolved = (
        current.join(by_sleeper, on="sleeper_id", how="left")
        .join(by_espn, on="espn_id", how="left")
        .with_columns(
            pl.coalesce("gsis_id", "_gsis_by_sleeper", "_gsis_by_espn").alias(RESOLVED_GSIS)
        )
        .drop("_gsis_by_sleeper", "_gsis_by_espn")
    )
    n_recovered = int(
        resolved.select(
            (pl.col("gsis_id").is_null() & pl.col(RESOLVED_GSIS).is_not_null()).sum()
        ).item()
    )
    log.info(
        "gsis_id: %d shipped by Sleeper, %d recovered via the players crosswalk, %d still unknown",
        resolved.get_column("gsis_id").is_not_null().sum(),
        n_recovered,
        resolved.get_column(RESOLVED_GSIS).is_null().sum(),
    )
    return resolved, n_recovered


def _read_previous_snapshot() -> pl.DataFrame | None:
    """Load the snapshot currently on disk, before it is overwritten. ``None`` on first run."""
    path = raw_path(PLAYERS_FILE)
    if not path.exists():
        return None
    try:
        return pl.read_parquet(path)
    except Exception:  # noqa: BLE001 - a corrupt snapshot just means "no diff this run"
        log.warning("could not read previous snapshot %s; skipping the diff", path, exc_info=True)
        return None


def _archive_raw(payload: dict[str, Any], fetched_at: datetime) -> Path | None:
    """Gzip the untouched JSON to ``data/cache/sleeper/{timestamp}.json.gz``. Best effort."""
    settings = get_settings()
    directory = settings.cache_dir / ARCHIVE_SUBDIR
    stamp = fetched_at.strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"{stamp}.json.gz"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        log.info("archived raw dump to %s (%.1f MB)", path, path.stat().st_size / 1e6)
        return path
    except OSError:
        log.warning("could not archive raw Sleeper dump to %s", path, exc_info=True)
        return None


def _event_id(identity: str, field: str, observed_at: datetime) -> str:
    """``md5(gsis_id|field|observed_at)``.

    ``identity`` is the player's ``gsis_id`` when it has one and ``sleeper:{sleeper_id}``
    otherwise -- thousands of players share a null ``gsis_id``, and they would all collide on the
    ``event_id`` primary key if the null were hashed literally.
    """
    return hashlib.md5(
        f"{identity}|{field}|{observed_at.isoformat()}".encode(), usedforsecurity=False
    ).hexdigest()


def _diff_snapshots(
    previous: pl.DataFrame | None, current: pl.DataFrame, observed_at: datetime
) -> pl.DataFrame:
    """One row per changed field between two snapshots, shaped like ``injury_events``.

    ``current`` must already carry :data:`RESOLVED_GSIS` (see :func:`_resolve_gsis`).

    Joined on ``sleeper_id`` (the dump's real primary key -- ``gsis_id`` is null on two thirds of
    records and duplicated on seven). A field counts as changed only when the two values actually
    differ; ``null -> null`` is not a change, while ``null -> "Questionable"`` and
    ``"Questionable" -> null`` both are.
    """
    empty = pl.DataFrame(
        schema={
            "event_id": pl.Utf8,
            "gsis_id": pl.Utf8,
            "sleeper_id": pl.Utf8,
            "player_name": pl.Utf8,
            "team": pl.Utf8,
            "position": pl.Utf8,
            "field": pl.Utf8,
            "old_value": pl.Utf8,
            "new_value": pl.Utf8,
            "observed_at": pl.Datetime(time_unit="us", time_zone="UTC"),
        }
    )
    if previous is None or previous.is_empty() or current.is_empty():
        return empty

    missing = [f for f in DIFF_FIELDS if f not in previous.columns]
    if missing or "sleeper_id" not in previous.columns:
        log.warning("previous snapshot is missing %s; skipping the diff", missing or "sleeper_id")
        return empty

    old = (
        previous.select(["sleeper_id", *DIFF_FIELDS])
        .unique(subset=["sleeper_id"], keep="first")
        .rename({f: f"{f}__old" for f in DIFF_FIELDS})
    )
    joined = current.join(old, on="sleeper_id", how="inner")

    frames: list[pl.DataFrame] = []
    for field in DIFF_FIELDS:
        old_col = pl.col(f"{field}__old").cast(pl.Utf8)
        new_col = pl.col(field).cast(pl.Utf8)
        # ne_missing treats null as a comparable value: null vs null is False (not a change),
        # null vs "Out" is True. That is exactly the "skip rows where both are null" rule.
        changed = joined.filter(old_col.ne_missing(new_col))
        if changed.is_empty():
            continue
        frames.append(
            changed.select(
                pl.col(RESOLVED_GSIS).alias("gsis_id"),
                pl.col("sleeper_id"),
                pl.col("full_name").alias("player_name"),
                pl.col("team"),
                pl.col("position"),
                pl.lit(field, dtype=pl.Utf8).alias("field"),
                old_col.alias("old_value"),
                new_col.alias("new_value"),
                pl.lit(observed_at).cast(SCHEMA["fetched_at"]).alias("observed_at"),
            )
        )

    if not frames:
        return empty

    events = pl.concat(frames, how="vertical")
    identities = [
        gsis or f"sleeper:{sleeper}"
        for gsis, sleeper in zip(
            events["gsis_id"].to_list(), events["sleeper_id"].to_list(), strict=True
        )
    ]
    event_ids = [
        _event_id(identity, field, observed_at)
        for identity, field in zip(identities, events["field"].to_list(), strict=True)
    ]
    return (
        events.with_columns(pl.Series("event_id", event_ids, dtype=pl.Utf8))
        .unique(subset=["event_id"], keep="first")
        .select(empty.columns)
    )


def _dedupe_by_gsis(df: pl.DataFrame, key: str) -> tuple[pl.DataFrame, int]:
    """Keep one row per ``key``, preferring the record most likely to be the live one.

    Sleeper carries a handful of duplicate ``gsis_id`` values (stale records that were never
    merged), and crosswalk recovery can map a stale Sleeper record onto an id another record
    already owns. ``injury_status`` is keyed on ``(gsis_id, source)``, and DuckDB refuses to
    update the same row twice in one statement, so the duplicates have to go. Preference order:
    on a roster, then flagged active, then the highest Sleeper id (their ids increase over time,
    so the higher one is the newer record).
    """
    ranked = df.with_columns(
        pl.col("team").is_not_null().cast(pl.Int8).alias("_has_team"),
        pl.col("active").fill_null(False).cast(pl.Int8).alias("_active"),
        pl.col("sleeper_id").cast(pl.Int64, strict=False).fill_null(-1).alias("_numeric_id"),
    ).sort(["_has_team", "_active", "_numeric_id"], descending=True)
    deduped = ranked.unique(subset=[key], keep="first", maintain_order=True)
    return deduped.drop("_has_team", "_active", "_numeric_id"), df.height - deduped.height


def _status_rows(current: pl.DataFrame, observed_at: datetime) -> tuple[pl.DataFrame, int, int]:
    """Build the ``injury_status`` upsert payload.

    Keyed on :data:`RESOLVED_GSIS`, so a player Sleeper ships without a ``gsis_id`` still lands in
    the table whenever the ``players`` crosswalk knows one (:func:`_resolve_gsis`).

    Returns the rows, how many players were dropped for having no resolvable ``gsis_id``, and how
    many were dropped as duplicates.
    """
    with_id = current.filter(pl.col(RESOLVED_GSIS).is_not_null())
    n_missing_gsis = current.height - with_id.height
    deduped, n_duplicates = _dedupe_by_gsis(with_id, RESOLVED_GSIS)

    rows = deduped.select(
        pl.col(RESOLVED_GSIS).alias("gsis_id"),
        pl.col("sleeper_id"),
        pl.lit(SOURCE, dtype=pl.Utf8).alias("source"),
        pl.lit(observed_at).cast(SCHEMA["fetched_at"]).alias("observed_at"),
        pl.col("team"),
        pl.col("position"),
        pl.col("injury_status"),
        pl.col("status").alias("roster_status"),
        pl.col("practice_participation"),
        pl.col("injury_body_part"),
        pl.col("injury_notes"),
        pl.col("depth_chart_position"),
        pl.col("depth_chart_order"),
    )
    return rows, n_missing_gsis, n_duplicates


def _persist(events: pl.DataFrame, statuses: pl.DataFrame) -> tuple[int, int]:
    """Write the diff to ``injury_events`` and the current state to ``injury_status``.

    Returns ``(events inserted, statuses upserted)``. Raises on a database failure so the caller
    can report it -- ``@resilient`` turns that into a failed result rather than a crash.
    """
    with connect() as con:
        con.register("sleeper_events", events.to_arrow())
        con.register("sleeper_statuses", statuses.to_arrow())

        con.execute(
            """
            INSERT INTO injury_events
                (event_id, gsis_id, sleeper_id, player_name, team, position,
                 field, old_value, new_value, observed_at)
            SELECT event_id, gsis_id, sleeper_id, player_name, team, position,
                   field, old_value, new_value, observed_at
            FROM sleeper_events
            ON CONFLICT (event_id) DO NOTHING
            """
        )
        con.execute(
            """
            INSERT INTO injury_status
                (gsis_id, sleeper_id, source, observed_at, team, position, injury_status,
                 roster_status, practice_participation, injury_body_part, injury_notes,
                 depth_chart_position, depth_chart_order)
            SELECT gsis_id, sleeper_id, source, observed_at, team, position, injury_status,
                   roster_status, practice_participation, injury_body_part, injury_notes,
                   depth_chart_position, depth_chart_order
            FROM sleeper_statuses
            ON CONFLICT (gsis_id, source) DO UPDATE SET
                sleeper_id             = excluded.sleeper_id,
                observed_at            = excluded.observed_at,
                team                   = excluded.team,
                position               = excluded.position,
                injury_status          = excluded.injury_status,
                roster_status          = excluded.roster_status,
                practice_participation = excluded.practice_participation,
                injury_body_part       = excluded.injury_body_part,
                injury_notes           = excluded.injury_notes,
                depth_chart_position   = excluded.depth_chart_position,
                depth_chart_order      = excluded.depth_chart_order
            """
        )
        con.unregister("sleeper_events")
        con.unregister("sleeper_statuses")
    return events.height, statuses.height


@resilient(SOURCE, DATASET)
def ingest_players(force: bool = False) -> IngestResult:
    """Fetch the Sleeper player dump, diff it against the previous snapshot, and store both.

    Enforces the once-per-day courtesy limit (D8): if ``data/raw/sleeper_players.parquet`` is
    younger than ``settings.sleeper_min_cache_age_hours`` this returns immediately with
    ``skipped=True`` and makes no network call.

    Args:
        force: bypass the cache-age guard and re-fetch regardless.

    Returns:
        An :class:`~backend.ingest.base.IngestResult`. ``extra`` carries ``n_events``,
        ``n_missing_gsis``, ``n_duplicate_gsis``, ``n_statuses`` and ``archive``.
    """
    settings = get_settings()
    min_age = settings.sleeper_min_cache_age_hours
    age_h = cache_age_hours()

    if not force and age_h < min_age:
        detail = f"cache is {age_h:.1f}h old, minimum is {min_age:.1f}h (D8)"
        log.info("skipping Sleeper player dump: %s", detail)
        return IngestResult(
            source=SOURCE,
            dataset=DATASET,
            skipped=True,
            path=raw_path(PLAYERS_FILE),
            detail=detail,
            extra={"cache_age_hours": round(age_h, 2)},
        )

    fetched_at = utcnow()
    payload = _get_json(PLAYERS_URL, PLAYERS_TIMEOUT)
    if not isinstance(payload, dict) or not payload:
        raise SleeperError(f"empty or unexpected /v1/players/nfl payload: {type(payload).__name__}")

    current = _flatten(payload, fetched_at)
    log.info("Sleeper dump: %d records", current.height)

    # Read the old snapshot before anything can overwrite it, then archive the raw JSON so the
    # diff is reconstructible even if the database write below fails.
    previous = _read_previous_snapshot()
    archive = _archive_raw(payload, fetched_at)

    # Fill Sleeper's missing gsis_ids from the crosswalk before anything keys on them; two thirds
    # of the dump, and ~80% of the live injury designations in it, arrive without one (D6).
    resolved, n_recovered_gsis = _resolve_gsis(current)

    events = _diff_snapshots(previous, resolved, fetched_at)
    statuses, n_missing_gsis, n_duplicate_gsis = _status_rows(resolved, fetched_at)
    log.info(
        "%d players have no resolvable gsis_id (dropped from injury_status); "
        "%d duplicate gsis_id dropped",
        n_missing_gsis,
        n_duplicate_gsis,
    )

    n_events, n_statuses = _persist(events, statuses)
    path = write_parquet_atomic(current, raw_path(PLAYERS_FILE))

    record_freshness(SOURCE, ok=True, detail=f"{n_events} change events", n_rows=current.height)
    log.info(
        "Sleeper ingest complete: %d players, %d injury_events, %d injury_status rows",
        current.height,
        n_events,
        n_statuses,
    )
    return IngestResult(
        source=SOURCE,
        dataset=DATASET,
        n_rows=current.height,
        path=path,
        detail=f"{n_events} change events, {n_statuses} live statuses",
        extra={
            "n_events": n_events,
            "n_statuses": n_statuses,
            "n_missing_gsis": n_missing_gsis,
            "n_recovered_gsis": n_recovered_gsis,
            "n_duplicate_gsis": n_duplicate_gsis,
            "archive": str(archive) if archive else None,
            "first_snapshot": previous is None,
        },
    )
