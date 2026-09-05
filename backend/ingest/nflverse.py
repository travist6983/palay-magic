"""nflverse ingest — the backbone dataset pull (docs/DECISIONS.md D1, D5, D6).

**Source.** nflverse is not a REST API. It is versioned Parquet published to GitHub Releases
(https://github.com/nflverse/nflverse-data/releases) and refreshed on a schedule. We read it
through ``nflreadpy`` (D1: ``nfl_data_py`` cannot install on Python 3.12) and fall back to the
release assets directly when the package chokes on a season it does not know about yet.

**Rate limit: none.** No key, no quota, no throttling — nflverse is static files behind a CDN.
The only thing worth economising is bandwidth, so every dataset carries a max cache age and a
re-pull is skipped when the Parquet on disk is younger than that (``force=True`` overrides).

**Upstream cadence** (docs/reference/free_nfl_data_sources.md §1), which is what the max ages
are derived from:

===================  ==========================================
dataset              published
===================  ==========================================
play-by-play         nightly after each game day, plus in-game
next gen stats       nightly, ~3-5am ET in season
snap counts          every 6 hours (0, 6, 12, 18 UTC)
pfr advanced stats   daily, 7am UTC
rosters              daily, 7am UTC
depth charts         daily, 7am UTC, year-round
schedules            weekly (lines move all week)
===================  ==========================================

Files land in ``data/raw/`` under exactly the names ``backend/db/views.py`` globs for; the DuckDB
views are recreated at the end of every top-level operation that wrote something, so a dataset
whose first file has just landed gets its view created rather than staying dropped.
"""

from __future__ import annotations

import io
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path

import httpx
import nflreadpy as nfl
import polars as pl

from backend.config import get_settings
from backend.db.connection import connect
from backend.db.views import raw_path, refresh_views, season_file
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

SOURCE = "nflverse"

RELEASE_URL = "https://github.com/nflverse/nflverse-data/releases/download/{tag}/{filename}"
"""Direct release-asset template used by :func:`_download_release_asset`."""

HTTP_TIMEOUT_S = 180.0
"""pbp assets run to ~40 MB; a short timeout would fail on a slow link, not on a broken source."""

_HOURS_PER_YEAR = 24.0 * 365.0

# Patterns that mean "nflverse has not published this", as opposed to "something else broke".
# Both forms verified against nflreadpy 0.1.5: it range-checks the season locally and raises
# ValueError("Season must be between 1999 and 2025"), and wraps a requests failure as
# ConnectionError("Failed to download <url>: 404 Client Error: Not Found for url: ...")
# (nflreadpy/downloader.py:97).
#
# Two markers were removed as unsafe. "no such file" matched OSError's strerror for errno 2, so
# a broken local nflreadpy cache read as "the season is not out yet"; "not found" matched every
# ``KeyError: column 'x' not found`` and ``ModuleNotFoundError`` too. Both misclassified a real
# failure as an empty upstream, which :func:`_fetch` then reported as a clean skip. ``404`` is
# matched on a word boundary so a digit run inside a release URL cannot trigger it.
_UNAVAILABLE_MARKERS = (re.compile(r"\b404\b"), re.compile(r"season must be between"))


class DatasetUnavailable(Exception):
    """nflverse has not published this dataset/season yet. Not a failure — a skip."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    """One nflverse dataset: how to load it, where it lands, and how long the cache is good for.

    ``name`` is also the raw-file stem, so it must match the glob in
    :data:`backend.db.views.RAW_VIEWS` (``pbp`` -> ``pbp_2025.parquet`` -> ``raw_pbp``).
    """

    name: str
    per_season: bool
    loader: Callable[..., pl.DataFrame]
    """Called as ``loader([season])`` when per-season, ``loader()`` otherwise."""

    release_tag: str
    release_file: str
    """Asset name inside the release; ``{season}`` is substituted for per-season datasets."""

    max_age_hours: float
    """Skip a re-pull while the cached file is younger than this."""

    final_max_age_hours: float = _HOURS_PER_YEAR
    """Max age once the season is over and the files stop changing."""

    min_season: int = 1999
    """First season nflverse publishes for this dataset."""

    rename: dict[str, str] = field(default_factory=dict)
    """Column renames applied before writing (D6: everything joins on ``gsis_id``)."""

    required_columns: tuple[str, ...] = ()
    """Join keys downstream depends on, checked *after* :func:`_normalise`.

    nflverse renames columns between releases (``player_stats`` -> ``stats_player`` in 2024,
    NGS ``player_gsis_id``). Without this check a rename upstream would write a Parquet file
    that no longer carries its D6 join key and every downstream join would quietly return
    nothing. A missing key fails the ingest and leaves the previous good file in place.
    """

    description: str = ""

    def path(self, season: int | None = None) -> Path:
        """Absolute path of this dataset's Parquet file in the raw cache."""
        if self.per_season:
            if season is None:
                raise ValueError(f"{self.name} is per-season; a season is required")
            return season_file(self.name, season)
        return raw_path(f"{self.name}.parquet")

    def asset_name(self, season: int | None = None) -> str:
        """Release-asset filename for this dataset (and season, when per-season)."""
        return self.release_file.format(season=season)


_pfr = partial(nfl.load_pfr_advstats, summary_level="week")
_ngs = partial(nfl.load_nextgen_stats, seasons=True)

DATASETS: dict[str, Dataset] = {
    # --- per season -------------------------------------------------------
    "pbp": Dataset(
        name="pbp",
        per_season=True,
        loader=nfl.load_pbp,
        release_tag="pbp",
        release_file="play_by_play_{season}.parquet",
        max_age_hours=12.0,
        min_season=1999,
        required_columns=("game_id", "season", "week", "posteam"),
        description="Play-by-play with EPA/WP/CPOE/xpass",
    ),
    "player_stats": Dataset(
        name="player_stats",
        per_season=True,
        loader=nfl.load_player_stats,
        release_tag="stats_player",
        release_file="stats_player_week_{season}.parquet",
        max_age_hours=12.0,
        min_season=1999,
        required_columns=("player_id", "season", "week"),
        description="Weekly player box scores",
    ),
    "snap_counts": Dataset(
        name="snap_counts",
        per_season=True,
        loader=nfl.load_snap_counts,
        release_tag="snap_counts",
        release_file="snap_counts_{season}.parquet",
        max_age_hours=6.0,
        min_season=2012,
        required_columns=("pfr_player_id", "season", "week"),
        description="Snap share (PFR), keyed on pfr_player_id",
    ),
    "injuries": Dataset(
        name="injuries",
        per_season=True,
        loader=nfl.load_injuries,
        release_tag="injuries",
        release_file="injuries_{season}.parquet",
        max_age_hours=6.0,
        min_season=2009,
        required_columns=("gsis_id", "season", "week"),
        description="Official weekly injury report",
    ),
    "depth_charts": Dataset(
        name="depth_charts",
        per_season=True,
        loader=nfl.load_depth_charts,
        release_tag="depth_charts",
        release_file="depth_charts_{season}.parquet",
        max_age_hours=6.0,
        min_season=2001,
        required_columns=("gsis_id",),
        description="Timestamped depth charts, no week column (D7)",
    ),
    "rosters": Dataset(
        name="rosters",
        per_season=True,
        loader=nfl.load_rosters,
        release_tag="rosters",
        release_file="roster_{season}.parquet",
        max_age_hours=6.0,
        min_season=1920,
        required_columns=("gsis_id", "season", "team"),
        description="Season rosters; source of sleeper_id/sportradar_id",
    ),
    "pfr_pass": Dataset(
        name="pfr_pass",
        per_season=True,
        loader=partial(_pfr, stat_type="pass"),
        release_tag="pfr_advstats",
        release_file="advstats_week_pass_{season}.parquet",
        max_age_hours=12.0,
        min_season=2018,
        required_columns=("pfr_player_id", "season", "week"),
        description="PFR advanced passing (pressure, time to throw)",
    ),
    "pfr_rush": Dataset(
        name="pfr_rush",
        per_season=True,
        loader=partial(_pfr, stat_type="rush"),
        release_tag="pfr_advstats",
        release_file="advstats_week_rush_{season}.parquet",
        max_age_hours=12.0,
        min_season=2018,
        required_columns=("pfr_player_id", "season", "week"),
        description="PFR advanced rushing (yards before/after contact)",
    ),
    "pfr_rec": Dataset(
        name="pfr_rec",
        per_season=True,
        loader=partial(_pfr, stat_type="rec"),
        release_tag="pfr_advstats",
        release_file="advstats_week_rec_{season}.parquet",
        max_age_hours=12.0,
        min_season=2018,
        required_columns=("pfr_player_id", "season", "week"),
        description="PFR advanced receiving (drops, broken tackles)",
    ),
    "pfr_def": Dataset(
        name="pfr_def",
        per_season=True,
        loader=partial(_pfr, stat_type="def"),
        release_tag="pfr_advstats",
        release_file="advstats_week_def_{season}.parquet",
        max_age_hours=12.0,
        min_season=2018,
        required_columns=("pfr_player_id", "season", "week"),
        description="PFR advanced defense (pressures, gamebook tackles) — D2",
    ),
    # --- single file ------------------------------------------------------
    "players": Dataset(
        name="players",
        per_season=False,
        loader=nfl.load_players,
        release_tag="players",
        release_file="players.parquet",
        max_age_hours=6.0,
        required_columns=("gsis_id",),
        description="Player master with espn_id/pfr_id/pff_id (D6)",
    ),
    "teams": Dataset(
        name="teams",
        per_season=False,
        loader=nfl.load_teams,
        release_tag="teams",
        release_file="teams_colors_logos.parquet",
        max_age_hours=24.0 * 7,
        required_columns=("team_abbr",),
        description="Team abbreviations, names, colors, logos",
    ),
    "schedules": Dataset(
        name="schedules",
        per_season=False,
        loader=nfl.load_schedules,
        release_tag="schedules",
        release_file="games.parquet",
        max_age_hours=6.0,
        required_columns=("game_id", "season", "week", "spread_line", "total_line"),
        description="Schedule + spread_line/total_line/roof/wind (D4)",
    ),
    "ngs_passing": Dataset(
        name="ngs_passing",
        per_season=False,
        loader=partial(_ngs, stat_type="passing"),
        release_tag="nextgen_stats",
        release_file="ngs_passing.parquet",
        max_age_hours=12.0,
        rename={"player_gsis_id": "gsis_id"},
        required_columns=("gsis_id", "season", "week"),
        description="Next Gen Stats passing, all seasons",
    ),
    "ngs_rushing": Dataset(
        name="ngs_rushing",
        per_season=False,
        loader=partial(_ngs, stat_type="rushing"),
        release_tag="nextgen_stats",
        release_file="ngs_rushing.parquet",
        max_age_hours=12.0,
        rename={"player_gsis_id": "gsis_id"},
        required_columns=("gsis_id", "season", "week"),
        description="Next Gen Stats rushing, all seasons",
    ),
    "ngs_receiving": Dataset(
        name="ngs_receiving",
        per_season=False,
        loader=partial(_ngs, stat_type="receiving"),
        release_tag="nextgen_stats",
        release_file="ngs_receiving.parquet",
        max_age_hours=12.0,
        rename={"player_gsis_id": "gsis_id"},
        required_columns=("gsis_id", "season", "week"),
        description="Next Gen Stats receiving, all seasons",
    ),
}


def per_season_datasets() -> list[Dataset]:
    """Every dataset that is stored one file per season, in registry order."""
    return [d for d in DATASETS.values() if d.per_season]


def static_datasets() -> list[Dataset]:
    """Every single-file dataset (players, teams, schedules, NGS), in registry order."""
    return [d for d in DATASETS.values() if not d.per_season]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def current_season(now: datetime | None = None) -> int:
    """The season nflverse is currently publishing into.

    The NFL league year rolls over in mid-March, so anything from March onwards belongs to that
    calendar year's season. Used only to decide how aggressively to re-pull; the authoritative
    current week comes from Sleeper (D3).
    """
    now = now or utcnow()
    return now.year if now.month >= 3 else now.year - 1


def _is_final_season(season: int) -> bool:
    """True when the season is over, so its files will not change again."""
    return season < current_season()


def _max_age_hours(spec: Dataset, season: int | None) -> float:
    """Cache lifetime for this dataset/season: long once the season is in the books."""
    if spec.per_season and season is not None and _is_final_season(season):
        return spec.final_max_age_hours
    return spec.max_age_hours


def _cache_age_hours(path: Path) -> float | None:
    """Hours since ``path`` was last written, or None if it does not exist."""
    if not path.exists():
        return None
    return (utcnow().timestamp() - path.stat().st_mtime) / 3600.0


def _cached_rows(path: Path) -> int:
    """Row count of a cached Parquet file, read from its metadata. 0 if unreadable."""
    try:
        return int(pl.scan_parquet(path).select(pl.len()).collect().item())
    except Exception:  # noqa: BLE001 - reporting nicety, never worth failing an ingest
        return 0


def _looks_unavailable(exc: BaseException) -> bool:
    """True when the exception means "nflverse has no such file", not "something else broke".

    Structure first, then message. A 404 is authoritative. An ``OSError`` carrying a real
    ``errno`` is a local filesystem or socket failure — never evidence about what nflverse
    publishes — which matters because ``FileNotFoundError``'s "[Errno 2] No such file or
    directory" from a corrupt nflreadpy cache would otherwise read as "the season is not out
    yet" and be reported as a clean skip. nflreadpy's own 404 wrapper is a ``ConnectionError``
    built from a message string, so its ``errno`` is ``None`` and it still reaches the markers.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 404
    if isinstance(exc, OSError) and exc.errno is not None:
        return False
    message = str(exc).lower()
    return any(marker.search(message) for marker in _UNAVAILABLE_MARKERS)


def _normalise(df: pl.DataFrame, spec: Dataset) -> pl.DataFrame:
    """Apply the dataset's column renames (D6: NGS ``player_gsis_id`` -> ``gsis_id``)."""
    renames = {
        old: new for old, new in spec.rename.items() if old in df.columns and new not in df.columns
    }
    if not renames:
        return df
    log.debug("%s: renaming %s", spec.name, renames)
    return df.rename(renames)


def _missing_columns(df: pl.DataFrame, spec: Dataset) -> list[str]:
    """Declared join keys (D6) that this frame does not carry. Empty is the happy path."""
    return [c for c in spec.required_columns if c not in df.columns]


@network_retry
def _load_via_nflreadpy(spec: Dataset, season: int | None) -> pl.DataFrame:
    """Load one dataset through nflreadpy, retrying transport errors.

    Raises :class:`DatasetUnavailable` (which is deliberately *not* retried) when nflverse simply
    has not published the file, so a missing 2026 season fails fast instead of backing off four
    times against a guaranteed 404.
    """
    try:
        if spec.per_season:
            return spec.loader([season])
        return spec.loader()
    except Exception as exc:
        if _looks_unavailable(exc):
            raise DatasetUnavailable(str(exc)[:200]) from exc
        raise


@network_retry
def _download_release_asset(tag: str, filename: str) -> pl.DataFrame | None:
    """Fetch a Parquet asset straight from the nflverse-data GitHub release (D1 fallback).

    Args:
        tag: release tag, e.g. ``pbp`` or ``pfr_advstats``.
        filename: asset name inside that release, e.g. ``play_by_play_2025.parquet``.

    Returns:
        The parsed frame, or ``None`` when the asset does not exist (HTTP 404) — that is the
        "nflverse has not published this season yet" case, not an error.
    """
    url = RELEASE_URL.format(tag=tag, filename=filename)
    log.info("release fallback: GET %s", url)
    try:
        with httpx.Client(follow_redirects=True, timeout=HTTP_TIMEOUT_S) as client:
            resp = client.get(url)
    except httpx.TransportError as exc:
        # Re-raised as ConnectionError so network_retry (OSError-based) actually retries it.
        raise ConnectionError(f"{url}: {exc}") from exc

    if resp.status_code == 404:
        log.info("release asset %s/%s does not exist (404)", tag, filename)
        return None
    if resp.status_code >= 500:
        raise ConnectionError(f"{url}: HTTP {resp.status_code}")
    resp.raise_for_status()
    return pl.read_parquet(io.BytesIO(resp.content))


def _fetch(spec: Dataset, season: int | None) -> pl.DataFrame | None:
    """Load a dataset, falling back to the raw release asset if nflreadpy cannot.

    Returns ``None`` only for the one case that is genuinely not an error: nflreadpy said the
    file does not exist *and* the release asset agrees (404). That is "the season is not
    published yet".

    When nflreadpy failed for any other reason the release asset is still tried — it is the D1
    fallback for a broken package — but if it comes back empty the original exception is
    re-raised. Swallowing it would report a renamed asset or a broken nflreadpy as "nflverse has
    not published this yet": a green badge over a dataset that will never arrive.
    """
    label = f"{spec.name} {season}" if season is not None else spec.name
    loader_error: Exception | None = None
    try:
        return _load_via_nflreadpy(spec, season)
    except DatasetUnavailable as exc:
        log.info("nflreadpy has no %s (%s); checking the release asset", label, exc)
    except Exception as exc:  # noqa: BLE001 - the release asset is exactly the fallback for this
        log.warning("nflreadpy failed on %s (%s); checking the release asset", label, exc)
        loader_error = exc

    df = _download_release_asset(spec.release_tag, spec.asset_name(season))
    if df is None and loader_error is not None:
        log.error(
            "%s: nflreadpy failed and release asset %s/%s is absent — treating as a failure, "
            "not a missing season",
            label,
            spec.release_tag,
            spec.asset_name(season),
        )
        raise loader_error
    return df


def _ingest(spec: Dataset, season: int | None, force: bool) -> IngestResult:
    """Pull one dataset (one season if per-season) and write it to the raw cache.

    Never raises: a broken source returns ``ok=False`` and a season with no data yet returns
    ``skipped=True``. Freshness is recorded by the caller, once per top-level operation.
    """
    label = f"{spec.name}_{season}" if spec.per_season else spec.name
    path = spec.path(season)

    if spec.per_season and season is not None and season < spec.min_season:
        return IngestResult(
            source=SOURCE,
            dataset=label,
            skipped=True,
            detail=f"nflverse publishes {spec.name} from {spec.min_season} onwards",
        )

    age = _cache_age_hours(path)
    max_age = _max_age_hours(spec, season)
    if not force and age is not None and age < max_age:
        return IngestResult(
            source=SOURCE,
            dataset=label,
            n_rows=_cached_rows(path),
            path=path,
            skipped=True,
            detail=f"cache is {age:.1f}h old, under the {max_age:.0f}h max age",
        )

    try:
        df = _fetch(spec, season)
    except Exception as exc:  # noqa: BLE001 - D8: a dead source degrades, it never crashes
        log.exception("nflverse %s failed", label)
        return IngestResult(
            source=SOURCE, dataset=label, ok=False, detail=f"{type(exc).__name__}: {exc}"
        )

    if df is None:
        return IngestResult(
            source=SOURCE,
            dataset=label,
            skipped=True,
            detail="nflverse has not published this yet",
        )
    if df.height == 0:
        return IngestResult(
            source=SOURCE,
            dataset=label,
            skipped=True,
            detail="source returned 0 rows; leaving any existing cache in place",
        )

    try:
        df = _normalise(df, spec)
        missing = _missing_columns(df, spec)
        if missing:
            # Do not write: a file without its join key is worse than a stale one, because
            # every downstream join would return nothing and no stage would report an error.
            log.error("nflverse %s is missing join columns %s; refusing to write", label, missing)
            return IngestResult(
                source=SOURCE,
                dataset=label,
                ok=False,
                detail=f"missing required columns {missing}; kept the previous file",
            )
        write_parquet_atomic(df, path)
    except Exception as exc:  # noqa: BLE001 - D8: a full disk degrades, it never crashes
        log.exception("nflverse %s could not be written", label)
        return IngestResult(
            source=SOURCE, dataset=label, ok=False, detail=f"{type(exc).__name__}: {exc}"
        )

    return IngestResult(
        source=SOURCE,
        dataset=label,
        n_rows=df.height,
        path=path,
        detail=f"{df.width} columns",
    )


def _safe_ingest(spec: Dataset, season: int | None, force: bool) -> IngestResult:
    """:func:`_ingest` with a belt-and-braces guard, for the multi-dataset loops.

    ``backfill``/``refresh_current`` return a list, so they cannot carry ``@resilient`` (which
    substitutes a single ``IngestResult``). Anything escaping ``_ingest`` — a settings error, an
    unreadable raw directory — would otherwise abort the remaining datasets *and* skip
    ``_record``/``_refresh_views``, leaving ``source_freshness`` stamped green over a run that
    wrote nothing. One dataset's problem stays one dataset's problem.
    """
    label = f"{spec.name}_{season}" if spec.per_season else spec.name
    try:
        return _ingest(spec, season, force)
    except Exception as exc:  # noqa: BLE001 - D8: nothing here is allowed to be fatal
        log.exception("nflverse %s raised out of _ingest", label)
        return IngestResult(
            source=SOURCE, dataset=label, ok=False, detail=f"{type(exc).__name__}: {exc}"
        )


def _lookup(name: str, per_season: bool) -> Dataset | None:
    """The dataset by name, or ``None`` if the caller named the wrong one.

    Deliberately does not raise. Raising here reached ``@resilient``, which cannot tell a caller
    typo from a dead source and stamped ``source_freshness`` with ``ok=False`` — so a misspelled
    argument turned the nflverse badge yellow and overwrote the detail with a Python type error
    while every byte of data on disk was fine.
    """
    spec = DATASETS.get(name)
    if spec is None or spec.per_season != per_season:
        return None
    return spec


def _bad_dataset(name: str, per_season: bool) -> IngestResult:
    """A failed result for a dataset name that does not exist, or is of the other kind."""
    wanted = "per-season" if per_season else "single-file"
    known = sorted(d.name for d in DATASETS.values() if d.per_season == per_season)
    if name in DATASETS:
        detail = f"{name!r} is not {wanted}; use the other entry point"
    else:
        detail = f"unknown nflverse dataset {name!r}; {wanted} datasets are {known}"
    log.error("nflverse: %s", detail)
    return IngestResult(source=SOURCE, dataset=name, ok=False, detail=detail)


def _record(results: list[IngestResult]) -> None:
    """Write one ``source_freshness`` row for a whole operation (D8)."""
    written = [r for r in results if r.ok and not r.skipped]
    skipped = [r for r in results if r.skipped]
    failed = [r for r in results if not r.ok]
    # Rows on hand, not rows downloaded: a cache-fresh refresh writes nothing but the corpus is
    # still 1.5M rows, and reporting the newly-written count made the badge read like the data
    # had shrunk to whatever the last touched dataset happened to be.
    total = sum(r.n_rows for r in results if r.ok)
    detail = f"{len(written)} written, {len(skipped)} skipped, {len(failed)} failed" + (
        f"; failed: {', '.join(r.dataset for r in failed[:6])}" if failed else ""
    )
    record_freshness(SOURCE, ok=not failed, detail=detail, n_rows=total or None)


def _refresh_views() -> None:
    """Recreate the DuckDB raw views so they see the files we just wrote (D5)."""
    try:
        with connect() as con:
            created = refresh_views(con)
        log.info("nflverse ingest: %d raw views now backed by files", len(created))
    except Exception:  # noqa: BLE001 - bookkeeping must never break ingest
        log.exception("could not refresh DuckDB views after nflverse ingest")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@resilient(SOURCE, "ingest_season")
def ingest_season(dataset: str, season: int, force: bool = False) -> IngestResult:
    """Pull one per-season nflverse dataset and cache it as Parquet.

    Args:
        dataset: registry key, e.g. ``pbp``, ``player_stats``, ``pfr_def``.
        season: the season to pull.
        force: re-pull even when the cached file is still within its max age.

    Returns:
        An :class:`IngestResult`. ``skipped=True`` when the cache is fresh or nflverse has no
        data for that season yet; ``ok=False`` when the source failed.
    """
    spec = _lookup(dataset, per_season=True)
    if spec is None:
        return _bad_dataset(dataset, per_season=True)
    result = _ingest(spec, season, force)
    log.info("nflverse %s", result)
    _record([result])
    if result:
        _refresh_views()
    return result


@resilient(SOURCE, "ingest_static")
def ingest_static(dataset: str, force: bool = False) -> IngestResult:
    """Pull one single-file nflverse dataset and cache it as Parquet.

    Single-file datasets are ``players``, ``teams``, ``schedules``, ``ngs_passing``,
    ``ngs_rushing`` and ``ngs_receiving`` — each covers every season in one file.

    Args:
        dataset: registry key.
        force: re-pull even when the cached file is still within its max age.
    """
    spec = _lookup(dataset, per_season=False)
    if spec is None:
        return _bad_dataset(dataset, per_season=False)
    result = _ingest(spec, None, force)
    log.info("nflverse %s", result)
    _record([result])
    if result:
        _refresh_views()
    return result


def backfill(seasons: list[int] | None = None, force: bool = False) -> list[IngestResult]:
    """One-time full historical pull: every dataset, every configured season.

    Args:
        seasons: seasons to pull. Defaults to ``get_settings().seasons``.
        force: ignore cache max ages and re-download everything.

    Returns:
        One :class:`IngestResult` per (dataset, season). Seasons nflverse has not published yet
        come back skipped, so a run that includes the upcoming season still succeeds.
    """
    targets = sorted(set(seasons if seasons is not None else get_settings().seasons))
    log.info("nflverse backfill: seasons %s (force=%s)", targets, force)

    results: list[IngestResult] = []
    for spec in static_datasets():
        result = _safe_ingest(spec, None, force)
        log.info("nflverse %s", result)
        results.append(result)

    for season in targets:
        for spec in per_season_datasets():
            result = _safe_ingest(spec, season, force)
            log.info("nflverse %s", result)
            results.append(result)

    _record(results)
    _refresh_views()
    return results


def refresh_current(season: int, force: bool = False) -> list[IngestResult]:
    """Re-pull the current season plus every single-file dataset. This is ``make refresh``.

    Historical seasons are deliberately left alone — their files stop changing once the season
    ends. Schedules move every week and players/rosters/depth charts move daily, so the
    single-file datasets are always in scope.

    Args:
        season: the season in progress.
        force: ignore cache max ages and re-download everything.
    """
    log.info("nflverse refresh: season %d (force=%s)", season, force)

    results: list[IngestResult] = []
    for spec in per_season_datasets():
        result = _safe_ingest(spec, season, force)
        log.info("nflverse %s", result)
        results.append(result)

    for spec in static_datasets():
        result = _safe_ingest(spec, None, force)
        log.info("nflverse %s", result)
        results.append(result)

    _record(results)
    _refresh_views()
    return results
