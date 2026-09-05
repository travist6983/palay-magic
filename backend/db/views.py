"""DuckDB views over the raw Parquet cache.

Raw nflverse/Sleeper/ESPN data is the Parquet on disk; DuckDB just reads it (D5). This module is
the single registry of what a raw dataset is called, which files back it, and how it is exposed
to SQL. Ingest modules write the files; nothing else needs to know the paths.

``union_by_name=true`` matters: nflverse adds and renames columns between seasons, so a glob
across 2023-2026 must align on names rather than position.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

from backend.config import get_settings
from backend.logging_setup import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class RawView:
    """One raw dataset: a DuckDB view name and the Parquet glob that backs it."""

    view: str
    pattern: str
    """Glob relative to ``data/raw/``. ``{season}`` files use a ``*`` wildcard."""

    per_season: bool
    description: str


RAW_VIEWS: tuple[RawView, ...] = (
    # --- nflverse, per season ---------------------------------------------
    RawView("raw_pbp", "pbp_*.parquet", True, "Play-by-play with EPA/WP/CPOE/xpass"),
    RawView("raw_player_stats", "player_stats_*.parquet", True, "Weekly player box scores"),
    RawView("raw_snap_counts", "snap_counts_*.parquet", True, "Snap share (PFR), keyed on pfr_player_id"),
    RawView("raw_injuries", "injuries_*.parquet", True, "Official weekly injury report"),
    RawView("raw_depth_charts", "depth_charts_*.parquet", True, "Timestamped depth charts (no week column, D7)"),
    RawView("raw_rosters", "rosters_*.parquet", True, "Season rosters; source of sleeper_id/sportradar_id"),
    RawView("raw_pfr_pass", "pfr_pass_*.parquet", True, "PFR advanced passing (pressure, time to throw)"),
    RawView("raw_pfr_rush", "pfr_rush_*.parquet", True, "PFR advanced rushing (yards before/after contact)"),
    RawView("raw_pfr_rec", "pfr_rec_*.parquet", True, "PFR advanced receiving (drops, broken tackles)"),
    RawView("raw_pfr_def", "pfr_def_*.parquet", True, "PFR advanced defense (pressures, gamebook tackles)"),
    # --- nflverse, single file --------------------------------------------
    RawView("raw_players", "players.parquet", False, "Player master with espn_id/pfr_id/pff_id"),
    RawView("raw_teams", "teams.parquet", False, "Team abbreviations, names, colors, logos"),
    RawView("raw_schedules", "schedules.parquet", False, "Schedule + closing spread_line/total_line/roof/wind"),
    RawView("raw_ngs_passing", "ngs_passing.parquet", False, "Next Gen Stats passing (all seasons)"),
    RawView("raw_ngs_rushing", "ngs_rushing.parquet", False, "Next Gen Stats rushing (all seasons)"),
    RawView("raw_ngs_receiving", "ngs_receiving.parquet", False, "Next Gen Stats receiving (all seasons)"),
    # --- other sources ------------------------------------------------------
    RawView("raw_sleeper_players", "sleeper_players.parquet", False, "Latest Sleeper player dump"),
    RawView("raw_espn_injuries", "espn_injuries.parquet", False, "Latest ESPN per-team injury hydrate"),
    RawView("raw_espn_weather", "espn_weather.parquet", False, "ESPN scoreboard weather + venue indoor flag"),
    RawView("raw_odds", "odds_*.parquet", True, "The Odds API spreads/totals, one file per (season, week)"),
)

VIEW_BY_NAME: dict[str, RawView] = {v.view: v for v in RAW_VIEWS}


def raw_path(filename: str) -> Path:
    """Absolute path of a file in the raw Parquet cache."""
    return get_settings().raw_dir / filename


def season_file(dataset: str, season: int) -> Path:
    """Path for a per-season raw file, e.g. ``pbp_2025.parquet``."""
    return raw_path(f"{dataset}_{season}.parquet")


def _matches(spec: RawView) -> list[Path]:
    return sorted(get_settings().raw_dir.glob(spec.pattern))


def refresh_views(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """(Re)create a view for every raw dataset that has at least one file on disk.

    Returns a mapping of view name to the number of Parquet files backing it. Datasets with no
    files are skipped and their view dropped, so a partial ingest still yields a usable database.
    """
    settings = get_settings()
    created: dict[str, int] = {}

    for spec in RAW_VIEWS:
        files = _matches(spec)
        if not files:
            con.execute(f"DROP VIEW IF EXISTS {spec.view}")
            log.debug("no files for %s (%s) - view dropped", spec.view, spec.pattern)
            continue

        glob = str(settings.raw_dir / spec.pattern).replace("'", "''")
        con.execute(
            f"CREATE OR REPLACE VIEW {spec.view} AS "
            f"SELECT * FROM read_parquet('{glob}', union_by_name = true)"
        )
        created[spec.view] = len(files)

    log.info("refreshed %d raw views", len(created))
    return created


def view_exists(con: duckdb.DuckDBPyConnection, view: str) -> bool:
    """True if the named view or table is queryable."""
    rows = con.execute(
        "SELECT 1 FROM duckdb_views() WHERE view_name = ? "
        "UNION ALL SELECT 1 FROM duckdb_tables() WHERE table_name = ?",
        [view, view],
    ).fetchall()
    return bool(rows)


def table_counts(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str, int]]:
    """(kind, name, row_count) for every PropLab view and table. Drives the §9 checkpoint."""
    out: list[tuple[str, str, int]] = []

    for spec in RAW_VIEWS:
        if not view_exists(con, spec.view):
            out.append(("view", spec.view, -1))
            continue
        n = con.execute(f"SELECT count(*) FROM {spec.view}").fetchone()[0]
        out.append(("view", spec.view, int(n)))

    tables = con.execute(
        "SELECT table_name FROM duckdb_tables() WHERE schema_name = 'main' ORDER BY table_name"
    ).fetchall()
    for (name,) in tables:
        n = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
        out.append(("table", name, int(n)))

    return out
