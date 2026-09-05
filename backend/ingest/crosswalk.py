"""Build the ``players`` ID crosswalk — the table every other join goes through (D6).

This module makes **no network calls**. It reads Parquet that ``backend/ingest/nflverse.py`` and
``backend/ingest/sleeper.py`` already put on disk, through the DuckDB raw views (D5), so it has no
rate limit of its own and must run *after* those two ingest steps. The upstream limits it inherits
are documented in D8: nflverse is unlimited, the Sleeper full dump is one call per day.

Field precedence (highest first), per docs/DECISIONS.md D6:

1. ``raw_players`` (nflverse ``players.parquet``) — the master list and the source of
   ``espn_id`` / ``pfr_id`` / ``pff_id`` plus all biographical fields.
2. ``raw_rosters`` for the **latest season present** — adds ``sleeper_id`` and ``sportradar_id``,
   and supplies the current team. Roster team beats ``players.latest_team`` because rosters are
   season-scoped while ``latest_team`` lags. A player who is on a current roster but has not yet
   been added to ``players.parquet`` (undrafted rookies, practice-squad callups) is carried in
   from the roster so the crosswalk is not blind to him.
3. ``raw_sleeper_players`` — authority for ``sleeper_id`` when the roster has none, joined on
   ``gsis_id`` wherever Sleeper supplies one.

``gsis_id`` is the primary key, so rows without one cannot be stored; they are counted and logged
rather than silently dropped.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Final

import duckdb
import polars as pl

from backend.db.connection import connect
from backend.db.views import refresh_views, view_exists
from backend.ingest.base import (
    IngestResult,
    network_retry,
    record_freshness,
    resilient,
    utcnow,
)
from backend.logging_setup import get_logger

log = get_logger(__name__)

SOURCE: Final[str] = "crosswalk"
"""Freshness key. Deliberately distinct from 'nflverse'/'sleeper' so this build's badge cannot
overwrite the badge of the ingest step that produced its inputs."""

DATASET: Final[str] = "players"

PLAYER_COLUMNS: Final[tuple[str, ...]] = (
    "gsis_id",
    "display_name",
    "first_name",
    "last_name",
    "position",
    "position_group",
    "team",
    "espn_id",
    "pfr_id",
    "pff_id",
    "sleeper_id",
    "sportradar_id",
    "birth_date",
    "height",
    "weight",
    "headshot_url",
    "rookie_season",
    "last_season",
    "years_exp",
    "draft_year",
    "draft_round",
    "draft_pick",
    "status",
    "updated_at",
)
"""Exact column list of ``players`` in 001_initial.sql, in order. Named explicitly in both halves
of the INSERT so a schema change fails loudly instead of shifting columns."""

ID_COLUMNS: Final[tuple[str, ...]] = (
    "gsis_id",
    "espn_id",
    "pfr_id",
    "pff_id",
    "sleeper_id",
    "sportradar_id",
)
"""Every ID column :func:`id_map` and :func:`coverage_report` know about."""

_NULLISH: Final[frozenset[str]] = frozenset({"", "na", "n/a", "nan", "none", "null", "-"})
"""Placeholder strings that upstream feeds use to mean "missing"."""


# ---------------------------------------------------------------------------
# Column readers. Each tolerates the column being absent from the source frame,
# because sibling ingest modules own those files and may add or rename columns.
# ---------------------------------------------------------------------------


def _has(df: pl.DataFrame, col: str) -> bool:
    return col in df.columns


def _text(df: pl.DataFrame, col: str) -> pl.Expr:
    """Read ``col`` as trimmed text, mapping placeholder strings to null."""
    if not _has(df, col):
        return pl.lit(None, dtype=pl.Utf8)
    value = pl.col(col).cast(pl.Utf8, strict=False).str.strip_chars()
    return pl.when(value.str.to_lowercase().is_in(list(_NULLISH))).then(None).otherwise(value)


def _string_id(df: pl.DataFrame, col: str) -> pl.Expr:
    """Read an external ID as text.

    Feeds disagree on type for the same ID: nflverse ships ``espn_id`` as a string, Sleeper ships
    it as an integer, and a Parquet round-trip can turn either into a float. Casting a float
    straight to text would produce ``"14856.0"``, which joins to nothing, so numerics are rounded
    through Int64 first.
    """
    if not _has(df, col):
        return pl.lit(None, dtype=pl.Utf8)
    dtype = df.schema[col]
    value = pl.col(col)
    if dtype.is_float():
        value = value.round(0).cast(pl.Int64, strict=False)
    value = value.cast(pl.Utf8, strict=False).str.strip_chars()
    return pl.when(value.str.to_lowercase().is_in(list(_NULLISH))).then(None).otherwise(value)


def _integer(
    df: pl.DataFrame,
    col: str,
    *,
    feet_inches: bool = False,
    min_valid: int | None = None,
) -> pl.Expr:
    """Read ``col`` as a nullable Int64.

    ``height`` and ``weight`` are INTEGER in the schema but arrive as Int32 from nflverse and as
    strings from Sleeper — and Sleeper's height is sometimes ``"74"`` and sometimes ``"6'2\\""``.
    A value that cannot be parsed becomes null; it never drops the row.

    Args:
        feet_inches: also accept ``F'I"`` notation and convert it to total inches.
        min_valid: values below this become null. Use ``1`` for height/weight (Sleeper writes
            ``0'0"`` for unknown); leave ``None`` for years_exp, where 0 means rookie.
    """
    if not _has(df, col):
        return pl.lit(None, dtype=pl.Int64)

    if df.schema[col] == pl.Utf8:
        text = pl.col(col).str.strip_chars()
        plain = (
            text.str.extract(r"^(\d+(?:\.\d+)?)$", 1)
            .cast(pl.Float64, strict=False)
            .round(0)
            .cast(pl.Int64, strict=False)
        )
        if feet_inches:
            feet = text.str.extract(r"^(\d+)\s*'", 1).cast(pl.Int64, strict=False)
            inches = text.str.extract(r"'\s*(\d+)", 1).cast(pl.Int64, strict=False)
            value = (
                pl.when(feet.is_not_null())
                .then(feet * 12 + inches.fill_null(0))
                .otherwise(plain)
            )
        else:
            value = plain
    else:
        value = pl.col(col).cast(pl.Float64, strict=False).round(0).cast(pl.Int64, strict=False)

    if min_valid is not None:
        value = pl.when(value >= min_valid).then(value).otherwise(None)
    return value


def _date(df: pl.DataFrame, col: str) -> pl.Expr:
    """Read ``col`` as a nullable Date. nflverse ships ISO strings, rosters ship a real Date."""
    if not _has(df, col):
        return pl.lit(None, dtype=pl.Date)
    dtype = df.schema[col]
    if dtype == pl.Date:
        return pl.col(col)
    if isinstance(dtype, pl.Datetime):
        return pl.col(col).cast(pl.Date, strict=False)
    return (
        pl.col(col)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
    )


def _coalesce(*names: str) -> pl.Expr:
    """First non-null of the named prepared columns."""
    return pl.coalesce([pl.col(n) for n in names])


# ---------------------------------------------------------------------------
# Reading the raw views
# ---------------------------------------------------------------------------


@network_retry
def _read(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    params: list[object] | None = None,
) -> pl.DataFrame:
    """Run ``sql`` and return polars.

    Retried on transient I/O: the raw Parquet is replaced atomically by other ingest steps, so a
    read that starts while a file is being swapped can fail once and succeed immediately after.
    """
    return con.execute(sql, params or []).pl()


def _latest_roster_season(con: duckdb.DuckDBPyConnection) -> int | None:
    """Newest season present in ``raw_rosters``, or None when no roster file exists."""
    if not view_exists(con, "raw_rosters"):
        return None
    row = _read(con, "SELECT max(season) AS season FROM raw_rosters").row(0)
    return int(row[0]) if row[0] is not None else None


def _prepare_players(df: pl.DataFrame) -> pl.DataFrame:
    """Normalise ``raw_players`` into ``p_*`` columns keyed on a cleaned ``gsis_id``."""
    return df.select(
        _string_id(df, "gsis_id").alias("gsis_id"),
        _text(df, "display_name").alias("p_display_name"),
        _text(df, "first_name").alias("p_first_name"),
        _text(df, "last_name").alias("p_last_name"),
        _text(df, "position").alias("p_position"),
        _text(df, "position_group").alias("p_position_group"),
        _text(df, "latest_team").alias("p_team"),
        _string_id(df, "espn_id").alias("p_espn_id"),
        _string_id(df, "pfr_id").alias("p_pfr_id"),
        _string_id(df, "pff_id").alias("p_pff_id"),
        _date(df, "birth_date").alias("p_birth_date"),
        _integer(df, "height", feet_inches=True, min_valid=1).alias("p_height"),
        _integer(df, "weight", min_valid=1).alias("p_weight"),
        _text(df, "headshot").alias("p_headshot_url"),
        _integer(df, "rookie_season").alias("p_rookie_season"),
        _integer(df, "last_season").alias("p_last_season"),
        _integer(df, "years_of_experience").alias("p_years_exp"),
        _integer(df, "draft_year").alias("p_draft_year"),
        _integer(df, "draft_round").alias("p_draft_round"),
        _integer(df, "draft_pick").alias("p_draft_pick"),
        _text(df, "status").alias("p_status"),
    )


def _prepare_rosters(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Normalise one season of ``raw_rosters`` into one ``r_*`` row per player.

    A season's roster file has a row per player *per week* (2025 carries weeks 1-22), so the newest
    week wins; within one week a row that still knows the team is preferred over one that does not,
    and the remaining ties break on the source row order so two runs agree.

    Rows without a ``gsis_id`` are dropped here rather than collapsed by the de-duplication below,
    which would otherwise fold *all* of them into a single row and under-report the loss.

    Returns:
        The prepared frame and the number of source rows dropped for a missing ``gsis_id``.
    """
    week = _integer(df, "week").fill_null(0).alias("_week")
    prepared = df.select(
        _string_id(df, "gsis_id").alias("gsis_id"),
        week,
        _text(df, "team").alias("r_team"),
        _string_id(df, "sleeper_id").alias("r_sleeper_id"),
        _string_id(df, "sportradar_id").alias("r_sportradar_id"),
        _string_id(df, "espn_id").alias("r_espn_id"),
        _string_id(df, "pfr_id").alias("r_pfr_id"),
        _string_id(df, "pff_id").alias("r_pff_id"),
        _text(df, "full_name").alias("r_display_name"),
        _text(df, "first_name").alias("r_first_name"),
        _text(df, "last_name").alias("r_last_name"),
        _text(df, "position").alias("r_position"),
        _text(df, "status").alias("r_status"),
        _date(df, "birth_date").alias("r_birth_date"),
        _integer(df, "height", feet_inches=True, min_valid=1).alias("r_height"),
        _integer(df, "weight", min_valid=1).alias("r_weight"),
        _text(df, "headshot_url").alias("r_headshot_url"),
        _integer(df, "years_exp").alias("r_years_exp"),
        # Marks a row as coming from the current roster. Survives the full join in _assemble, so
        # _drop_ambiguous_ids can tell a current player from one who last played decades ago.
        pl.lit(True).alias("_on_roster"),
    )
    n_no_gsis = int(prepared.select(pl.col("gsis_id").is_null().sum()).item())
    prepared = prepared.filter(pl.col("gsis_id").is_not_null())
    return (
        prepared.sort(
            ["_week", "r_team"],
            descending=[True, False],
            nulls_last=True,
            maintain_order=True,
        )
        .unique(subset=["gsis_id"], keep="first", maintain_order=True)
        .drop("_week"),
        n_no_gsis,
    )


_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})


def normalise_name(name: str | None) -> str | None:
    """Reduce a display name to a comparison key: lowercase, no punctuation, no generational suffix.

    ``D.J. Johnson``, ``DJ Johnson`` and ``D J Johnson`` all become ``dj johnson``;
    ``Kenneth Murray, Jr.`` becomes ``kenneth murray``.

    This is the scalar spelling of the name component of :func:`_name_key_expr`, which is what
    :func:`_resolve_sleeper_by_name` actually runs. Keep the two in step: they must agree for a
    caller to reason about a match the fallback made.
    """
    if not name:
        return None
    cleaned = re.sub(r"[^a-z\s]", "", name.lower())
    parts = [w for w in cleaned.split() if w and w not in _SUFFIXES]
    return " ".join(parts) or None


def _name_key_expr(name_col: str, team_col: str, position_col: str) -> pl.Expr:
    """``normalised name | team | position`` as a single join key."""
    return pl.concat_str(
        [
            pl.col(name_col)
            .str.to_lowercase()
            .str.replace_all(r"[^a-z\s]", "")
            .str.split(" ")
            .list.eval(
                pl.element().filter(
                    ~pl.element().is_in(list(_SUFFIXES)) & (pl.element() != "")
                )
            )
            .list.join(" "),
            pl.col(team_col).str.to_uppercase(),
            pl.col(position_col).str.to_uppercase(),
        ],
        separator="|",
        ignore_nulls=False,
    )


def _resolve_sleeper_by_name(
    sleeper_raw: pl.DataFrame,
    players: pl.DataFrame,
    rosters: pl.DataFrame,
) -> pl.DataFrame:
    """Recover a ``gsis_id`` for Sleeper rows that do not carry one.

    Sleeper supplies ``gsis_id`` for only about a third of its dump — in the 2026 snapshot
    CeeDee Lamb, Bucky Irving and Brandon Aubrey all lack one. Left unresolved, roughly a fifth of
    rostered skill players get no injury designation or practice status at all, which guts §5.7.

    Two fallback keys, applied in order and only when the match is **unique on both sides**:

    1. ``espn_id``, when Sleeper has one and the crosswalk knows it.
    2. ``normalised name | team | position`` against the current roster.

    Ambiguous matches (the same key on two players, e.g. two ``Michael Carter``s) are dropped
    rather than guessed — a wrong crosswalk row silently attaches one player's injury to another.

    Returns:
        The Sleeper frame with ``gsis_id`` filled in where it could be resolved unambiguously.
    """
    sleeper_id_col = "player_id" if _has(sleeper_raw, "player_id") else "sleeper_id"

    work = sleeper_raw.select(
        _string_id(sleeper_raw, "gsis_id").alias("gsis_id"),
        _string_id(sleeper_raw, sleeper_id_col).alias("sleeper_key"),
        _string_id(sleeper_raw, "espn_id").alias("espn_id"),
        _text(sleeper_raw, "full_name").alias("full_name"),
        _text(sleeper_raw, "team").alias("team"),
        _text(sleeper_raw, "position").alias("position"),
    ).filter(pl.col("sleeper_key").is_not_null())

    known = work.filter(pl.col("gsis_id").is_not_null())
    unknown = work.filter(pl.col("gsis_id").is_null()).drop("gsis_id")
    if unknown.is_empty():
        return sleeper_raw

    taken = set(known["gsis_id"].to_list())

    # --- 1. espn_id ---------------------------------------------------------
    espn_lookup = (
        players.select(pl.col("gsis_id"), pl.col("p_espn_id").alias("espn_id"))
        .filter(pl.col("espn_id").is_not_null())
        .unique(subset=["espn_id"], keep="none")  # drop any espn_id shared by two players
    )
    unknown = unknown.join(espn_lookup, on="espn_id", how="left")
    by_espn = unknown.filter(
        pl.col("gsis_id").is_not_null() & ~pl.col("gsis_id").is_in(list(taken))
    )
    # Sleeper ships duplicate entries for the same person (two rows, one espn_id -- De'Jon Harris
    # is 7493 and 7504). Without this the same player is claimed twice and _prepare_sleeper picks
    # between them arbitrarily. Prefer the row Sleeper still has on a team, then the lowest key,
    # so the choice is the live account and is the same on every run.
    by_espn = by_espn.sort(
        ["gsis_id", "team", "sleeper_key"], nulls_last=True, maintain_order=True
    ).unique(subset=["gsis_id"], keep="first", maintain_order=True)
    taken |= set(by_espn["gsis_id"].to_list())

    # --- 2. normalised name | team | position -------------------------------
    remaining = unknown.filter(pl.col("gsis_id").is_null()).drop("gsis_id")

    roster_names = (
        rosters.select(
            pl.col("gsis_id"),
            _coalesce("r_display_name").alias("full_name"),
            pl.col("r_team").alias("team"),
            pl.col("r_position").alias("position"),
        )
        .filter(
            pl.col("gsis_id").is_not_null()
            & pl.col("full_name").is_not_null()
            & pl.col("team").is_not_null()
            & pl.col("position").is_not_null()
        )
        .with_columns(_name_key_expr("full_name", "team", "position").alias("name_key"))
        .select("gsis_id", "name_key")
        .unique(subset=["name_key"], keep="none")  # a duplicated key is not a usable match
    )

    remaining = (
        remaining.with_columns(_name_key_expr("full_name", "team", "position").alias("name_key"))
        .unique(subset=["name_key"], keep="none")
        .join(roster_names, on="name_key", how="inner")
        .filter(~pl.col("gsis_id").is_in(list(taken)))
        .unique(subset=["gsis_id"], keep="none")
    )

    recovered = pl.concat(
        [
            by_espn.select("sleeper_key", "gsis_id"),
            remaining.select("sleeper_key", "gsis_id"),
        ],
        how="vertical",
    ).unique(subset=["sleeper_key"], keep="none")

    log.info(
        "sleeper id recovery: %d rows had a gsis_id, recovered %d more"
        " (%d via espn_id, %d via name)",
        known.height,
        recovered.height,
        by_espn.height,
        remaining.height,
    )
    if recovered.is_empty():
        return sleeper_raw

    return (
        sleeper_raw.with_columns(_string_id(sleeper_raw, sleeper_id_col).alias("_sleeper_key"))
        .join(
            recovered.rename({"sleeper_key": "_sleeper_key", "gsis_id": "_recovered_gsis"}),
            on="_sleeper_key",
            how="left",
        )
        .with_columns(
            pl.coalesce(
                _string_id(sleeper_raw, "gsis_id"), pl.col("_recovered_gsis")
            ).alias("gsis_id")
        )
        .drop("_sleeper_key", "_recovered_gsis")
    )


def _prepare_sleeper(df: pl.DataFrame) -> pl.DataFrame:
    """Normalise ``raw_sleeper_players`` into one ``s_*`` row per ``gsis_id``.

    Rows without a ``gsis_id`` are unusable, so :func:`_resolve_sleeper_by_name` runs first to
    recover as many as it safely can. Sleeper carries a handful of duplicate ``gsis_id`` values
    (retired/duplicate entries); the row that still has a team wins.
    """
    sleeper_id_col = "player_id" if _has(df, "player_id") else "sleeper_id"
    prepared = df.select(
        _string_id(df, "gsis_id").alias("gsis_id"),
        _string_id(df, sleeper_id_col).alias("s_sleeper_id"),
        _string_id(df, "sportradar_id").alias("s_sportradar_id"),
        _string_id(df, "espn_id").alias("s_espn_id"),
        _text(df, "team").alias("s_team"),
        _text(df, "position").alias("s_position"),
        _date(df, "birth_date").alias("s_birth_date"),
        _integer(df, "height", feet_inches=True, min_valid=1).alias("s_height"),
        _integer(df, "weight", min_valid=1).alias("s_weight"),
        _integer(df, "years_exp").alias("s_years_exp"),
    ).filter(pl.col("gsis_id").is_not_null() & pl.col("s_sleeper_id").is_not_null())

    return (
        # s_sleeper_id breaks the ties s_team leaves, so a player with two Sleeper rows gets the
        # same one on every run instead of whichever the unstable sort happened to surface.
        prepared.sort(["s_team", "s_sleeper_id"], nulls_last=True, maintain_order=True)
        .unique(subset=["gsis_id"], keep="first", maintain_order=True)
        .drop("s_team")
    )


def _assemble(
    players: pl.DataFrame,
    rosters: pl.DataFrame,
    sleeper: pl.DataFrame,
    now: datetime,
) -> pl.DataFrame:
    """Join the three prepared frames and resolve every field by precedence."""
    joined = players.join(rosters, on="gsis_id", how="full", coalesce=True).join(
        sleeper, on="gsis_id", how="left"
    )

    return joined.select(
        pl.col("gsis_id"),
        _coalesce("p_display_name", "r_display_name").alias("display_name"),
        _coalesce("p_first_name", "r_first_name").alias("first_name"),
        _coalesce("p_last_name", "r_last_name").alias("last_name"),
        _coalesce("p_position", "r_position", "s_position").alias("position"),
        pl.col("p_position_group").alias("position_group"),
        # Roster team wins: rosters are season-scoped, players.latest_team lags.
        _coalesce("r_team", "p_team").alias("team"),
        _coalesce("p_espn_id", "r_espn_id", "s_espn_id").alias("espn_id"),
        _coalesce("p_pfr_id", "r_pfr_id").alias("pfr_id"),
        _coalesce("p_pff_id", "r_pff_id").alias("pff_id"),
        _coalesce("r_sleeper_id", "s_sleeper_id").alias("sleeper_id"),
        _coalesce("r_sportradar_id", "s_sportradar_id").alias("sportradar_id"),
        _coalesce("p_birth_date", "r_birth_date", "s_birth_date").alias("birth_date"),
        _coalesce("p_height", "r_height", "s_height").alias("height"),
        _coalesce("p_weight", "r_weight", "s_weight").alias("weight"),
        _coalesce("p_headshot_url", "r_headshot_url").alias("headshot_url"),
        pl.col("p_rookie_season").alias("rookie_season"),
        pl.col("p_last_season").alias("last_season"),
        _coalesce("p_years_exp", "r_years_exp", "s_years_exp").alias("years_exp"),
        pl.col("p_draft_year").alias("draft_year"),
        pl.col("p_draft_round").alias("draft_round"),
        pl.col("p_draft_pick").alias("draft_pick"),
        _coalesce("p_status", "r_status").alias("status"),
        pl.lit(now).alias("updated_at"),
        # Not a players column; consumed by _drop_ambiguous_ids and then discarded.
        pl.col("_on_roster").fill_null(False).alias("on_current_roster"),
    )


def _drop_ambiguous_ids(frame: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    """Null an external ID wherever it maps to more than one ``gsis_id``.

    PFR reuses slugs across eras: ``WoodPe00`` is both Peter Woods, a 2025 first-round DT, and
    Pete Woods, last seen in 1980. Left alone, every ``snap_counts``/``pfr_advstats`` join on that
    ID fans out to two rows and double-counts the player. The most recently active player keeps
    the ID; everyone else loses it.

    ``on_current_roster`` leads the ordering because ``last_season`` alone gets it backwards for
    the players this module works hardest to include: someone carried in from the roster is not in
    ``players.parquet`` yet, so his ``last_season`` is null, and a null sorts *below* a player last
    seen in 1980. Ranking roster membership first hands the ID to the man playing this season.

    Returns:
        The cleaned frame and a per-column count of the IDs that were cleared.
    """
    ordered = frame.sort(
        ["on_current_roster", "last_season", "rookie_season", "gsis_id"],
        descending=[True, True, True, False],
        nulls_last=True,
        maintain_order=True,
    )
    cleared: dict[str, int] = {}

    for col in ID_COLUMNS:
        if col == "gsis_id":  # the primary key; duplicates are removed, not blanked
            continue
        before = int(ordered.select(pl.col(col).is_not_null().sum()).item())
        ordered = ordered.with_columns(
            pl.when(pl.col(col).is_null() | (pl.int_range(pl.len()).over(col) == 0))
            .then(pl.col(col))
            .otherwise(None)
            .alias(col)
        )
        after = int(ordered.select(pl.col(col).is_not_null().sum()).item())
        if before != after:
            cleared[col] = before - after

    return ordered, cleared


def _no_rows() -> pl.DataFrame:
    """An empty frame with just a ``gsis_id`` column, for a source that is not on disk yet."""
    return pl.DataFrame({"gsis_id": []}, schema={"gsis_id": pl.Utf8})


def _replace_players(con: duckdb.DuckDBPyConnection, frame: pl.DataFrame) -> None:
    """DELETE + INSERT the whole table inside one transaction.

    Columns are named on both sides of the INSERT so a migration that adds or reorders a column
    raises here rather than shifting every value one place to the left.
    """
    columns = ", ".join(PLAYER_COLUMNS)
    con.register("tmp_players", frame.select(list(PLAYER_COLUMNS)))
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute("DELETE FROM players")
        con.execute(f"INSERT INTO players ({columns}) SELECT {columns} FROM tmp_players")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.unregister("tmp_players")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@resilient(SOURCE, DATASET)
def build_players() -> IngestResult:
    """Rebuild the ``players`` crosswalk from the raw views. Never raises.

    Reads ``raw_players``, the newest season of ``raw_rosters`` and ``raw_sleeper_players``,
    resolves every field by the precedence documented at the top of this module, and replaces the
    whole table in one transaction.

    Returns:
        An :class:`IngestResult` whose ``extra`` carries ``dropped_null_gsis``,
        ``dropped_duplicate_gsis``, ``cleared_ambiguous_ids``, ``roster_season`` and the
        per-ID fill counts.
    """
    now = utcnow()

    with connect() as con:
        refresh_views(con)

        if not view_exists(con, "raw_players"):
            log.warning("raw_players view is missing - run the nflverse ingest first")
            record_freshness(SOURCE, ok=False, detail="raw_players view missing")
            return IngestResult(
                source=SOURCE,
                dataset=DATASET,
                skipped=True,
                detail="raw_players view missing; run nflverse ingest first",
            )

        players = _prepare_players(_read(con, "SELECT * FROM raw_players"))
        log.info("raw_players: %d rows", players.height)

        roster_season = _latest_roster_season(con)
        if roster_season is None:
            log.warning("no raw_rosters files - sleeper_id/sportradar_id will be sparse")
            rosters, n_roster_no_gsis = _prepare_rosters(_no_rows())
        else:
            rosters, n_roster_no_gsis = _prepare_rosters(
                _read(con, "SELECT * FROM raw_rosters WHERE season = ?", [roster_season])
            )
            log.info("raw_rosters season %d: %d players", roster_season, rosters.height)

        if view_exists(con, "raw_sleeper_players"):
            sleeper_raw = _read(con, "SELECT * FROM raw_sleeper_players")
            sleeper_raw = _resolve_sleeper_by_name(sleeper_raw, players, rosters)
            sleeper = _prepare_sleeper(sleeper_raw)
            log.info("raw_sleeper_players: %d rows carrying a gsis_id", sleeper.height)
        else:
            log.warning("no raw_sleeper_players view - falling back to roster sleeper_ids only")
            sleeper = _prepare_sleeper(_no_rows())

        frame = _assemble(players, rosters, sleeper, now)

        # Roster rows with no gsis_id were already counted and dropped by _prepare_rosters; what
        # is left here comes from players.parquet. Adding them keeps the reported figure equal to
        # the number of source rows lost, not the number of surviving null-keyed rows.
        n_null_gsis = (
            int(frame.select(pl.col("gsis_id").is_null().sum()).item()) + n_roster_no_gsis
        )
        frame = frame.filter(pl.col("gsis_id").is_not_null())

        before_dedupe = frame.height
        frame = frame.unique(subset=["gsis_id"], keep="first", maintain_order=True)
        n_duplicates = before_dedupe - frame.height

        if n_null_gsis:
            log.warning(
                "dropped %d row(s) with a null gsis_id - it is the primary key", n_null_gsis
            )
        if n_duplicates:
            log.warning("dropped %d duplicate gsis_id row(s)", n_duplicates)

        frame, cleared = _drop_ambiguous_ids(frame)
        for col, n in cleared.items():
            log.warning("cleared %d ambiguous %s value(s) shared by several players", n, col)

        fill = {
            f"n_{col}": int(frame.select(pl.col(col).is_not_null().sum()).item())
            for col in ID_COLUMNS
        }
        log.info("crosswalk fill: %s", fill)

        _replace_players(con, frame)

    record_freshness(
        SOURCE,
        ok=True,
        detail=f"rebuilt from roster season {roster_season}",
        n_rows=frame.height,
    )
    log.info("players rebuilt: %d rows", frame.height)

    return IngestResult(
        source=SOURCE,
        dataset=DATASET,
        n_rows=frame.height,
        detail=f"roster season {roster_season}",
        extra={
            "dropped_null_gsis": n_null_gsis,
            "dropped_duplicate_gsis": n_duplicates,
            "cleared_ambiguous_ids": cleared,
            "roster_season": roster_season,
            **fill,
        },
    )


def id_map(from_id: str, to_id: str) -> dict[str, str]:
    """Map one ID space onto another, skipping players missing either side.

    ``snap_counts`` and ``pfr_advstats`` key on ``pfr_player_id``, so they resolve to gsis_id with
    ``id_map("pfr_id", "gsis_id")`` (D6). :func:`build_players` guarantees each external ID belongs
    to at most one player, so the mapping is unambiguous in both directions.

    Args:
        from_id: key column, one of :data:`ID_COLUMNS`.
        to_id: value column, one of :data:`ID_COLUMNS`.

    Returns:
        ``{from_value: to_value}``. Empty when the table has not been built yet.

    Raises:
        ValueError: if either name is not an ID column, or the two are the same.
    """
    for name in (from_id, to_id):
        if name not in ID_COLUMNS:
            raise ValueError(f"{name!r} is not an ID column; expected one of {ID_COLUMNS}")
    if from_id == to_id:
        raise ValueError("from_id and to_id must differ")

    with connect() as con:
        if not view_exists(con, "players"):
            log.warning("players table does not exist yet - run build_players()")
            return {}
        rows = con.execute(
            f"SELECT {from_id}, {to_id} FROM players "
            f"WHERE {from_id} IS NOT NULL AND {to_id} IS NOT NULL"
        ).fetchall()

    mapping = {str(key): str(value) for key, value in rows}
    if len(mapping) != len(rows):
        log.warning(
            "%s -> %s: %d duplicate %s values collapsed",
            from_id,
            to_id,
            len(rows) - len(mapping),
            from_id,
        )
    return mapping


def coverage_report() -> pl.DataFrame:
    """Fill rate of every ID column, overall and among players who actually played.

    The overall rate is nearly meaningless — ``players.parquet`` carries every player back to
    1999, so a low rate mostly reflects history. The number that matters is ``pct_active``:
    the share of players with a game row in ``raw_player_stats`` for the most recent season that
    has data. A gap there breaks real joins.

    Returns:
        One row per ID column with ``id_column``, ``n_total``, ``n_with_id``, ``pct_all``,
        ``active_season``, ``n_active``, ``n_active_with_id`` and ``pct_active``. ``pct_active``
        is null when no ``raw_player_stats`` season is available.
    """
    with connect() as con:
        if not view_exists(con, "players"):
            log.warning("players table does not exist yet - run build_players()")
            return pl.DataFrame(schema={"id_column": pl.Utf8})

        refresh_views(con)
        active_season: int | None = None
        if view_exists(con, "raw_player_stats"):
            row = _read(con, "SELECT max(season) AS season FROM raw_player_stats").row(0)
            active_season = int(row[0]) if row[0] is not None else None

        columns = ", ".join(ID_COLUMNS)
        if active_season is None:
            log.warning("no raw_player_stats season available - pct_active will be null")
            frame = _read(con, f"SELECT {columns}, FALSE AS is_active FROM players")
        else:
            selected = ", ".join(f"p.{c}" for c in ID_COLUMNS)
            frame = _read(
                con,
                f"""
                SELECT {selected}, (a.player_id IS NOT NULL) AS is_active
                FROM players p
                LEFT JOIN (
                    SELECT DISTINCT player_id FROM raw_player_stats WHERE season = ?
                ) a ON p.gsis_id = a.player_id
                """,
                [active_season],
            )

    n_total = frame.height
    n_active = int(frame.select(pl.col("is_active").sum()).item()) if n_total else 0

    rows = []
    for col in ID_COLUMNS:
        n_with = int(frame.select(pl.col(col).is_not_null().sum()).item()) if n_total else 0
        n_active_with = (
            int(frame.filter(pl.col("is_active")).select(pl.col(col).is_not_null().sum()).item())
            if n_active
            else 0
        )
        rows.append(
            {
                "id_column": col,
                "n_total": n_total,
                "n_with_id": n_with,
                "pct_all": round(100.0 * n_with / n_total, 2) if n_total else None,
                "active_season": active_season,
                "n_active": n_active,
                "n_active_with_id": n_active_with,
                "pct_active": round(100.0 * n_active_with / n_active, 2) if n_active else None,
            }
        )

    return pl.DataFrame(
        rows,
        schema={
            "id_column": pl.Utf8,
            "n_total": pl.Int64,
            "n_with_id": pl.Int64,
            "pct_all": pl.Float64,
            "active_season": pl.Int64,
            "n_active": pl.Int64,
            "n_active_with_id": pl.Int64,
            "pct_active": pl.Float64,
        },
    )


if __name__ == "__main__":  # pragma: no cover - operator convenience
    result = build_players()
    print(result)
    print(result.extra)
    with pl.Config(tbl_cols=-1, tbl_rows=-1):
        print(coverage_report())
