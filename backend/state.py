"""What week is it? One answer, resolved once, shared by every module.

The current week is never hardcoded (docs/DECISIONS.md D3). It is resolved in this order:

1. **Sleeper** ``/v1/state/nfl`` -- authoritative, and the only source that knows the week has
   flipped before any game of it has been played.
2. **nflverse schedules** (``raw_schedules``) -- when Sleeper is unreachable, the current week is
   the lowest week of the latest season that still has a null ``home_score``, ``gameday``
   breaking ties. If every game already has a score, the season is over and we report its final
   week.
3. **Persisted / calendar fallback** -- the last value written to ``refresh_state``, or, failing
   even that, the season implied by today's date at week 1.

Whatever is resolved is written back to ``refresh_state`` (``season``, ``week``, ``season_type``,
``state_source``, ``state_updated_at``) so the read-only API can answer without a network call,
and cached in-process so a refresh run resolves it once.

``games_played_this_season`` comes from ``raw_schedules`` and is the flag callers need for the
"nothing has happened yet" case: as of 2026-09-05 the state is 2026 week 1 with **zero** games
played, so every projection has to be built from prior seasons.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from backend.db.connection import connect
from backend.ingest.sleeper import fetch_state
from backend.logging_setup import get_logger

log = get_logger(__name__)

SEASON_TYPES: tuple[str, ...] = ("pre", "regular", "post", "off")

#: nflverse ``game_type`` -> our ``season_type``.
_GAME_TYPE_TO_SEASON_TYPE: dict[str, str] = {
    "PRE": "pre",
    "REG": "regular",
    "WC": "post",
    "DIV": "post",
    "CON": "post",
    "SB": "post",
}

#: Only consulted when ``raw_schedules`` cannot be read at all. The league went to 17 regular
#: season games (week 18) in 2021; before that the final regular-season week was 17.
_EXPANSION_SEASON = 2021

#: A new league year begins in March, so Jan/Feb still belong to the previous season.
_NEW_LEAGUE_YEAR_MONTH = 3

_REFRESH_KEYS = ("season", "week", "season_type", "state_source", "state_updated_at")

_cache: SeasonState | None = None
_lock = threading.Lock()


@dataclass(frozen=True)
class SeasonState:
    """The league week the whole app is pinned to for this run."""

    season: int
    week: int
    season_type: str
    source: str
    season_start_date: date | None
    games_played_this_season: int

    @property
    def no_games_played(self) -> bool:
        """True when the season has not started, so every input must come from prior seasons."""
        return self.games_played_this_season == 0

    def __str__(self) -> str:
        return (
            f"{self.season} week {self.week} ({self.season_type}, via {self.source}; "
            f"{self.games_played_this_season} games played)"
        )


# ---------------------------------------------------------------------------
# small, failure-tolerant database helpers
# ---------------------------------------------------------------------------


def _query(sql: str, params: list[Any] | None = None) -> list[tuple] | None:
    """Run a read query, returning ``None`` if the database or view is unavailable.

    Another process may hold the DuckDB write lock, ``raw_schedules`` may not exist yet (nothing
    ingested), or the file itself may be missing. None of that is fatal here -- it just means we
    fall through to the next source.
    """
    for read_only in (True, False):
        try:
            with connect(read_only=read_only) as con:
                return con.execute(sql, params or []).fetchall()
        except Exception as exc:  # noqa: BLE001 - degrade to the next source
            log.debug("state query failed (read_only=%s): %s", read_only, exc)
    return None


def _persist(state: SeasonState) -> None:
    """Write the resolved state into ``refresh_state``. Best effort; never raises."""
    values = {
        "season": str(state.season),
        "week": str(state.week),
        "season_type": state.season_type,
        "state_source": state.source,
        "state_updated_at": datetime.now().astimezone().isoformat(),
    }
    try:
        with connect() as con:
            for key, value in values.items():
                con.execute(
                    "INSERT INTO refresh_state (key, value, updated_at) VALUES (?, ?, now()) "
                    "ON CONFLICT (key) DO UPDATE SET "
                    "value = excluded.value, updated_at = excluded.updated_at",
                    [key, value],
                )
    except Exception as exc:  # noqa: BLE001 - bookkeeping must not break callers
        log.warning("could not persist season state to refresh_state: %s", exc)


def _persisted() -> tuple[int, int, str] | None:
    """Read ``(season, week, season_type)`` back out of ``refresh_state``, if it is there."""
    rows = _query(
        "SELECT key, value FROM refresh_state WHERE key IN (?, ?, ?)",
        ["season", "week", "season_type"],
    )
    if not rows:
        return None
    stored = dict(rows)
    try:
        return (
            int(stored["season"]),
            int(stored["week"]),
            _clean_season_type(stored.get("season_type")),
        )
    except (KeyError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def _clean_season_type(value: object) -> str:
    """Normalise a season type to one of :data:`SEASON_TYPES`, defaulting to ``'regular'``."""
    text = str(value or "").strip().lower()
    if text in SEASON_TYPES:
        return text
    if text in {"post_season", "postseason", "playoffs"}:
        return "post"
    if text in {"pre_season", "preseason"}:
        return "pre"
    if text in {"off_season", "offseason"}:
        return "off"
    return "regular"


def _parse_date(value: object) -> date | None:
    """Parse a ``YYYY-MM-DD`` string into a date, tolerating junk and nulls."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _from_sleeper(refresh: bool) -> tuple[int, int, str, date | None] | None:
    """``(season, week, season_type, season_start_date)`` from Sleeper, or ``None``."""
    payload = fetch_state(force=refresh)
    if not isinstance(payload, dict):
        return None
    try:
        season = int(str(payload["season"]).strip())
        week = int(payload["week"])
    except (KeyError, TypeError, ValueError):
        log.warning("Sleeper state payload lacks a usable season/week: %r", payload)
        return None
    if week < 1:
        # Sleeper reports week 0 during the dead period between the Super Bowl and the draft.
        week = 1
    season_type = _clean_season_type(payload.get("season_type"))
    start = _parse_date(payload.get("season_start_date"))
    return season, week, season_type, start


def _from_schedules() -> tuple[int, int, str, date | None] | None:
    """``(season, week, season_type, season_start_date)`` derived from ``raw_schedules``.

    The current week is the lowest week of the latest season with an unplayed game (null
    ``home_score``), ``gameday`` breaking ties. When every game of that season has a score the
    season is finished, so we report its highest week instead.
    """
    rows = _query(
        """
        SELECT season, week, game_type, gameday
        FROM raw_schedules
        WHERE season = (SELECT max(season) FROM raw_schedules)
          AND home_score IS NULL
        ORDER BY week ASC, gameday ASC NULLS LAST
        LIMIT 1
        """
    )
    if rows is None:
        return None
    if not rows:
        rows = _query(
            """
            SELECT season, week, game_type, gameday
            FROM raw_schedules
            WHERE season = (SELECT max(season) FROM raw_schedules)
            ORDER BY week DESC, gameday DESC NULLS LAST
            LIMIT 1
            """
        )
    if not rows:
        return None

    season, week, game_type, _gameday = rows[0]
    season_type = _GAME_TYPE_TO_SEASON_TYPE.get(str(game_type or "").upper(), "regular")
    return int(season), int(week), season_type, _season_start_date(int(season))


def _season_start_date(season: int) -> date | None:
    """First scheduled regular-season kickoff date for ``season``, from ``raw_schedules``."""
    rows = _query(
        "SELECT min(gameday) FROM raw_schedules WHERE season = ? AND game_type = 'REG'",
        [season],
    )
    if not rows or rows[0][0] is None:
        return None
    return _parse_date(rows[0][0])


def _from_calendar(today: date | None = None) -> tuple[int, int, str, date | None]:
    """Last resort: the season implied by today's date, at week 1.

    Not a hardcoded week so much as an admission that we know nothing -- it exists only so the app
    still renders when Sleeper is down *and* no schedule has ever been ingested.
    """
    today = today or date.today()
    season = today.year if today.month >= _NEW_LEAGUE_YEAR_MONTH else today.year - 1
    return season, 1, "regular", None


def _games_played(season: int) -> int:
    """How many games of ``season`` have a final result in ``raw_schedules``."""
    rows = _query(
        "SELECT count(*) FROM raw_schedules WHERE season = ? AND result IS NOT NULL", [season]
    )
    if not rows or rows[0][0] is None:
        return 0
    return int(rows[0][0])


def current_state(refresh: bool = False) -> SeasonState:
    """Resolve the current season, week and season type.

    Cached in-process after the first call. The result is also written to ``refresh_state`` so the
    API layer can read it without touching the network.

    Args:
        refresh: ignore both the in-process cache and Sleeper's one-hour state cache.

    Returns:
        A :class:`SeasonState`. This never raises: with every source unavailable it still returns
        a calendar-derived state with ``source='fallback'``.
    """
    global _cache
    with _lock:
        if _cache is not None and not refresh:
            return _cache

        resolved = _from_sleeper(refresh)
        source = "sleeper"
        if resolved is None:
            log.warning("Sleeper state unavailable; deriving the week from raw_schedules")
            resolved = _from_schedules()
            source = "schedules"
        if resolved is None:
            persisted = _persisted()
            if persisted is not None:
                season, week, season_type = persisted
                resolved = (season, week, season_type, _season_start_date(season))
                log.warning("using the last persisted state: %s week %s", season, week)
            else:
                resolved = _from_calendar()
                log.warning("no state source available; falling back to the calendar")
            source = "fallback"

        season, week, season_type, season_start = resolved
        if season_start is None:
            season_start = _season_start_date(season)

        state = SeasonState(
            season=season,
            week=week,
            season_type=season_type,
            source=source,
            season_start_date=season_start,
            games_played_this_season=_games_played(season),
        )
        _persist(state)
        _cache = state

    log.info("season state: %s", state)
    return state


def clear_cache() -> None:
    """Drop the in-process cache. Mainly for tests and long-lived processes."""
    global _cache
    with _lock:
        _cache = None


def max_regular_week(season: int) -> int:
    """Highest regular-season week actually scheduled in ``season``.

    Read from ``raw_schedules`` rather than assumed: the league played 17 weeks through 2020 and
    18 from 2021, and a season still being ingested may have fewer. The constant is used only when
    the schedule cannot be read at all.
    """
    rows = _query(
        "SELECT max(week) FROM raw_schedules WHERE season = ? AND game_type = 'REG'", [season]
    )
    if rows and rows[0][0] is not None:
        return int(rows[0][0])
    fallback = 18 if season >= _EXPANSION_SEASON else 17
    log.debug("no schedule rows for %s; assuming %d regular-season weeks", season, fallback)
    return fallback


def previous_week(state: SeasonState) -> tuple[int, int]:
    """The ``(season, week)`` immediately before ``state``.

    Rolls back across the season boundary using the last regular-season week actually present in
    ``raw_schedules`` for the previous season, so 2026 week 1 becomes 2025 week 18 (or 17, for a
    pre-2021 season) without anything being assumed.
    """
    if state.week > 1:
        return state.season, state.week - 1
    previous_season = state.season - 1
    return previous_season, max_regular_week(previous_season)
