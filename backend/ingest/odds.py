"""Implied team totals and the game environment (§5.3).

Two sources, in priority order (docs/DECISIONS.md D4):

1. **The Odds API** — ``https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds``,
   markets ``spreads,totals`` only (player props are a paid feature on that service). The free
   tier is **500 requests per month**, so :func:`ingest_odds` issues **at most one request per
   call**, is never invoked from a request handler, and is gated by the persisted counter in
   ``api_budget`` (``backend/ingest/budget.py``): ``check()`` before the request, ``consume()``
   after a successful one. The response header ``x-requests-remaining`` is authoritative — when
   it disagrees with our counter we keep the *smaller* remaining number and write it back.

2. **nflverse ``schedules``** — ``spread_line`` / ``total_line`` / ``roof`` / ``surface`` /
   ``temp`` / ``wind``. Free, unlimited, and the **only** source the backtest (§6) reads, so a
   historical replay costs zero API requests.

When ``ODDS_API_KEY`` is absent, the monthly budget is spent, or the request fails, we fall back
to schedules and mark the ``the-odds-api`` freshness badge yellow. Nothing here raises (D8).
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Collection, Iterable, Iterator
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import httpx
import nflreadpy as nfl
import polars as pl

from backend.config import get_settings
from backend.db.connection import connect
from backend.db.views import raw_path, view_exists
from backend.ingest.base import (
    IngestResult,
    network_retry,
    record_freshness,
    resilient,
    utcnow,
    write_parquet_atomic,
)
from backend.ingest.budget import BudgetExceeded, check, consume, get_state
from backend.logging_setup import get_logger

log = get_logger(__name__)

ODDS_API = "the-odds-api"
ODDS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
ODDS_REGIONS = "us"
ODDS_MARKETS = "spreads,totals"
ODDS_FORMAT = "american"
REQUEST_TIMEOUT = 20.0

#: Roof values that make wind meaningless. §5.5 forces wind to 0 for these.
INDOOR_ROOFS = frozenset({"dome", "closed"})

#: Columns pulled from nflverse ``schedules`` (docs/nflverse_schema.json).
SCHEDULE_COLUMNS = (
    "game_id",
    "season",
    "week",
    "gameday",
    "gametime",
    "away_team",
    "home_team",
    "spread_line",
    "total_line",
    "roof",
    "surface",
    "temp",
    "wind",
)

#: Schema of ``data/raw/odds_{season}_{week}.parquet``. Declared so an empty pull still writes a
#: readable file and the ``raw_odds`` view keeps stable types across weeks.
ODDS_SCHEMA: dict[str, pl.DataType] = {
    "season": pl.Int64,
    "week": pl.Int64,
    "game_id": pl.String,
    "event_id": pl.String,
    "commence_time": pl.Datetime(time_unit="us", time_zone="UTC"),
    "home_team": pl.String,
    "away_team": pl.String,
    "home_team_book": pl.String,
    "away_team_book": pl.String,
    "spread_line": pl.Float64,
    "total_line": pl.Float64,
    "home_spread_price": pl.Float64,
    "away_spread_price": pl.Float64,
    "over_price": pl.Float64,
    "under_price": pl.Float64,
    "n_books": pl.Int64,
    "books": pl.String,
    "fetched_at": pl.Datetime(time_unit="us", time_zone="UTC"),
}

#: Schema returned by :func:`implied_totals`, so callers can rely on it even for an empty week.
IMPLIED_SCHEMA: dict[str, pl.DataType] = {
    "season": pl.Int64,
    "week": pl.Int64,
    "game_id": pl.String,
    "team": pl.String,
    "opponent": pl.String,
    "is_home": pl.Boolean,
    "implied_total": pl.Float64,
    "spread": pl.Float64,
    "total_line": pl.Float64,
    "kickoff": pl.Datetime(time_unit="us"),
    "roof": pl.String,
    "surface": pl.String,
    "temp": pl.Float64,
    "wind": pl.Float64,
    "odds_source": pl.String,
}


# ---------------------------------------------------------------------------
# raw nflverse access (view -> Parquet cache -> network, in that order)
# ---------------------------------------------------------------------------


@network_retry
def _download_schedules() -> pl.DataFrame:
    """Download nflverse ``schedules`` (all seasons). Retried on transport errors."""
    return nfl.load_schedules()


@network_retry
def _download_teams() -> pl.DataFrame:
    """Download the nflverse ``teams`` file. Retried on transport errors."""
    return nfl.load_teams()


@contextmanager
def _read_connection() -> Iterator[duckdb.DuckDBPyConnection]:
    """Open a connection for reading, tolerating a read-write connection already open.

    DuckDB refuses a read-only connection when the same process already holds a read-write one
    ("Can't open a connection to same database file with a different configuration"), which is
    exactly the situation inside the FastAPI app, whose ``get_connection()`` singleton is
    read-write. Falling back to the default configuration reads the same data instead of
    reporting the database as unreadable and silently returning nothing.
    """
    with ExitStack() as stack:
        try:
            con = stack.enter_context(connect(read_only=True))
        except duckdb.Error as exc:
            log.debug("read-only connection refused (%s); reading with a read-write one", exc)
            con = stack.enter_context(connect())
        yield con


def _load_raw(view: str, filename: str, loader: Callable[[], pl.DataFrame]) -> pl.DataFrame:
    """Read a raw nflverse dataset, preferring local data over the network.

    Order: the DuckDB view created by ``backend/db/views.py``, then the Parquet file in
    ``data/raw/``, then a fresh download. The first two cost nothing and are what a normal
    refresh uses; the download keeps this module usable before the nflverse ingest has run.
    """
    try:
        with _read_connection() as con:
            if view_exists(con, view):
                return con.execute(f"SELECT * FROM {view}").pl()  # noqa: S608 - fixed view names
    except duckdb.Error as exc:
        log.debug("view %s unavailable (%s); falling back to the Parquet cache", view, exc)

    path = raw_path(filename)
    if path.exists():
        return pl.read_parquet(path)

    log.info("no local %s yet; loading %s from nflverse", filename, view)
    return loader()


def _load_schedules() -> pl.DataFrame:
    """All nflverse schedule rows, from whichever local or remote source is available."""
    return _load_raw("raw_schedules", "schedules.parquet", _download_schedules)


def _load_teams() -> pl.DataFrame:
    """The nflverse team master (abbreviation, name, nickname), including historical clubs."""
    return _load_raw("raw_teams", "teams.parquet", _download_teams)


def _kickoff_expr() -> pl.Expr:
    """``gameday`` + ``gametime`` as a naive timestamp.

    nflverse publishes kickoff as a local (US Eastern) wall clock in two string columns. It is
    stored naive, exactly as published — the Odds API's UTC ``commence_time`` is kept separately
    in the raw odds file rather than mixed into this column.
    """
    return (
        pl.concat_str(
            [pl.col("gameday").cast(pl.String), pl.col("gametime").cast(pl.String)],
            separator=" ",
        )
        .str.to_datetime(format="%Y-%m-%d %H:%M", strict=False)
        .alias("kickoff")
    )


def _week_schedule(season: int, week: int) -> pl.DataFrame:
    """Schedule rows for one (season, week), typed for the ``game_environment`` table.

    Returns an empty frame — never raises — when nflverse has nothing for that week yet.
    """
    sched = _load_schedules()
    missing = [c for c in SCHEDULE_COLUMNS if c not in sched.columns]
    if missing:
        raise ValueError(f"nflverse schedules is missing expected columns: {missing}")

    return (
        sched.select(SCHEDULE_COLUMNS)
        .filter((pl.col("season") == season) & (pl.col("week") == week))
        .with_columns(
            _kickoff_expr(),
            pl.col("spread_line").cast(pl.Float64),
            pl.col("total_line").cast(pl.Float64),
            pl.col("temp").cast(pl.Float64),
            pl.col("wind").cast(pl.Float64),
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
        )
        .drop("gameday", "gametime")
        .sort("kickoff", "game_id")
    )


# ---------------------------------------------------------------------------
# team-name mapping
# ---------------------------------------------------------------------------


def _norm(value: str | None) -> str:
    """Normalise a team label for lookup: lowercase, no periods, single spaces."""
    if not value:
        return ""
    return " ".join(str(value).replace(".", "").lower().split())


def team_abbreviation_map(valid_abbrs: Collection[str] | None = None) -> dict[str, str]:
    """Map bookmaker team labels to nflverse abbreviations, built from ``raw_teams``.

    Keys are the normalised ``team_abbr``, ``team_name`` ("Kansas City Chiefs") and ``team_nick``
    ("Chiefs"). ``raw_teams`` also carries relocated franchises whose names collide with current
    clubs (LA/LAR/STL all map to "Rams"; LV/OAK to "Raiders"; LAC/SD to "Chargers"), so pass
    ``valid_abbrs`` — normally the abbreviations in this week's schedule — to resolve a collision
    to the abbreviation nflverse actually uses today. Without it, the alphabetically first
    abbreviation wins, which is deterministic but not necessarily current.
    """
    teams = _load_teams()
    mapping: dict[str, str] = {}
    allowed = {a.upper() for a in valid_abbrs} if valid_abbrs else None

    for row in teams.sort("team_abbr").iter_rows(named=True):
        abbr = (row.get("team_abbr") or "").strip().upper()
        if not abbr or (allowed is not None and abbr not in allowed):
            continue
        for label in (abbr, row.get("team_name"), row.get("team_nick")):
            key = _norm(label)
            if key:
                mapping.setdefault(key, abbr)

    return mapping


# ---------------------------------------------------------------------------
# The Odds API
# ---------------------------------------------------------------------------


class OddsRequestError(RuntimeError):
    """The server answered and the answer is unusable. Deliberately **not** retried.

    Not an ``OSError``/``TimeoutError``, so :data:`~backend.ingest.base.network_retry` leaves it
    alone — which is the point: the request was already served and counted by the API.
    """


@network_retry
def _fetch_odds(api_key: str) -> tuple[list[dict[str, Any]], str | None]:
    """Make the single Odds API request and return ``(events, x-requests-remaining)``.

    **At most one request reaches the server per call**, which is what D4 promises and what the
    ``api_budget`` counter assumes. Only a failure to *establish* the connection is retried
    (``httpx.ConnectError``/``ConnectTimeout``, re-raised as ``ConnectionError`` because
    ``network_retry`` retries ``OSError``/``TimeoutError``): nothing was served, so those retries
    cost no quota. Anything the server did answer — 4xx, 5xx, a read timeout, a body that is not
    a JSON list — raises :class:`OddsRequestError` and is *not* retried, because ``network_retry``
    would otherwise issue up to four metered requests while ``consume()`` recorded one.
    The caller falls back to nflverse schedules for this refresh and tries again on the next.
    """
    params = {
        "apiKey": api_key,
        "regions": ODDS_REGIONS,
        "markets": ODDS_MARKETS,
        "oddsFormat": ODDS_FORMAT,
    }
    try:
        response = httpx.get(ODDS_URL, params=params, timeout=REQUEST_TIMEOUT)
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise ConnectionError(f"the-odds-api unreachable: {exc}") from exc
    except httpx.TimeoutException as exc:
        raise OddsRequestError(f"the-odds-api timed out mid-request: {exc}") from exc
    except httpx.TransportError as exc:
        raise OddsRequestError(f"the-odds-api transport error: {exc}") from exc

    remaining = response.headers.get("x-requests-remaining")
    used = response.headers.get("x-requests-used")
    log.info(
        "the-odds-api: HTTP %s x-requests-remaining=%s x-requests-used=%s",
        response.status_code,
        remaining,
        used,
    )

    if response.status_code >= 400:
        body = response.text[:200].replace("\n", " ")
        raise OddsRequestError(f"the-odds-api returned HTTP {response.status_code}: {body}")

    payload = response.json()
    if not isinstance(payload, list):
        raise OddsRequestError(f"unexpected the-odds-api payload of type {type(payload).__name__}")
    return payload, remaining


def _reconcile_budget(remaining_header: str | None, budget: int) -> None:
    """Reconcile our persisted counter with ``x-requests-remaining``; keep the smaller number.

    The header is authoritative for the account, and its quota window is the account's own
    billing cycle rather than the calendar month our counter keys on, so the two can legitimately
    diverge. We only ever move our counter *up* — never hand ourselves extra requests.
    """
    if remaining_header is None:
        return
    try:
        remaining = int(remaining_header)
    except (TypeError, ValueError):
        log.warning("the-odds-api: unparseable x-requests-remaining %r", remaining_header)
        return

    state = get_state(ODDS_API, budget)
    if remaining >= state.remaining:
        return

    n_calls = min(budget, max(state.n_calls, budget - remaining))
    with connect() as con:
        con.execute(
            "UPDATE api_budget SET n_calls = ? WHERE api = ? AND period = ?",
            [n_calls, ODDS_API, state.period],
        )
    log.warning(
        "the-odds-api reports %d requests remaining but our counter said %d; "
        "trusting the smaller number (n_calls %d -> %d)",
        remaining,
        state.remaining,
        state.n_calls,
        n_calls,
    )


def _median(values: Iterable[float]) -> float | None:
    """Median of the collected book values, or ``None`` when no book quoted the market."""
    vals = list(values)
    return float(statistics.median(vals)) if vals else None


def _parse_commence_time(value: str | None) -> datetime | None:
    """Parse the Odds API's ISO-8601 UTC ``commence_time`` (``2026-09-14T00:20:00Z``)."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        log.warning("the-odds-api: unparseable commence_time %r", value)
        return None


def _consensus_from_bookmakers(
    event: dict[str, Any], home_name: str, away_name: str
) -> dict[str, Any]:
    """Collapse one event's bookmakers into a median consensus spread and total.

    The spread is flipped into the nflverse convention: a book quoting the home team at -3.5
    (home favoured) becomes ``spread_line = +3.5``.
    """
    spreads: list[float] = []
    totals: list[float] = []
    home_prices: list[float] = []
    away_prices: list[float] = []
    over_prices: list[float] = []
    under_prices: list[float] = []
    books: list[str] = []

    home_key, away_key = _norm(home_name), _norm(away_name)

    for book in event.get("bookmakers") or []:
        contributed = False
        for market in book.get("markets") or []:
            outcomes = market.get("outcomes") or []
            if market.get("key") == "spreads":
                by_team = {_norm(o.get("name")): o for o in outcomes}
                home_out = by_team.get(home_key)
                away_out = by_team.get(away_key)
                if home_out and home_out.get("point") is not None:
                    spreads.append(-float(home_out["point"]))
                    contributed = True
                    if home_out.get("price") is not None:
                        home_prices.append(float(home_out["price"]))
                    if away_out and away_out.get("price") is not None:
                        away_prices.append(float(away_out["price"]))
            elif market.get("key") == "totals":
                for outcome in outcomes:
                    side = _norm(outcome.get("name"))
                    price = outcome.get("price")
                    if side == "over" and outcome.get("point") is not None:
                        totals.append(float(outcome["point"]))
                        contributed = True
                        if price is not None:
                            over_prices.append(float(price))
                    elif side == "under" and price is not None:
                        under_prices.append(float(price))
        if contributed and book.get("key"):
            books.append(str(book["key"]))

    return {
        "spread_line": _median(spreads),
        "total_line": _median(totals),
        "home_spread_price": _median(home_prices),
        "away_spread_price": _median(away_prices),
        "over_price": _median(over_prices),
        "under_price": _median(under_prices),
        "n_books": len(books),
        "books": ",".join(sorted(books)) or None,
    }


def _flip_orientation(consensus: dict[str, Any]) -> dict[str, Any]:
    """Restate a consensus quoted from the other team's point of view.

    Used when the book calls the nflverse away team the home team (neutral-site games). The
    spread negates and the two spread prices swap; the total is orientation-free.
    """
    flipped = dict(consensus)
    if flipped.get("spread_line") is not None:
        flipped["spread_line"] = -float(flipped["spread_line"])
    flipped["home_spread_price"] = consensus.get("away_spread_price")
    flipped["away_spread_price"] = consensus.get("home_spread_price")
    return flipped


def _kickoff_gap(commence: datetime | None, kickoff: datetime | None) -> float:
    """Seconds between an event's ``commence_time`` and the scheduled kickoff.

    ``commence_time`` is UTC and ``kickoff`` is the naive US-Eastern wall clock nflverse
    publishes, so the two are offset by a constant 4-5 hours. That is irrelevant here: this only
    ever ranks candidate events for the *same* matchup, where the right one is hours away and a
    wrong one (a preseason meeting, a later-season rematch) is weeks away. ``inf`` when either
    timestamp is missing, so a dated candidate always beats an undated one.
    """
    if commence is None or kickoff is None:
        return float("inf")
    return abs((commence.replace(tzinfo=None) - kickoff).total_seconds())


def _parse_odds_payload(
    payload: list[dict[str, Any]], schedule: pl.DataFrame, season: int, week: int
) -> pl.DataFrame:
    """Turn the Odds API payload into **exactly one** consensus row per scheduled game.

    Team labels are mapped to nflverse abbreviations via :func:`team_abbreviation_map`, built
    from ``raw_teams`` and narrowed to the teams playing this week. A game whose team we cannot
    map, or that has no schedule row for this week (a different week's event, or a preseason
    fixture), is **logged** and dropped — never dropped silently.

    Two things the endpoint does that a naive parse gets wrong:

    * It returns *every* upcoming NFL event, so the same matchup can appear twice (a preseason
      meeting, or an event relisted under a new id). One row per ``game_id`` is a hard
      requirement — ``game_environment.game_id`` is a primary key and the join in
      :func:`build_game_environment` would otherwise fan out — so duplicates are resolved to the
      candidate whose ``commence_time`` is closest to the scheduled kickoff, and the loser is
      logged.
    * For a neutral-site game (the international series, the Super Bowl) the book may designate
      the other side as home. Rather than dropping the game, the schedule's orientation wins and
      the consensus is flipped through :func:`_flip_orientation`.
    """
    abbrs = set(schedule["home_team"].to_list()) | set(schedule["away_team"].to_list())
    name_map = team_abbreviation_map(abbrs)
    by_pair = {(row["home_team"], row["away_team"]): row for row in schedule.iter_rows(named=True)}
    fetched_at = utcnow()
    #: game_id -> (rank key, row). Lower rank key wins.
    best: dict[str, tuple[tuple[float, int], dict[str, Any]]] = {}

    for event in payload:
        home_name = event.get("home_team")
        away_name = event.get("away_team")
        home = name_map.get(_norm(home_name))
        away = name_map.get(_norm(away_name))
        if home is None or away is None:
            unmapped = [n for n, a in ((home_name, home), (away_name, away)) if a is None]
            log.warning(
                "the-odds-api: no nflverse abbreviation for %s; dropping event %s (%s @ %s)",
                unmapped,
                event.get("id"),
                away_name,
                home_name,
            )
            continue

        game = by_pair.get((home, away))
        consensus = _consensus_from_bookmakers(event, str(home_name), str(away_name))
        if game is None:
            game = by_pair.get((away, home))
            if game is not None:
                log.warning(
                    "the-odds-api: event %s lists %s at %s but nflverse has %s at %s "
                    "(neutral site?); keeping the schedule orientation and flipping the spread",
                    event.get("id"),
                    away,
                    home,
                    away,
                    home,
                )
                home, away = away, home
                home_name, away_name = away_name, home_name
                consensus = _flip_orientation(consensus)

        if game is None:
            log.warning(
                "the-odds-api: %s @ %s has no %s week %s schedule row; dropping event %s",
                away,
                home,
                season,
                week,
                event.get("id"),
            )
            continue

        game_id = game["game_id"]
        commence = _parse_commence_time(event.get("commence_time"))
        row = {
            "season": season,
            "week": week,
            "game_id": game_id,
            "event_id": event.get("id"),
            "commence_time": commence,
            "home_team": home,
            "away_team": away,
            "home_team_book": home_name,
            "away_team_book": away_name,
            "fetched_at": fetched_at,
            **consensus,
        }
        rank = (_kickoff_gap(commence, game.get("kickoff")), -int(consensus["n_books"]))

        previous = best.get(game_id)
        if previous is None:
            best[game_id] = (rank, row)
            continue
        keep, drop = ((rank, row), previous) if rank < previous[0] else (previous, (rank, row))
        log.warning(
            "the-odds-api: %s is priced by two events; keeping %s (kicking off nearest the "
            "scheduled time) and dropping %s",
            game_id,
            keep[1]["event_id"],
            drop[1]["event_id"],
        )
        best[game_id] = keep

    return pl.DataFrame([row for _, row in best.values()], schema=ODDS_SCHEMA)


def odds_file(season: int, week: int) -> Path:
    """Path of the raw odds Parquet for one week (``data/raw/odds_{season}_{week}.parquet``)."""
    return raw_path(f"odds_{season}_{week}.parquet")


def _read_odds_file(season: int, week: int) -> pl.DataFrame | None:
    """Read this week's raw odds file, or ``None`` when it is absent or unreadable."""
    path = odds_file(season, week)
    if not path.exists():
        return None
    try:
        return pl.read_parquet(path)
    except (OSError, pl.exceptions.PolarsError) as exc:
        log.warning("could not read %s (%s); treating it as absent", path.name, exc)
        return None


def _have_full_week(season: int, week: int, schedule: pl.DataFrame) -> bool:
    """True when the cached odds file already prices every scheduled game this week."""
    cached = _read_odds_file(season, week)
    if cached is None or cached.is_empty():
        return False
    priced = set(
        cached.filter(pl.col("spread_line").is_not_null() & pl.col("total_line").is_not_null())[
            "game_id"
        ].to_list()
    )
    return set(schedule["game_id"].to_list()).issubset(priced)


def _mark_odds_fallback(detail: str) -> None:
    """Record a deliberate fall-back to nflverse schedules as a **yellow** badge (D4).

    ``record_freshness(ok=False)`` would go red once the last success ages out, but choosing not
    to spend a metered request is not an outage: D4 says the badge is yellow while we are serving
    schedule lines instead. We record the attempt, then pin the status.
    """
    record_freshness(ODDS_API, ok=False, detail=detail)
    try:
        with connect() as con:
            con.execute(
                "UPDATE source_freshness SET status = 'yellow' WHERE source = ?", [ODDS_API]
            )
    except duckdb.Error:
        log.exception("could not mark %s freshness yellow", ODDS_API)


@resilient(ODDS_API, "odds")
def ingest_odds(season: int, week: int, force: bool = False) -> IngestResult:
    """Pull spreads and totals for one week from The Odds API. At most one request.

    Skips (and leaves :func:`build_game_environment` to fall back to nflverse schedules) when:

    * nflverse has no schedule rows for that week yet;
    * ``ODDS_API_KEY`` is not configured;
    * the persisted monthly budget (500 by default, D8) would be exceeded;
    * ``force`` is False and the cached file already prices every game this week.

    On success it writes ``data/raw/odds_{season}_{week}.parquet`` (one consensus row per game,
    the median across books), consumes one unit of budget, reconciles the counter against
    ``x-requests-remaining``, and records green freshness. It never raises: a failure logs, marks
    the badge, and returns a failed :class:`IngestResult`.
    """
    settings = get_settings()
    schedule = _week_schedule(season, week)
    if schedule.is_empty():
        detail = f"no nflverse schedule rows for {season} week {week}"
        log.info("odds: %s", detail)
        return IngestResult(source=ODDS_API, dataset="odds", skipped=True, detail=detail)

    if not settings.has_odds:
        detail = "ODDS_API_KEY not set; using nflverse schedules"
        log.info("odds: %s", detail)
        _mark_odds_fallback(detail)
        return IngestResult(source=ODDS_API, dataset="odds", skipped=True, detail=detail)

    budget = settings.odds_monthly_budget
    try:
        state = check(ODDS_API, budget)
    except BudgetExceeded as exc:
        detail = f"{exc}; using nflverse schedules"
        log.warning("odds: %s", detail)
        _mark_odds_fallback(detail)
        return IngestResult(source=ODDS_API, dataset="odds", skipped=True, detail=detail)

    if not force and _have_full_week(season, week, schedule):
        detail = f"{odds_file(season, week).name} already prices all {schedule.height} games"
        log.info("odds: %s", detail)
        return IngestResult(source=ODDS_API, dataset="odds", skipped=True, detail=detail)

    log.info(
        "odds: requesting %s week %s (%d/%d monthly requests used)",
        season,
        week,
        state.n_calls,
        state.budget,
    )
    payload, remaining = _fetch_odds(str(settings.odds_api_key))
    consume(ODDS_API, budget)
    _reconcile_budget(remaining, budget)

    df = _parse_odds_payload(payload, schedule, season, week)
    if df.is_empty():
        detail = "the-odds-api returned no usable games; falling back to nflverse schedules"
        log.warning(detail)
        _mark_odds_fallback(detail)
        return IngestResult(source=ODDS_API, dataset="odds", skipped=True, detail=detail)

    path = write_parquet_atomic(df, odds_file(season, week))
    record_freshness(ODDS_API, ok=True, detail=f"{season} week {week}", n_rows=df.height)
    log.info("odds: wrote %d of %d games to %s", df.height, schedule.height, path.name)

    return IngestResult(
        source=ODDS_API,
        dataset="odds",
        n_rows=df.height,
        path=path,
        detail=f"{df.height}/{schedule.height} games priced",
        extra={"events": len(payload), "requests_remaining": remaining},
    )


# ---------------------------------------------------------------------------
# game environment
# ---------------------------------------------------------------------------


def _replace_week(out: pl.DataFrame, season: int, week: int) -> None:
    """Swap one week's ``game_environment`` rows for ``out``, all or nothing.

    The DELETE and the INSERT share one transaction. Without it a rejected INSERT (a duplicate
    ``game_id``, a value the table will not take) leaves the week *deleted* and nothing written:
    the app is then left with no game environment at all, which is worse than the stale rows it
    was replacing and is precisely what D8 forbids.
    """
    with connect() as con:
        con.register("ge_df", out)
        try:
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    "DELETE FROM game_environment WHERE season = ? AND week = ?", [season, week]
                )
                con.execute(
                    "INSERT INTO game_environment "
                    "(game_id, season, week, home_team, away_team, kickoff, spread_line, "
                    " total_line, home_implied_total, away_implied_total, roof, surface, temp, "
                    " wind, odds_source, observed_at) "
                    "SELECT game_id, season, week, home_team, away_team, kickoff, spread_line, "
                    "       total_line, home_implied_total, away_implied_total, roof, surface, "
                    "       temp, wind, odds_source, observed_at FROM ge_df"
                )
            except Exception:
                con.execute("ROLLBACK")
                raise
            con.execute("COMMIT")
        finally:
            con.unregister("ge_df")


@resilient("nflverse", "game_environment")
def build_game_environment(season: int, week: int, prefer_odds: bool = True) -> IngestResult:
    """Populate ``game_environment`` for one week: lines, implied totals, and weather.

    Implied totals, with nflverse's convention that ``spread_line`` is **positive when the home
    team is favoured**::

        home_implied_total = total_line / 2 + spread_line / 2
        away_implied_total = total_line / 2 - spread_line / 2

    Verified on 2025 week 10 ``2025_10_LV_DEN`` (DEN home, ``spread_line=9.5``,
    ``total_line=42.5``): home 42.5/2 + 9.5/2 = **26.0**, away 42.5/2 - 9.5/2 = **16.5**. They sum
    to the total (42.5) and differ by the spread (9.5), with the home favourite on the higher
    number — which is the direction the data confirms: across 2025, ``corr(spread_line, result)``
    is +0.51 and home teams favoured by the line won by 6.9 points on average.

    Sources per game: the cached Odds API file when ``prefer_odds`` and it prices that game
    (``odds_source='the-odds-api'``), otherwise the nflverse schedule line
    (``odds_source='nflverse-schedules'``). Games with no line at all are still written, with null
    implied totals, so the weather and kickoff are available downstream.

    Weather comes from ``schedules``. Per §5.5 ``wind`` is forced to 0 for a ``dome`` or
    ``closed`` roof. Future games have null ``temp``/``wind`` in nflverse and are left null here —
    the ESPN weather ingest fills them in later; nothing is invented.
    """
    schedule = _week_schedule(season, week)
    if schedule.is_empty():
        detail = f"no nflverse schedule rows for {season} week {week}"
        log.info("game_environment: %s", detail)
        return IngestResult(
            source="nflverse", dataset="game_environment", skipped=True, detail=detail
        )

    odds = _read_odds_file(season, week) if prefer_odds else None
    if odds is not None and not odds.is_empty():
        priced = odds.select(
            "game_id",
            pl.col("spread_line").cast(pl.Float64).alias("odds_spread"),
            pl.col("total_line").cast(pl.Float64).alias("odds_total"),
        )
        # One row per game or the left join below fans out, duplicating schedule rows and
        # tripping the game_id primary key. _parse_odds_payload guarantees it; a file written
        # by an older build (or edited by hand) may not, and being sure costs one call.
        deduped = priced.unique(subset="game_id", keep="first", maintain_order=True)
        if deduped.height != priced.height:
            log.warning(
                "%s holds %d rows for %d games; keeping the first row per game",
                odds_file(season, week).name,
                priced.height,
                deduped.height,
            )
        schedule = schedule.join(deduped, on="game_id", how="left")
    else:
        schedule = schedule.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("odds_spread"),
            pl.lit(None, dtype=pl.Float64).alias("odds_total"),
        )

    use_odds = pl.col("odds_spread").is_not_null() & pl.col("odds_total").is_not_null()
    resolved = schedule.with_columns(
        pl.when(use_odds)
        .then(pl.col("odds_spread"))
        .otherwise(pl.col("spread_line"))
        .alias("spread_line"),
        pl.when(use_odds)
        .then(pl.col("odds_total"))
        .otherwise(pl.col("total_line"))
        .alias("total_line"),
        pl.when(use_odds)
        .then(pl.lit(ODDS_API))
        .otherwise(pl.lit("nflverse-schedules"))
        .alias("odds_source"),
    ).with_columns(
        (pl.col("total_line") / 2 + pl.col("spread_line") / 2).alias("home_implied_total"),
        (pl.col("total_line") / 2 - pl.col("spread_line") / 2).alias("away_implied_total"),
        pl.when(pl.col("roof").str.to_lowercase().is_in(list(INDOOR_ROOFS)))
        .then(pl.lit(0.0))
        .otherwise(pl.col("wind"))
        .alias("wind"),
        pl.lit(utcnow()).alias("observed_at"),
    )

    out = resolved.select(
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "kickoff",
        "spread_line",
        "total_line",
        "home_implied_total",
        "away_implied_total",
        "roof",
        "surface",
        "temp",
        "wind",
        "odds_source",
        "observed_at",
    )

    # Refuse to replace a week with nothing. The write below deletes before it inserts, so an
    # empty frame here -- a transient schedules read failure, say -- would silently wipe a week
    # of good rows and leave the UI with no game environment at all.
    if out.is_empty():
        log.warning(
            "game_environment: %s week %s produced no rows; leaving the existing week untouched",
            season,
            week,
        )
        return IngestResult(
            source="nflverse",
            dataset="game_environment",
            n_rows=0,
            skipped=True,
            detail=f"no schedule rows for {season} week {week}; existing rows preserved",
        )

    try:
        _replace_week(out, season, week)
    except duckdb.Error as exc:
        # A rejected write is a database problem, not an nflverse outage. Handling it here
        # instead of letting @resilient see it keeps the shared 'nflverse' freshness badge --
        # which the nflverse ingest owns -- from being repainted with a DuckDB error string.
        log.exception("game_environment: could not write %s week %s", season, week)
        return IngestResult(
            source="nflverse",
            dataset="game_environment",
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
        )

    n_api = int(out.filter(pl.col("odds_source") == ODDS_API).height)
    n_lines = int(out.filter(pl.col("total_line").is_not_null()).height)
    log.info(
        "game_environment: %s week %s -> %d games (%d from the-odds-api, %d with a line)",
        season,
        week,
        out.height,
        n_api,
        n_lines,
    )

    return IngestResult(
        source="nflverse",
        dataset="game_environment",
        n_rows=out.height,
        detail=f"{n_api} the-odds-api / {out.height - n_api} nflverse-schedules",
        extra={"games_with_line": n_lines},
    )


def implied_totals(season: int, week: int) -> pl.DataFrame:
    """One row per team for a week: implied total, its own spread, opponent, and weather.

    ``spread`` is from the team's own point of view — **negative means favoured** — so the home
    row carries ``-spread_line`` and the away row ``+spread_line``. Reads ``game_environment``;
    run :func:`build_game_environment` first. An unpopulated week returns an empty frame with the
    documented schema rather than raising, so the ranking and projection layers can degrade.
    """
    try:
        with _read_connection() as con:
            games = con.execute(
                "SELECT game_id, season, week, home_team, away_team, kickoff, spread_line, "
                "       total_line, home_implied_total, away_implied_total, roof, surface, temp, "
                "       wind, odds_source "
                "FROM game_environment WHERE season = ? AND week = ? ORDER BY kickoff, game_id",
                [season, week],
            ).pl()
    except duckdb.Error as exc:
        log.warning("implied_totals: cannot read game_environment (%s)", exc)
        return pl.DataFrame(schema=IMPLIED_SCHEMA)

    if games.is_empty():
        log.warning(
            "implied_totals: game_environment has no rows for %s week %s "
            "(run build_game_environment first)",
            season,
            week,
        )
        return pl.DataFrame(schema=IMPLIED_SCHEMA)

    shared = [
        "season",
        "week",
        "game_id",
        "total_line",
        "kickoff",
        "roof",
        "surface",
        "temp",
        "wind",
        "odds_source",
    ]

    home = games.select(
        *shared,
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
        pl.lit(True).alias("is_home"),
        pl.col("home_implied_total").alias("implied_total"),
        (-pl.col("spread_line")).alias("spread"),
    )
    away = games.select(
        *shared,
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent"),
        pl.lit(False).alias("is_home"),
        pl.col("away_implied_total").alias("implied_total"),
        pl.col("spread_line").alias("spread"),
    )

    return (
        pl.concat([home, away])
        .select(list(IMPLIED_SCHEMA))
        .cast(IMPLIED_SCHEMA)  # type: ignore[arg-type]
        .sort("kickoff", "team")
    )
