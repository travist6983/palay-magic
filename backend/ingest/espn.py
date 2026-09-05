"""ESPN hidden API ingest: near-live injuries and per-game weather/venue.

Source
------
ESPN's undocumented "hidden" API. Free, no key, no published terms, and **no published rate
limit** (docs/reference/free_nfl_data_sources.md §3, docs/DECISIONS.md D8). It can change shape or
disappear without notice, so every function here is written to degrade rather than fail: a dead
endpoint logs, records a freshness badge, and returns a failed :class:`IngestResult`.

Endpoints used
--------------
``site.api.espn.com/apis/site/v2/sports/football/nfl/teams``
    Team id / abbreviation / name. Cached to ``data/cache/espn_teams.json`` with a long TTL.
``site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{id}?enable=roster``
    One request per team, used purely to resolve ``athlete_id -> (name, position)`` locally.
    Without it we would have to hydrate ~2,000 athlete ``$ref`` links per refresh.
``sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{id}/injuries``
    A ``$ref`` collection. It is **paginated** (``pageSize`` defaults to 25, and a single team can
    carry 60+ records), so we walk ``page=1..pageCount`` at ``limit=100``. Every ``$ref`` must be
    followed to hydrate the record, and the links come back as ``http://`` — we rewrite to https.
``site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates=YYYYMMDD``
    Per-event weather, venue (with an ``indoor`` flag) and the ESPN-listed spread / over-under.

Rate limiting
-------------
Undocumented, so we self-limit: at most ``_MAX_IN_FLIGHT`` (6) concurrent requests and a
``_REQUEST_DELAY_S`` pause inside each worker before it fires, which caps us near 40 req/s.
Transport errors, 429 and 5xx are retried by :data:`network_retry`; 4xx are not.

TODO(owner of backend/db/views.py): ``ingest_weather()`` writes ``data/raw/espn_weather.parquet``
and ``ingest_injuries()`` writes ``data/raw/espn_injuries.parquet``. The exact RAW_VIEWS entries
these files need are::

    RawView("raw_espn_injuries", "espn_injuries.parquet", False, "Latest ESPN per-team injury hydrate"),
    RawView("raw_espn_weather", "espn_weather.parquet", False, "ESPN scoreboard weather + venue indoor flag"),

Verified 2026-09-05: both lines are **already present** in ``RAW_VIEWS``, so no edit is needed.
This note stays so the owner of that module can confirm the file names have not drifted.
"""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
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

SOURCE = "espn"

SITE_API = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
CORE_API = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
TEAMS_URL = f"{SITE_API}/teams"
SCOREBOARD_URL = f"{SITE_API}/scoreboard"

TEAMS_CACHE_NAME = "espn_teams.json"
TEAMS_CACHE_TTL_HOURS = 24.0 * 7.0
"""Team ids change roughly never; a week is already conservative."""

INJURIES_FILE = "espn_injuries.parquet"
WEATHER_FILE = "espn_weather.parquet"

_MAX_IN_FLIGHT = 6
"""Concurrent requests to ESPN. Politeness ceiling, not a throughput target."""

_REQUEST_DELAY_S = 0.15
"""Pause each worker takes before firing. With 6 workers this caps us near 40 req/s."""

_TIMEOUT_S = 25.0
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

# Deliberately no User-Agent override. ``site.api.espn.com`` sits behind a WAF that 403s
# unrecognised UA strings (verified 2026-09-05: "PropLab/0.1" -> 403, httpx's own default,
# curl/* and python-requests/* -> 200), so we let httpx send its default "python-httpx/<ver>",
# which identifies the client honestly and is accepted by both ESPN hosts.
_HEADERS = {"Accept": "application/json"}

_ATHLETE_ID_RE = re.compile(r"/athletes/(-?\d+)/injuries/")

# ESPN abbreviations that differ from the nflverse canon (verified against
# nflreadpy.load_rosters([2026]): the Rams are "LA" and Washington is "WAS").
_ESPN_TO_NFLVERSE_TEAM: dict[str, str] = {
    "WSH": "WAS",
    "LAR": "LA",
}

# ESPN's INJURY_STATUS_* vocabulary mapped onto the designations Sleeper writes into
# ``injury_status.injury_status`` (Questionable | Doubtful | Out | IR | PUP | Sus | NA | NULL),
# so rows from source='espn' and source='sleeper' are directly comparable.
_STATUS_TO_DESIGNATION: dict[str, str | None] = {
    "active": None,
    "healthy": None,
    "practice squad": None,
    "day-to-day": "Questionable",
    "day to day": "Questionable",
    "probable": "Questionable",  # NFL retired Probable in 2016; Questionable is the nearest peer.
    "questionable": "Questionable",
    "doubtful": "Doubtful",
    "out": "Out",
    "injured reserve": "IR",
    "injured reserve - designated for return": "IR",
    "ir": "IR",
    "physically unable to perform": "PUP",
    "pup": "PUP",
    "non football injury": "NA",
    "non football illness": "NA",
    "non-football injury": "NA",
    "nfi": "NA",
    "suspension": "Sus",
    "suspended": "Sus",
}

_INJURY_COLUMNS: tuple[str, ...] = (
    "espn_athlete_id",
    "espn_injury_id",
    "team_abbr",
    "player_name",
    "position",
    "status",
    "injury_type",
    "detail",
    "side",
    "return_date",
    "short_comment",
    "long_comment",
    "fetched_at",
)


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------


def _make_client() -> httpx.Client:
    """Build the shared HTTP client. ``httpx.Client`` is thread-safe, so one serves the pool."""
    return httpx.Client(
        timeout=_TIMEOUT_S,
        headers=_HEADERS,
        follow_redirects=True,
        limits=httpx.Limits(
            max_connections=_MAX_IN_FLIGHT,
            max_keepalive_connections=_MAX_IN_FLIGHT,
        ),
    )


@network_retry
def _get_json(
    client: httpx.Client, url: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """GET ``url`` and parse JSON, retrying transport errors, 429 and 5xx.

    ``network_retry`` only retries ``OSError``/``TimeoutError``/``ConnectionError``, so httpx
    transport failures and retryable status codes are re-raised as :class:`ConnectionError`.
    A 4xx other than 429 raises ``httpx.HTTPStatusError`` immediately — retrying a 404 is waste.
    """
    time.sleep(_REQUEST_DELAY_S)
    try:
        response = client.get(url, params=params)
    except httpx.HTTPError as exc:
        raise ConnectionError(f"ESPN transport error for {url}: {exc}") from exc

    if response.status_code in _RETRYABLE_STATUS:
        raise ConnectionError(f"ESPN returned {response.status_code} for {url}")
    response.raise_for_status()

    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise ConnectionError(f"ESPN returned non-JSON for {url}: {exc}") from exc


def _https(ref: str) -> str:
    """ESPN hands back ``http://`` ``$ref`` links; rewrite them so we never downgrade."""
    return ref.replace("http://", "https://", 1) if ref.startswith("http://") else ref


def _nflverse_team(espn_abbr: str | None) -> str | None:
    """Translate an ESPN team abbreviation to the nflverse canon (WSH -> WAS, LAR -> LA)."""
    if not espn_abbr:
        return None
    return _ESPN_TO_NFLVERSE_TEAM.get(espn_abbr.upper(), espn_abbr.upper())


# ---------------------------------------------------------------------------
# Status vocabulary
# ---------------------------------------------------------------------------


def espn_status_to_designation(status: str) -> str | None:
    """Map an ESPN injury status onto the designation vocabulary Sleeper uses.

    ESPN emits things like ``"Out"``, ``"Questionable"``, ``"Injured Reserve"``, ``"Suspension"``
    and ``"Active"``. ``injury_status.injury_status`` stores the Sleeper designations
    (``Questionable`` | ``Doubtful`` | ``Out`` | ``IR`` | ``PUP`` | ``Sus`` | ``NA`` | ``NULL``),
    so both sources land in the same alphabet and can be compared row for row.

    Args:
        status: the raw ESPN status string. Case and surrounding whitespace do not matter.

    Returns:
        The Sleeper-style designation, or ``None`` for a healthy/unknown status. ``None`` is the
        correct answer for "Active": the player carries no designation at all.
    """
    if not status:
        return None
    key = " ".join(status.strip().lower().split())
    if key in _STATUS_TO_DESIGNATION:
        return _STATUS_TO_DESIGNATION[key]
    log.debug("unmapped ESPN injury status %r -> NULL designation", status)
    return None


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------


def _teams_cache_path() -> Path:
    return get_settings().cache_dir / TEAMS_CACHE_NAME


def _read_teams_cache(max_age_hours: float | None) -> list[dict[str, Any]] | None:
    """Return cached teams, or ``None`` when the cache is missing, unreadable or too old.

    ``max_age_hours=None`` means "any age will do" — that is the last-ditch fallback when the
    live endpoint is down.
    """
    path = _teams_cache_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        teams = payload["teams"]
        fetched_at = datetime.fromisoformat(payload["fetched_at"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.warning("ignoring unreadable ESPN team cache %s: %s", path, exc)
        return None

    if max_age_hours is not None:
        age_h = (utcnow() - fetched_at).total_seconds() / 3600.0
        if age_h > max_age_hours:
            return None
    return list(teams)


def _write_teams_cache(teams: list[dict[str, Any]]) -> None:
    """Persist the team list next to the other long-lived caches. Failure is non-fatal."""
    path = _teams_cache_path()
    payload = {"fetched_at": utcnow().isoformat(), "teams": teams}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        log.warning("could not write ESPN team cache %s: %s", path, exc)


@resilient(SOURCE, "espn_teams")
def _fetch_teams_from_api() -> list[dict[str, Any]]:
    """Pull the live team list. Wrapped in ``@resilient`` so a dead endpoint never raises."""
    with _make_client() as client:
        payload = _get_json(client, TEAMS_URL)

    entries = payload["sports"][0]["leagues"][0]["teams"]
    teams: list[dict[str, Any]] = []
    for entry in entries:
        team = entry["team"]
        espn_abbr = team.get("abbreviation")
        teams.append(
            {
                "espn_id": str(team["id"]),
                "abbreviation": espn_abbr,
                "nflverse_abbr": _nflverse_team(espn_abbr),
                "display_name": team.get("displayName"),
                "short_display_name": team.get("shortDisplayName"),
                "location": team.get("location"),
                "nickname": team.get("nickname"),
                "slug": team.get("slug"),
            }
        )
    teams.sort(key=lambda t: int(t["espn_id"]))
    return teams


def fetch_teams(force: bool = False) -> list[dict[str, Any]]:
    """Return ESPN's NFL teams, reading a long-TTL cache before hitting the network.

    Each entry carries ``espn_id`` (the id the injuries endpoint is keyed on), ``abbreviation``,
    ``nflverse_abbr`` (WSH -> WAS, LAR -> LA) and the display names.

    Args:
        force: ignore a fresh cache and re-fetch from ESPN.

    Returns:
        The team list, newest first-party data if the fetch worked, otherwise the cached copy at
        any age, otherwise an empty list. It never raises.
    """
    if not force:
        cached = _read_teams_cache(TEAMS_CACHE_TTL_HOURS)
        if cached:
            log.debug("ESPN teams: %d from cache", len(cached))
            return cached

    fetched = _fetch_teams_from_api()
    # ``@resilient`` hands back an IngestResult instead of raising when the source is down.
    if isinstance(fetched, list) and fetched:
        _write_teams_cache(fetched)
        log.info("ESPN teams: fetched %d", len(fetched))
        return fetched

    stale = _read_teams_cache(None)
    if stale:
        log.warning("ESPN team list unavailable; falling back to the stale cache (%d)", len(stale))
        return stale

    log.error("ESPN team list unavailable and no cache on disk")
    return []


# ---------------------------------------------------------------------------
# Injuries
# ---------------------------------------------------------------------------


def _list_injury_refs(client: httpx.Client, espn_team_id: str) -> list[str]:
    """Walk every page of a team's injuries collection and return the hydrate links.

    The collection defaults to ``pageSize=25`` while a single team routinely carries 60+ records,
    so paging is mandatory — reading page 1 alone silently drops two thirds of the injuries.
    """
    url = f"{CORE_API}/teams/{espn_team_id}/injuries"
    refs: list[str] = []
    page = 1
    page_count = 1

    while page <= page_count:
        payload = _get_json(client, url, params={"limit": 100, "page": page})
        page_count = int(payload.get("pageCount") or 1)
        items = payload.get("items") or []
        if not items:
            break
        refs.extend(_https(item["$ref"]) for item in items if item.get("$ref"))
        page += 1

    return refs


def _fetch_roster_index(client: httpx.Client, espn_team_id: str) -> dict[str, dict[str, Any]]:
    """One request per team giving ``athlete_id -> {name, position}``.

    Cheaper and politer than hydrating an ``athlete`` ``$ref`` for every single injury record
    (~2,000 extra requests per full refresh).
    """
    payload = _get_json(client, f"{SITE_API}/teams/{espn_team_id}", params={"enable": "roster"})
    index: dict[str, dict[str, Any]] = {}
    for athlete in payload.get("team", {}).get("athletes") or []:
        position = athlete.get("position") or {}
        index[str(athlete["id"])] = {
            "player_name": athlete.get("fullName") or athlete.get("displayName"),
            "position": position.get("abbreviation") or position.get("name"),
        }
    return index


def _fetch_athlete(client: httpx.Client, athlete_id: str) -> dict[str, Any]:
    """Hydrate one athlete by id. Only used for injured players missing from the team roster."""
    payload = _get_json(client, f"{CORE_API}/seasons/{utcnow().year}/athletes/{athlete_id}")
    position = payload.get("position") or {}
    return {
        "player_name": payload.get("fullName") or payload.get("displayName"),
        "position": position.get("abbreviation") or position.get("name"),
    }


def _athlete_id_from_ref(ref: str) -> str | None:
    """Pull the athlete id out of an injury ``$ref`` without an extra request."""
    match = _ATHLETE_ID_RE.search(ref)
    return match.group(1) if match else None


def _parse_injury(
    record: dict[str, Any],
    ref: str,
    team_abbr: str | None,
    roster: dict[str, dict[str, Any]],
    fetched_at: datetime,
) -> dict[str, Any]:
    """Flatten one hydrated ESPN injury record into a row of the injuries frame."""
    details = record.get("details") or {}
    athlete = record.get("athlete") or {}
    athlete_id = str(athlete.get("id") or "") or _athlete_id_from_ref(ref) or ""
    who = roster.get(athlete_id, {})
    return {
        "espn_athlete_id": athlete_id or None,
        "espn_injury_id": str(record["id"]) if record.get("id") is not None else None,
        "team_abbr": team_abbr,
        "player_name": who.get("player_name"),
        "position": who.get("position"),
        "status": record.get("status"),
        "injury_type": details.get("type"),
        "detail": details.get("detail"),
        "side": details.get("side"),
        "return_date": details.get("returnDate"),
        "short_comment": record.get("shortComment"),
        "long_comment": record.get("longComment"),
        "fetched_at": record.get("date") or fetched_at.isoformat(),
    }


def _collect_team_injuries(
    client: httpx.Client,
    pool: ThreadPoolExecutor,
    team: dict[str, Any],
    fetched_at: datetime,
) -> list[dict[str, Any]]:
    """Page, hydrate and flatten one team's injuries. Raises so the caller can count failures."""
    espn_team_id = team["espn_id"]
    team_abbr = _nflverse_team(team.get("abbreviation"))

    refs = _list_injury_refs(client, espn_team_id)
    if not refs:
        return []

    roster = _fetch_roster_index(client, espn_team_id)
    records = list(pool.map(lambda r: (r, _get_json(client, r)), refs))

    missing = sorted(
        {
            aid
            for ref, rec in records
            if (aid := str((rec.get("athlete") or {}).get("id") or "") or _athlete_id_from_ref(ref))
            and aid not in roster
        }
    )
    if missing:
        log.debug("team %s: hydrating %d off-roster athletes", team_abbr, len(missing))
        hydrated = list(pool.map(lambda a: _fetch_athlete(client, a), missing))
        for athlete_id, info in zip(missing, hydrated, strict=True):
            roster[athlete_id] = info

    rows = [_parse_injury(rec, ref, team_abbr, roster, fetched_at) for ref, rec in records]
    log.debug("team %s: %d injuries over %d refs", team_abbr, len(rows), len(refs))
    return rows


def _load_espn_to_gsis() -> dict[str, tuple[str, str | None]]:
    """Read ``players`` and return ``espn_id -> (gsis_id, sleeper_id)``.

    Returns an empty map (and logs) when the ``players`` table does not exist yet, which is the
    normal state before the nflverse ingest has run for the first time.
    """
    try:
        with connect() as con:
            exists = con.execute(
                "SELECT count(*) FROM duckdb_tables() WHERE table_name = 'players'"
            ).fetchone()
            if not exists or not exists[0]:
                log.warning("players table does not exist yet; ESPN injuries stay Parquet-only")
                return {}
            rows = con.execute(
                "SELECT espn_id, gsis_id, sleeper_id FROM players "
                "WHERE espn_id IS NOT NULL AND gsis_id IS NOT NULL"
            ).fetchall()
    except Exception as exc:  # noqa: BLE001 - the crosswalk is optional, never fatal
        log.warning("could not read the players crosswalk: %s", exc)
        return {}

    return {str(espn_id): (str(gsis_id), sleeper_id) for espn_id, gsis_id, sleeper_id in rows}


def _upsert_injury_status(df: pl.DataFrame) -> tuple[int, int]:
    """Upsert the mapped ESPN rows into ``injury_status`` with ``source='espn'``.

    Rows whose ``espn_athlete_id`` has no ``players.espn_id`` match stay in the Parquet file and
    are skipped here; the counts are reported so the caller can surface them.

    Returns:
        ``(n_written, n_unmapped)``.
    """
    if df.is_empty():
        return 0, 0

    crosswalk = _load_espn_to_gsis()
    if not crosswalk:
        return 0, df.height

    # One row per player: the most recent record wins.
    latest = (
        df.filter(pl.col("espn_athlete_id").is_not_null())
        .sort("fetched_at", descending=True)
        .unique(subset=["espn_athlete_id"], keep="first")
    )

    params: list[tuple[Any, ...]] = []
    unmapped = 0
    for row in latest.iter_rows(named=True):
        mapped = crosswalk.get(str(row["espn_athlete_id"]))
        if mapped is None:
            unmapped += 1
            continue
        gsis_id, sleeper_id = mapped
        observed_at = _parse_timestamp(row["fetched_at"])
        params.append(
            (
                gsis_id,
                sleeper_id,
                SOURCE,
                observed_at,
                row["team_abbr"],
                row["position"],
                espn_status_to_designation(row["status"] or ""),
                row["status"],
                None,  # ESPN carries no practice participation; Sleeper owns that column.
                row["injury_type"],
                (row["short_comment"] or "")[:1000] or None,
                None,
                None,
            )
        )

    # Duplicate records for one athlete, plus any record with no resolvable athlete id.
    unmapped += df.height - latest.height
    if not params:
        return 0, unmapped

    with connect() as con:
        con.executemany(
            """
            INSERT INTO injury_status (
                gsis_id, sleeper_id, source, observed_at, team, position,
                injury_status, roster_status, practice_participation,
                injury_body_part, injury_notes, depth_chart_position, depth_chart_order
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (gsis_id, source) DO UPDATE SET
                sleeper_id             = excluded.sleeper_id,
                observed_at            = excluded.observed_at,
                team                   = excluded.team,
                position               = excluded.position,
                injury_status          = excluded.injury_status,
                roster_status          = excluded.roster_status,
                practice_participation = excluded.practice_participation,
                injury_body_part       = excluded.injury_body_part,
                injury_notes           = excluded.injury_notes
            """,
            params,
        )

    return len(params), unmapped


def _parse_timestamp(value: str | None) -> datetime:
    """Parse an ESPN timestamp (``2026-09-04T21:36Z``) to an aware datetime, defaulting to now."""
    if not value:
        return utcnow()
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return utcnow()


@resilient(SOURCE, "espn_injuries")
def ingest_injuries(max_teams: int | None = None) -> IngestResult:
    """Pull every team's ESPN injury list, write Parquet, and upsert into ``injury_status``.

    For each team we page through the ``/injuries`` ``$ref`` collection, hydrate every reference,
    and resolve player names and positions from that team's roster. Teams that fail are logged and
    skipped: whatever succeeded is still written and the freshness badge is downgraded to yellow
    rather than red, because partial ESPN data is far better than none (D8). If *every* team
    fails, the previous Parquet is left in place rather than truncated to zero rows.

    Args:
        max_teams: only process the first N teams. Handy for smoke tests; ``None`` means all 32.

    Returns:
        An :class:`IngestResult` whose ``extra`` carries ``n_teams``, ``n_failed_teams``,
        ``n_mapped`` (rows written to ``injury_status``) and ``n_unmapped``.
    """
    teams = fetch_teams()
    if not teams:
        record_freshness(SOURCE, ok=False, detail="team list unavailable")
        return IngestResult(
            source=SOURCE,
            dataset="espn_injuries",
            ok=False,
            detail="ESPN team list unavailable; cannot enumerate injuries",
        )

    if max_teams is not None:
        teams = teams[:max_teams]

    fetched_at = utcnow()
    rows: list[dict[str, Any]] = []
    failed: list[str] = []
    lock = threading.Lock()

    with _make_client() as client, ThreadPoolExecutor(max_workers=_MAX_IN_FLIGHT) as pool:
        for team in teams:
            label = team.get("abbreviation") or team["espn_id"]
            try:
                team_rows = _collect_team_injuries(client, pool, team, fetched_at)
            except Exception as exc:  # noqa: BLE001 - one bad team must not lose the other 31
                log.warning("ESPN injuries failed for team %s: %s", label, exc)
                failed.append(str(label))
                continue
            with lock:
                rows.extend(team_rows)

    if failed and not rows:
        # Total failure: keep the previous good Parquet rather than replacing it with nothing.
        detail = f"all {len(failed)} teams failed ({', '.join(failed)})"
        log.error("ESPN injuries: %s; existing Parquet left untouched", detail)
        record_freshness(SOURCE, ok=False, detail=detail)
        return IngestResult(
            source=SOURCE,
            dataset="espn_injuries",
            ok=False,
            detail=detail,
            extra={"n_teams": len(teams), "n_failed_teams": len(failed), "failed_teams": failed},
        )

    df = pl.DataFrame(rows, schema={c: pl.Utf8 for c in _INJURY_COLUMNS})
    path = write_parquet_atomic(df, raw_path(INJURIES_FILE))

    n_mapped, n_unmapped = _upsert_injury_status(df)

    detail = (
        f"{len(teams) - len(failed)}/{len(teams)} teams, "
        f"{n_mapped} mapped to gsis_id, {n_unmapped} unmapped"
    )
    log.info("ESPN injuries: %d rows (%s)", df.height, detail)

    # Stamp the success first so the badge has a fresh last_success_at, then downgrade to yellow
    # if some teams were lost. ``record_freshness`` derives yellow from "ok=False, success recent".
    record_freshness(SOURCE, ok=True, detail=detail, n_rows=df.height)
    if failed:
        record_freshness(
            SOURCE, ok=False, detail=f"partial: {len(failed)} teams failed ({', '.join(failed)})"
        )

    return IngestResult(
        source=SOURCE,
        dataset="espn_injuries",
        n_rows=df.height,
        path=path,
        ok=True,
        detail=detail,
        extra={
            "n_teams": len(teams),
            "n_failed_teams": len(failed),
            "failed_teams": failed,
            "n_mapped": n_mapped,
            "n_unmapped": n_unmapped,
        },
    )


# ---------------------------------------------------------------------------
# Weather / venue
# ---------------------------------------------------------------------------


def _parse_event(
    event: dict[str, Any], requested_day: date, fetched_at: datetime
) -> dict[str, Any]:
    """Flatten one scoreboard event into a weather/venue row."""
    competition = (event.get("competitions") or [{}])[0]
    venue = competition.get("venue") or {}
    address = venue.get("address") or {}
    weather = event.get("weather") or {}
    season = event.get("season") or {}
    week = event.get("week") or {}

    home_abbr = away_abbr = None
    for competitor in competition.get("competitors") or []:
        abbr = (competitor.get("team") or {}).get("abbreviation")
        if competitor.get("homeAway") == "home":
            home_abbr = abbr
        elif competitor.get("homeAway") == "away":
            away_abbr = abbr

    odds = (competition.get("odds") or [{}])[0]
    espn_spread = odds.get("spread")
    provider = (odds.get("provider") or {}).get("name")

    return {
        "espn_event_id": str(event["id"]),
        "requested_date": requested_day.isoformat(),
        "kickoff_utc": event.get("date"),
        "season": int(season["year"]) if season.get("year") is not None else None,
        "season_type": int(season["type"]) if season.get("type") is not None else None,
        "week": int(week["number"]) if week.get("number") is not None else None,
        "short_name": event.get("shortName"),
        "home_team": _nflverse_team(home_abbr),
        "away_team": _nflverse_team(away_abbr),
        "home_team_espn": home_abbr,
        "away_team_espn": away_abbr,
        "venue_name": venue.get("fullName"),
        "venue_city": address.get("city"),
        "venue_state": address.get("state"),
        "indoor": bool(venue["indoor"]) if venue.get("indoor") is not None else None,
        "temperature": (
            float(weather["temperature"]) if weather.get("temperature") is not None else None
        ),
        "high_temperature": (
            float(weather["highTemperature"])
            if weather.get("highTemperature") is not None
            else None
        ),
        "weather_description": weather.get("displayValue"),
        # Cross-check only. The projection prices off nflverse schedules / The Odds API (D4).
        "odds_provider": provider,
        "odds_details": odds.get("details"),
        "espn_spread": float(espn_spread) if espn_spread is not None else None,
        # nflverse convention: positive = home favored. ESPN quotes the home team's line.
        "spread_line_home": -float(espn_spread) if espn_spread is not None else None,
        "over_under": float(odds["overUnder"]) if odds.get("overUnder") is not None else None,
        "status_state": ((competition.get("status") or {}).get("type") or {}).get("state"),
        "fetched_at": fetched_at.isoformat(),
    }


_WEATHER_SCHEMA: dict[str, Any] = {
    "espn_event_id": pl.Utf8,
    "requested_date": pl.Utf8,
    "kickoff_utc": pl.Utf8,
    "season": pl.Int32,
    "season_type": pl.Int32,
    "week": pl.Int32,
    "short_name": pl.Utf8,
    "home_team": pl.Utf8,
    "away_team": pl.Utf8,
    "home_team_espn": pl.Utf8,
    "away_team_espn": pl.Utf8,
    "venue_name": pl.Utf8,
    "venue_city": pl.Utf8,
    "venue_state": pl.Utf8,
    "indoor": pl.Boolean,
    "temperature": pl.Float64,
    "high_temperature": pl.Float64,
    "weather_description": pl.Utf8,
    "odds_provider": pl.Utf8,
    "odds_details": pl.Utf8,
    "espn_spread": pl.Float64,
    "spread_line_home": pl.Float64,
    "over_under": pl.Float64,
    "status_state": pl.Utf8,
    "fetched_at": pl.Utf8,
}


@resilient(SOURCE, "espn_weather")
def ingest_weather(dates: list[date]) -> IngestResult:
    """Pull the ESPN scoreboard for each date and write per-event weather, venue and odds.

    The venue ``indoor`` flag and the forecast temperature feed the §5.3 game-environment model.
    The ESPN spread / over-under are stored as a **cross-check only** — pricing comes from
    nflverse ``schedules`` and The Odds API (D4).

    Args:
        dates: calendar days to query, one scoreboard request each. Days with no games are fine.

    Returns:
        An :class:`IngestResult`; ``extra`` carries ``n_dates`` and ``failed_dates``. A date that
        fails is skipped and the badge goes yellow, not red. If *every* date fails the previous
        Parquet is left in place and the result is ``ok=False``.
    """
    if not dates:
        return IngestResult(
            source=SOURCE, dataset="espn_weather", skipped=True, detail="no dates requested"
        )

    fetched_at = utcnow()
    rows: list[dict[str, Any]] = []
    failed: list[str] = []

    with _make_client() as client:
        for day in dates:
            try:
                payload = _get_json(
                    client, SCOREBOARD_URL, params={"dates": day.strftime("%Y%m%d")}
                )
            except Exception as exc:  # noqa: BLE001 - one bad day must not lose the week
                log.warning("ESPN scoreboard failed for %s: %s", day.isoformat(), exc)
                failed.append(day.isoformat())
                continue
            for event in payload.get("events") or []:
                try:
                    rows.append(_parse_event(event, day, fetched_at))
                except (KeyError, TypeError, ValueError) as exc:
                    log.warning("skipping malformed ESPN event on %s: %s", day.isoformat(), exc)

    if failed and not rows:
        # Total failure: keep the previous good Parquet rather than replacing it with nothing.
        detail = f"all {len(failed)} dates failed ({', '.join(failed)})"
        log.error("ESPN weather: %s; existing Parquet left untouched", detail)
        record_freshness(SOURCE, ok=False, detail=detail)
        return IngestResult(
            source=SOURCE,
            dataset="espn_weather",
            ok=False,
            detail=detail,
            extra={"n_dates": len(dates), "failed_dates": failed},
        )

    df = pl.DataFrame(rows, schema=_WEATHER_SCHEMA)
    path = write_parquet_atomic(df, raw_path(WEATHER_FILE))

    detail = f"{len(dates) - len(failed)}/{len(dates)} dates, {df.height} events"
    log.info("ESPN weather: %s", detail)

    record_freshness(SOURCE, ok=True, detail=detail, n_rows=df.height)
    if failed:
        record_freshness(
            SOURCE, ok=False, detail=f"partial: {len(failed)} dates failed ({', '.join(failed)})"
        )

    return IngestResult(
        source=SOURCE,
        dataset="espn_weather",
        n_rows=df.height,
        path=path,
        ok=True,
        detail=detail,
        extra={"n_dates": len(dates), "failed_dates": failed},
    )
