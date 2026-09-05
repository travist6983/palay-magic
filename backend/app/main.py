"""FastAPI app. A thin read layer over DuckDB — everything is precomputed on refresh (§8).

Single-user and local by design: no auth, no rate limiting, bound to localhost. CORS is open to
the Vite dev server only.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.app import queries
from backend.app.schemas import (
    Board,
    CalibrationOut,
    DefenseTable,
    Meta,
    PlayerDetail,
)
from backend.config import get_settings
from backend.db.connection import (
    active_db_path,
    close_connection,
    set_read_only,
    use_serve_snapshot,
)
from backend.logging_setup import get_logger, setup_logging

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    # Read from the published snapshot, read-only, so `make refresh` can run underneath a live
    # `make dev` without either process losing the database lock.
    use_serve_snapshot(True)
    set_read_only(True)
    log.info("PropLab API starting (read-only, serving %s)", active_db_path().name)
    yield
    close_connection()


app = FastAPI(
    title="PropLab",
    description="Local NFL player-prop projections. Read-only over a precomputed DuckDB.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _resolve(season: int | None, week: int | None) -> tuple[int, int]:
    """Fill in the current season/week when the caller does not pin them (§10)."""
    if season and week:
        return season, week
    from backend.state import current_state

    state = current_state()
    return season or state.season, week or state.week


@app.get("/api/meta", response_model=Meta)
def get_meta(season: int | None = None, week: int | None = None) -> Any:
    """Current week, freshness badges and refresh timings — the header (§8)."""
    s, w = _resolve(season, week)
    return queries.meta(s, w)


@app.get("/api/boards", response_model=list[Board])
def get_boards(season: int | None = None, week: int | None = None) -> Any:
    """All six position boards in one round trip."""
    s, w = _resolve(season, week)
    return queries.all_boards(s, w)


@app.get("/api/board/{position}", response_model=Board)
def get_board(position: str, season: int | None = None, week: int | None = None) -> Any:
    """One position board (§8)."""
    s, w = _resolve(season, week)
    from backend.models.stats import Position

    if position.upper() not in {p.value for p in Position}:
        raise HTTPException(404, f"unknown position {position!r}")
    return queries.board(s, w, position)


@app.get("/api/player/{gsis_id}", response_model=PlayerDetail)
def get_player(gsis_id: str, season: int | None = None, week: int | None = None) -> Any:
    """The deep-dive page: game log, projections, matchup, and the full math trace (§8)."""
    s, w = _resolve(season, week)
    detail = queries.player_detail(gsis_id, s, w)
    if detail is None:
        raise HTTPException(404, f"no projectable player {gsis_id!r}")
    return detail


@app.get("/api/defense/{position}", response_model=DefenseTable)
def get_defense(
    position: str,
    metric: str | None = None,
    season: int | None = None,
    week: int | None = None,
) -> Any:
    """Every defence graded against one position, softest first."""
    s, w = _resolve(season, week)
    return queries.defense_table(s, w, position, metric)


@app.get("/api/search")
def get_search(q: Annotated[str, Query(min_length=2)], limit: int = 12) -> Any:
    """Player name search for the header's jump box."""
    return queries.search_players(q, limit)


@app.get("/api/games")
def get_games(season: int | None = None, week: int | None = None) -> Any:
    """The week's schedule with lines and weather."""
    s, w = _resolve(season, week)
    return queries.week_games(s, w).to_dicts()


@app.get("/api/calibration", response_model=CalibrationOut | None)
def get_calibration(run_id: str | None = None) -> Any:
    """The most recent backtest's calibration table (§6). Null before a backtest has run."""
    return queries.calibration(run_id)


@app.get("/api/correlations")
def get_correlations(
    season: int | None = None, week: int | None = None, limit: int = 40
) -> Any:
    """Strongest simulated same-game relationships (§5.8). Empty until `proplab simulate` runs."""
    from backend.models.simulate import top_correlations

    s, w = _resolve(season, week)
    return top_correlations(s, w, limit).to_dicts()


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Liveness plus a one-line summary of whether the database is populated."""
    from backend.db.connection import query

    try:
        n_rank = query("SELECT count(*) FROM rankings")[0][0]
        n_proj = query("SELECT count(*) FROM projections")[0][0]
    except Exception as exc:  # noqa: BLE001 - health must answer even when the DB is broken
        return {"ok": False, "detail": str(exc)}

    get_settings()
    return {
        "ok": bool(n_rank and n_proj),
        "rankings": int(n_rank),
        "projections": int(n_proj),
        "db": str(active_db_path()),
        "detail": "run `make refresh`" if not (n_rank and n_proj) else "ready",
    }
