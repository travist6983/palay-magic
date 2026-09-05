"""The refresh pipeline: the ordered stages that turn raw sources into projections.

``make backfill`` runs the historical ingest once. ``make refresh`` runs the weekly loop and must
finish in under five minutes on a laptop (§1), which it does by re-pulling only the current season
and precomputing everything the API will serve (§8).

Each stage is independently runnable and reports what it did. A stage that fails logs and lets the
pipeline continue, so one broken source cannot leave the app with nothing (D8).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from backend.config import get_settings
from backend.db.connection import connect
from backend.db.migrate import migrate
from backend.db.views import refresh_views
from backend.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class StageResult:
    """Outcome of one pipeline stage."""

    name: str
    ok: bool
    seconds: float
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        mark = "ok " if self.ok else "FAIL"
        return f"[{mark}] {self.name:<24} {self.seconds:6.1f}s  {self.detail}"


def _run(name: str, fn, *args, **kwargs) -> StageResult:
    """Run one stage, timing it and converting any exception into a failed result."""
    t0 = time.perf_counter()
    try:
        out = fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - a broken stage must not abort the pipeline
        log.exception("stage %s failed", name)
        return StageResult(name, False, time.perf_counter() - t0, f"{type(exc).__name__}: {exc}")
    detail = _summarise(out)
    return StageResult(name, True, time.perf_counter() - t0, detail, {"result": out})


def _summarise(out: Any) -> str:
    """Render a stage's return value as one short line."""
    if out is None:
        return ""
    if isinstance(out, list):
        parts = [str(x) for x in out]
        shown = ", ".join(parts[:4])
        return shown + (f", +{len(parts) - 4} more" if len(parts) > 4 else "")
    return str(out)


def ensure_schema() -> list[str]:
    """Apply pending migrations and rebuild the raw views. Cheap; safe to run every time."""
    applied = migrate()
    with connect() as con:
        views = refresh_views(con)
    return [f"{len(applied)} migrations applied", f"{len(views)} raw views"]


def backfill(seasons: list[int] | None = None, force: bool = False) -> list[StageResult]:
    """One-time historical ingest: every nflverse season, the Sleeper dump, the crosswalk."""
    from backend.ingest import crosswalk, nflverse, sleeper

    seasons = seasons or get_settings().seasons
    stages = [
        _run("schema", ensure_schema),
        _run("nflverse.backfill", nflverse.backfill, seasons, force),
        _run("sleeper.players", sleeper.ingest_players, force),
        _run("views", lambda: ensure_schema()[1]),
        _run("crosswalk.players", crosswalk.build_players),
    ]
    return stages


def refresh(force: bool = False) -> list[StageResult]:
    """The weekly loop. Pull, recompute, precompute. Must stay under five minutes (§1)."""
    from backend.ingest import crosswalk, espn, nflverse, odds, sleeper
    from backend.state import current_state

    stages: list[StageResult] = [_run("schema", ensure_schema)]

    state = current_state(refresh=True)
    log.info("refreshing for %s week %s (%s)", state.season, state.week, state.source)

    stages += [
        _run("nflverse.current", nflverse.refresh_current, state.season, force),
        _run("sleeper.players", sleeper.ingest_players, force),
        _run("views", lambda: ensure_schema()[1]),
        _run("crosswalk.players", crosswalk.build_players),
        _run("espn.injuries", espn.ingest_injuries),
        _run("odds.fetch", odds.ingest_odds, state.season, state.week, force),
        _run("odds.environment", odds.build_game_environment, state.season, state.week),
    ]

    stages += _model_stages(state.season, state.week)
    _record_refresh(state, stages)
    return stages


def _model_stages(season: int, week: int) -> list[StageResult]:
    """Modelling stages (§5). Each is skipped with a clear note until its milestone lands."""
    stages: list[StageResult] = []

    for name, importer in (
        ("defense.multipliers", _defense_stage),
        ("adjust.game_logs", _adjust_stage),
        ("rank.top10", _rank_stage),
        ("project.week", _project_stage),
    ):
        try:
            fn = importer()
        except ImportError as exc:
            stages.append(StageResult(name, True, 0.0, f"not built yet ({exc.name})"))
            continue
        stages.append(_run(name, fn, season, week))

    return stages


def _defense_stage():
    from backend.models.adjust import compute_defense_multipliers

    return compute_defense_multipliers


def _adjust_stage():
    from backend.models.adjust import build_adjusted_game_logs

    return build_adjusted_game_logs


def _rank_stage():
    from backend.models.ranking import build_rankings

    return build_rankings


def _project_stage():
    from backend.models.project import project_week

    return project_week


def _record_refresh(state, stages: list[StageResult]) -> None:
    """Stamp the refresh into ``refresh_state`` so the UI header can show when data last moved."""
    total = sum(s.seconds for s in stages)
    failed = [s.name for s in stages if not s.ok]
    rows = [
        ("last_refresh_at", "now"),
        ("last_refresh_seconds", f"{total:.1f}"),
        ("last_refresh_season", str(state.season)),
        ("last_refresh_week", str(state.week)),
        ("last_refresh_failed_stages", ",".join(failed)),
    ]
    try:
        with connect() as con:
            for key, value in rows:
                v = None if value == "now" else value
                con.execute(
                    "INSERT INTO refresh_state (key, value, updated_at) VALUES (?, ?, now()) "
                    "ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = now()",
                    [key, v if v is not None else ""],
                )
    except Exception:  # noqa: BLE001 - bookkeeping only
        log.exception("could not record refresh state")
