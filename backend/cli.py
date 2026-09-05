"""PropLab command line.

    uv run proplab --help

Every command is thin: it resolves arguments, calls into :mod:`backend.pipeline` or a model
module, and renders the result. No business logic lives here.
"""

from __future__ import annotations

import json
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from backend.config import get_settings
from backend.logging_setup import setup_logging

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Local NFL player-prop projections. See docs/DECISIONS.md for the conventions.",
)
console = Console()


@app.callback()
def _root(
    log_level: Annotated[str, typer.Option(help="DEBUG, INFO, WARNING, ERROR")] = "",
) -> None:
    setup_logging(log_level or None)


def _print_stages(stages) -> None:
    for s in stages:
        style = "green" if s.ok else "red"
        console.print(f"[{style}]{s}[/{style}]")
    total = sum(s.seconds for s in stages)
    failed = [s.name for s in stages if not s.ok]
    console.print(f"\n[bold]{total:.1f}s total[/bold]" + (f"  [red]{len(failed)} failed[/red]" if failed else ""))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@app.command()
def backfill(
    seasons: Annotated[str, typer.Option(help="Comma-separated, e.g. 2023,2024,2025,2026")] = "",
    force: Annotated[bool, typer.Option(help="Re-download even if the cache is fresh")] = False,
) -> None:
    """One-time historical ingest. Downloads a few hundred MB; run it once."""
    from backend.pipeline import backfill as run

    season_list = [int(s) for s in seasons.split(",") if s.strip()] or None
    _print_stages(run(season_list, force))


@app.command()
def refresh(
    force: Annotated[bool, typer.Option(help="Ignore cache-age guards and re-pull everything")] = False,
) -> None:
    """Weekly refresh: pull new data, recompute defenses, rankings, and projections."""
    from backend.pipeline import refresh as run

    _print_stages(run(force))


@app.command()
def counts() -> None:
    """SELECT count(*) from every PropLab view and table (the §9 checkpoint)."""
    from backend.db.connection import connect
    from backend.db.views import table_counts

    with connect() as con:
        rows = table_counts(con)

    table = Table("kind", "name", "rows", title="PropLab storage")
    for kind, name, n in rows:
        if n < 0:
            table.add_row(kind, name, "[dim]— no files —[/dim]")
        else:
            table.add_row(kind, name, f"{n:,}")
    console.print(table)


@app.command()
def state(
    refresh_: Annotated[bool, typer.Option("--refresh", help="Re-query Sleeper")] = False,
) -> None:
    """Show the current season/week and where that answer came from."""
    from backend.state import current_state

    st = current_state(refresh=refresh_)
    console.print_json(json.dumps(st.__dict__, default=str))


@app.command()
def freshness() -> None:
    """Per-source data freshness badges (what the UI header shows)."""
    from backend.db.connection import connect

    with connect() as con:
        rows = con.execute(
            "SELECT source, status, last_success_at, last_attempt_at, n_rows, detail "
            "FROM source_freshness ORDER BY source"
        ).fetchall()

    table = Table("source", "status", "last success", "last attempt", "rows", "detail")
    colours = {"green": "green", "yellow": "yellow", "red": "red"}
    for source, status, ok_at, try_at, n, detail in rows:
        c = colours.get(status, "white")
        table.add_row(
            source,
            f"[{c}]{status}[/{c}]",
            str(ok_at or "—"),
            str(try_at or "—"),
            f"{n:,}" if n else "—",
            (detail or "")[:60],
        )
    console.print(table)


@app.command()
def budget() -> None:
    """Persisted API request budgets (Odds API 500/month, Anthropic per refresh)."""
    from backend.db.connection import connect

    with connect() as con:
        rows = con.execute(
            "SELECT api, period, n_calls, budget, last_call_at FROM api_budget "
            "ORDER BY api, period DESC"
        ).fetchall()

    table = Table("api", "period", "used", "budget", "remaining", "last call")
    for api, period, used, budget_, last in rows:
        table.add_row(api, period, str(used), str(budget_), str(max(0, budget_ - used)), str(last or "—"))
    console.print(table)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@app.command()
def defense(
    season: Annotated[int, typer.Option()] = 0,
    week: Annotated[int, typer.Option()] = 0,
    position: Annotated[str, typer.Option(help="QB, RB, WR, TE, K, LB")] = "",
    top: Annotated[int, typer.Option(help="How many at each end to show")] = 5,
) -> None:
    """Top/bottom defenses vs. each position (the §9 milestone-2 checkpoint)."""
    from backend.models.adjust import print_defense_report

    season, week = _resolve_week(season, week)
    print_defense_report(season, week, position or None, top)


@app.command()
def rank(
    season: Annotated[int, typer.Option()] = 0,
    week: Annotated[int, typer.Option()] = 0,
    position: Annotated[str, typer.Option(help="QB, RB, WR, TE, K, LB. Empty = all six")] = "",
    recompute: Annotated[bool, typer.Option(help="Rebuild instead of reading the cache")] = False,
) -> None:
    """Print the top-10 board for each position (the §9 milestone-3 checkpoint)."""
    from backend.models.ranking import print_rankings

    season, week = _resolve_week(season, week)
    print_rankings(season, week, position or None, recompute=recompute)


@app.command()
def project(
    player: Annotated[str, typer.Option(help="Player name or gsis_id")] = "",
    season: Annotated[int, typer.Option()] = 0,
    week: Annotated[int, typer.Option()] = 0,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full deep-dive as JSON")] = False,
) -> None:
    """Project one player for the given week, with the full 'show math' trace."""
    from backend.models.project import print_projection, project_player_json

    season, week = _resolve_week(season, week)
    if not player:
        raise typer.BadParameter("--player is required")
    if as_json:
        console.print_json(json.dumps(project_player_json(player, season, week), default=str))
    else:
        print_projection(player, season, week)


@app.command()
def backtest(
    season: Annotated[int, typer.Option()] = 2025,
    weeks: Annotated[str, typer.Option(help="Range like 5-18, or a single week")] = "5-18",
    positions: Annotated[str, typer.Option(help="Comma-separated; empty = all")] = "",
) -> None:
    """Score projections against actuals and print the calibration table (§6)."""
    from backend.models.backtest import run_backtest

    if "-" in weeks:
        lo, hi = (int(x) for x in weeks.split("-", 1))
        week_list = list(range(lo, hi + 1))
    else:
        week_list = [int(weeks)]
    pos = [p.strip().upper() for p in positions.split(",") if p.strip()] or None
    run_backtest(season=season, weeks=week_list, positions=pos)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_week(season: int, week: int) -> tuple[int, int]:
    """Fill in season/week from the live state when either is left at 0 (§10: never hardcoded)."""
    if season and week:
        return season, week
    from backend.state import current_state

    st = current_state()
    return (season or st.season, week or st.week)


@app.command()
def config() -> None:
    """Print the resolved configuration, with secrets masked."""
    s = get_settings()
    data = s.model_dump()
    for key in ("anthropic_api_key", "odds_api_key"):
        data[key] = "set" if data.get(key) else None
    console.print_json(json.dumps(data, default=str))


if __name__ == "__main__":
    app()
