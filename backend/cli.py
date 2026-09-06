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
def calibrate(
    seasons: Annotated[str, typer.Option(help="Comma-separated; empty = configured seasons")] = "",
) -> None:
    """Re-tune the opponent-adjustment model per metric by walk-forward evaluation (§5.2)."""
    from backend.models.adjust import calibrate_metrics

    season_list = [int(s) for s in seasons.split(",") if s.strip()] or None
    frame = calibrate_metrics(season_list)
    if frame.is_empty():
        console.print("[yellow]no facts to calibrate against — run `proplab backfill` first[/yellow]")
        return

    table = Table("metric", "unit", "ridge λ", "MSE reduction", "raw-rate baseline", "β",
                  title="Opponent-adjustment calibration (walk-forward, out of sample)")
    for r in frame.to_dicts():
        table.add_row(
            r["metric"], r["unit"], f"{r['ridge_lambda']:.0f}",
            f"{r['mse_reduction_pct']:+.2f}%", f"{r['raw_mse_reduction_pct']:+.2f}%",
            f"{r['beta']:.3f}",
        )
    console.print(table)


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


@app.command(name="calibrate-bias")
def calibrate_bias_cmd(
    season: Annotated[int, typer.Option(help="Season to FIT on; use a different one to report")] = 2024,
    weeks: Annotated[str, typer.Option()] = "5-18",
    reuse_run: Annotated[str, typer.Option(help="Score an existing run id instead of replaying")] = "",
) -> None:
    """Fit the per-stat mean correction on a training season (§6)."""
    from backend.models.backtest import calibrate_bias

    lo, hi = (int(x) for x in weeks.split("-", 1)) if "-" in weeks else (int(weeks), int(weeks))
    frame = calibrate_bias(season, list(range(lo, hi + 1)), reuse_run or None)
    if frame.is_empty():
        console.print("[yellow]nothing to calibrate[/yellow]")
        return
    table = Table("position", "stat", "ratio", "n", "projected", "actual",
                  title=f"Mean-bias calibration (fitted on {season} weeks {lo}-{hi})")
    for r in frame.to_dicts():
        table.add_row(r["position"], r["stat"], f"{r['ratio']:.3f}", str(r["n"]),
                      f"{r['projected_mean']:.2f}", f"{r['actual_mean']:.2f}")
    console.print(table)


@app.command(name="calibrate-dispersion")
def calibrate_dispersion_cmd(
    seasons: Annotated[str, typer.Option(help="Comma-separated seasons to FIT on; report on another")] = "2023,2024",
    weeks: Annotated[str, typer.Option()] = "5-18",
) -> None:
    """Fit the p25–p75 width per stat on the training seasons (§6)."""
    from backend.models.backtest import calibrate_dispersion

    lo, hi = (int(x) for x in weeks.split("-", 1)) if "-" in weeks else (int(weeks), int(weeks))
    season_list = [int(x) for x in seasons.split(",") if x.strip()]
    frame = calibrate_dispersion(season_list, list(range(lo, hi + 1)))
    if frame.is_empty():
        console.print("[yellow]nothing to calibrate[/yellow]")
        return
    table = Table("position", "stat", "scale", "n", "coverage before", "coverage after",
                  title=f"Dispersion calibration (fitted on {seasons} weeks {lo}-{hi})")
    for r in frame.to_dicts():
        table.add_row(r["position"], r["stat"], f"{r['scale']:.2f}", str(r["n"]),
                      f"{r['coverage_before']:.1%}", f"{r['coverage_after']:.1%}")
    console.print(table)


@app.command()
def backtest(
    season: Annotated[int, typer.Option()] = 2025,
    weeks: Annotated[str, typer.Option(help="Range like 5-18, or a single week")] = "5-18",
    positions: Annotated[str, typer.Option(help="Comma-separated; empty = all")] = "",
    refit: Annotated[bool, typer.Option(help="Refit hyperparameters on prior seasons only")] = True,
) -> None:
    """Score projections against actuals and print the calibration table (§6)."""
    from backend.models.backtest import run_backtest

    if "-" in weeks:
        lo, hi = (int(x) for x in weeks.split("-", 1))
        week_list = list(range(lo, hi + 1))
    else:
        week_list = [int(weeks)]
    pos = [p.strip().upper() for p in positions.split(",") if p.strip()] or None
    run_backtest(season=season, weeks=week_list, positions=pos, refit=refit)


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
def simulate(
    season: Annotated[int, typer.Option()] = 0,
    week: Annotated[int, typer.Option()] = 0,
    sims: Annotated[int, typer.Option(help="Simulations per game")] = 5000,
    top: Annotated[int, typer.Option(help="How many correlations to print")] = 20,
) -> None:
    """Drive-level joint simulation, for correlation between players in the same game (§5.8)."""
    from backend.models.simulate import run_simulation, top_correlations

    season, week = _resolve_week(season, week)
    summary = run_simulation(season, week, n_sims=sims)
    if not summary.get("run_id"):
        console.print("[yellow]nothing to simulate — run `make refresh` first[/yellow]")
        return
    console.print_json(json.dumps(summary, default=str))

    df = top_correlations(season, week, top)
    table = Table("game", "player A", "stat", "player B", "stat", "r", "same team",
                  title="Strongest simulated relationships")
    for r in df.to_dicts():
        table.add_row(
            r["game_id"], r["player_a"] or "?", r["stat_a"], r["player_b"] or "?", r["stat_b"],
            f"{r['correlation']:+.3f}", "yes" if r["same_team"] else "no",
        )
    console.print(table)
    console.print(
        "\n[dim]The marginals come from the analytic model; the simulation supplies only the "
        "dependence between them.[/dim]"
    )


@app.command()
def parlay(
    legs: Annotated[
        str,
        typer.Argument(
            help='Semicolon-separated "Player Name:stat:line:over|under", e.g. '
            '"Joe Burrow:passing_yards:265.5:over;Ja\'Marr Chase:receiving_yards:88.5:over"'
        ),
    ],
    season: Annotated[int, typer.Option()] = 0,
    week: Annotated[int, typer.Option()] = 0,
) -> None:
    """Price a multi-leg ticket off the simulation, against the naive independent product."""
    from backend.models.project import resolve_player
    from backend.models.simulate import joint_probability

    season, week = _resolve_week(season, week)
    parsed: list[tuple[str, str, float, str]] = []
    for raw in legs.split(";"):
        parts = [x.strip() for x in raw.split(":")]
        if len(parts) != 4:
            raise typer.BadParameter(f"could not parse leg {raw!r}")
        gsis = resolve_player(parts[0])
        if not gsis:
            raise typer.BadParameter(f"no player matching {parts[0]!r}")
        parsed.append((gsis, parts[1], float(parts[2]), parts[3].lower()))

    result = joint_probability(season, week, parsed)
    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        return

    table = Table("leg", "line", "side", "P(leg)")
    for leg in result["legs"]:
        table.add_row(leg["stat"], str(leg["line"]), leg["side"], f"{leg['leg_probability']:.1%}")
    console.print(table)
    console.print(
        f"\njoint (correlated): [bold]{result['joint']:.2%}[/bold]\n"
        f"naive independent:  {result['independent']:.2%}\n"
        f"correlation multiple: [bold]{result['correlation_multiple']:.2f}x[/bold]"
    )
    console.print(
        "[dim]A multiple above 1 means the legs move together, so a fairly priced parlay pays "
        "LESS than multiplying the individual prices. Books charge 20-35% hold on same-game "
        "parlays for exactly this.[/dim]"
    )


@app.command()
def notes(
    season: Annotated[int, typer.Option()] = 0,
    week: Annotated[int, typer.Option()] = 0,
    player: Annotated[str, typer.Option(help="Just this player; empty = the whole board")] = "",
    deep_dives: Annotated[bool, typer.Option(help="Include the per-player narratives")] = True,
) -> None:
    """Generate the week's LLM notes (§7). Does nothing useful without ANTHROPIC_API_KEY."""
    from backend.llm.tasks import deep_dive_note, generate_week_notes, injury_note

    season, week = _resolve_week(season, week)
    if not get_settings().has_anthropic:
        console.print("[yellow]ANTHROPIC_API_KEY not set — narratives are skipped and the UI "
                      "omits them. Everything else works.[/yellow]")

    if player:
        from backend.models.project import resolve_player

        gsis = resolve_player(player)
        if not gsis:
            raise typer.BadParameter(f"no player matching {player!r}")
        console.print_json(json.dumps({"injury": injury_note(gsis, season, week),
                                       "deep_dive": deep_dive_note(gsis, season, week)}, default=str))
        return

    console.print_json(json.dumps(generate_week_notes(season, week, deep_dives), default=str))


@app.command()
def llm() -> None:
    """LLM token usage and the per-refresh cap (§7)."""
    from backend.llm.client import token_usage

    usage = token_usage()
    table = Table("task", "cached calls", "input tokens", "output tokens")
    for r in usage["by_task"]:
        table.add_row(r["task"], str(r["calls"]), f"{r['input_tokens']:,}", f"{r['output_tokens']:,}")
    console.print(table)
    console.print(f"this refresh: {usage['calls_this_refresh']}/{usage['cap']} calls")


@app.command()
def publish() -> None:
    """Publish the read-only snapshot the API serves from."""
    from backend.db.connection import publish_snapshot

    path = publish_snapshot()
    console.print(f"published {path}" if path else "[red]nothing to publish[/red]")


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
