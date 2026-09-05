"""Backtest and calibration (§6).

Replays a season week by week. For each week the pipeline is rebuilt using **only** information
available before that week's games — defensive multipliers, team environment, rankings and
projections all recomputed against a frozen cutoff — and then scored against what actually
happened.

Three things are reported, per position and per stat:

* **MAE** — how far off the median was.
* **p25–p75 coverage** — the actual should land inside the interval about **50%** of the time. If
  it does not, the dispersion is wrong, and §6 is explicit that this gets fixed before anything
  else is added.
* **Brier score** — for anytime TD, and for P(over) at the median, where a calibrated model scores
  0.25 and anything much worse means the median is not the median.

**Leakage.** The ridge penalties and the environment coefficients are hyperparameters fitted over
the whole cache, which includes the test season. Left alone that is a mild but real leak, so by
default the backtest refits both on seasons strictly before the test season and says so. Pass
``refit=False`` to reuse the production fit and accept the leak.

The Odds API is never called (§6): implied team totals come from nflverse ``schedules``, which
carries closing ``spread_line`` and ``total_line`` for every historical week.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from backend.config import get_settings
from backend.db.connection import connect
from backend.logging_setup import get_logger
from backend.models.stats import Family

log = get_logger(__name__)

# Stats whose actual is a 0/1 outcome rather than a count.
BINARY_STATS = frozenset({"anytime_td", "anytime_rush_td"})


@dataclass
class WeekResult:
    """What one replayed week produced."""

    season: int
    week: int
    n_players: int
    n_projections: int
    seconds: float
    error: str = ""


@dataclass
class Calibration:
    """Scored results for one (position, stat) cell."""

    position: str
    stat: str
    n: int
    mae: float
    bias: float
    coverage: float
    brier: float
    median_actual: float
    median_projected: float
    extras: dict[str, Any] = field(default_factory=dict)


def _rebuild_week(season: int, week: int) -> WeekResult:
    """Recompute the whole pipeline for one week using only prior data."""
    from backend.ingest.odds import build_game_environment
    from backend.models.adjust import build_adjusted_game_logs, compute_defense_multipliers
    from backend.models.environment import build_team_environment
    from backend.models.project import project_week
    from backend.models.ranking import build_rankings

    t0 = time.perf_counter()
    try:
        # prefer_odds=False: the backtest must never touch the Odds API (§6).
        build_game_environment(season, week, prefer_odds=False)
        compute_defense_multipliers(season, week, rebuild_facts=False)
        build_adjusted_game_logs(season, week)
        build_team_environment(season, week)
        n_players = build_rankings(season, week)
        n_proj = project_week(season, week)
    except Exception as exc:  # noqa: BLE001 - one bad week must not abort the run
        log.exception("backtest week %s/%s failed", season, week)
        return WeekResult(season, week, 0, 0, time.perf_counter() - t0, f"{type(exc).__name__}: {exc}")

    return WeekResult(season, week, n_players, n_proj, time.perf_counter() - t0)


def _score_week(run_id: str, season: int, week: int) -> int:
    """Join the week's stored projections to actuals and persist the scored rows."""
    from backend.models.distributions import Distribution as Dist

    with connect() as con:
        rows = con.execute(
            """
            SELECT p.gsis_id, p.position, p.stat, p.dist_family, p.params,
                   p.median, p.p25, p.p75, p.mean, a.value AS actual
            FROM projections p
            JOIN player_game_stats a
              ON a.gsis_id = p.gsis_id AND a.season = p.season AND a.week = p.week
             AND a.stat = p.stat
            WHERE p.season = ? AND p.week = ? AND p.conditional_on_playing = TRUE
            """,
            [season, week],
        ).pl()

    if rows.is_empty():
        return 0

    scored: list[dict[str, Any]] = []
    for r in rows.to_dicts():
        params = json.loads(r["params"])
        dist = Dist(
            family=Family(r["dist_family"]),
            params=params,
            mean=r["mean"],
            integer_valued=r["dist_family"] != "empirical_max",
        )
        actual = float(r["actual"])
        median = float(r["median"])
        p_over_median = dist.prob_over(median)

        # Randomised PIT: for a discrete distribution the CDF jumps, so F(x) alone cannot be
        # uniform even for a perfect model. Averaging F(x-) and F(x) is the standard fix, and it
        # makes PIT the one calibration metric that works across every family here.
        upper = 1.0 - dist.prob_over(actual)
        lower = upper - dist.prob_exact(actual)
        pit = (lower + upper) / 2.0

        # For a yes/no market the meaningful Brier is against P(at least one), not P(> median).
        # Scoring anytime-TD against its own median is degenerate: the median is 0 or 1 and
        # P(X > median) is ~0 either way, which scores 0.03 while saying nothing.
        if r["stat"] in BINARY_STATS:
            p_event = dist.prob_over(0.5)
            outcome_event = actual > 0
        else:
            p_event = p_over_median
            outcome_event = actual > median

        scored.append(
            {
                "run_id": run_id, "season": season, "week": week, "gsis_id": r["gsis_id"],
                "position": r["position"], "stat": r["stat"],
                "projected_median": median, "projected_p25": r["p25"], "projected_p75": r["p75"],
                "dist_family": r["dist_family"], "params": r["params"],
                "actual": actual, "abs_error": abs(actual - median),
                "in_interval": bool(r["p25"] <= actual <= r["p75"]),
                "p_over_median": p_event,
                "outcome_over": bool(outcome_event),
                "pit": pit,
                "degenerate_interval": bool(r["p25"] == r["p75"]),
            }
        )

    frame = pl.DataFrame(scored)
    with connect() as con:
        con.register("bt_df", frame)
        try:
            con.execute("BEGIN TRANSACTION")
            con.execute(
                "DELETE FROM backtest_results WHERE run_id = ? AND season = ? AND week = ?",
                [run_id, season, week],
            )
            con.execute(
                "INSERT INTO backtest_results "
                "(run_id, season, week, gsis_id, position, stat, projected_median, projected_p25, "
                " projected_p75, dist_family, params, actual, abs_error, in_interval, "
                " p_over_median, outcome_over, pit, degenerate_interval) "
                "SELECT run_id, season, week, gsis_id, position, stat, projected_median, "
                "       projected_p25, projected_p75, dist_family, params, actual, abs_error, "
                "       in_interval, p_over_median, outcome_over, pit, degenerate_interval "
                "FROM bt_df"
            )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            con.unregister("bt_df")

    return frame.height


def _refit_for(season: int) -> None:
    """Refit hyperparameters on seasons strictly before ``season``, to avoid leaking the test set."""
    from backend.models.adjust import calibrate_metrics
    from backend.models.environment import fit_environment_models
    from backend.models.project import _DISPERSION_CACHE, _RATE_PRIOR_CACHE

    train = [s for s in get_settings().seasons if s < season]
    if not train:
        log.warning("no seasons before %s to train on; keeping the production fit", season)
        return

    log.info("refitting hyperparameters on %s (test season %s held out)", train, season)
    _DISPERSION_CACHE.clear()
    _RATE_PRIOR_CACHE.clear()
    calibrate_metrics(train)
    fit_environment_models(train)


# Scales are bounded: a cell asking for 10x is telling us the mean is wrong, not the width.
MIN_SCALE, MAX_SCALE = 0.4, 5.0

# Below this many scored rows a cell borrows the scale fitted for its stat across all positions.
MIN_CELL_ROWS = 60


def _central_coverage(rows: list[dict[str, Any]], scale: float) -> float:
    """Share of outcomes inside p25-p75 when the model's variance is multiplied by ``scale``."""
    from scipy import stats as st

    hits = 0
    n = 0
    for r in rows:
        params = json.loads(r["params"])
        family = r["dist_family"]
        actual = float(r["actual"])

        if family == "negative_binomial":
            mean = float(params["mean"])
            var = max(float(params["variance"]) * scale, mean * 1.05)
            alpha = (var - mean) / (mean * mean)
            rr = 1.0 / max(alpha, 1e-9)
            pp = rr / (rr + mean)
            lo, hi = st.nbinom.ppf(0.25, rr, pp), st.nbinom.ppf(0.75, rr, pp)
        elif family == "poisson":
            mean = float(params["lam"])
            var = max(mean * scale, mean * 1.001)
            if var > mean * 1.05:
                alpha = (var - mean) / (mean * mean) if mean > 0 else 1e-6
                rr = 1.0 / max(alpha, 1e-9)
                pp = rr / (rr + mean)
                lo, hi = st.nbinom.ppf(0.25, rr, pp), st.nbinom.ppf(0.75, rr, pp)
            else:
                lo, hi = st.poisson.ppf(0.25, mean), st.poisson.ppf(0.75, mean)
        else:
            continue

        n += 1
        hits += int(lo <= actual <= hi)

    return hits / n if n else float("nan")


def _solve_scale(rows: list[dict[str, Any]], target: float = 0.5) -> tuple[float, float, float]:
    """Bisect for the variance scale whose p25-p75 covers ``target`` of outcomes.

    Returns ``(scale, coverage_before, coverage_after)``.
    """
    before = _central_coverage(rows, 1.0)
    if not np.isfinite(before):
        return 1.0, float("nan"), float("nan")

    lo, hi = MIN_SCALE, MAX_SCALE
    if _central_coverage(rows, hi) < target:
        return hi, before, _central_coverage(rows, hi)
    if _central_coverage(rows, lo) > target:
        return lo, before, _central_coverage(rows, lo)

    for _ in range(24):
        mid = (lo + hi) / 2
        if _central_coverage(rows, mid) < target:
            lo = mid
        else:
            hi = mid
    scale = (lo + hi) / 2
    return scale, before, _central_coverage(rows, scale)


def calibrate_dispersion(
    season: int = 2024,
    weeks: list[int] | None = None,
    reuse_run: str | None = None,
) -> pl.DataFrame:
    """Fit the per-(position, stat) dispersion scale on a training season (§6).

    Fit on one season and validate on another: calling this on 2024 and then backtesting 2025
    keeps the width honest, because a scale solved on the same rows it is scored against would
    report 50% coverage by construction.

    Args:
        season: the season to fit on. Should NOT be the season you intend to report.
        weeks: which weeks; defaults to 5-18.
        reuse_run: score an existing run id instead of replaying.

    Returns:
        The fitted scales, with coverage before and after.
    """
    weeks = weeks or list(range(5, 19))
    if reuse_run:
        run_id = reuse_run
    else:
        log.info("replaying %s weeks %s-%s to fit dispersion", season, min(weeks), max(weeks))
        run_backtest(season=season, weeks=weeks, refit=True, quiet=True)
        run_id = latest_run()

    with connect() as con:
        df = con.execute(
            "SELECT position, stat, dist_family, params, actual FROM backtest_results "
            "WHERE run_id = ? AND dist_family IN ('negative_binomial', 'poisson')",
            [run_id],
        ).pl()

    if df.is_empty():
        log.warning("no scored rows to calibrate dispersion from")
        return pl.DataFrame()

    label = f"{season} weeks {min(weeks)}-{max(weeks)}"
    by_stat: dict[str, tuple[float, float, float, int]] = {}
    for stat in df["stat"].unique():
        rows = df.filter(pl.col("stat") == stat).to_dicts()
        scale, before, after = _solve_scale(rows)
        by_stat[stat] = (scale, before, after, len(rows))

    out: list[dict[str, Any]] = []
    for key in df.select("position", "stat").unique().to_dicts():
        position, stat = key["position"], key["stat"]
        rows = df.filter((pl.col("position") == position) & (pl.col("stat") == stat)).to_dicts()
        if len(rows) >= MIN_CELL_ROWS:
            scale, before, after = _solve_scale(rows)
            n = len(rows)
        else:
            # Too thin to fit on its own: borrow the scale fitted for this stat league-wide.
            scale, before, after, n = by_stat.get(stat, (1.0, float("nan"), float("nan"), 0))
        out.append(
            {
                "position": position, "stat": stat, "scale": float(scale), "n": int(n),
                "coverage_before": float(before), "coverage_after": float(after),
                "fitted_on": label,
            }
        )

    frame = pl.DataFrame(out)
    with connect() as con:
        con.register("disp_df", frame)
        try:
            con.execute("BEGIN TRANSACTION")
            con.execute("DELETE FROM dispersion_calibration")
            con.execute(
                "INSERT INTO dispersion_calibration "
                "(position, stat, scale, n, coverage_before, coverage_after, fitted_on, computed_at) "
                "SELECT position, stat, scale, n, coverage_before, coverage_after, fitted_on, now() "
                "FROM disp_df"
            )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            con.unregister("disp_df")

    log.info("dispersion calibrated for %d (position, stat) cells on %s", frame.height, label)
    return frame.sort("scale", descending=True)


def run_backtest(
    season: int = 2025,
    weeks: list[int] | None = None,
    positions: list[str] | None = None,
    refit: bool = True,
    quiet: bool = False,
) -> pl.DataFrame:
    """Replay a season and print the calibration table (§6).

    Args:
        season: the season to replay.
        weeks: which weeks. Defaults to 5-18, so every projection has real trailing data.
        positions: restrict the report; the replay always projects everything.
        refit: refit hyperparameters on prior seasons only. Turn off to reuse the production fit
            and accept the leak.
        quiet: skip printing.

    Returns:
        The per-(position, stat) calibration table.
    """
    weeks = weeks or list(range(5, 19))
    run_id = uuid.uuid4().hex[:12]
    started = time.time()

    if refit:
        _refit_for(season)

    # Facts are just aggregates of games played, not a fitted model, so they can span everything.
    from backend.models.adjust import compute_unit_facts
    from backend.models.gamelog import build_all as build_gamelogs

    with connect() as con:
        if not con.execute("SELECT count(*) FROM unit_game_facts").fetchone()[0]:
            compute_unit_facts()
        if not con.execute("SELECT count(*) FROM player_game_stats").fetchone()[0]:
            build_gamelogs()

    week_results: list[WeekResult] = []
    total_scored = 0
    for week in weeks:
        wr = _rebuild_week(season, week)
        if wr.error:
            week_results.append(wr)
            continue
        n = _score_week(run_id, season, week)
        wr.n_projections = n
        week_results.append(wr)
        total_scored += n
        log.info(
            "backtest %s week %02d: %d players, %d scored rows, %.1fs",
            season, week, wr.n_players, n, wr.seconds,
        )

    with connect() as con:
        con.execute(
            "INSERT INTO backtest_runs "
            "(run_id, season, week_start, week_end, n_projections, started_at, finished_at, config) "
            "VALUES (?, ?, ?, ?, ?, to_timestamp(?), now(), ?)",
            [
                run_id, season, min(weeks), max(weeks), total_scored, started,
                json.dumps(
                    {
                        "weeks": weeks, "refit": refit,
                        "recency_window": get_settings().recency_window,
                        "efficiency_window": get_settings().efficiency_window,
                        "efficiency_prior_games": get_settings().efficiency_prior_games,
                        "defense_window": get_settings().defense_window,
                    }
                ),
            ],
        )

    table = calibration_table(run_id, positions)
    if not quiet:
        print_calibration(table, run_id, week_results)
    return table


def calibration_table(run_id: str, positions: list[str] | None = None) -> pl.DataFrame:
    """Score a stored run into a per-(position, stat) calibration table."""
    where = "WHERE run_id = ?"
    params: list[Any] = [run_id]
    if positions:
        where += " AND position IN ({})".format(",".join("?" * len(positions)))
        params.extend(positions)

    with connect() as con:
        df = con.execute(
            f"""
            SELECT position, stat,
                   count(*)                                   AS n,
                   avg(abs_error)                             AS mae,
                   avg(actual - projected_median)             AS bias,
                   avg(CASE WHEN in_interval THEN 1.0 ELSE 0 END) AS coverage,
                   avg((p_over_median - CASE WHEN outcome_over THEN 1.0 ELSE 0 END)
                       * (p_over_median - CASE WHEN outcome_over THEN 1.0 ELSE 0 END)) AS brier,
                   median(actual)                             AS median_actual,
                   median(projected_median)                   AS median_projected,
                   avg(CASE WHEN outcome_over THEN 1.0 ELSE 0 END) AS over_rate,
                   avg(CASE WHEN degenerate_interval THEN 1.0 ELSE 0 END) AS degenerate_rate,
                   avg(CASE WHEN pit BETWEEN 0.25 AND 0.75 THEN 1.0 ELSE 0 END) AS pit_central,
                   avg(pit)                                   AS pit_mean
            FROM backtest_results
            {where}
            GROUP BY 1, 2
            HAVING count(*) >= 20
            ORDER BY position, stat
            """,
            params,
        ).pl()
    return df


def print_calibration(
    table: pl.DataFrame, run_id: str, week_results: list[WeekResult] | None = None
) -> None:
    """Print the §6 calibration table and the verdict on whether to proceed."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    if week_results:
        failed = [w for w in week_results if w.error]
        ok = [w for w in week_results if not w.error]
        console.print(
            f"\n[bold]Backtest {run_id}[/bold] · {len(ok)} weeks replayed"
            + (f", [red]{len(failed)} failed[/red]" if failed else "")
            + f" · {sum(w.seconds for w in week_results):.0f}s"
        )
        for w in failed:
            console.print(f"  [red]week {w.week}: {w.error}[/red]")

    if table.is_empty():
        console.print("[yellow]no scored rows — the replay produced nothing to score[/yellow]")
        return

    t = Table(
        "position", "stat", "n", "MAE", "bias", "p25–p75", "PIT central", "Brier", "over rate",
        "median proj", "median actual",
        title="Calibration — target: coverage ≈ 50%, PIT central ≈ 50%, Brier ≈ 0.25, bias ≈ 0",
        title_justify="left",
    )
    for r in table.to_dicts():
        cov, pit = r["coverage"], r["pit_central"]
        degenerate = (r.get("degenerate_rate") or 0.0) > 0.25
        colour = "green" if 0.40 <= cov <= 0.60 else ("yellow" if 0.30 <= cov <= 0.70 else "red")
        pit_colour = "green" if 0.40 <= pit <= 0.60 else ("yellow" if 0.30 <= pit <= 0.70 else "red")
        cov_text = f"[{colour}]{cov:.1%}[/{colour}]" + (" [dim]†[/dim]" if degenerate else "")
        t.add_row(
            r["position"], r["stat"], str(r["n"]),
            f"{r['mae']:.2f}", f"{r['bias']:+.2f}", cov_text,
            f"[{pit_colour}]{pit:.1%}[/{pit_colour}]",
            f"{r['brier']:.3f}", f"{r['over_rate']:.1%}",
            f"{r['median_projected']:.1f}", f"{r['median_actual']:.1f}",
        )
    console.print(t)

    # A low-count stat's p25 and p75 collapse onto the same integer -- Poisson(0.13) has
    # p25 = p75 = 0 -- so "inside the interval" catches every zero and coverage reads 85%+ no
    # matter how good the model is. Those cells are marked † and judged on PIT instead, which is
    # the metric that survives a discrete support.
    continuous = table.filter(pl.col("degenerate_rate").fill_null(0.0) <= 0.25)
    degenerate = table.filter(pl.col("degenerate_rate").fill_null(0.0) > 0.25)

    if continuous.height:
        overall = float((continuous["coverage"] * continuous["n"]).sum() / continuous["n"].sum())
        console.print(
            f"\n[bold]p25–p75 coverage on non-degenerate cells: {overall:.1%}[/bold] "
            f"({continuous.height} of {table.height} cells, target 50%)"
        )
    else:
        overall = float("nan")

    pit_all = float((table["pit_central"] * table["n"]).sum() / table["n"].sum())
    console.print(f"[bold]PIT central mass (all cells): {pit_all:.1%}[/bold] (target 50%)")

    if degenerate.height:
        console.print(
            f"[dim]† {degenerate.height} cells have p25 = p75 for most rows "
            "(low-count stats). Their coverage number is not meaningful; judge them on PIT and "
            "Brier.[/dim]"
        )

    bad = continuous.filter((pl.col("coverage") < 0.40) | (pl.col("coverage") > 0.60))
    if not np.isfinite(overall):
        console.print("[yellow]No non-degenerate cells to judge coverage on.[/yellow]")
    elif overall < 0.40 or overall > 0.60:
        console.print(
            "[red]Coverage is outside 40–60%. §6 says fix the dispersion before adding "
            "features — run `proplab calibrate-dispersion`.[/red]"
        )
    elif bad.height:
        console.print(
            f"[yellow]{bad.height} of {continuous.height} continuous cells sit outside 40–60%: "
            + ", ".join(f"{r['position']}/{r['stat']} {r['coverage']:.0%}" for r in bad.to_dicts()[:8])
            + "[/yellow]"
        )
    else:
        console.print("[green]Every continuous cell is within 40–60%. Dispersion is calibrated.[/green]")


def latest_run() -> str | None:
    """The most recent backtest run id."""
    with connect() as con:
        row = con.execute(
            "SELECT run_id FROM backtest_runs ORDER BY finished_at DESC LIMIT 1"
        ).fetchone()
    return row[0] if row else None


def coverage_by_week(run_id: str) -> pl.DataFrame:
    """Coverage per week, to spot a systematic drift across a season."""
    with connect() as con:
        return con.execute(
            "SELECT week, count(*) AS n, "
            "       avg(CASE WHEN in_interval THEN 1.0 ELSE 0 END) AS coverage, "
            "       avg(actual - projected_median) AS bias "
            "FROM backtest_results WHERE run_id = ? GROUP BY 1 ORDER BY 1",
            [run_id],
        ).pl()
