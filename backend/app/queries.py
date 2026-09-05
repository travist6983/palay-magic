"""Read-only queries backing the API.

The refresh has already done every computation (§8), so these are joins and reshapes over DuckDB.
Nothing here fits a model or calls a network.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import polars as pl

from backend.config import get_settings
from backend.db.connection import query_df
from backend.models.adjust import METRIC_BY_KEY
from backend.models.stats import HEADLINE_STATS, Position, get_spec, stats_for


def _distribution(row: dict[str, Any]) -> dict[str, Any]:
    """Reshape a stored projection row into the wire format the browser evaluates (D10)."""
    params = row["params"]
    return {
        "family": row["dist_family"],
        "params": json.loads(params) if isinstance(params, str) else params,
        "mean": row["mean"],
        "median": row["median"],
        "p25": row["p25"],
        "p75": row["p75"],
        "integer_valued": row["dist_family"] != "empirical_max",
    }


def meta(season: int, week: int) -> dict[str, Any]:
    """Header state: current week, freshness badges, refresh timings (§8)."""
    from backend.state import current_state

    state = current_state()
    settings = get_settings()

    fresh = query_df(
        "SELECT source, status, last_success_at, last_attempt_at, detail, n_rows "
        "FROM source_freshness ORDER BY source"
    )
    now = datetime.now(UTC)
    sources = []
    for r in fresh.to_dicts():
        last = r["last_success_at"]
        age = None
        if last is not None:
            last_dt = last if isinstance(last, datetime) else None
            if last_dt is not None:
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=UTC)
                age = (now - last_dt).total_seconds() / 3600.0
        sources.append(
            {
                "source": r["source"],
                "status": r["status"] or "red",
                "last_success_at": str(last) if last else None,
                "last_attempt_at": str(r["last_attempt_at"]) if r["last_attempt_at"] else None,
                "detail": r["detail"],
                "rows": int(r["n_rows"]) if r["n_rows"] else None,
                "age_hours": age,
            }
        )

    refresh = {
        r["key"]: r["value"]
        for r in query_df("SELECT key, value FROM refresh_state").to_dicts()
    }
    refresh_at = query_df(
        "SELECT max(updated_at) AS t FROM refresh_state WHERE key = 'last_refresh_at'"
    )
    budget = query_df(
        "SELECT budget - n_calls AS remaining FROM api_budget "
        "WHERE api = 'the-odds-api' ORDER BY period DESC LIMIT 1"
    )

    return {
        "season": season,
        "week": week,
        "season_type": state.season_type,
        "state_source": state.source,
        "games_played_this_season": state.games_played_this_season,
        "season_start_date": str(state.season_start_date) if state.season_start_date else None,
        "last_refresh_at": str(refresh_at["t"][0]) if refresh_at.height and refresh_at["t"][0] else None,
        "last_refresh_seconds": float(refresh["last_refresh_seconds"])
        if refresh.get("last_refresh_seconds")
        else None,
        "failed_stages": [s for s in (refresh.get("last_refresh_failed_stages") or "").split(",") if s],
        "sources": sources,
        "positions": [p.value for p in Position],
        "all_history_prior_season": state.games_played_this_season == 0,
        "odds_budget_remaining": int(budget["remaining"][0]) if budget.height else None,
        "llm_available": settings.has_anthropic,
    }


def board(season: int, week: int, position: str) -> dict[str, Any]:
    """One position board with its headline projections (§8)."""
    position = position.upper()
    rows = query_df(
        """
        SELECT r.rank, r.gsis_id, r.score, r.team, r.opponent, r.injury_status,
               r.play_probability, r.insufficient_history, r.changed_team, r.changed_coach,
               r.pass_rate_shift, p.display_name, p.headshot_url
        FROM rankings r LEFT JOIN players p USING (gsis_id)
        WHERE r.season = ? AND r.week = ? AND r.position = ?
        ORDER BY r.rank
        """,
        [season, week, position],
    )
    if rows.is_empty():
        return {"position": position, "season": season, "week": week, "rows": []}

    headline_stats = HEADLINE_STATS[Position(position)]
    ids = rows["gsis_id"].to_list()
    placeholders = ",".join("?" * len(ids))
    projections = query_df(
        f"""
        SELECT gsis_id, stat, dist_family, params, mean, median, p25, p75, high_variance,
               play_probability, conditional_on_playing
        FROM projections
        WHERE season = ? AND week = ? AND gsis_id IN ({placeholders})
          AND stat IN ({",".join("?" * len(headline_stats))})
        """,
        [season, week, *ids, *headline_stats],
    )

    by_player: dict[str, dict[str, dict]] = {}
    for r in projections.to_dicts():
        bucket = by_player.setdefault(r["gsis_id"], {})
        key = ("cond" if r["conditional_on_playing"] else "uncond") + ":" + r["stat"]
        bucket[key] = r

    # The opponent multiplier shown on the board is the one for the position's primary stat.
    # That metric is NOT always the projected stat: a kicker's headline is kicking points but the
    # matchup metric is FG attempts allowed, and a linebacker's is tackles but the metric is
    # offensive plays run. The board therefore ships the metric's own name so the UI can say what
    # the number actually measures instead of inferring it from the projection label.
    primary_metric = get_spec(position, headline_stats[0]).defense_metric
    primary_spec = METRIC_BY_KEY.get(primary_metric)
    mults = {
        (r["team"]): (r["multiplier"], r["rank"])
        for r in query_df(
            "SELECT team, multiplier, rank FROM defense_multipliers "
            "WHERE season = ? AND week = ? AND metric = ?",
            [season, week, primary_metric],
        ).to_dicts()
    }

    out_rows = []
    for r in rows.to_dicts():
        bucket = by_player.get(r["gsis_id"], {})
        headline = []
        for stat in headline_stats:
            cond = bucket.get(f"cond:{stat}")
            if not cond:
                continue
            uncond = bucket.get(f"uncond:{stat}")
            spec = get_spec(position, stat)
            headline.append(
                {
                    "stat": stat,
                    "label": spec.label,
                    "distribution": _distribution(cond),
                    "unconditional": _distribution(uncond) if uncond else None,
                    "high_variance": bool(cond["high_variance"]),
                    "play_probability": cond["play_probability"],
                    "settlement_note": spec.settlement_note,
                }
            )
        mult, rank = mults.get(r["opponent"], (1.0, None))
        out_rows.append(
            {
                "rank": r["rank"], "gsis_id": r["gsis_id"],
                "display_name": r["display_name"] or r["gsis_id"],
                "team": r["team"], "opponent": r["opponent"],
                "headshot_url": r["headshot_url"], "score": r["score"],
                "injury_status": r["injury_status"],
                "play_probability": r["play_probability"] or 1.0,
                "insufficient_history": bool(r["insufficient_history"]),
                "changed_team": bool(r["changed_team"]),
                "changed_coach": bool(r["changed_coach"]),
                "pass_rate_shift": r["pass_rate_shift"] or 0.0,
                "opponent_multiplier": mult, "opponent_rank": rank,
                "opponent_metric": primary_metric,
                "opponent_metric_label": primary_spec.label if primary_spec else primary_metric,
                "headline": headline,
            }
        )

    return {"position": position, "season": season, "week": week, "rows": out_rows}


def player_detail(gsis_id: str, season: int, week: int) -> dict[str, Any] | None:
    """Everything the deep-dive page renders (§8)."""
    bio = query_df(
        "SELECT gsis_id, display_name, position, team, headshot_url, height, weight, years_exp "
        "FROM players WHERE gsis_id = ?",
        [gsis_id],
    )
    if bio.is_empty():
        return None
    b = bio.to_dicts()[0]
    position = b["position"]
    if position not in {p.value for p in Position}:
        return None

    rank_row = query_df(
        "SELECT team, opponent, injury_status, play_probability, changed_team, changed_coach, "
        "       pass_rate_shift, insufficient_history, components "
        "FROM rankings WHERE season = ? AND week = ? AND gsis_id = ?",
        [season, week, gsis_id],
    )
    team = rank_row["team"][0] if rank_row.height else b["team"]
    opponent = rank_row["opponent"][0] if rank_row.height else None

    env_df = query_df(
        "SELECT * FROM team_environment WHERE season = ? AND week = ? AND team = ?",
        [season, week, team],
    )
    game_df = query_df(
        "SELECT roof, wind, temp, total_line, odds_source FROM game_environment "
        "WHERE season = ? AND week = ? AND (home_team = ? OR away_team = ?)",
        [season, week, team, team],
    )
    env = env_df.to_dicts()[0] if env_df.height else {}
    game = game_df.to_dicts()[0] if game_df.height else {}
    indoor = (game.get("roof") or "").lower() in {"dome", "closed"}

    projections = query_df(
        "SELECT stat, dist_family, params, mean, median, p25, p75, high_variance, "
        "       play_probability, conditional_on_playing "
        "FROM projections WHERE season = ? AND week = ? AND gsis_id = ?",
        [season, week, gsis_id],
    )
    cond = {r["stat"]: r for r in projections.to_dicts() if r["conditional_on_playing"]}
    uncond = {r["stat"]: r for r in projections.to_dicts() if not r["conditional_on_playing"]}

    stat_order = [s.key for s in stats_for(position)]
    proj_out = []
    for stat in stat_order:
        r = cond.get(stat)
        if not r:
            continue
        spec = get_spec(position, stat)
        u = uncond.get(stat)
        proj_out.append(
            {
                "stat": stat, "label": spec.label,
                "distribution": _distribution(r),
                "unconditional": _distribution(u) if u else None,
                "high_variance": bool(r["high_variance"]),
                "play_probability": r["play_probability"],
                "settlement_note": spec.settlement_note,
            }
        )

    log = query_df(
        """
        SELECT season, week, opponent, stat, raw_value, adjusted_value, opponent_multiplier,
               opponent_rank
        FROM adjusted_game_log
        WHERE gsis_id = ? AND ((season < ?) OR (season = ? AND week < ?))
        ORDER BY season DESC, week DESC
        """,
        [gsis_id, season, season, week],
    )
    window = get_settings().recency_window
    grouped: dict[tuple[int, int], dict] = {}
    for r in log.to_dicts():
        key = (r["season"], r["week"])
        entry = grouped.setdefault(
            key,
            {
                "season": r["season"], "week": r["week"], "opponent": r["opponent"],
                "prior_season": r["season"] < season, "cells": {},
            },
        )
        if r["stat"] in stat_order:
            entry["cells"][r["stat"]] = {
                "stat": r["stat"], "raw_value": r["raw_value"],
                "adjusted_value": r["adjusted_value"],
                "opponent_multiplier": r["opponent_multiplier"],
                "opponent_rank": r["opponent_rank"],
            }
    game_log = sorted(grouped.values(), key=lambda x: (-x["season"], -x["week"]))[:window]

    metrics = list(dict.fromkeys(s.defense_metric for s in stats_for(position)))
    matchup = []
    if opponent and metrics:
        rows = query_df(
            f"""
            SELECT d.metric, d.multiplier, d.raw_multiplier, d.rank, d.percentile, d.z_score,
                   d.beta, m.mse_reduction_pct
            FROM defense_multipliers d
            LEFT JOIN metric_reliability m ON m.unit = d.unit AND m.metric = d.metric
            WHERE d.season = ? AND d.week = ? AND d.team = ?
              AND d.metric IN ({",".join("?" * len(metrics))})
            """,
            [season, week, opponent, *metrics],
        )
        for r in rows.to_dicts():
            spec = METRIC_BY_KEY.get(r["metric"])
            matchup.append(
                {
                    "metric": r["metric"], "label": spec.label if spec else r["metric"],
                    "multiplier": r["multiplier"], "raw_multiplier": r["raw_multiplier"],
                    "rank": r["rank"], "percentile": r["percentile"] or 0.0,
                    "z_score": r["z_score"] or 0.0,
                    "predictive_weight": r["beta"],
                    "mse_reduction_pct": r["mse_reduction_pct"],
                }
            )

    math_rows = query_df(
        "SELECT stat, step, section, label, value, detail FROM projection_math "
        "WHERE season = ? AND week = ? AND gsis_id = ? ORDER BY stat, step",
        [season, week, gsis_id],
    )
    math: dict[str, list[dict]] = {}
    for r in math_rows.to_dicts():
        math.setdefault(r["stat"], []).append(
            {"section": r["section"], "label": r["label"], "value": r["value"],
             "detail": r["detail"] or ""}
        )

    inj_row = query_df(
        "SELECT injury_status, practice_participation FROM injury_status "
        "WHERE gsis_id = ? ORDER BY CASE WHEN source = 'sleeper' THEN 0 ELSE 1 END LIMIT 1",
        [gsis_id],
    )
    components = json.loads(rank_row["components"][0]) if rank_row.height and rank_row["components"][0] else {}
    injury = {
        "report_status": inj_row["injury_status"][0] if inj_row.height else None,
        "practice_status": inj_row["practice_participation"][0] if inj_row.height else None,
        "p_played": (rank_row["play_probability"][0] if rank_row.height else 1.0) or 1.0,
        "expected_snap_share": components.get("prior_snap_share") or 1.0,
        "role": components.get("role", "starter"),
        "source": components.get("play_probability_source", "healthy"),
        "n_observations": 0,
    }
    narrative_row = query_df(
        "SELECT output_json FROM llm_cache WHERE cache_key = ?",
        [f"deep_dive:{gsis_id}:{season}:{week}"],
    )
    narrative = None
    if narrative_row.height:
        try:
            narrative = json.loads(narrative_row["output_json"][0]).get("summary")
        except (json.JSONDecodeError, TypeError):
            narrative = None

    return {
        "gsis_id": gsis_id, "display_name": b["display_name"], "position": position,
        "team": team, "opponent": opponent, "season": season, "week": week,
        "headshot_url": b["headshot_url"], "height": b["height"], "weight": b["weight"],
        "years_exp": b["years_exp"],
        "environment": {
            "implied_total": env.get("implied_total"),
            "spread": env.get("spread"),
            "total_line": game.get("total_line"),
            "expected_plays": env.get("expected_plays"),
            "expected_pass_rate": env.get("expected_pass_rate"),
            "expected_pass_attempts": env.get("expected_pass_attempts"),
            "expected_rush_attempts": env.get("expected_rush_attempts"),
            "expected_team_tds": env.get("expected_team_tds"),
            "roof": game.get("roof"), "wind": 0.0 if indoor else game.get("wind"),
            "temp": game.get("temp"), "indoor": indoor,
            "wind_multiplier": 1.0, "is_home": env.get("is_home"),
            "odds_source": game.get("odds_source"),
        },
        "injury": injury,
        "flags": {
            "changed_team": bool(rank_row["changed_team"][0]) if rank_row.height else False,
            "changed_coach": bool(rank_row["changed_coach"][0]) if rank_row.height else False,
            "pass_rate_shift": (rank_row["pass_rate_shift"][0] if rank_row.height else 0.0) or 0.0,
            "insufficient_history": bool(rank_row["insufficient_history"][0]) if rank_row.height else False,
            "all_history_prior_season": bool(game_log and game_log[0]["prior_season"]),
        },
        "projections": proj_out,
        "game_log": game_log,
        "matchup": matchup,
        "math": math,
        "narrative": narrative,
        "stat_order": stat_order,
    }


def defense_table(season: int, week: int, position: str, metric: str | None = None) -> dict[str, Any]:
    """Every defence graded on one metric, for the matchup view."""
    position = position.upper()
    metrics = list(dict.fromkeys(s.defense_metric for s in stats_for(position)))
    metric = metric or metrics[0]

    rows = query_df(
        "SELECT team, multiplier, raw_value, league_avg, rank, percentile, n_games "
        "FROM defense_multipliers WHERE season = ? AND week = ? AND metric = ? ORDER BY rank",
        [season, week, metric],
    )
    rel = query_df(
        "SELECT mse_reduction_pct FROM metric_reliability WHERE metric = ?", [metric]
    )
    spec = METRIC_BY_KEY.get(metric)
    return {
        "position": position, "metric": metric,
        "label": spec.label if spec else metric,
        "season": season, "week": week,
        "mse_reduction_pct": float(rel["mse_reduction_pct"][0]) if rel.height else None,
        "rows": rows.to_dicts(),
    }


def search_players(term: str, limit: int = 12) -> list[dict[str, Any]]:
    """Name search for the header's jump-to box."""
    rows = query_df(
        """
        SELECT gsis_id, display_name, position, team, headshot_url
        FROM players
        WHERE lower(display_name) LIKE lower(?)
          AND position IN ('QB','RB','WR','TE','K','LB')
        ORDER BY last_season DESC NULLS LAST, display_name
        LIMIT ?
        """,
        [f"%{term}%", limit],
    )
    return rows.to_dicts()


def calibration(run_id: str | None = None) -> dict[str, Any] | None:
    """The most recent backtest's calibration table, for the model-health page (§6)."""
    from backend.models.backtest import calibration_table, latest_run

    run_id = run_id or latest_run()
    if not run_id:
        return None

    run = query_df(
        "SELECT run_id, season, week_start, week_end FROM backtest_runs WHERE run_id = ?", [run_id]
    )
    if run.is_empty():
        return None

    table = calibration_table(run_id)
    if table.is_empty():
        return None

    rows = [
        {
            "position": r["position"], "stat": r["stat"], "n": int(r["n"]),
            "mae": r["mae"], "bias": r["bias"], "coverage": r["coverage"],
            "pit_central": r["pit_central"], "brier": r["brier"],
            "degenerate": (r.get("degenerate_rate") or 0.0) > 0.25,
        }
        for r in table.to_dicts()
    ]
    continuous = [r for r in rows if not r["degenerate"]]
    total_n = sum(r["n"] for r in continuous) or 1
    return {
        "run_id": run_id,
        "season": int(run["season"][0]),
        "week_start": int(run["week_start"][0]),
        "week_end": int(run["week_end"][0]),
        "coverage": sum(r["coverage"] * r["n"] for r in continuous) / total_n,
        "pit_central": float((table["pit_central"] * table["n"]).sum() / table["n"].sum()),
        "rows": rows,
    }


def all_boards(season: int, week: int) -> list[dict[str, Any]]:
    """Every position board in one call, so the app loads in a single round trip."""
    return [board(season, week, p.value) for p in Position]


def week_games(season: int, week: int) -> pl.DataFrame:
    """The week's schedule with lines and weather, for the environment strip."""
    return query_df(
        "SELECT * FROM game_environment WHERE season = ? AND week = ? ORDER BY kickoff, home_team",
        [season, week],
    )
