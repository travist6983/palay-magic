"""Opponent adjustment: defensive strength and opponent-adjusted game logs (§5.2).

This is the piece that tells us whether 110 receiving yards against a bottom-5 pass defense is
really a 92-yard performance. Two stages:

1. :func:`compute_unit_facts` reduces play-by-play and weekly box scores to one row per
   (team, unit, game, metric) with a numerator and a denominator.
2. :func:`compute_defense_multipliers` aggregates the trailing ``defense_window`` games as of a
   given week, converts each metric to a **multiplier relative to league average**, and shrinks it
   toward 1.0 with ``k`` games of league-average pseudo-data so a two-game sample cannot claim a
   defense allows 40% more than everyone else.

A multiplier reads the same way for every metric: **how much more of this stat happens against
this unit than against an average one.** 1.12 means 12% more. That holds even for
``int_rate_generated``, where "more" is bad for the quarterback.

Position-split allowed stats come from ``raw_player_stats`` rather than being re-derived from
play-by-play, because that table *is* the official gamebook — which is what the books settle on
(``sharp_bettor_prop_reference.md``). Play-level rates that the box score cannot express
(explosive-play rates, red-zone conversion, pace, pressure) come from ``raw_pbp``.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from backend.config import get_settings
from backend.db.connection import connect
from backend.db.views import view_exists
from backend.logging_setup import get_logger
from backend.models.baseline import shrink_multiplier
from backend.models.twoway import WEEKS_PER_SEASON, fit_two_way, tune_ridge

log = get_logger(__name__)

DEFENSE = "defense"
OFFENSE = "offense"

RATIO = "ratio"
"""Metric with a meaningful zero, where "12% more than average" is a sentence that parses."""

ADDITIVE = "additive"
"""Metric centred near zero (EPA). A ratio to league average is meaningless -- see migration 004.

Additive metrics are pinned to ``multiplier = 1.0`` so they can never scale a projection, and
carry their signal in ``z_score``.
"""


@dataclass(frozen=True)
class Metric:
    """One defensive/offensive strength metric.

    Attributes:
        key: the metric name referenced by :mod:`backend.models.stats`.
        unit: ``defense`` (what this team allows) or ``offense`` (what this team's offense does).
        position: which position's stat it adjusts, or ``ALL``.
        label: human-readable, for the UI and "Show math".
        higher_is_softer: True when a larger value means a friendlier matchup for the player whose
            prop we are projecting. Only used for how the UI colours the cell -- the multiplier
            itself always means "more of this stat happens here".
        scale: ``ratio`` or ``additive``. Additive metrics never produce a usable multiplier.
    """

    key: str
    unit: str
    position: str
    label: str
    higher_is_softer: bool = True
    scale: str = RATIO


METRICS: tuple[Metric, ...] = (
    # --- pass defense -------------------------------------------------------
    Metric("pass_volume_allowed", DEFENSE, "QB", "Pass attempts allowed / game"),
    Metric("completion_rate_allowed", DEFENSE, "QB", "Completion % allowed"),
    Metric("pass_yards_allowed", DEFENSE, "QB", "Passing yards allowed / game"),
    Metric("pass_td_rate_allowed", DEFENSE, "QB", "Passing TDs allowed / game"),
    Metric("int_rate_generated", DEFENSE, "QB", "INTs forced / attempt", higher_is_softer=False),
    Metric("explosive_pass_allowed", DEFENSE, "ALL", "20+ yard completions / completion"),
    Metric("sack_rate_generated", DEFENSE, "QB", "Sacks / dropback", higher_is_softer=False),
    # --- run defense --------------------------------------------------------
    Metric("rush_volume_allowed", DEFENSE, "ALL", "Rush attempts allowed / game"),
    Metric("rush_yards_allowed", DEFENSE, "ALL", "Rush yards allowed / game"),
    Metric("rush_yards_allowed_rb", DEFENSE, "RB", "Yards per carry allowed to RBs"),
    Metric("explosive_rush_allowed", DEFENSE, "RB", "10+ yard runs / carry"),
    Metric("rush_td_rate_allowed", DEFENSE, "ALL", "Rush TDs allowed / game"),
    # --- coverage, split by receiver position -------------------------------
    Metric("rec_volume_allowed_rb", DEFENSE, "RB", "Receptions allowed to RBs / game"),
    Metric("rec_yards_allowed_rb", DEFENSE, "RB", "Yards per target allowed to RBs"),
    Metric("target_volume_allowed_wr", DEFENSE, "WR", "Targets to WRs / game"),
    Metric("rec_volume_allowed_wr", DEFENSE, "WR", "Receptions allowed to WRs / game"),
    Metric("rec_yards_allowed_wr", DEFENSE, "WR", "Yards per target allowed to WRs"),
    Metric("target_volume_allowed_te", DEFENSE, "TE", "Targets to TEs / game"),
    Metric("rec_volume_allowed_te", DEFENSE, "TE", "Receptions allowed to TEs / game"),
    Metric("rec_yards_allowed_te", DEFENSE, "TE", "Yards per target allowed to TEs"),
    # --- scoring and kicking ------------------------------------------------
    Metric("rz_td_rate_allowed", DEFENSE, "ALL", "Red-zone TD rate allowed"),
    Metric("fg_attempts_allowed", DEFENSE, "K", "FG attempts allowed / game"),
    # --- efficiency, for the UI and the narrative ---------------------------
    Metric("epa_per_pass_allowed", DEFENSE, "ALL", "EPA per dropback allowed", scale=ADDITIVE),
    Metric("epa_per_rush_allowed", DEFENSE, "ALL", "EPA per rush allowed", scale=ADDITIVE),
    Metric("success_rate_allowed", DEFENSE, "ALL", "Success rate allowed"),
    # --- opposing OFFENCE, which is what drives LB props --------------------
    Metric("opp_plays_allowed", OFFENSE, "LB", "Offensive plays run / game"),
    Metric("opp_pass_rate", OFFENSE, "LB", "Pass rate", higher_is_softer=False),
    Metric("pressure_allowed", OFFENSE, "LB", "Sacks allowed / dropback"),
    Metric("opp_rush_volume", OFFENSE, "LB", "Rush attempts run / game"),
)

METRIC_BY_KEY: dict[str, Metric] = {m.key: m for m in METRICS}


# ---------------------------------------------------------------------------
# Stage 1 — per-game facts
# ---------------------------------------------------------------------------

# Position-split allowed stats, straight off the official gamebook (raw_player_stats).
# Each entry is (metric key, numerator SQL, denominator SQL). `1` as a denominator means per game.
_BOX_METRICS: tuple[tuple[str, str, str], ...] = (
    ("pass_volume_allowed", "sum(CASE WHEN position='QB' THEN attempts END)", "1"),
    ("completion_rate_allowed",
     "sum(CASE WHEN position='QB' THEN completions END)",
     "sum(CASE WHEN position='QB' THEN attempts END)"),
    ("pass_yards_allowed", "sum(CASE WHEN position='QB' THEN passing_yards END)", "1"),
    ("pass_td_rate_allowed", "sum(CASE WHEN position='QB' THEN passing_tds END)", "1"),
    ("int_rate_generated",
     "sum(CASE WHEN position='QB' THEN passing_interceptions END)",
     "sum(CASE WHEN position='QB' THEN attempts END)"),
    ("rush_volume_allowed", "sum(carries)", "1"),
    ("rush_yards_allowed", "sum(rushing_yards)", "1"),
    ("rush_td_rate_allowed", "sum(rushing_tds)", "1"),
    ("rush_yards_allowed_rb",
     "sum(CASE WHEN position='RB' THEN rushing_yards END)",
     "sum(CASE WHEN position='RB' THEN carries END)"),
    ("rec_volume_allowed_rb", "sum(CASE WHEN position='RB' THEN receptions END)", "1"),
    ("rec_yards_allowed_rb",
     "sum(CASE WHEN position='RB' THEN receiving_yards END)",
     "sum(CASE WHEN position='RB' THEN targets END)"),
    ("target_volume_allowed_wr", "sum(CASE WHEN position='WR' THEN targets END)", "1"),
    ("rec_volume_allowed_wr", "sum(CASE WHEN position='WR' THEN receptions END)", "1"),
    ("rec_yards_allowed_wr",
     "sum(CASE WHEN position='WR' THEN receiving_yards END)",
     "sum(CASE WHEN position='WR' THEN targets END)"),
    ("target_volume_allowed_te", "sum(CASE WHEN position='TE' THEN targets END)", "1"),
    ("rec_volume_allowed_te", "sum(CASE WHEN position='TE' THEN receptions END)", "1"),
    ("rec_yards_allowed_te",
     "sum(CASE WHEN position='TE' THEN receiving_yards END)",
     "sum(CASE WHEN position='TE' THEN targets END)"),
    ("fg_attempts_allowed", "sum(CASE WHEN position='K' THEN fg_att END)", "1"),
)

# Play-level metrics the box score cannot express, from raw_pbp.
# nflverse quirk verified on 2025: `pass_attempt` is 1 on sacks too, so official pass attempts are
# `pass_attempt - sack`. `qb_dropback` additionally includes scrambles, which is what a sack rate
# should be measured against.
_PBP_DEFENSE_METRICS: tuple[tuple[str, str, str], ...] = (
    ("explosive_pass_allowed",
     "sum(CASE WHEN complete_pass = 1 AND yards_gained >= 20 THEN 1 ELSE 0 END)",
     "sum(complete_pass)"),
    ("explosive_rush_allowed",
     "sum(CASE WHEN rush_attempt = 1 AND yards_gained >= 10 THEN 1 ELSE 0 END)",
     "sum(rush_attempt)"),
    ("sack_rate_generated", "sum(sack)", "sum(qb_dropback)"),
    ("epa_per_pass_allowed", "sum(CASE WHEN qb_dropback = 1 THEN epa END)", "sum(qb_dropback)"),
    ("epa_per_rush_allowed", "sum(CASE WHEN rush_attempt = 1 THEN epa END)", "sum(rush_attempt)"),
    ("success_rate_allowed",
     "sum(CASE WHEN (qb_dropback = 1 OR rush_attempt = 1) THEN success END)",
     "sum(CASE WHEN (qb_dropback = 1 OR rush_attempt = 1) THEN 1 ELSE 0 END)"),
)

_PBP_OFFENSE_METRICS: tuple[tuple[str, str, str], ...] = (
    ("opp_plays_allowed",
     "sum(CASE WHEN qb_dropback = 1 OR rush_attempt = 1 THEN 1 ELSE 0 END)", "1"),
    ("opp_rush_volume", "sum(rush_attempt)", "1"),
    ("opp_pass_rate", "sum(qb_dropback)",
     "sum(CASE WHEN qb_dropback = 1 OR rush_attempt = 1 THEN 1 ELSE 0 END)"),
    ("pressure_allowed", "sum(sack)", "sum(qb_dropback)"),
)


def _box_facts_sql(seasons: str) -> str:
    """Per-defense, per-game facts from the weekly box scores."""
    selects = ",\n            ".join(
        f"{num} AS n_{key}, {den} AS d_{key}" for key, num, den in _BOX_METRICS
    )
    unpivot = " UNION ALL ".join(
        f"SELECT season, week, game_id, team, opponent, '{key}' AS metric, "
        f"n_{key} AS numerator, d_{key} AS denominator FROM agg"
        for key, _, _ in _BOX_METRICS
    )
    return f"""
    WITH agg AS (
        SELECT
            season,
            week,
            game_id,
            opponent_team AS team,       -- the DEFENSE
            team           AS opponent,  -- the offence it faced
            {selects}
        FROM raw_player_stats
        WHERE season IN ({seasons}) AND season_type = 'REG' AND opponent_team IS NOT NULL
        GROUP BY 1, 2, 3, 4, 5
    )
    {unpivot}
    """


def _pbp_facts_sql(seasons: str, unit: str) -> str:
    """Per-team, per-game facts from play-by-play, for either unit."""
    metrics = _PBP_DEFENSE_METRICS if unit == DEFENSE else _PBP_OFFENSE_METRICS
    team_col = "defteam" if unit == DEFENSE else "posteam"
    opp_col = "posteam" if unit == DEFENSE else "defteam"

    selects = ",\n            ".join(
        f"{num} AS n_{key}, {den} AS d_{key}" for key, num, den in metrics
    )
    unpivot = " UNION ALL ".join(
        f"SELECT season, week, game_id, team, opponent, '{key}' AS metric, "
        f"n_{key} AS numerator, d_{key} AS denominator FROM agg"
        for key, _, _ in metrics
    )
    return f"""
    WITH agg AS (
        SELECT
            season, week, game_id,
            {team_col} AS team,
            any_value({opp_col}) AS opponent,
            {selects}
        FROM raw_pbp
        WHERE season IN ({seasons}) AND season_type = 'REG'
          AND {team_col} IS NOT NULL
          AND (qb_dropback = 1 OR rush_attempt = 1)
        GROUP BY 1, 2, 3, 4
    )
    {unpivot}
    """


def _rz_facts_sql(seasons: str) -> str:
    """Red-zone TD rate allowed, at the drive level.

    A red-zone trip is a drive that reached the opponent's 20. The numerator is the drives that
    ended in a touchdown. Drive-level, not play-level: a team that snaps six plays inside the 10
    and scores once converted one trip, not one in six.
    """
    return f"""
    WITH drives AS (
        SELECT
            season, week, game_id, defteam AS team, any_value(posteam) AS opponent,
            fixed_drive,
            max(CASE WHEN yardline_100 <= 20 THEN 1 ELSE 0 END) AS reached_rz,
            max(CASE WHEN fixed_drive_result = 'Touchdown' THEN 1 ELSE 0 END) AS scored_td
        FROM raw_pbp
        WHERE season IN ({seasons}) AND season_type = 'REG'
          AND defteam IS NOT NULL AND fixed_drive IS NOT NULL
        GROUP BY 1, 2, 3, 4, 6
    )
    SELECT season, week, game_id, team, any_value(opponent) AS opponent,
           'rz_td_rate_allowed' AS metric,
           sum(CASE WHEN reached_rz = 1 THEN scored_td ELSE 0 END)::DOUBLE AS numerator,
           sum(reached_rz)::DOUBLE AS denominator
    FROM drives
    GROUP BY 1, 2, 3, 4
    """


def compute_unit_facts(seasons: list[int] | None = None) -> int:
    """Materialise every per-(team, unit, game, metric) fact. Rebuilt in full; cheap enough.

    Returns:
        Number of fact rows written.
    """
    seasons = seasons or list(get_settings().seasons)
    season_list = ",".join(str(s) for s in seasons)

    with connect() as con:
        for view in ("raw_player_stats", "raw_pbp", "raw_schedules"):
            if not view_exists(con, view):
                log.warning("%s missing - run the nflverse ingest first", view)
                return 0

        sql = f"""
        WITH facts AS (
            SELECT season, week, game_id, team, '{DEFENSE}' AS unit, opponent, metric,
                   numerator::DOUBLE AS numerator, denominator::DOUBLE AS denominator
            FROM ({_box_facts_sql(season_list)})
            UNION ALL
            SELECT season, week, game_id, team, '{DEFENSE}', opponent, metric,
                   numerator::DOUBLE, denominator::DOUBLE
            FROM ({_pbp_facts_sql(season_list, DEFENSE)})
            UNION ALL
            SELECT season, week, game_id, team, '{DEFENSE}', opponent, metric,
                   numerator::DOUBLE, denominator::DOUBLE
            FROM ({_rz_facts_sql(season_list)})
            UNION ALL
            SELECT season, week, game_id, team, '{OFFENSE}', opponent, metric,
                   numerator::DOUBLE, denominator::DOUBLE
            FROM ({_pbp_facts_sql(season_list, OFFENSE)})
        )
        SELECT f.season, f.week, f.game_id, f.team, f.unit, f.opponent,
               s.kickoff, f.metric, f.numerator, f.denominator
        FROM facts f
        LEFT JOIN (
            SELECT game_id,
                   try_cast(gameday AS TIMESTAMP) + coalesce(
                       try_cast(gametime || ':00' AS INTERVAL), INTERVAL 0 SECOND) AS kickoff
            FROM raw_schedules
        ) s ON s.game_id = f.game_id
        WHERE f.numerator IS NOT NULL
        """

        con.execute("BEGIN TRANSACTION")
        try:
            con.execute("DELETE FROM unit_game_facts")
            con.execute(
                "INSERT INTO unit_game_facts "
                "(season, week, game_id, team, unit, opponent, kickoff, metric, numerator, denominator) "
                f"{sql}"
            )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

        n = con.execute("SELECT count(*) FROM unit_game_facts").fetchone()[0]

    log.info("unit_game_facts: %d rows across seasons %s", n, seasons)
    return int(n)


# ---------------------------------------------------------------------------
# Stage 2 — multipliers as of a week
# ---------------------------------------------------------------------------


def estimate_metric_reliability(
    seasons: list[int] | None = None,
    window: int | None = None,
    min_pairs: int = 100,
) -> pl.DataFrame:
    """Measure how much of a metric's team-to-team spread actually predicts the next game (§5.2).

    §5.2 prescribes shrinking every multiplier toward 1.0 with a flat ``k = 6`` games of
    league-average pseudo-data. One number for every metric is demonstrably wrong: over the 2025
    trailing-8 window, ``int_rate_generated`` multipliers spread with an SD of 0.25 while
    ``opp_plays_allowed`` spread with 0.03. Almost all of the first is sampling noise -- the Jets
    forced zero interceptions in eight games, which is a small sample, not a defence incapable of
    intercepting anybody -- and almost none of the second is.

    Rather than infer the split from a variance decomposition (which is badly biased for ratio
    metrics like yards-per-target, where per-game denominators are small and unequal), we measure
    it directly and out of sample:

        For every (team, metric, week) with a full trailing window, pair
            x = trailing-window rate / league average - 1      (the multiplier's claim)
            y = the NEXT game's rate / league average - 1      (what actually happened)
        and fit y = beta * x through the origin, weighted by the next game's denominator.

    ``beta`` is exactly the optimal shrinkage weight: it is how much of the claimed edge survives
    into the next game. A metric that is pure noise regresses to ``beta ~ 0`` and collapses to
    league average; a metric that is genuinely stable keeps ``beta ~ 1``. The shrunk multiplier is
    then ``1 + beta * (raw_multiplier - 1)``, and the equivalent pseudo-games ``k = n(1-beta)/beta``
    is reported so "Show math" can put a number next to the spec's 6.

    Fitting through the origin is deliberate: a defence exactly at league average must project to
    league average, so the intercept is not free.

    Args:
        seasons: seasons to learn from. Defaults to every season in the cache.
        window: trailing games, default ``settings.defense_window`` (8).
        min_pairs: below this many (x, y) pairs a metric falls back to the flat ``k``.

    Returns:
        One row per metric with ``beta``, ``r_squared``, ``n_pairs`` and the implied ``k``.
    """
    settings = get_settings()
    window = settings.defense_window if window is None else window
    seasons = seasons or list(settings.seasons)

    with connect() as con:
        facts = con.execute(
            "SELECT team, unit, metric, season, week, numerator, denominator "
            "FROM unit_game_facts WHERE season IN ({}) ".format(",".join(str(x) for x in seasons))
        ).pl()

    if facts.is_empty():
        return pl.DataFrame(
            schema={"unit": pl.Utf8, "metric": pl.Utf8, "beta": pl.Float64,
                    "r_squared": pl.Float64, "n_pairs": pl.Int64, "k": pl.Float64}
        )

    ordered = facts.sort(["team", "unit", "metric", "season", "week"]).with_columns(
        pl.int_range(pl.len()).over(["team", "unit", "metric"]).alias("game_index")
    )

    # Trailing-window numerator/denominator, EXCLUDING the current game: shift by one so x is
    # strictly prior information, exactly as it will be at prediction time.
    trailing = ordered.with_columns(
        pl.col("numerator").shift(1).rolling_sum(window, min_periods=window)
        .over(["team", "unit", "metric"]).alias("prior_num"),
        pl.col("denominator").shift(1).rolling_sum(window, min_periods=window)
        .over(["team", "unit", "metric"]).alias("prior_den"),
    ).filter(
        pl.col("prior_num").is_not_null()
        & (pl.col("prior_den") > 0)
        & (pl.col("denominator") > 0)
    )

    # League average is computed per (unit, metric) over the same rows, so beta is not contaminated
    # by a league-wide drift between seasons.
    league = trailing.group_by(["unit", "metric"]).agg(
        (pl.col("numerator").sum() / pl.col("denominator").sum()).alias("league_avg")
    )

    paired = (
        trailing.join(league, on=["unit", "metric"])
        .filter(pl.col("league_avg").abs() > 1e-9)
        .with_columns(
            ((pl.col("prior_num") / pl.col("prior_den")) / pl.col("league_avg") - 1.0).alias("x"),
            ((pl.col("numerator") / pl.col("denominator")) / pl.col("league_avg") - 1.0).alias("y"),
            pl.col("denominator").alias("w"),
        )
        .filter(pl.col("x").is_finite() & pl.col("y").is_finite())
    )

    # Weighted least squares through the origin: beta = sum(w x y) / sum(w x^2).
    fit = paired.group_by(["unit", "metric"]).agg(
        (pl.col("w") * pl.col("x") * pl.col("y")).sum().alias("sxy"),
        (pl.col("w") * pl.col("x") * pl.col("x")).sum().alias("sxx"),
        (pl.col("w") * pl.col("y") * pl.col("y")).sum().alias("syy"),
        pl.len().alias("n_pairs"),
    ).with_columns(
        pl.when(pl.col("sxx") > 1e-12)
        .then(pl.col("sxy") / pl.col("sxx"))
        .otherwise(0.0)
        .alias("beta_raw")
    ).with_columns(
        # A negative beta means the signal anti-predicts, which at these sample sizes is noise:
        # clamp to zero (fall back to league average) rather than invert the adjustment.
        pl.col("beta_raw").clip(0.0, 1.0).alias("beta"),
        pl.when(pl.col("syy") > 1e-12)
        .then(pl.col("sxy") * pl.col("sxy") / (pl.col("sxx") * pl.col("syy")))
        .otherwise(0.0)
        .alias("r_squared"),
    ).with_columns(
        pl.when(pl.col("beta") > 1e-6)
        .then(pl.lit(float(window)) * (1.0 - pl.col("beta")) / pl.col("beta"))
        .otherwise(pl.lit(1e6))
        .alias("k")
    )

    thin = fit.filter(pl.col("n_pairs") < min_pairs)
    if thin.height:
        log.info(
            "%d metric(s) had fewer than %d pairs and will use the flat k: %s",
            thin.height, min_pairs, thin["metric"].to_list(),
        )

    return fit.select("unit", "metric", "beta", "r_squared", "n_pairs", "k", "beta_raw")


def persist_metric_reliability(seasons: list[int] | None = None) -> pl.DataFrame:
    """Compute and store the per-metric shrinkage weights. Run once per refresh."""
    fit = estimate_metric_reliability(seasons)
    if fit.is_empty():
        return fit
    with connect() as con:
        con.register("rel_df", fit)
        try:
            con.execute("BEGIN TRANSACTION")
            con.execute("DELETE FROM metric_reliability")
            con.execute(
                "INSERT INTO metric_reliability "
                "(unit, metric, beta, beta_raw, r_squared, n_pairs, implied_k, computed_at) "
                "SELECT unit, metric, beta, beta_raw, r_squared, n_pairs, k, now() FROM rel_df"
            )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            con.unregister("rel_df")
    log.info("metric_reliability: %d metrics fitted", fit.height)
    return fit


def load_metric_reliability() -> dict[tuple[str, str], float]:
    """``(unit, metric) -> beta``. Empty when the fit has not been run yet."""
    with connect() as con:
        try:
            rows = con.execute("SELECT unit, metric, beta, n_pairs FROM metric_reliability").fetchall()
        except Exception:  # noqa: BLE001 - table may not exist on a fresh database
            return {}
    return {(u, m): float(b) for u, m, b, n in rows if n and n >= 100}


DEFAULT_RIDGE_LAMBDA = 25.0
"""Used when a metric has not been calibrated yet. Mid-range of the tuned values."""


def calibrate_metrics(
    seasons: list[int] | None = None,
    candidates: tuple[float, ...] = (5.0, 15.0, 40.0, 100.0, 250.0, 600.0),
) -> pl.DataFrame:
    """Tune the ridge penalty per metric by walk-forward evaluation, and persist it (§5.2).

    Expensive -- it refits every candidate penalty at every week of history -- so it runs on
    backfill and on demand (``proplab calibrate``), not on every weekly refresh. The penalties it
    produces are stable; the data behind them moves slowly.

    Also records, per metric, how much weighted MSE the model removes against a league-average
    prediction. That number is the honest answer to "how much does the matchup actually matter",
    and it is small for exactly the stats the public treats as matchup-driven: receiving yards
    per target allowed to WRs improves on league average by about 2%.

    Returns:
        One row per metric with ``ridge_lambda``, ``mse_reduction_pct`` and the raw-rate baseline.
    """
    seasons = seasons or list(get_settings().seasons)
    season_list = ",".join(str(x) for x in seasons)

    with connect() as con:
        facts = con.execute(
            "SELECT team, unit, metric, season, week, opponent, numerator, denominator "
            f"FROM unit_game_facts WHERE season IN ({season_list})"
        ).pl()

    if facts.is_empty():
        log.warning("no unit facts to calibrate against")
        return pl.DataFrame()

    reliability = estimate_metric_reliability(seasons)
    beta_by_key = {
        (r["unit"], r["metric"]): r["beta"] for r in reliability.to_dicts()
    } if not reliability.is_empty() else {}

    rows: list[dict[str, object]] = []
    for key in facts.select("unit", "metric").unique().sort(["unit", "metric"]).to_dicts():
        unit, metric = key["unit"], key["metric"]
        history = facts.filter(
            (pl.col("unit") == unit) & (pl.col("metric") == metric) & (pl.col("denominator") > 0)
        ).sort(["season", "week"])
        if history.height < 400:
            continue

        lam, _ = tune_ridge(history, metric, unit, candidates=candidates)
        beta = beta_by_key.get((unit, metric), 0.3)
        scores = _walk_forward_scores(history, metric, unit, lam, beta)
        rel = reliability.filter(
            (pl.col("unit") == unit) & (pl.col("metric") == metric)
        ).to_dicts()
        rel = rel[0] if rel else {}

        rows.append(
            {
                "unit": unit,
                "metric": metric,
                "beta": float(rel.get("beta", beta)),
                "beta_raw": float(rel.get("beta_raw", beta)),
                "r_squared": float(rel.get("r_squared", 0.0)),
                "n_pairs": int(rel.get("n_pairs", 0)),
                "implied_k": float(rel.get("k", 0.0)),
                "ridge_lambda": float(lam),
                "mse_reduction_pct": scores["ridge"],
                "raw_mse_reduction_pct": scores["raw"],
            }
        )

    frame = pl.DataFrame(rows)
    if frame.is_empty():
        return frame

    with connect() as con:
        con.register("cal_df", frame)
        try:
            con.execute("BEGIN TRANSACTION")
            con.execute("DELETE FROM metric_reliability")
            con.execute(
                "INSERT INTO metric_reliability "
                "(unit, metric, beta, beta_raw, r_squared, n_pairs, implied_k, computed_at, "
                " ridge_lambda, mse_reduction_pct, raw_mse_reduction_pct) "
                "SELECT unit, metric, beta, beta_raw, r_squared, n_pairs, implied_k, now(), "
                "       ridge_lambda, mse_reduction_pct, raw_mse_reduction_pct FROM cal_df"
            )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            con.unregister("cal_df")

    log.info("calibrated %d metrics", frame.height)
    return frame.sort("mse_reduction_pct", descending=True)


def _walk_forward_scores(
    history: pl.DataFrame, metric: str, unit: str, ridge_lambda: float, beta: float
) -> dict[str, float]:
    """% weighted-MSE reduction against a league-average prediction, for ridge and shrunk-raw."""
    import numpy as np

    window = get_settings().defense_window
    marks = history.select("season", "week").unique().sort(["season", "week"]).to_dicts()
    marks = marks[len(marks) // 3 :]
    acc = {"league": [0.0, 0.0], "raw": [0.0, 0.0], "ridge": [0.0, 0.0]}

    for mark in marks:
        season, week = mark["season"], mark["week"]
        train = history.filter(
            (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week))
        )
        test = history.filter((pl.col("season") == season) & (pl.col("week") == week))
        if train.height < 400 or test.is_empty():
            continue

        league_avg = train["numerator"].sum() / train["denominator"].sum()
        if not league_avg or abs(league_avg) < 1e-12:
            continue

        trailing = (
            train.sort(["team", "season", "week"], descending=[False, True, True])
            .with_columns(pl.int_range(pl.len()).over("team").alias("ago"))
            .filter(pl.col("ago") < window)
            .group_by("team")
            .agg(pl.col("numerator").sum(), pl.col("denominator").sum())
        )
        raw_rate = {
            t: (n / d if d > 0 else league_avg)
            for t, n, d in zip(
                trailing["team"], trailing["numerator"], trailing["denominator"], strict=True
            )
        }

        train2 = train.with_columns(
            ((season - pl.col("season")) * WEEKS_PER_SEASON + (week - pl.col("week")))
            .cast(pl.Float64)
            .alias("games_ago")
        )
        fit = fit_two_way(train2, metric, unit, ridge_lambda=ridge_lambda)

        y = (test["numerator"] / test["denominator"]).to_numpy()
        w = test["denominator"].to_numpy().astype(float)
        teams, opps = test["team"].to_list(), test["opponent"].to_list()

        preds = {
            "league": np.full(len(y), league_avg),
            "raw": np.array(
                [league_avg * (1 + beta * ((raw_rate.get(t, league_avg) / league_avg) - 1))
                 for t in teams]
            ),
        }
        if fit is None:
            preds["ridge"] = preds["league"]
        else:
            rated = fit.defense if unit == DEFENSE else fit.offense
            faced = fit.offense if unit == DEFENSE else fit.defense
            preds["ridge"] = np.array(
                [fit.intercept + rated.get(t, 0.0) + faced.get(o, 0.0)
                 for t, o in zip(teams, opps, strict=True)]
            )

        for key, pred in preds.items():
            acc[key][0] += float((w * (y - pred) ** 2).sum())
            acc[key][1] += float(w.sum())

    base = acc["league"][0] / acc["league"][1] if acc["league"][1] else 0.0
    if base <= 0:
        return {"ridge": 0.0, "raw": 0.0}
    return {
        key: 100.0 * (1.0 - (acc[key][0] / acc[key][1]) / base)
        for key in ("ridge", "raw")
        if acc[key][1]
    }


def load_ridge_lambdas() -> dict[tuple[str, str], float]:
    """``(unit, metric) -> calibrated ridge penalty``. Empty before the first calibration."""
    with connect() as con:
        try:
            rows = con.execute(
                "SELECT unit, metric, ridge_lambda FROM metric_reliability "
                "WHERE ridge_lambda IS NOT NULL"
            ).fetchall()
        except Exception:  # noqa: BLE001 - table may not exist yet
            return {}
    return {(u, m): float(v) for u, m, v in rows}


def compute_defense_multipliers(
    season: int,
    week: int,
    window: int | None = None,
    k: float | None = None,
    rebuild_facts: bool = True,
    model: str = "two_way_ridge",
) -> int:
    """Opponent multipliers for every team and metric as of ``(season, week)`` (§5.2).

    "As of" means every game played **strictly before** this week, which lets the window cross a
    season boundary. That matters right now: at 2026 Week 1 nothing has been played this season,
    so every multiplier is built from 2025 and earlier.

    §5.2 specifies a trailing-8 rate shrunk toward 1.0 with ``k = 6`` games of league-average
    pseudo-data. Walk-forward evaluation over 2023-2025 says that estimator does not work: the raw
    trailing-8 rate is **worse than simply predicting league average** on all 29 metrics
    (-5.8% to -11.3% MSE), and shrinking it by its measured predictive weight only claws back to
    roughly break-even (+0.05% to +6.3%). The problem is a confound the spec does not address --
    a defence that drew four backup quarterbacks looks elite -- plus the small-sample noise that
    shrinkage alone cannot separate from signal.

    So the production estimator is a two-way ridge model, ``rate = mu + offence + defence``, fitted
    over all prior games with recency-decayed, volume-weighted observations
    (:mod:`backend.models.twoway`). It beats the shrunk raw rate on every metric, typically by
    5-10x. The ridge penalty is calibrated per metric by :func:`calibrate_metrics`.

    The trailing-8 raw ratio is still computed and stored as ``raw_multiplier`` so "Show math" can
    show the number a naive model would have used next to the one we actually used.

    Args:
        season: the season the target week belongs to.
        week: the target week. Games in this week are excluded.
        window: trailing games for the displayed raw ratio, default 8.
        k: flat pseudo-games, used only by the ``shrunk_raw`` fallback model.
        rebuild_facts: recompute ``unit_game_facts`` first.
        model: ``two_way_ridge`` (default) or ``shrunk_raw`` to reproduce the spec's estimator.

    Returns:
        Number of multiplier rows written.
    """
    settings = get_settings()
    window = settings.defense_window if window is None else window
    k = settings.defense_shrink_k if k is None else k

    if rebuild_facts:
        with connect() as con:
            have = con.execute("SELECT count(*) FROM unit_game_facts").fetchone()[0]
        if not have:
            compute_unit_facts()

    with connect() as con:
        facts = con.execute(
            """
            SELECT team, unit, metric, season, week, game_id, opponent, numerator, denominator
            FROM unit_game_facts
            WHERE (season < ?) OR (season = ? AND week < ?)
            """,
            [season, season, week],
        ).pl()

    if facts.is_empty():
        log.warning("no unit facts before %s week %s", season, week)
        return 0

    # --- the displayed raw trailing-window ratio ---------------------------
    windowed = (
        facts.sort(
            ["team", "unit", "metric", "season", "week"], descending=[False, False, False, True, True]
        )
        .with_columns(pl.int_range(pl.len()).over(["team", "unit", "metric"]).alias("games_ago"))
        .filter(pl.col("games_ago") < window)
    )
    agg = windowed.group_by(["team", "unit", "metric"]).agg(
        pl.col("numerator").sum().alias("numerator"),
        pl.col("denominator").sum().alias("denominator"),
        pl.len().alias("n_games"),
    )
    league = agg.group_by(["unit", "metric"]).agg(
        pl.col("numerator").sum().alias("lg_num"),
        pl.col("denominator").sum().alias("lg_den"),
    )
    out = (
        agg.join(league, on=["unit", "metric"])
        .with_columns(
            pl.when(pl.col("denominator") > 0)
            .then(pl.col("numerator") / pl.col("denominator"))
            .otherwise(None)
            .alias("raw_value"),
            pl.when(pl.col("lg_den") > 0)
            .then(pl.col("lg_num") / pl.col("lg_den"))
            .otherwise(None)
            .alias("league_avg"),
        )
        .filter(pl.col("raw_value").is_not_null() & pl.col("league_avg").is_not_null())
    )
    if out.is_empty():
        log.warning("every metric was null before %s week %s", season, week)
        return 0

    # --- the two-way ridge fit, which is what the multiplier actually comes from ---
    lambdas = load_ridge_lambdas()
    betas = load_metric_reliability()
    fits: dict[tuple[str, str], object] = {}

    if model == "two_way_ridge":
        dated = facts.with_columns(
            ((season - pl.col("season")) * WEEKS_PER_SEASON + (week - pl.col("week")))
            .cast(pl.Float64)
            .alias("games_ago")
        )
        for key in facts.select("unit", "metric").unique().to_dicts():
            unit, metric = key["unit"], key["metric"]
            subset = dated.filter((pl.col("unit") == unit) & (pl.col("metric") == metric))
            lam = lambdas.get((unit, metric), DEFAULT_RIDGE_LAMBDA)
            fit = fit_two_way(subset, metric, unit, ridge_lambda=lam)
            if fit is not None:
                fits[(unit, metric)] = fit

    scales = {m.key: m.scale for m in METRICS}
    rows = out.to_dicts()
    for r in rows:
        unit, metric = r["unit"], r["metric"]
        scale = scales.get(metric, RATIO)
        r["scale"] = scale
        r["season"] = season
        r["week"] = week
        r["beta"] = betas.get((unit, metric))
        r["ridge_lambda"] = lambdas.get((unit, metric), DEFAULT_RIDGE_LAMBDA)
        r["raw_multiplier"] = (
            r["raw_value"] / r["league_avg"] if r["league_avg"] not in (0, None) else 1.0
        )

        if scale == ADDITIVE:
            # EPA and friends are centred near zero; a ratio to league average is nonsense
            # (migration 004). Pin the multiplier so nothing can divide by it.
            r["raw_multiplier"] = 1.0
            r["multiplier"] = 1.0
            r["effective_k"] = None
            r["model"] = "pinned"
            continue

        fit = fits.get((unit, metric))
        if fit is not None:
            r["multiplier"] = (
                fit.defense_multiplier(r["team"])
                if unit == DEFENSE
                else fit.offense_multiplier(r["team"])
            )
            r["model"] = "two_way_ridge"
            # Equivalent pseudo-games, purely so "Show math" can compare against the spec's k = 6.
            gap = r["raw_multiplier"] - 1.0
            implied = (r["multiplier"] - 1.0) / gap if abs(gap) > 1e-9 else None
            n = int(r["n_games"])
            r["effective_k"] = (
                n * (1.0 - implied) / implied if implied and implied > 1e-6 else None
            )
        else:
            r["multiplier"] = shrink_multiplier(
                raw_value=r["raw_value"],
                league_average=r["league_avg"],
                n_games=int(r["n_games"]),
                k=k,
            )
            r["model"] = "shrunk_raw"
            r["effective_k"] = float(k)

    frame = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("metric")
        .replace_strict({m.key: m.position for m in METRICS}, default="ALL", return_dtype=pl.Utf8)
        .alias("position"),
    )

    frame = frame.with_columns(
        (
            (pl.col("raw_value") - pl.col("raw_value").mean().over("metric"))
            / pl.col("raw_value").std().over("metric")
        ).alias("z_score"),
        (pl.col("raw_value").rank("average").over("metric") / pl.len().over("metric")).alias(
            "percentile"
        ),
        # 1 = allows the most of this stat, i.e. the softest matchup. Ranked on the model's own
        # multiplier, not the raw rate, so the rank the UI shows is the one we actually used.
        pl.col("multiplier").rank("dense", descending=True).over("metric").cast(pl.Int32).alias("rank"),
    ).with_columns(pl.col("z_score").fill_nan(0.0).fill_null(0.0))

    with connect() as con:
        con.register("mult_df", frame)
        try:
            con.execute("BEGIN TRANSACTION")
            con.execute(
                "DELETE FROM defense_multipliers WHERE season = ? AND week = ?", [season, week]
            )
            con.execute(
                """
                INSERT INTO defense_multipliers
                    (season, week, team, unit, position, metric, numerator, denominator,
                     raw_value, league_avg, multiplier, raw_multiplier, n_games, rank, computed_at,
                     scale, z_score, percentile, effective_k, beta, ridge_lambda, model)
                SELECT season, week, team, unit, position, metric, numerator, denominator,
                       raw_value, league_avg, multiplier, raw_multiplier, n_games, rank, now(),
                       scale, z_score, percentile, effective_k, beta, ridge_lambda, model
                FROM mult_df
                """
            )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            con.unregister("mult_df")

    log.info(
        "defense_multipliers: %s week %s -> %d rows (%d teams x %d metrics, model=%s)",
        season, week, frame.height, frame["team"].n_unique(), frame["metric"].n_unique(), model,
    )
    return frame.height


def multiplier_lookup(season: int, week: int) -> dict[tuple[str, str], float]:
    """``(team, metric) -> multiplier`` for one week. One query, used by every projection."""
    with connect() as con:
        rows = con.execute(
            "SELECT team, metric, multiplier FROM defense_multipliers WHERE season = ? AND week = ?",
            [season, week],
        ).fetchall()
    return {(t, m): float(v) for t, m, v in rows}


def get_multiplier(
    lookup: dict[tuple[str, str], float], team: str | None, metric: str, default: float = 1.0
) -> float:
    """One multiplier, defaulting to neutral when the matchup is unknown."""
    if not team:
        return default
    return lookup.get((team, metric), default)


# ---------------------------------------------------------------------------
# Stage 3 — opponent-adjusted game logs (§5.2)
# ---------------------------------------------------------------------------


def build_adjusted_game_logs(
    season: int,
    week: int,
    lookback: int | None = None,
    positions: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K", "LB"),
) -> int:
    """Raw and opponent-adjusted values for every recent game (§5.2).

        adjusted_stat = raw_stat / opponent_multiplier

    So 110 receiving yards against a defence that allows 7% more than average is a 103-yard
    performance, and the same 110 against one that allows 7% less is 118.

    **Which multiplier.** The log is scored with the multipliers computed *as of the target week*,
    which is the best estimate of each opponent's strength given everything known at projection
    time. That is a retrodiction, and it is the right one: the question the log answers is "given
    what we now understand about these defences, how good was this performance?" Those multipliers
    are still built only from games before the target week, so the backtest stays honest -- freeze
    the week and the same rule reproduces exactly what we would have had.

    Args:
        season: target season.
        week: target week; multipliers are taken as of here.
        lookback: how many recent games per player, default ``settings.recency_window`` (6).
        positions: which positions to build logs for.

    Returns:
        Number of rows written.
    """
    from backend.models.stats import STAT_SPECS

    settings = get_settings()
    lookback = settings.recency_window if lookback is None else lookback

    # (position, stat) -> the defensive metric that adjusts it, from the canonical registry.
    mapping = pl.DataFrame(
        [
            {"position": pos, "stat": spec.key, "metric": spec.defense_metric}
            for spec in STAT_SPECS
            for pos in spec.positions
            if pos in positions
        ]
    )
    if mapping.is_empty():
        return 0

    with connect() as con:
        con.register("stat_metric_map", mapping)
        try:
            con.execute("BEGIN TRANSACTION")
            con.execute(
                "DELETE FROM adjusted_game_log WHERE season <= ? OR (season = ? AND week <= ?)",
                [season, season, week],
            )
            con.execute(
                f"""
                INSERT INTO adjusted_game_log
                    (gsis_id, season, week, game_id, team, opponent, position, stat, raw_value,
                     opponent_multiplier, adjusted_value, opponent_rank, metric, computed_at)
                WITH recent AS (
                    SELECT g.*,
                           row_number() OVER (
                               PARTITION BY g.gsis_id, g.stat
                               ORDER BY g.season DESC, g.week DESC
                           ) AS games_ago
                    FROM player_game_stats g
                    WHERE ((g.season < ?) OR (g.season = ? AND g.week < ?))
                      AND g.position IN ({",".join("?" * len(positions))})
                )
                SELECT
                    r.gsis_id, r.season, r.week, r.game_id, r.team, r.opponent, r.position,
                    r.stat, r.value,
                    coalesce(d.multiplier, 1.0),
                    r.value / nullif(coalesce(d.multiplier, 1.0), 0),
                    d.rank,
                    m.metric,
                    now()
                FROM recent r
                JOIN stat_metric_map m ON m.position = r.position AND m.stat = r.stat
                LEFT JOIN defense_multipliers d
                       ON d.season = ? AND d.week = ? AND d.team = r.opponent AND d.metric = m.metric
                WHERE r.games_ago <= ?
                """,
                [season, season, week, *positions, season, week, lookback],
            )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            con.unregister("stat_metric_map")

        n = con.execute("SELECT count(*) FROM adjusted_game_log").fetchone()[0]

    log.info("adjusted_game_log: %d rows for %s week %s (lookback %d)", n, season, week, lookback)
    return int(n)


def print_defense_report(
    season: int, week: int, position: str | None = None, top: int = 5
) -> None:
    """Print the softest and toughest defences vs. each position (§9 milestone-2 checkpoint)."""
    from rich.console import Console
    from rich.table import Table

    from backend.models.stats import Position, stats_for

    console = Console()
    positions = [position.upper()] if position else [p.value for p in Position]

    with connect() as con:
        reliability = {
            (u, m): (b, r)
            for u, m, b, r in con.execute(
                "SELECT unit, metric, beta, mse_reduction_pct FROM metric_reliability"
            ).fetchall()
        }

        for pos in positions:
            metrics = list(dict.fromkeys(s.defense_metric for s in stats_for(pos)))
            for metric in metrics:
                rows = con.execute(
                    """
                    SELECT team, multiplier, raw_multiplier, raw_value, league_avg, rank, unit
                    FROM defense_multipliers
                    WHERE season = ? AND week = ? AND metric = ?
                    ORDER BY rank
                    """,
                    [season, week, metric],
                ).fetchall()
                if not rows:
                    continue

                unit = rows[0][6]
                beta, mse = reliability.get((unit, metric), (None, None))
                spec = METRIC_BY_KEY.get(metric)
                title = f"{pos} · {spec.label if spec else metric}"
                if mse is not None:
                    title += f"   [dim](removes {mse:.1f}% of MSE vs league average)[/dim]"

                table = Table(title=title, title_justify="left", show_edge=False, pad_edge=False)
                table.add_column("softest", style="red")
                table.add_column("mult", justify="right")
                table.add_column("raw", justify="right")
                table.add_column("   ")
                table.add_column("toughest", style="green")
                table.add_column("mult", justify="right")
                table.add_column("raw", justify="right")

                head, tail = rows[:top], rows[-top:][::-1]
                for soft, tough in zip(head, tail, strict=False):
                    table.add_row(
                        soft[0], f"{soft[1]:.3f}", f"{soft[3]:.2f}", "",
                        tough[0], f"{tough[1]:.3f}", f"{tough[3]:.2f}",
                    )
                console.print(table)
                console.print()
