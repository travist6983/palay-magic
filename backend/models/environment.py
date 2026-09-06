"""Game environment: how many plays, how pass-heavy, how many scores (§5.3).

The market line is the anchor. ``implied_team_total = (total ± spread) / 2`` converts a spread and
a total into each side's expected points, and everything downstream — expected plays, pass
attempts, rush attempts, drives, red-zone trips, touchdowns, field goals — is derived from there.

§5.3 specifies the *shape* of these relationships but not their magnitudes. Rather than hard-code
constants, every relationship is **fitted on 2023-2025 team-games** and the coefficients stored in
``environment_models`` so "Show math" can show them and the backtest can refit them. In particular
the game-script rule ("favourites of 7+ shift toward run, underdogs of 7+ shift toward pass") is
not asserted — it falls out of the fitted spread coefficient, and the fit reports how large it
actually is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from backend.config import get_settings
from backend.db.connection import connect
from backend.logging_setup import get_logger

log = get_logger(__name__)

INDOOR_ROOFS = frozenset({"dome", "closed"})

# Points per touchdown drive including the kick, allowing for missed XPs and two-point tries.
POINTS_PER_TD = 6.95


@dataclass(frozen=True)
class LinearModel:
    """A fitted linear model: feature name -> coefficient, plus fit diagnostics."""

    name: str
    terms: dict[str, float]
    n_observations: int
    r_squared: float
    rmse: float

    def predict(self, features: dict[str, float]) -> float:
        """Evaluate the model. Missing features are treated as zero, the intercept always applies."""
        out = self.terms.get("intercept", 0.0)
        for term, coef in self.terms.items():
            if term == "intercept":
                continue
            out += coef * float(features.get(term, 0.0))
        return out


def _fit(name: str, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> LinearModel:
    """Ordinary least squares with an intercept prepended."""
    design = np.column_stack([np.ones(len(y)), X])
    coefs, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coefs
    resid = y - pred
    ss_res = float((resid**2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return LinearModel(
        name=name,
        terms={"intercept": float(coefs[0])} | dict(zip(feature_names, map(float, coefs[1:]), strict=True)),
        n_observations=len(y),
        r_squared=1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        rmse=float(np.sqrt(ss_res / len(y))) if len(y) else 0.0,
    )


def _pace_by_team_game(season_filter: str, params: list[Any] | None = None) -> pl.DataFrame:
    """Median snap-to-snap seconds per (season, week, team). ONE definition for fit and serve.

    The plays model was fitted on a rolling mean of per-game medians but served an all-time mean
    over every play, a different statistic by ~5 seconds -- which biased expected plays by +2.3 a
    game for every team. Both paths now call this.
    """
    sql = f"""
    SELECT season, week, posteam AS team,
           median(CASE WHEN play_clock_seconds BETWEEN 1 AND 60 THEN play_clock_seconds END) AS med_clock
    FROM (
        SELECT season, week, posteam,
               lag(game_seconds_remaining) OVER (
                   PARTITION BY game_id, posteam, fixed_drive ORDER BY play_id
               ) - game_seconds_remaining AS play_clock_seconds
        FROM raw_pbp
        WHERE {season_filter} AND season_type = 'REG' AND posteam IS NOT NULL
          AND (qb_dropback = 1 OR rush_attempt = 1)
    )
    GROUP BY 1, 2, 3
    """
    with connect() as con:
        return con.execute(sql, params or []).pl()


def measure_constants(seasons: list[int]) -> dict[str, float]:
    """The scalar ratios the usage model needs, measured rather than assumed.

    * ``attempts_per_dropback`` -- official attempts / dropbacks (0.88; was hard-coded 0.885).
    * ``targets_per_attempt`` -- official targets / attempts (0.953). target_share is defined
      against team TARGETS, so multiplying it by expected ATTEMPTS overstated every target
      projection by ~4.9%.
    * ``scrambles_per_dropback`` -- carry_share is measured against official carries, which
      include scrambles; plays minus dropbacks does not, so RB carries came out ~7% low.
    * ``xp_made_per_td`` -- extra points MADE per offensive touchdown (0.905). The old 0.94 was
      attempts, which omitted the make rate.
    * ``rz_trips_per_drive`` -- measured 0.31, not the assumed 0.38.
    """
    season_list = ",".join(str(x) for x in seasons)
    with connect() as con:
        row = con.execute(
            f"""
            SELECT
              sum(CASE WHEN pass_attempt = 1 AND sack = 0 AND coalesce(two_point_attempt,0) = 0 THEN 1 ELSE 0 END)::DOUBLE
                / nullif(sum(qb_dropback), 0)                                            AS attempts_per_dropback,
              sum(CASE WHEN qb_dropback = 1 AND rush_attempt = 1 THEN 1 ELSE 0 END)::DOUBLE
                / nullif(sum(qb_dropback), 0)                                            AS scrambles_per_dropback,
              sum(CASE WHEN extra_point_result = 'good' THEN 1 ELSE 0 END)::DOUBLE
                / nullif(sum(CASE WHEN touchdown = 1 AND td_team = posteam
                                   AND coalesce(play_type,'') <> 'kickoff' THEN 1 ELSE 0 END), 0) AS xp_made_per_td
            FROM raw_pbp WHERE season IN ({season_list}) AND season_type = 'REG' AND posteam IS NOT NULL
            """
        ).fetchone()
        tgt = con.execute(
            f"""
            SELECT sum(targets)::DOUBLE / nullif(sum(attempts), 0)
            FROM raw_player_stats WHERE season IN ({season_list}) AND season_type = 'REG'
            """
        ).fetchone()
        rz = con.execute(
            f"""
            WITH d AS (
                SELECT game_id, fixed_drive,
                       max(CASE WHEN yardline_100 <= 20 AND (pass_attempt = 1 OR rush_attempt = 1
                                 OR field_goal_attempt = 1) AND coalesce(extra_point_attempt,0) = 0
                                THEN 1 ELSE 0 END) AS rz
                FROM raw_pbp
                WHERE season IN ({season_list}) AND season_type = 'REG' AND posteam IS NOT NULL
                  AND (qb_dropback = 1 OR rush_attempt = 1 OR field_goal_attempt = 1)
                GROUP BY 1, 2
            ) SELECT avg(rz) FROM d
            """
        ).fetchone()
    return {
        "attempts_per_dropback": float(row[0] or 0.881),
        "scrambles_per_dropback": float(row[1] or 0.055),
        "xp_made_per_td": float(row[2] or 0.905),
        "targets_per_attempt": float(tgt[0] or 0.953),
        "rz_trips_per_drive": float(rz[0] or 0.31),
    }


def _training_frame(seasons: list[int]) -> pl.DataFrame:
    """One row per team-game with the market line, what the team actually did, and its recent form."""
    season_list = ",".join(str(s) for s in seasons)
    sql = f"""
    WITH team_game AS (
        SELECT
            p.season, p.week, p.game_id, p.posteam AS team, p.defteam AS opponent,
            sum(CASE WHEN p.qb_dropback = 1 OR p.rush_attempt = 1 THEN 1 ELSE 0 END) AS plays,
            sum(p.qb_dropback)   AS dropbacks,
            sum(p.rush_attempt)  AS rush_attempts,
            sum(CASE WHEN p.pass_attempt = 1 AND p.sack = 0 THEN 1 ELSE 0 END) AS pass_attempts,
            avg(p.xpass)         AS mean_xpass,
            count(DISTINCT p.fixed_drive) AS drives,
            sum(CASE WHEN p.yardline_100 <= 20 THEN 1 ELSE 0 END) AS rz_plays
        FROM raw_pbp p
        WHERE p.season IN ({season_list}) AND p.season_type = 'REG' AND p.posteam IS NOT NULL
          AND (p.qb_dropback = 1 OR p.rush_attempt = 1)
        GROUP BY 1, 2, 3, 4, 5
    ),
    scoring AS (
        SELECT season, week, game_id, posteam AS team,
               -- Kickoff rows carry the RECEIVING team as posteam, so a kick-return TD would
               -- be credited to the offence's touchdown count that feeds every TD lambda.
               sum(CASE WHEN touchdown = 1 AND td_team = posteam
                         AND coalesce(play_type, '') <> 'kickoff' THEN 1 ELSE 0 END) AS tds,
               sum(CASE WHEN field_goal_result = 'made' THEN 1 ELSE 0 END)          AS fgs,
               sum(CASE WHEN field_goal_attempt = 1 THEN 1 ELSE 0 END)              AS fg_atts
        FROM raw_pbp
        WHERE season IN ({season_list}) AND season_type = 'REG' AND posteam IS NOT NULL
        GROUP BY 1, 2, 3, 4
    ),
    lines AS (
        SELECT game_id, season, week, home_team, away_team, spread_line, total_line,
               roof, temp, wind
        FROM raw_schedules
        WHERE season IN ({season_list}) AND game_type = 'REG'
    )
    SELECT
        tg.season, tg.week, tg.game_id, tg.team, tg.opponent,
        tg.plays, tg.dropbacks, tg.rush_attempts, tg.pass_attempts, tg.mean_xpass,
        tg.drives, tg.rz_plays,
        s.tds, s.fgs, s.fg_atts,
        l.total_line,
        -- The team's OWN spread: negative when it is favoured.
        CASE WHEN tg.team = l.home_team THEN -l.spread_line ELSE l.spread_line END AS spread,
        l.roof, l.temp, l.wind,
        (tg.team = l.home_team) AS is_home
    FROM team_game tg
    LEFT JOIN scoring s ON s.season=tg.season AND s.week=tg.week AND s.game_id=tg.game_id AND s.team=tg.team
    JOIN lines        l  ON l.game_id = tg.game_id
    WHERE l.total_line IS NOT NULL AND l.spread_line IS NOT NULL
    """
    with connect() as con:
        frame = con.execute(sql).pl()
    pace = _pace_by_team_game(f"season IN ({season_list})")
    return frame.join(pace, on=["season", "week", "team"], how="left")


def fit_environment_models(seasons: list[int] | None = None) -> dict[str, LinearModel]:
    """Fit and persist the four environment models (§5.3).

    * ``plays`` — offensive plays run, from the game total, the absolute spread (blowouts shorten
      games) and the team's own recent pace.
    * ``pass_rate`` — dropbacks / plays, from the team's recent pass rate, its expected pass rate
      (nflverse ``xpass``, which already conditions on down/distance/score/time) and its spread.
      The spread coefficient **is** the game-script rule of §5.3, measured rather than assumed.
    * ``team_tds`` / ``team_fgs`` — from the implied team total. Splitting the total into
      touchdowns and field goals is what turns a market line into an anytime-TD lambda and a
      kicking-points projection.

    Returns:
        The fitted models, keyed by name.
    """
    seasons = seasons or [s for s in get_settings().seasons]
    df = _training_frame(seasons)
    if df.height < 500:
        log.warning("only %d team-games available; environment models not fitted", df.height)
        return {}

    df = df.with_columns(
        (pl.col("total_line") / 2 - pl.col("spread") / 2).alias("implied_total"),
        (pl.col("dropbacks") / pl.col("plays")).alias("pass_rate"),
        pl.col("spread").abs().alias("abs_spread"),
    ).filter(pl.col("plays") > 20)

    # Trailing form, strictly prior: what we would actually know at projection time.
    df = df.sort(["team", "season", "week"]).with_columns(
        pl.col("plays").shift(1).rolling_mean(6, min_periods=2).over("team").alias("prior_plays"),
        pl.col("pass_rate").shift(1).rolling_mean(6, min_periods=2).over("team").alias("prior_pass_rate"),
        pl.col("mean_xpass").shift(1).rolling_mean(6, min_periods=2).over("team").alias("prior_xpass"),
        pl.col("med_clock").shift(1).rolling_mean(6, min_periods=2).over("team").alias("prior_sec_per_play"),
        pl.col("rush_attempts").shift(1).rolling_mean(6, min_periods=2).over("team").alias("prior_rush_attempts"),
    )

    models: dict[str, LinearModel] = {}

    plays_df = df.drop_nulls(["prior_plays", "prior_sec_per_play"])
    models["plays"] = _fit(
        "plays",
        plays_df.select("prior_plays", "total_line", "abs_spread", "prior_sec_per_play").to_numpy(),
        plays_df["plays"].to_numpy().astype(float),
        ["prior_plays", "total_line", "abs_spread", "prior_sec_per_play"],
    )

    pr_df = df.drop_nulls(["prior_pass_rate", "prior_xpass"])
    models["pass_rate"] = _fit(
        "pass_rate",
        pr_df.select("prior_pass_rate", "prior_xpass", "spread", "total_line").to_numpy(),
        pr_df["pass_rate"].to_numpy().astype(float),
        ["prior_pass_rate", "prior_xpass", "spread", "total_line"],
    )

    td_df = df.drop_nulls(["tds"])
    models["team_tds"] = _fit(
        "team_tds",
        td_df.select("implied_total").to_numpy(),
        td_df["tds"].to_numpy().astype(float),
        ["implied_total"],
    )

    fg_df = df.drop_nulls(["fg_atts"])
    models["team_fg_attempts"] = _fit(
        "team_fg_attempts",
        fg_df.select("implied_total").to_numpy(),
        fg_df["fg_atts"].to_numpy().astype(float),
        ["implied_total"],
    )
    models["team_fgs"] = _fit(
        "team_fgs",
        fg_df.select("implied_total").to_numpy(),
        fg_df["fgs"].to_numpy().astype(float),
        ["implied_total"],
    )

    # Team rush attempts INCLUDING scrambles, on the same footing carry_share is measured on.
    rush_df = df.drop_nulls(["prior_rush_attempts"])
    models["rush_attempts"] = _fit(
        "rush_attempts",
        rush_df.select("prior_rush_attempts", "spread", "total_line").to_numpy(),
        rush_df["rush_attempts"].to_numpy().astype(float),
        ["prior_rush_attempts", "spread", "total_line"],
    )

    drive_df = df.drop_nulls(["drives"])
    models["drives"] = _fit(
        "drives",
        drive_df.select("total_line", "abs_spread").to_numpy(),
        drive_df["drives"].to_numpy().astype(float),
        ["total_line", "abs_spread"],
    )

    rows = [
        {
            "model": m.name, "term": term, "coefficient": coef,
            "n_observations": m.n_observations, "r_squared": m.r_squared, "rmse": m.rmse,
        }
        for m in models.values()
        for term, coef in m.terms.items()
    ]
    for term, value in measure_constants(seasons).items():
        rows.append({"model": "constants", "term": term, "coefficient": value,
                     "n_observations": df.height, "r_squared": None, "rmse": None})
    frame = pl.DataFrame(rows)
    with connect() as con:
        con.register("env_df", frame)
        try:
            con.execute("BEGIN TRANSACTION")
            con.execute("DELETE FROM environment_models WHERE model NOT LIKE 'td_%'")
            con.execute(
                "INSERT INTO environment_models "
                "(model, term, coefficient, n_observations, r_squared, rmse, computed_at) "
                "SELECT model, term, coefficient, n_observations, r_squared, rmse, now() FROM env_df"
            )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            con.unregister("env_df")

    for m in models.values():
        log.info("%s: R2=%.3f rmse=%.2f n=%d %s", m.name, m.r_squared, m.rmse, m.n_observations, m.terms)
    return models


def load_environment_models() -> dict[str, LinearModel]:
    """Read the fitted coefficients back. Empty before the first fit."""
    with connect() as con:
        try:
            rows = con.execute(
                "SELECT model, term, coefficient, n_observations, r_squared, rmse "
                "FROM environment_models"
            ).fetchall()
        except Exception:  # noqa: BLE001 - table may not exist yet
            return {}

    grouped: dict[str, dict] = {}
    for model, term, coef, n, r2, rmse in rows:
        entry = grouped.setdefault(model, {"terms": {}, "n": n, "r2": r2, "rmse": rmse})
        entry["terms"][term] = float(coef)
    return {
        name: LinearModel(name, e["terms"], int(e["n"] or 0), float(e["r2"] or 0), float(e["rmse"] or 0))
        for name, e in grouped.items()
    }


_DEFAULT_CONSTANTS = {
    "attempts_per_dropback": 0.881,
    "scrambles_per_dropback": 0.055,
    "xp_made_per_td": 0.905,
    "targets_per_attempt": 0.953,
    "rz_trips_per_drive": 0.31,
}


def load_constants() -> dict[str, float]:
    """The measured ratios, with the 2023-25 values as a fallback before the first fit."""
    out = dict(_DEFAULT_CONSTANTS)
    with connect() as con:
        try:
            rows = con.execute(
                "SELECT term, coefficient FROM environment_models WHERE model = 'constants'"
            ).fetchall()
        except Exception:  # noqa: BLE001 - table may not exist yet
            return out
    for term, value in rows:
        if value is not None:
            out[term] = float(value)
    return out


def build_team_environment(season: int, week: int) -> int:
    """Populate ``team_environment`` for every team playing this week (§5.3).

    Wind is forced to 0 for a dome or closed roof (§5.5). An outdoor game with no forecast yet
    leaves wind null rather than inventing a value; the ESPN weather ingest fills it in later.

    Returns:
        Number of rows written.
    """
    models = load_environment_models()
    if not models:
        models = fit_environment_models()
    if not models:
        log.warning("environment models unavailable; team_environment not built")
        return 0

    with connect() as con:
        games = con.execute(
            """
            SELECT game_id, season, week, home_team, away_team, spread_line, total_line,
                   home_implied_total, away_implied_total, roof, wind, temp
            FROM game_environment WHERE season = ? AND week = ?
            """,
            [season, week],
        ).pl()

        form = con.execute(
            """
            WITH team_game AS (
                SELECT season, week, posteam AS team,
                       sum(CASE WHEN qb_dropback = 1 OR rush_attempt = 1 THEN 1 ELSE 0 END) AS plays,
                       sum(qb_dropback) AS dropbacks,
                       avg(xpass) AS mean_xpass
                FROM raw_pbp
                WHERE season_type = 'REG' AND posteam IS NOT NULL
                  AND (qb_dropback = 1 OR rush_attempt = 1)
                  AND ((season < ?) OR (season = ? AND week < ?))
                GROUP BY 1, 2, 3
            ),
            ranked AS (
                SELECT *, row_number() OVER (PARTITION BY team ORDER BY season DESC, week DESC) AS ago
                FROM team_game
            )
            SELECT team,
                   avg(plays)                          AS prior_plays,
                   avg(dropbacks::DOUBLE / plays)      AS prior_pass_rate,
                   avg(mean_xpass)                     AS prior_xpass,
                   count(*)                            AS n_games
            FROM ranked WHERE ago <= 6 GROUP BY 1
            """,
            [season, season, week],
        ).pl()

        prior_rush = con.execute(
            """
            SELECT posteam AS team, avg(r) AS prior_rush_attempts FROM (
                SELECT posteam, season, week, sum(rush_attempt) AS r,
                       row_number() OVER (PARTITION BY posteam ORDER BY season DESC, week DESC) AS ago
                FROM raw_pbp
                WHERE season_type = 'REG' AND posteam IS NOT NULL
                  AND ((season < ?) OR (season = ? AND week < ?))
                GROUP BY 1, 2, 3
            ) WHERE ago <= 6 GROUP BY 1
            """,
            [season, season, week],
        ).pl()

    # Rolling mean of the last six per-game MEDIANS -- the statistic the plays model was fitted on.
    pace_games = _pace_by_team_game(
        "((season < ?) OR (season = ? AND week < ?))", [season, season, week]
    )
    pace = (
        pace_games.sort(["team", "season", "week"], descending=[False, True, True])
        .with_columns(pl.int_range(pl.len()).over("team").alias("ago"))
        .filter(pl.col("ago") < 6)
        .group_by("team")
        .agg(pl.col("med_clock").mean().alias("prior_sec_per_play"))
    )

    if games.is_empty():
        log.warning("no game_environment rows for %s week %s", season, week)
        return 0

    form_map = {r["team"]: r for r in form.to_dicts()}
    pace_map = {r["team"]: r["prior_sec_per_play"] for r in pace.to_dicts()}
    rush_map = {r["team"]: r["prior_rush_attempts"] for r in prior_rush.to_dicts()}
    league_pace = float(pace["prior_sec_per_play"].mean()) if pace.height else 35.0
    league_plays = float(form["prior_plays"].mean()) if form.height else 62.0
    league_pass_rate = float(form["prior_pass_rate"].mean()) if form.height else 0.57
    league_rush = float(prior_rush["prior_rush_attempts"].mean()) if prior_rush.height else 27.0
    constants = load_constants()

    rows: list[dict[str, object]] = []
    for g in games.to_dicts():
        indoor = (g["roof"] or "").lower() in INDOOR_ROOFS
        wind = 0.0 if indoor else g["wind"]

        for side in ("home", "away"):
            team = g[f"{side}_team"]
            opponent = g["away_team"] if side == "home" else g["home_team"]
            implied = g[f"{side}_implied_total"]
            # nflverse spread_line is positive when the HOME team is favoured.
            spread = -g["spread_line"] if side == "home" else g["spread_line"]

            f = form_map.get(team, {})
            prior_plays = f.get("prior_plays") or league_plays
            prior_pass_rate = f.get("prior_pass_rate") or league_pass_rate
            prior_xpass = f.get("prior_xpass") or league_pass_rate
            sec_per_play = pace_map.get(team) or league_pace

            features = {
                "prior_plays": prior_plays,
                "prior_pass_rate": prior_pass_rate,
                "prior_xpass": prior_xpass,
                "prior_sec_per_play": sec_per_play,
                "prior_rush_attempts": rush_map.get(team) or league_rush,
                "total_line": g["total_line"] or 44.0,
                "spread": spread or 0.0,
                "abs_spread": abs(spread or 0.0),
                "implied_total": implied or 22.0,
            }

            plays = max(35.0, models["plays"].predict(features))
            pass_rate = float(np.clip(models["pass_rate"].predict(features), 0.30, 0.80))
            drives = max(6.0, models["drives"].predict(features))
            team_tds = max(0.2, models["team_tds"].predict(features))
            fg_atts = max(0.2, models["team_fg_attempts"].predict(features))

            dropbacks = plays * pass_rate
            # Attempts are dropbacks less sacks and scrambles; the ratio is measured, not assumed.
            pass_attempts = dropbacks * constants["attempts_per_dropback"]
            # Official team carries include scrambles. plays - dropbacks does not, and carry_share
            # is measured against the official count, so the rush model is fitted on that count.
            rush_attempts = (
                max(8.0, models["rush_attempts"].predict(features))
                if "rush_attempts" in models
                else plays - dropbacks + dropbacks * constants["scrambles_per_dropback"]
            )

            rows.append(
                {
                    "season": season, "week": week, "team": team, "opponent": opponent,
                    "game_id": g["game_id"], "is_home": side == "home",
                    "implied_total": implied, "spread": spread,
                    "sec_per_play": sec_per_play,
                    "expected_plays": plays,
                    "base_pass_rate": prior_pass_rate,
                    "proe": prior_pass_rate - prior_xpass,
                    "script_adjustment": pass_rate - prior_pass_rate,
                    "expected_pass_rate": pass_rate,
                    "expected_pass_attempts": pass_attempts,
                    "expected_rush_attempts": rush_attempts,
                    "expected_dropbacks": dropbacks,
                    "expected_drives": drives,
                    "expected_rz_trips": drives * constants["rz_trips_per_drive"],
                    "expected_team_tds": team_tds,
                    "expected_team_fgs": fg_atts,
                    "wind": wind,
                }
            )

    frame = pl.DataFrame(rows)
    with connect() as con:
        con.register("te_df", frame)
        try:
            con.execute("BEGIN TRANSACTION")
            con.execute("DELETE FROM team_environment WHERE season = ? AND week = ?", [season, week])
            con.execute(
                """
                INSERT INTO team_environment
                    (season, week, team, opponent, game_id, is_home, implied_total, spread,
                     sec_per_play, expected_plays, base_pass_rate, proe, script_adjustment,
                     expected_pass_rate, expected_pass_attempts, expected_rush_attempts,
                     expected_dropbacks, expected_drives, expected_rz_trips, expected_team_tds,
                     expected_team_fgs, computed_at)
                SELECT season, week, team, opponent, game_id, is_home, implied_total, spread,
                       sec_per_play, expected_plays, base_pass_rate, proe, script_adjustment,
                       expected_pass_rate, expected_pass_attempts, expected_rush_attempts,
                       expected_dropbacks, expected_drives, expected_rz_trips, expected_team_tds,
                       expected_team_fgs, now()
                FROM te_df
                """
            )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            con.unregister("te_df")

    log.info("team_environment: %s week %s -> %d teams", season, week, frame.height)
    return frame.height


def team_environment(season: int, week: int) -> pl.DataFrame:
    """Read the week's team environment. One row per team."""
    with connect() as con:
        return con.execute(
            "SELECT * FROM team_environment WHERE season = ? AND week = ?", [season, week]
        ).pl()
