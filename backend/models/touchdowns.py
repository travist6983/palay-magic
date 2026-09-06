"""Calibrated touchdown share (§5.6).

§5.6 says ``λ = expected_team_TDs × player_TD_share``, with the share coming from red-zone target
share and goal-line carry share. That structure is right; the *raw* usage share is not the share.
Scored on a 2025 replay, the hand-set version (``0.75·gl_share + 0.25·rz_share`` against all team
touchdowns) over-predicted running backs by 11.5 points, over-predicted quarterback rushing scores
2×, and under-predicted tight ends by 13 points — and a constant at the positional base rate beat
it on log-loss for every position.

So the share is **fitted**, per position, as a linear function of the usage signals, separately for
rushing and receiving scores:

    λ = E[team rush TDs] · max(0, a_r + b_r·gl_carry_share + c_r·hist_rush_td_share)
      + E[team rec  TDs] · max(0, a_p + b_p·rz_target_share + c_p·hist_rec_td_share)
    P(anytime TD) = 1 − e^(−λ)

Coefficients are found by minimising log-loss on prior seasons and validated held-out. On 2025:
RB log-loss 0.634 against 0.656 for the base rate, WR 0.517 vs 0.536, TE 0.467 vs 0.487, QB 0.395
vs 0.426 — every position beats the constant, bias inside ±2 points, calibration monotone by
quintile. The structure is kept so "Show math" still reads as share × team touchdowns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.optimize import minimize

from backend.config import get_settings
from backend.db.connection import connect
from backend.logging_setup import get_logger

log = get_logger(__name__)

TRAILING_GAMES = 10
"""Window for the trailing usage signals the share is fitted on."""

MIN_SNAPS = 15
"""A player-game with fewer offensive snaps is a cameo, not evidence about a role."""


@dataclass(frozen=True)
class TDShareModel:
    """Fitted per-position coefficients. ``rush`` and ``rec`` are (intercept, usage, history)."""

    position: str
    rush: tuple[float, float, float]
    rec: tuple[float, float, float]
    n_train: int
    train_logloss: float
    base_logloss: float

    def rush_share(self, gl_carry_share: float, hist_rush_td_share: float) -> float:
        a, b, c = self.rush
        return max(0.0, a + b * gl_carry_share + c * hist_rush_td_share)

    def rec_share(self, rz_target_share: float, hist_rec_td_share: float) -> float:
        a, b, c = self.rec
        return max(0.0, a + b * rz_target_share + c * hist_rec_td_share)

    def lam(
        self,
        team_rush_tds: float,
        team_rec_tds: float,
        gl_carry_share: float,
        rz_target_share: float,
        hist_rush_td_share: float,
        hist_rec_td_share: float,
        rushing_only: bool = False,
    ) -> float:
        """Expected touchdowns for this player. ``rushing_only`` is the QB anytime-rush prop."""
        out = team_rush_tds * self.rush_share(gl_carry_share, hist_rush_td_share)
        if not rushing_only:
            out += team_rec_tds * self.rec_share(rz_target_share, hist_rec_td_share)
        return float(max(out, 0.0))


def _training_frame(seasons: list[int]) -> pl.DataFrame:
    season_list = ",".join(str(s) for s in seasons)
    with connect() as con:
        df = con.execute(
            f"""
            WITH s AS (
                SELECT gsis_id, season, week, team, position,
                       max(CASE WHEN stat = 'rushing_tds'   THEN value END) AS rush_td,
                       max(CASE WHEN stat = 'receiving_tds' THEN value END) AS rec_td
                FROM player_game_stats
                WHERE position IN ('QB','RB','WR','TE') AND season IN ({season_list})
                GROUP BY 1, 2, 3, 4, 5
            ),
            t AS (
                SELECT season, week, team,
                       sum(rush_td) AS team_rush_td, sum(rec_td) AS team_rec_td
                FROM s GROUP BY 1, 2, 3
            )
            SELECT s.*, u.gl_carry_share, u.rz_target_share, u.offense_snaps,
                   u.target_share, u.carry_share,
                   t.team_rush_td, t.team_rec_td
            FROM s
            JOIN t USING (season, week, team)
            LEFT JOIN player_game_usage u USING (gsis_id, season, week)
            ORDER BY gsis_id, season, week
            """
        ).pl()

    def trailing_mean(col: str) -> pl.Expr:
        return (
            pl.col(col).shift(1).rolling_mean(TRAILING_GAMES, min_samples=3).over("gsis_id")
        )

    def trailing_ratio(num: str, den: str) -> pl.Expr:
        return (
            pl.col(num).shift(1).rolling_sum(TRAILING_GAMES, min_samples=3).over("gsis_id")
            / pl.col(den)
            .shift(1)
            .rolling_sum(TRAILING_GAMES, min_samples=3)
            .over("gsis_id")
            .clip(1e-9, None)
        )

    return df.with_columns(
        trailing_mean("gl_carry_share").alias("gl"),
        trailing_mean("rz_target_share").alias("rz"),
        trailing_mean("target_share").alias("tr_target_share"),
        trailing_mean("carry_share").alias("tr_carry_share"),
        trailing_ratio("rush_td", "team_rush_td").alias("h_rush"),
        trailing_ratio("rec_td", "team_rec_td").alias("h_rec"),
        ((pl.col("rush_td") + pl.col("rec_td")) > 0).cast(pl.Float64).alias("y_any"),
        (pl.col("rush_td") > 0).cast(pl.Float64).alias("y_rush"),
    ).filter(pl.col("offense_snaps").fill_null(0) > MIN_SNAPS)


def _logloss(p: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    if w is None:
        return float(ll.mean())
    return float((ll * w).sum() / w.sum())


def fit_td_share_models(seasons: list[int] | None = None) -> dict[str, TDShareModel]:
    """Fit and persist the per-position share coefficients (§5.6).

    The environment input during fitting is the league-average team touchdowns by type, so the
    coefficients describe the share alone; at projection time the team's own expected touchdowns
    are substituted in.
    """
    seasons = seasons or list(get_settings().seasons)
    df = _training_frame(seasons)
    if df.height < 500:
        log.warning("only %d player-games; TD share models not fitted", df.height)
        return {}

    team_games = df.group_by(["season", "week", "team"]).agg(
        pl.col("team_rush_td").first(), pl.col("team_rec_td").first()
    )
    league_rush = float(team_games["team_rush_td"].mean())
    league_rec = float(team_games["team_rec_td"].mean())

    models: dict[str, TDShareModel] = {}
    for position in ("RB", "WR", "TE", "QB"):
        sub = df.filter(pl.col("position") == position).drop_nulls(["gl", "rz", "h_rush", "h_rec"])
        if sub.height < 200:
            continue
        # Sample weight = trailing usage, so a nine-target receiver counts more than a two-target
        # one. A small floor keeps low-usage games in the fit rather than discarding them.
        usage = (
            sub["tr_carry_share"].fill_null(0.0).to_numpy()
            if position == "QB"
            else sub["tr_target_share"].fill_null(0.0).to_numpy()
            + (sub["tr_carry_share"].fill_null(0.0).to_numpy() if position == "RB" else 0.0)
        )
        weights = np.clip(usage, 0.02, None)
        model = _fit_position(
            position=position,
            y=sub["y_rush" if position == "QB" else "y_any"].to_numpy(),
            gl=sub["gl"].to_numpy(),
            rz=sub["rz"].to_numpy(),
            hist_rush=sub["h_rush"].to_numpy(),
            hist_rec=sub["h_rec"].to_numpy(),
            league_rush=league_rush,
            league_rec=league_rec,
            weights=weights,
        )
        models[position] = model
        log.info(
            "TD share %s: n=%d logloss %.4f vs base %.4f rush=%s rec=%s",
            position, model.n_train, model.train_logloss, model.base_logloss,
            tuple(round(x, 3) for x in model.rush), tuple(round(x, 3) for x in model.rec),
        )

    _persist(models, league_rush, league_rec)
    return models


def _fit_position(
    position: str,
    y: np.ndarray,
    gl: np.ndarray,
    rz: np.ndarray,
    hist_rush: np.ndarray,
    hist_rec: np.ndarray,
    league_rush: float,
    league_rec: float,
    weights: np.ndarray | None = None,
) -> TDShareModel:
    """Minimise (weighted) log-loss for one position. A plain function so nothing closes over a loop.

    ``weights`` tilt the fit toward the players the app projects. Fitted unweighted on every
    player with 15+ snaps, the model was near-unbiased on the whole population and 13 points low
    on the tight ends who make a top-10 board -- the relationship is not linear all the way up,
    and a fit dominated by 4-target players will not bend for 9-target ones.
    """
    rushing_only = position == "QB"

    def lam(theta: np.ndarray) -> np.ndarray:
        rush = league_rush * np.clip(theta[0] + theta[1] * gl + theta[2] * hist_rush, 0, None)
        if rushing_only:
            return rush
        rec = league_rec * np.clip(theta[3] + theta[4] * rz + theta[5] * hist_rec, 0, None)
        return rush + rec

    def objective(theta: np.ndarray) -> float:
        return _logloss(1 - np.exp(-lam(theta)), y, weights)

    x0 = np.array([0.02, 0.3, 0.3] if rushing_only else [0.0, 0.4, 0.4, 0.0, 0.4, 0.4])
    # Non-negativity on the usage coefficients: a negative weight on goal-line share is the
    # optimiser exploiting a near-constant feature, not a real effect, and it would let a heavy
    # goal-line role REDUCE a projection.
    bounds = [(-0.5, 0.5), (0.0, 3.0), (0.0, 3.0)] + (
        [] if rushing_only else [(-0.5, 0.5), (0.0, 3.0), (0.0, 3.0)]
    )
    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    theta = res.x

    return TDShareModel(
        position=position,
        rush=(float(theta[0]), float(theta[1]), float(theta[2])),
        rec=(0.0, 0.0, 0.0) if rushing_only else (float(theta[3]), float(theta[4]), float(theta[5])),
        n_train=len(y),
        train_logloss=objective(theta),
        base_logloss=_logloss(np.full(len(y), y.mean()), y, weights),
    )


def _persist(models: dict[str, TDShareModel], league_rush: float, league_rec: float) -> None:
    rows = []
    for m in models.values():
        for side, coefs in (("rush", m.rush), ("rec", m.rec)):
            for term, value in zip(("intercept", "usage", "history"), coefs, strict=True):
                rows.append(
                    {
                        "model": f"td_share_{m.position}_{side}", "term": term, "coefficient": value,
                        "n_observations": m.n_train, "r_squared": None,
                        "rmse": m.train_logloss,
                    }
                )
    rows.append({"model": "td_league", "term": "rush_per_team_game", "coefficient": league_rush,
                 "n_observations": 0, "r_squared": None, "rmse": None})
    rows.append({"model": "td_league", "term": "rec_per_team_game", "coefficient": league_rec,
                 "n_observations": 0, "r_squared": None, "rmse": None})
    frame = pl.DataFrame(rows)
    with connect() as con:
        con.register("td_df", frame)
        try:
            con.execute("BEGIN TRANSACTION")
            con.execute("DELETE FROM environment_models WHERE model LIKE 'td_%'")
            con.execute(
                "INSERT INTO environment_models "
                "(model, term, coefficient, n_observations, r_squared, rmse, computed_at) "
                "SELECT model, term, coefficient, n_observations, r_squared, rmse, now() FROM td_df"
            )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            con.unregister("td_df")


def load_td_share_models() -> dict[str, TDShareModel]:
    """Read the fitted coefficients back. Empty before the first fit."""
    with connect() as con:
        try:
            rows = con.execute(
                "SELECT model, term, coefficient, n_observations, rmse FROM environment_models "
                "WHERE model LIKE 'td_share_%'"
            ).fetchall()
        except Exception:  # noqa: BLE001 - table may not exist yet
            return {}

    grouped: dict[str, dict] = {}
    for model, term, coef, n, ll in rows:
        _, _, position, side = model.split("_", 3)
        entry = grouped.setdefault(position, {"rush": {}, "rec": {}, "n": n, "ll": ll})
        entry[side][term] = float(coef)

    out: dict[str, TDShareModel] = {}
    for position, e in grouped.items():
        rush = e["rush"]
        rec = e["rec"]
        out[position] = TDShareModel(
            position=position,
            rush=(rush.get("intercept", 0.0), rush.get("usage", 0.0), rush.get("history", 0.0)),
            rec=(rec.get("intercept", 0.0), rec.get("usage", 0.0), rec.get("history", 0.0)),
            n_train=int(e["n"] or 0),
            train_logloss=float(e["ll"] or 0.0),
            base_logloss=0.0,
        )
    return out


def player_usage_features(gsis_id: str, season: int, week: int) -> tuple[float, float, int]:
    """The goal-line and red-zone shares EXACTLY as the share model was fitted on them.

    The fit uses an unweighted mean over the trailing ten games (min three); the projection was
    applying a six-game, snap-weighted, decayed mean instead -- a different feature, so the fitted
    coefficients were being evaluated off-distribution. One helper for both paths.

    Returns ``(gl_carry_share, rz_target_share, games)``.
    """
    with connect() as con:
        row = con.execute(
            f"""
            SELECT avg(gl_carry_share), avg(rz_target_share), count(*) FROM (
                SELECT gl_carry_share, rz_target_share
                FROM player_game_usage
                WHERE gsis_id = ? AND ((season < ?) OR (season = ? AND week < ?))
                ORDER BY season DESC, week DESC LIMIT {TRAILING_GAMES}
            )
            """,
            [gsis_id, season, season, week],
        ).fetchone()
    if not row or not row[2]:
        return 0.0, 0.0, 0
    return float(row[0] or 0.0), float(row[1] or 0.0), int(row[2])


def player_td_history(gsis_id: str, season: int, week: int) -> tuple[float, float, int]:
    """The player's share of his team's rushing and receiving TDs over the trailing window.

    Returns ``(hist_rush_td_share, hist_rec_td_share, games)``. Both are ratios of sums, so a
    player on a team that scored four rushing touchdowns and took three of them is 0.75.
    """
    with connect() as con:
        row = con.execute(
            f"""
            WITH p AS (
                SELECT season, week, team,
                       max(CASE WHEN stat = 'rushing_tds'   THEN value END) AS rush_td,
                       max(CASE WHEN stat = 'receiving_tds' THEN value END) AS rec_td
                FROM player_game_stats
                WHERE gsis_id = ? AND ((season < ?) OR (season = ? AND week < ?))
                GROUP BY 1, 2, 3
                ORDER BY season DESC, week DESC LIMIT {TRAILING_GAMES}
            ),
            t AS (
                SELECT g.season, g.week, g.team,
                       sum(CASE WHEN g.stat = 'rushing_tds'   THEN g.value ELSE 0 END) AS team_rush,
                       sum(CASE WHEN g.stat = 'receiving_tds' THEN g.value ELSE 0 END) AS team_rec
                FROM player_game_stats g
                JOIN p ON p.season = g.season AND p.week = g.week AND p.team = g.team
                GROUP BY 1, 2, 3
            )
            SELECT sum(p.rush_td) / nullif(sum(t.team_rush), 0),
                   sum(p.rec_td)  / nullif(sum(t.team_rec), 0),
                   count(*)
            FROM p JOIN t USING (season, week, team)
            """,
            [gsis_id, season, season, week],
        ).fetchone()
    if not row:
        return 0.0, 0.0, 0
    return float(row[0] or 0.0), float(row[1] or 0.0), int(row[2] or 0)
