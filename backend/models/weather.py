"""Weather and kicking-distance models, fitted rather than assumed (§5.5, §5.6).

§5.5 says "wind > 15 mph reduces passing/kicking efficiency; dome sets wind to 0" without giving
magnitudes, and the prop reference quotes a rule of thumb (a 50-yarder "80%+ in a dome, ~60-65% in
15+ mph wind") that our own data does not support — 2023-2025 says 71.6% indoors and 65.3% in a
breeze. So both effects are estimated from play-by-play joined to the schedule's wind and roof.

Two models:

* **Passing efficiency vs. wind** — a linear yards-per-attempt penalty. Measured on outdoor games:
  7.21 Y/A at 0-4 mph falling monotonically to 6.49 at 15-19 mph.
* **Field-goal make probability** — a logistic model on distance, wind and the indoor flag. This
  is what turns an expected attempt distribution into expected makes, and therefore kicking points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from backend.config import get_settings
from backend.db.connection import connect
from backend.logging_setup import get_logger

log = get_logger(__name__)

INDOOR_ROOFS = frozenset({"dome", "closed"})

# Beyond this the sample is too thin to trust and the fit is extrapolating.
MAX_MODELLED_WIND = 25.0

MEAN_OUTDOOR_WIND = 7.9
"""Used when an outdoor game has no forecast yet. Pricing "unknown" as 0 mph gave every outdoor
kicker in a live week the calm make rate."""


@dataclass(frozen=True)
class WindEffect:
    """Linear wind penalty on passing efficiency."""

    intercept: float
    per_mph: float
    n_games: int
    baseline_ypa: float

    def multiplier(self, wind_mph: float | None, indoor: bool = False) -> float:
        """Passing-efficiency multiplier relative to a calm outdoor game.

        A dome or closed roof is calm by definition (§5.5), so it returns 1.0 rather than being
        given a bonus — the baseline already reflects average conditions.
        """
        if indoor or wind_mph is None:
            return 1.0
        wind = float(np.clip(wind_mph, 0.0, MAX_MODELLED_WIND))
        predicted = self.intercept + self.per_mph * wind
        if self.baseline_ypa <= 0:
            return 1.0
        return float(np.clip(predicted / self.baseline_ypa, 0.75, 1.10))


@dataclass(frozen=True)
class FieldGoalModel:
    """Logistic make probability: ``logit(p) = b0 + b_dist*distance + b_wind*wind + b_indoor``."""

    intercept: float
    per_yard: float
    per_mph: float
    indoor_bonus: float
    n_attempts: int

    def make_probability(self, distance: float, wind_mph: float | None, indoor: bool) -> float:
        """Probability this attempt is good. Unknown outdoor wind is priced as average, not calm."""
        if indoor:
            wind = 0.0
        elif wind_mph is None:
            wind = MEAN_OUTDOOR_WIND
        else:
            wind = float(np.clip(wind_mph, 0.0, MAX_MODELLED_WIND))
        z = (
            self.intercept
            + self.per_yard * float(distance)
            + self.per_mph * wind
            + (self.indoor_bonus if indoor else 0.0)
        )
        return float(1.0 / (1.0 + np.exp(-z)))

    def expected_make_rate(
        self, distances: np.ndarray, weights: np.ndarray, wind_mph: float | None, indoor: bool
    ) -> float:
        """Make rate over a distribution of attempt distances."""
        probs = np.array([self.make_probability(d, wind_mph, indoor) for d in distances])
        total = weights.sum()
        return float((probs * weights).sum() / total) if total > 0 else 0.0


def fit_wind_effect(seasons: list[int] | None = None) -> WindEffect:
    """Fit yards-per-attempt against wind on outdoor games."""
    seasons = seasons or list(get_settings().seasons)
    season_list = ",".join(str(s) for s in seasons)
    with connect() as con:
        df = con.execute(
            f"""
            SELECT s.wind,
                   sum(p.passing_yards)::DOUBLE AS yards,
                   sum(p.attempts)::DOUBLE      AS attempts
            FROM raw_player_stats p
            JOIN raw_schedules s USING (game_id)
            WHERE p.season_type = 'REG' AND p.position = 'QB'
              AND p.season IN ({season_list})
              AND lower(coalesce(s.roof, '')) = 'outdoors'
              AND s.wind IS NOT NULL
            GROUP BY s.wind, p.game_id
            HAVING sum(p.attempts) >= 10
            """
        ).pl()

    if df.height < 100:
        log.warning("only %d outdoor game-halves with wind; using a neutral wind effect", df.height)
        return WindEffect(intercept=7.0, per_mph=0.0, n_games=df.height, baseline_ypa=7.0)

    wind = df["wind"].to_numpy().astype(float)
    ypa = (df["yards"] / df["attempts"]).to_numpy()
    weights = df["attempts"].to_numpy().astype(float)

    mask = wind <= MAX_MODELLED_WIND
    wind, ypa, weights = wind[mask], ypa[mask], weights[mask]

    design = np.column_stack([np.ones(len(wind)), wind])
    sw = np.sqrt(weights)
    coefs, *_ = np.linalg.lstsq(design * sw[:, None], ypa * sw, rcond=None)

    baseline = float((ypa * weights).sum() / weights.sum())
    effect = WindEffect(
        intercept=float(coefs[0]), per_mph=float(coefs[1]), n_games=len(wind), baseline_ypa=baseline
    )
    log.info(
        "wind effect: %.4f Y/A per mph (baseline %.2f, n=%d) -> %.1f%% at 15 mph",
        effect.per_mph, baseline, len(wind), 100 * (effect.multiplier(15.0) - 1),
    )
    return effect


def fit_field_goal_model(seasons: list[int] | None = None) -> FieldGoalModel:
    """Fit a logistic field-goal make model on distance, wind and roof."""
    seasons = seasons or list(get_settings().seasons)
    season_list = ",".join(str(s) for s in seasons)
    with connect() as con:
        df = con.execute(
            f"""
            SELECT p.kick_distance AS distance,
                   CASE WHEN p.field_goal_result = 'made' THEN 1 ELSE 0 END AS made,
                   CASE WHEN lower(coalesce(s.roof, '')) IN ('dome', 'closed') THEN 1 ELSE 0 END AS indoor,
                   coalesce(s.wind, 0) AS wind
            FROM raw_pbp p
            JOIN raw_schedules s USING (game_id)
            WHERE p.season_type = 'REG' AND p.field_goal_attempt = 1
              AND p.kick_distance IS NOT NULL AND p.field_goal_result IS NOT NULL
              AND p.season IN ({season_list})
              -- An outdoor try with no recorded wind is not a 0-mph try; 7% of rows coded that
              -- way steepened the wind slope by 22%. Indoor rows keep wind = 0 by definition.
              AND (lower(coalesce(s.roof, '')) IN ('dome', 'closed') OR s.wind IS NOT NULL)
            """
        ).pl()

    if df.height < 500:
        log.warning("only %d field goals; using a distance-only fallback", df.height)
        return FieldGoalModel(6.0, -0.105, 0.0, 0.0, df.height)

    y = df["made"].to_numpy().astype(float)
    indoor = df["indoor"].to_numpy().astype(float)
    wind = np.clip(df["wind"].to_numpy().astype(float), 0, MAX_MODELLED_WIND) * (1 - indoor)
    X = np.column_stack([np.ones(len(y)), df["distance"].to_numpy().astype(float), wind, indoor])

    # Newton-Raphson; the design is tiny and well conditioned.
    beta = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        w = np.clip(p * (1 - p), 1e-8, None)
        gradient = X.T @ (y - p)
        hessian = (X * w[:, None]).T @ X
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:  # pragma: no cover
            break
        beta += step
        if np.abs(step).max() < 1e-8:
            break

    model = FieldGoalModel(
        intercept=float(beta[0]), per_yard=float(beta[1]), per_mph=float(beta[2]),
        indoor_bonus=float(beta[3]), n_attempts=len(y),
    )
    log.info(
        "field goal model (n=%d): 40y calm %.3f, 40y 18mph %.3f, 50y indoor %.3f, 50y 18mph %.3f",
        len(y),
        model.make_probability(40, 0, False), model.make_probability(40, 18, False),
        model.make_probability(50, 0, True), model.make_probability(50, 18, False),
    )
    return model


def attempt_distance_distribution(seasons: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """League distribution of field-goal attempt distances, as (distances, weights).

    Used to turn an expected number of attempts into expected makes without pretending to know
    where each one will come from.
    """
    seasons = seasons or list(get_settings().seasons)
    season_list = ",".join(str(s) for s in seasons)
    with connect() as con:
        df = con.execute(
            f"""
            SELECT kick_distance AS distance, count(*) AS n
            FROM raw_pbp
            WHERE season_type = 'REG' AND field_goal_attempt = 1
              AND kick_distance BETWEEN 17 AND 70 AND season IN ({season_list})
            GROUP BY 1 ORDER BY 1
            """
        ).pl()
    if df.is_empty():
        return np.arange(20, 56, 5.0), np.ones(8)
    return df["distance"].to_numpy().astype(float), df["n"].to_numpy().astype(float)


def kicking_context(season: int, week: int) -> dict[str, dict]:
    """Per-team wind and indoor flags for the week, for the kicking projection."""
    with connect() as con:
        rows = con.execute(
            """
            SELECT home_team, away_team, roof, wind
            FROM game_environment WHERE season = ? AND week = ?
            """,
            [season, week],
        ).fetchall()
    out: dict[str, dict] = {}
    for home, away, roof, wind in rows:
        indoor = (roof or "").lower() in INDOOR_ROOFS
        for team in (home, away):
            out[team] = {"indoor": indoor, "wind": 0.0 if indoor else wind, "roof": roof}
    return out


def team_weather(season: int, week: int) -> pl.DataFrame:
    """The week's roof/wind/temp per game, for the UI's environment card."""
    with connect() as con:
        return con.execute(
            "SELECT game_id, home_team, away_team, roof, wind, temp "
            "FROM game_environment WHERE season = ? AND week = ?",
            [season, week],
        ).pl()
