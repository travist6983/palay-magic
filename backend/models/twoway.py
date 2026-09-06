"""Two-way (offense x defense) ridge effects for opponent adjustment (§5.2).

A raw trailing-window rate confounds two things: how good a defense is, and how good the offenses
it happened to face were. A defense that drew four backup quarterbacks looks elite; one that drew
Buffalo and Detroit looks porous. "Opponent-adjusted" should mean that confound is removed.

So instead of a raw team rate we fit, over every game in a trailing window,

    rate_g  =  mu  +  offense_effect[o_g]  +  defense_effect[d_g]  +  error_g

with an L2 penalty on the effect vectors and observation weights that combine the play volume
behind each rate with exponential recency decay. The defense's multiplier is then
``(mu + defense_effect[d]) / mu`` -- what this defence would allow against a league-average
offence.

Ridge does double duty here: it removes the collinearity that makes a 64-parameter model
unidentifiable on a thin schedule, and the penalty *is* the shrinkage toward league average, so a
defence with little evidence behind it lands near a multiplier of 1.0 without a separate step.

Whether this actually beats the raw rate is not assumed -- :mod:`backend.models.adjust` measures
both out of sample and keeps the winner per metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from backend.logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_HALFLIFE_GAMES = 17.0
"""Recency half-life, in league games per team. One season."""

WEEKS_PER_SEASON = 18
"""Used to turn a (season, week) gap into a games-ago distance for recency weighting."""


@dataclass(frozen=True)
class TwoWayFit:
    """Fitted offense and defense effects for one metric."""

    metric: str
    unit: str
    intercept: float
    offense: dict[str, float]
    defense: dict[str, float]
    ridge_lambda: float
    n_observations: int
    teams: tuple[str, ...] = field(default_factory=tuple)

    def defense_multiplier(self, team: str) -> float:
        """What this defence allows relative to league average, offence held constant."""
        if self.intercept == 0:
            return 1.0
        return (self.intercept + self.defense.get(team, 0.0)) / self.intercept

    def offense_multiplier(self, team: str) -> float:
        """What this offence produces relative to league average, defence held constant."""
        if self.intercept == 0:
            return 1.0
        return (self.intercept + self.offense.get(team, 0.0)) / self.intercept


def fit_two_way(
    games: pl.DataFrame,
    metric: str,
    unit: str,
    ridge_lambda: float = 20.0,
    halflife_games: float = DEFAULT_HALFLIFE_GAMES,
) -> TwoWayFit | None:
    """Fit ``rate = mu + offense_effect + defense_effect`` by weighted ridge regression.

    Args:
        games: one row per game with ``team`` (the unit being rated), ``opponent``,
            ``numerator``, ``denominator``, and ``games_ago`` (0 = most recent).
        metric: metric name, carried onto the fit.
        unit: ``defense`` or ``offense``; determines which side ``team`` is.
        ridge_lambda: L2 penalty on the effect vectors. Larger shrinks harder toward league
            average. Tuned per metric by :func:`tune_ridge`.
        halflife_games: recency half-life in games. Older games still inform the fit but weigh
            less, which matters across a season boundary.

    Returns:
        A :class:`TwoWayFit`, or None when there is not enough data to identify the model.
    """
    df = games.filter(
        (pl.col("denominator") > 0)
        & pl.col("team").is_not_null()
        & pl.col("opponent").is_not_null()
    )
    if df.height < 32:
        return None

    rate = (df["numerator"] / df["denominator"]).to_numpy()
    volume = df["denominator"].to_numpy().astype(float)
    games_ago = df["games_ago"].to_numpy().astype(float)

    # Weight by play volume (a 40-target game says more than a 12-target one) and by recency.
    recency = 0.5 ** (games_ago / max(halflife_games, 1e-6))
    weights = np.clip(volume, 1e-9, None) * recency
    weights = weights / weights.mean()

    rated = df["team"].to_list()
    faced = df["opponent"].to_list()
    teams = sorted(set(rated) | set(faced))
    index = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    if n_teams < 8:
        return None

    # Columns: [intercept | rated-side effects | faced-side effects]
    n_rows = len(rate)
    X = np.zeros((n_rows, 1 + 2 * n_teams))
    X[:, 0] = 1.0
    X[np.arange(n_rows), 1 + np.array([index[t] for t in rated])] = 1.0
    X[np.arange(n_rows), 1 + n_teams + np.array([index[t] for t in faced])] = 1.0

    sw = np.sqrt(weights)
    Xw = X * sw[:, None]
    yw = rate * sw

    # Penalise the effects but never the intercept: league average is not something to shrink.
    penalty = np.full(1 + 2 * n_teams, ridge_lambda)
    penalty[0] = 0.0

    gram = Xw.T @ Xw + np.diag(penalty)
    try:
        coefs = np.linalg.solve(gram, Xw.T @ yw)
    except np.linalg.LinAlgError:
        log.warning("two-way fit for %s/%s is singular", unit, metric)
        return None

    intercept = float(coefs[0])
    rated_eff = coefs[1 : 1 + n_teams]
    faced_eff = coefs[1 + n_teams :]

    # Ridge does not enforce sum-to-zero; re-centre so the intercept really is league average and
    # a team with no effect gets a multiplier of exactly 1.0.
    intercept += float(rated_eff.mean() + faced_eff.mean())
    rated_eff = rated_eff - rated_eff.mean()
    faced_eff = faced_eff - faced_eff.mean()

    if unit == "defense":
        defense = {t: float(rated_eff[index[t]]) for t in teams}
        offense = {t: float(faced_eff[index[t]]) for t in teams}
    else:
        offense = {t: float(rated_eff[index[t]]) for t in teams}
        defense = {t: float(faced_eff[index[t]]) for t in teams}

    return TwoWayFit(
        metric=metric,
        unit=unit,
        intercept=intercept,
        offense=offense,
        defense=defense,
        ridge_lambda=ridge_lambda,
        n_observations=n_rows,
        teams=tuple(teams),
    )


def tune_ridge(
    history: pl.DataFrame,
    metric: str,
    unit: str,
    candidates: tuple[float, ...] = (2.0, 5.0, 10.0, 25.0, 60.0, 150.0, 400.0),
    halflife_games: float = DEFAULT_HALFLIFE_GAMES,
    min_train_games: int = 400,
) -> tuple[float, float]:
    """Pick the ridge penalty that best predicts the next game, walking forward in time.

    For each candidate penalty, replay the season: fit on everything before week ``w``, predict
    every game in week ``w``, and accumulate the weighted squared error. The penalty with the
    lowest out-of-sample error wins. Nothing here is fitted on data it is later scored against.

    Args:
        history: game rows with ``season``, ``week``, ``team``, ``opponent``, ``numerator``,
            ``denominator``.
        metric: metric name.
        unit: ``defense`` or ``offense``.
        candidates: penalties to try.
        halflife_games: recency half-life passed through to the fit.
        min_train_games: skip weeks with less history than this, so early weeks do not dominate.

    Returns:
        ``(best_lambda, best_weighted_mse)``. Falls back to ``(25.0, inf)`` when there is not
        enough history to evaluate anything.
    """
    df = history.filter(pl.col("denominator") > 0).sort(["season", "week"])
    if df.height < min_train_games * 2:
        return 25.0, float("inf")

    marks = (
        df.select("season", "week").unique().sort(["season", "week"]).to_dicts()
    )
    # Evaluate on the back half of the timeline, where the training set is realistic.
    marks = marks[len(marks) // 2 :]
    if not marks:
        return 25.0, float("inf")

    errors = {lam: [0.0, 0.0] for lam in candidates}

    for mark in marks:
        season, week = mark["season"], mark["week"]
        train = df.filter(
            (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week))
        )
        if train.height < min_train_games:
            continue
        test = df.filter((pl.col("season") == season) & (pl.col("week") == week))
        if test.is_empty():
            continue

        # Each team plays about one game a week, so weeks-until-the-target-week is games-ago.
        train = train.with_columns(
            (
                (season - pl.col("season")) * WEEKS_PER_SEASON + (week - pl.col("week"))
            ).cast(pl.Float64).alias("games_ago")
        )

        y_true = (test["numerator"] / test["denominator"]).to_numpy()
        w_test = test["denominator"].to_numpy().astype(float)
        rated = test["team"].to_list()

        for lam in candidates:
            fit = fit_two_way(train, metric, unit, ridge_lambda=lam, halflife_games=halflife_games)
            if fit is None:
                continue
            # Only the rated side becomes a multiplier, so only the rated side is scored
            # (see adjust._walk_forward_scores). Tuning on the full prediction picked penalties
            # for the offence effect's benefit, not the defence's.
            rated_map = fit.defense if unit == "defense" else fit.offense
            pred = np.array([fit.intercept + rated_map.get(r, 0.0) for r in rated])
            errors[lam][0] += float((w_test * (y_true - pred) ** 2).sum())
            errors[lam][1] += float(w_test.sum())

    scored = {
        lam: (num / den) for lam, (num, den) in errors.items() if den > 0
    }
    if not scored:
        return 25.0, float("inf")

    best = min(scored, key=scored.get)
    return best, scored[best]
