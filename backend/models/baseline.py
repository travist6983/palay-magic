"""Recency-weighted baselines and the season-boundary rule (§5.1, §4).

Pure functions over plain values. Given a player's recent games most-recent-first, these produce
the weighted mean and variance that every projection starts from, with prior-season games
discounted (§4) and a documented fallback when a player has too little history.

The weights are ``w_i = decay ** i`` with ``i = 0`` for the most recent game. A game from a prior
season carries an additional multiplicative ``prior_season_discount``, because a player's role,
scheme, and supporting cast may all have changed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from backend.config import get_settings


@dataclass(frozen=True)
class GameObservation:
    """One game in a player's recent history."""

    season: int
    week: int
    value: float
    is_prior_season: bool = False
    game_id: str = ""
    opponent: str = ""


@dataclass(frozen=True)
class Baseline:
    """A recency-weighted baseline for one player-stat.

    Attributes:
        mean: weighted mean.
        variance: weighted (unbiased-corrected) variance.
        n_games: how many observations went in.
        effective_n: Kish effective sample size, ``(sum w)^2 / sum(w^2)``. Six games at decay 0.8
            are worth about 5.3 independent games, which is what the dispersion fallback keys off.
        weights: the weight actually applied to each observation, most recent first.
        insufficient_history: fewer than ``min_games_for_history`` observations (§4).
    """

    mean: float
    variance: float
    n_games: int
    effective_n: float
    weights: tuple[float, ...] = field(default_factory=tuple)
    values: tuple[float, ...] = field(default_factory=tuple)
    insufficient_history: bool = False
    n_prior_season: int = 0

    @property
    def sd(self) -> float:
        return math.sqrt(max(self.variance, 0.0))

    @property
    def dispersion_ratio(self) -> float:
        """``variance / mean``. Above 1.3 a count stat gets a negative binomial (§5.6)."""
        return self.variance / self.mean if self.mean > 0 else 0.0

    def to_json(self) -> dict[str, object]:
        return {
            "mean": self.mean,
            "variance": self.variance,
            "sd": self.sd,
            "n_games": self.n_games,
            "effective_n": self.effective_n,
            "n_prior_season": self.n_prior_season,
            "insufficient_history": self.insufficient_history,
            "weights": list(self.weights),
            "values": list(self.values),
        }


def recency_weights(
    n: int,
    decay: float | None = None,
    prior_season_flags: Sequence[bool] | None = None,
    prior_season_discount: float | None = None,
) -> np.ndarray:
    """Exponential recency weights, most recent first (§5.1).

        w_i = decay ** i          for the i-th most recent game (i = 0 is the latest)

    A game flagged as prior-season is multiplied by an additional ``prior_season_discount`` (§4).
    Weights are returned unnormalised; the weighted-mean helpers normalise them.

    Args:
        n: number of observations.
        decay: per-game decay, default ``settings.recency_decay`` (0.8).
        prior_season_flags: parallel to the observations; True marks a prior-season game.
        prior_season_discount: default ``settings.prior_season_discount`` (0.85).

    Returns:
        Array of length ``n``, most recent first.
    """
    settings = get_settings()
    decay = settings.recency_decay if decay is None else decay
    discount = settings.prior_season_discount if prior_season_discount is None else prior_season_discount

    w = np.array([decay**i for i in range(n)], dtype=float)

    if prior_season_flags is not None:
        flags = np.asarray(list(prior_season_flags)[:n], dtype=bool)
        if flags.size != n:
            raise ValueError(f"prior_season_flags has {flags.size} entries, expected {n}")
        w = np.where(flags, w * discount, w)

    return w


def weighted_mean_variance(values: Sequence[float], weights: Sequence[float]) -> tuple[float, float]:
    """Weighted mean and weighted variance (§5.1).

    The variance uses the reliability-weight correction

        var = sum(w_i (x_i - mu)^2) / (V1 - V2 / V1)

    where ``V1 = sum(w)`` and ``V2 = sum(w^2)``. That reduces to the usual ``n / (n - 1)``
    correction when every weight is equal, so a six-game sample is not reported as more certain
    than it is. A single observation has no variance estimate and returns 0.0.
    """
    v = np.asarray(list(values), dtype=float)
    w = np.asarray(list(weights), dtype=float)
    if v.size == 0:
        return 0.0, 0.0
    if v.size != w.size:
        raise ValueError(f"{v.size} values but {w.size} weights")

    v1 = w.sum()
    if v1 <= 0:
        return 0.0, 0.0

    mean = float((w * v).sum() / v1)

    if v.size == 1:
        return mean, 0.0

    v2 = (w * w).sum()
    denom = v1 - v2 / v1
    if denom <= 0:
        return mean, 0.0

    variance = float((w * (v - mean) ** 2).sum() / denom)
    return mean, max(variance, 0.0)


def effective_sample_size(weights: Sequence[float]) -> float:
    """Kish effective sample size: ``(sum w)^2 / sum(w^2)``.

    Six games at decay 0.8 give about 5.3 effective games. Below ``min_games_for_history`` we
    fall back to position priors instead of pretending (§4).
    """
    w = np.asarray(list(weights), dtype=float)
    if w.size == 0:
        return 0.0
    v2 = float((w * w).sum())
    return float(w.sum() ** 2 / v2) if v2 > 0 else 0.0


def compute_baseline(
    observations: Sequence[GameObservation],
    window: int | None = None,
    decay: float | None = None,
    prior_season_discount: float | None = None,
    min_games: int | None = None,
) -> Baseline:
    """Recency-weighted baseline over a player's last ``window`` games (§5.1).

    ``observations`` must be ordered most-recent-first and may span a season boundary; games
    flagged ``is_prior_season`` take the extra 0.85 discount (§4).

    Note for Week 1: when *every* observation is from the prior season the discount multiplies all
    weights equally, so it cancels in the mean and is a no-op for relative ranking. It stays in the
    code because it becomes live the moment current-season games exist (D3).

    Args:
        observations: recent games, most recent first.
        window: how many to use, default ``settings.recency_window`` (6).
        decay: per-game decay, default 0.8.
        prior_season_discount: default 0.85.
        min_games: below this the baseline is flagged ``insufficient_history``, default 3.

    Returns:
        A :class:`Baseline`. An empty history returns zeros with the flag set.
    """
    settings = get_settings()
    window = settings.recency_window if window is None else window
    min_games = settings.min_games_for_history if min_games is None else min_games

    used = list(observations)[:window]
    if not used:
        return Baseline(0.0, 0.0, 0, 0.0, insufficient_history=True)

    flags = [o.is_prior_season for o in used]
    values = [float(o.value) for o in used]
    w = recency_weights(len(used), decay, flags, prior_season_discount)
    mean, variance = weighted_mean_variance(values, w)

    return Baseline(
        mean=mean,
        variance=variance,
        n_games=len(used),
        effective_n=effective_sample_size(w),
        weights=tuple(float(x) for x in w),
        values=tuple(values),
        insufficient_history=len(used) < min_games,
        n_prior_season=sum(flags),
    )


def blend_with_prior(
    baseline: Baseline,
    prior_mean: float,
    prior_variance: float,
    prior_weight_games: float = 3.0,
) -> Baseline:
    """Shrink a thin baseline toward a position/depth-chart prior (§4).

    Weighting is by effective sample size: the prior counts as ``prior_weight_games`` games, so a
    player with one game is mostly prior and a player with six is mostly himself.

    Args:
        baseline: the player's own weighted baseline.
        prior_mean: the positional or depth-chart prior mean.
        prior_variance: the prior's variance.
        prior_weight_games: how many games of pseudo-data the prior is worth.

    Returns:
        A new :class:`Baseline` with the blended moments and the original counts preserved.
    """
    n_eff = baseline.effective_n
    total = n_eff + prior_weight_games
    if total <= 0:
        return baseline

    w_player = n_eff / total
    w_prior = prior_weight_games / total

    mean = w_player * baseline.mean + w_prior * prior_mean

    # Law of total variance across the two-component mixture: the spread between the two means is
    # genuine uncertainty and must not vanish in the blend.
    within = w_player * baseline.variance + w_prior * prior_variance
    between = w_player * (baseline.mean - mean) ** 2 + w_prior * (prior_mean - mean) ** 2

    return Baseline(
        mean=mean,
        variance=within + between,
        n_games=baseline.n_games,
        effective_n=n_eff,
        weights=baseline.weights,
        values=baseline.values,
        insufficient_history=baseline.insufficient_history,
        n_prior_season=baseline.n_prior_season,
    )


def shrink_multiplier(
    raw_value: float,
    league_average: float,
    n_games: int,
    k: float | None = None,
) -> float:
    """Shrink a defensive multiplier toward 1.0 with a Bayesian prior (§5.2).

        shrunk = (n * raw + k * league_avg) / (n + k)   then divided by league_avg

    ``k`` games of league-average pseudo-data keep an early-season two-game sample from claiming a
    defense allows 40% more than everyone else.

    Args:
        raw_value: the defense's observed per-game (or per-opportunity) value.
        league_average: the league mean for that metric over the same window.
        n_games: how many games the observation is based on.
        k: pseudo-games of prior, default ``settings.defense_shrink_k`` (6.0).

    Returns:
        The shrunk multiplier, 1.0 meaning exactly league average.
    """
    k = get_settings().defense_shrink_k if k is None else k
    if league_average <= 0 or n_games <= 0:
        return 1.0
    shrunk_value = (n_games * raw_value + k * league_average) / (n_games + k)
    return shrunk_value / league_average
