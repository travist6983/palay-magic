"""Tests for the recency-weighted baselines and the season-boundary rule (§5.1, §4)."""

from __future__ import annotations

import numpy as np
import pytest

from backend.models.baseline import (
    Baseline,
    GameObservation,
    blend_with_prior,
    compute_baseline,
    effective_sample_size,
    recency_weights,
    shrink_multiplier,
    weighted_mean_variance,
)


def games(*values: float, prior: bool = False) -> list[GameObservation]:
    """Build observations most-recent-first from bare values."""
    return [
        GameObservation(season=2025, week=18 - i, value=v, is_prior_season=prior)
        for i, v in enumerate(values)
    ]


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def test_weights_decay_geometrically_from_the_most_recent_game():
    """§5.1: w_i = 0.8 ** i, with i = 0 the most recent."""
    w = recency_weights(6, decay=0.8)
    assert w == pytest.approx([1.0, 0.8, 0.64, 0.512, 0.4096, 0.32768])


def test_prior_season_games_take_the_extra_discount():
    """§4: prior-season games get 0.85 on top of recency decay."""
    w = recency_weights(3, decay=0.8, prior_season_flags=[False, True, True], prior_season_discount=0.85)
    assert w == pytest.approx([1.0, 0.8 * 0.85, 0.64 * 0.85])


def test_flag_length_mismatch_is_an_error_not_a_silent_truncation():
    with pytest.raises(ValueError):
        recency_weights(3, prior_season_flags=[True, False])


def test_effective_sample_size_of_six_decayed_games():
    """Six games at 0.8 are worth ~5.3 independent games, not 6."""
    n_eff = effective_sample_size(recency_weights(6, decay=0.8))
    assert n_eff == pytest.approx(5.26, abs=0.02)
    assert n_eff < 6


def test_effective_sample_size_equals_n_for_flat_weights():
    assert effective_sample_size([1, 1, 1, 1]) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Weighted moments
# ---------------------------------------------------------------------------


def test_flat_weights_reproduce_the_ordinary_mean_and_sample_variance():
    values = [10.0, 20.0, 30.0, 40.0]
    mean, var = weighted_mean_variance(values, [1, 1, 1, 1])

    assert mean == pytest.approx(np.mean(values))
    assert var == pytest.approx(np.var(values, ddof=1))


def test_recent_games_move_the_mean_more_than_old_ones():
    hot_recently = compute_baseline(games(140, 40, 40, 40, 40, 40))
    hot_long_ago = compute_baseline(games(40, 40, 40, 40, 40, 140))

    assert hot_recently.mean > hot_long_ago.mean


def test_single_observation_has_no_variance_estimate():
    b = compute_baseline(games(80.0))
    assert b.mean == 80.0
    assert b.variance == 0.0


def test_empty_history_is_flagged_not_crashed():
    b = compute_baseline([])
    assert b.n_games == 0
    assert b.insufficient_history


def test_value_mismatch_is_an_error():
    with pytest.raises(ValueError):
        weighted_mean_variance([1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------
# The season boundary (§4)
# ---------------------------------------------------------------------------


def test_uniform_prior_season_discount_cancels_out_of_the_mean():
    """Week 1 2026: every game is prior-season, so the 0.85 is a no-op for ranking (D3)."""
    current = compute_baseline(games(110, 80, 95, 60, 130, 70))
    all_prior = compute_baseline(games(110, 80, 95, 60, 130, 70, prior=True))

    assert all_prior.mean == pytest.approx(current.mean)
    assert all_prior.n_prior_season == 6


def test_mixed_history_downweights_the_prior_season_games():
    """One current-season game plus five prior-season ones must lean toward the current game."""
    obs = [GameObservation(2026, 1, 110.0)] + [
        GameObservation(2025, 18 - i, 80.0, is_prior_season=True) for i in range(5)
    ]
    mixed = compute_baseline(obs)
    same_values_no_discount = compute_baseline(
        [GameObservation(2026, 1, 110.0)]
        + [GameObservation(2026, 1 - i, 80.0) for i in range(1, 6)]
    )

    assert mixed.mean > same_values_no_discount.mean


def test_window_caps_how_many_games_are_used():
    """§5.1: N = 6. A longer history must not sneak in."""
    b = compute_baseline(games(*([100.0] * 10)), window=6)
    assert b.n_games == 6


# ---------------------------------------------------------------------------
# Insufficient history (§4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,expected", [(0, True), (1, True), (2, True), (3, False), (6, False)])
def test_insufficient_history_threshold(n, expected):
    """§4: fewer than 3 games means we project from priors instead of pretending."""
    b = compute_baseline(games(*([50.0] * n)), min_games=3)
    assert b.insufficient_history is expected


def test_blending_pulls_a_thin_baseline_toward_the_prior():
    thin = compute_baseline(games(120.0))
    blended = blend_with_prior(thin, prior_mean=60.0, prior_variance=400.0, prior_weight_games=3.0)

    assert 60.0 < blended.mean < 120.0


def test_blending_barely_moves_a_full_baseline():
    full = compute_baseline(games(100, 100, 100, 100, 100, 100))
    blended = blend_with_prior(full, prior_mean=50.0, prior_variance=400.0, prior_weight_games=3.0)

    assert blended.mean == pytest.approx(100 * full.effective_n / (full.effective_n + 3) + 50 * 3 / (full.effective_n + 3))
    assert blended.mean > 80.0


def test_blending_keeps_the_disagreement_as_variance():
    """A player and prior that disagree are genuinely uncertain; the blend must not hide that."""
    thin = compute_baseline(games(120.0, 118.0))
    blended = blend_with_prior(thin, prior_mean=40.0, prior_variance=100.0, prior_weight_games=3.0)

    assert blended.variance > max(thin.variance, 100.0)


# ---------------------------------------------------------------------------
# Defensive shrinkage (§5.2)
# ---------------------------------------------------------------------------


def test_shrinkage_pulls_a_small_sample_hard_toward_league_average():
    """§5.2: k = 6 games of league-average pseudo-data."""
    two_games = shrink_multiplier(raw_value=140, league_average=100, n_games=2, k=6)
    eight_games = shrink_multiplier(raw_value=140, league_average=100, n_games=8, k=6)

    assert 1.0 < two_games < eight_games < 1.4
    assert two_games == pytest.approx((2 * 140 + 6 * 100) / (2 + 6) / 100)


def test_shrinkage_is_symmetric_around_one():
    stingy = shrink_multiplier(raw_value=60, league_average=100, n_games=4, k=6)
    generous = shrink_multiplier(raw_value=140, league_average=100, n_games=4, k=6)

    assert stingy < 1.0 < generous
    assert (1.0 - stingy) == pytest.approx(generous - 1.0)


def test_league_average_defense_gets_a_multiplier_of_one():
    assert shrink_multiplier(100, 100, n_games=5) == pytest.approx(1.0)


def test_zero_games_falls_back_to_league_average():
    """An unplayed defense must be neutral, never a divide-by-zero or an extreme."""
    assert shrink_multiplier(140, 100, n_games=0) == 1.0
    assert shrink_multiplier(140, 0, n_games=5) == 1.0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_dispersion_ratio_feeds_the_overdispersion_test():
    """§5.6 tests variance/mean > 1.3 to pick negative binomial over Poisson."""
    b = compute_baseline(games(1, 9, 1, 9, 1, 9))
    assert b.dispersion_ratio == pytest.approx(b.variance / b.mean)


def test_baseline_serialises_every_input_for_show_math():
    """§1.4: every projection must be reproducible from the values shown."""
    b = compute_baseline(games(110, 80, 95))
    payload = b.to_json()

    assert payload["values"] == [110.0, 80.0, 95.0]
    assert len(payload["weights"]) == 3
    assert payload["n_games"] == 3


def test_zero_weights_do_not_produce_nan():
    assert weighted_mean_variance([1.0, 2.0], [0.0, 0.0]) == (0.0, 0.0)
    assert isinstance(Baseline(0.0, 0.0, 0, 0.0).sd, float)
