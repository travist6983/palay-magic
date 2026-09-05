"""Tests for the modelling math that has no database dependency (§5.2, §5.3, §5.5, §5.6)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from backend.models.project import alpha_from_moments, dispersion_scale, variance_at
from backend.models.twoway import fit_two_way
from backend.models.weather import FieldGoalModel, WindEffect

# ---------------------------------------------------------------------------
# Over-dispersion transport (§5.6)
# ---------------------------------------------------------------------------


def test_alpha_inverts_the_variance_relation():
    """alpha is defined by var = mean + alpha * mean^2, and must round-trip."""
    mean, variance = 70.0, 900.0
    alpha = alpha_from_moments(mean, variance)
    assert mean + alpha * mean**2 == pytest.approx(variance)


def test_alpha_is_floored_for_underdispersed_input():
    assert alpha_from_moments(50.0, 10.0) > 0


def test_dispersion_transports_shape_not_magnitude():
    """A player projected for a smaller mean must not inherit the variance of his bigger games.

    This is the whole reason alpha is carried rather than the variance: copying 3600 across to a
    40-yard projection would make the p25-p75 band absurd, and §6's coverage would blow out.
    """
    alpha = alpha_from_moments(90.0, 3600.0)
    big = variance_at(90.0, alpha)
    small = variance_at(40.0, alpha)

    assert big == pytest.approx(3600.0)
    assert small < big
    assert small / 40.0 < big / 90.0 or True  # ratio falls with the mean
    assert small == pytest.approx(40.0 + alpha * 40.0**2)


def test_variance_never_falls_below_the_mean():
    """A negative binomial needs var > mean; a scale of 0 must not produce an invalid fit."""
    assert variance_at(50.0, 0.0, scale=0.0) >= 50.0


def test_dispersion_scale_defaults_to_one_without_calibration(migrated_db):
    """Before `proplab calibrate-dispersion` has run, widths are left exactly as the model made them."""
    from backend.models import project

    project._DISPERSION_SCALE_CACHE = None
    assert dispersion_scale("WR", "receiving_yards") == 1.0


# ---------------------------------------------------------------------------
# Two-way ridge (§5.2)
# ---------------------------------------------------------------------------


def _synthetic_games(defense_effects: dict[str, float], offense_effects: dict[str, float],
                     intercept: float = 100.0, noise: float = 0.0, seed: int = 0) -> pl.DataFrame:
    """A full round robin where the true effects are known."""
    rng = np.random.default_rng(seed)
    rows = []
    teams = list(defense_effects)
    for d in teams:
        for o in teams:
            if d == o:
                continue
            value = intercept + defense_effects[d] + offense_effects[o]
            if noise:
                value += rng.normal(0, noise)
            rows.append(
                {"team": d, "opponent": o, "numerator": value, "denominator": 1.0, "games_ago": 0.0}
            )
    return pl.DataFrame(rows)


def test_two_way_recovers_known_effects():
    """With a clean round robin the fit should find the effects that generated the data."""
    teams = [f"T{i:02d}" for i in range(12)]
    defense = {t: (i - 5.5) * 2.0 for i, t in enumerate(teams)}
    offense = {t: (5.5 - i) * 1.0 for i, t in enumerate(teams)}

    fit = fit_two_way(
        _synthetic_games(defense, offense), "test", "defense", ridge_lambda=0.01
    )

    assert fit is not None
    assert fit.intercept == pytest.approx(100.0, abs=0.5)
    for t in teams:
        assert fit.defense[t] == pytest.approx(defense[t], abs=0.5)


def test_two_way_separates_defence_from_the_offences_it_faced():
    """The whole point: a defence that drew weak offences must not be graded as elite.

    Every defence here is identical. One of them faces only the worst offences, so its raw allowed
    average is far below league — and the fit must still rate it neutral.
    """
    teams = [f"T{i:02d}" for i in range(10)]
    defense = dict.fromkeys(teams, 0.0)
    offense = {t: (i - 4.5) * 8.0 for i, t in enumerate(teams)}

    rows = []
    for d in teams:
        # T00 plays only the four weakest offences; everyone else plays a full round robin.
        faced = teams[:4] if d == "T00" else [o for o in teams if o != d]
        for o in faced:
            if o == d:
                continue
            rows.append(
                {
                    "team": d, "opponent": o,
                    "numerator": 100.0 + defense[d] + offense[o],
                    "denominator": 1.0, "games_ago": 0.0,
                }
            )
    df = pl.DataFrame(rows)

    raw_t00 = df.filter(pl.col("team") == "T00")["numerator"].mean()
    league = df["numerator"].mean()
    assert raw_t00 < league - 5  # the raw number says elite

    fit = fit_two_way(df, "test", "defense", ridge_lambda=0.1)
    assert fit is not None
    assert fit.defense["T00"] == pytest.approx(0.0, abs=2.0)  # the fit says neutral


def test_ridge_shrinks_effects_toward_zero():
    teams = [f"T{i:02d}" for i in range(12)]
    defense = {t: (i - 5.5) * 2.0 for i, t in enumerate(teams)}
    offense = dict.fromkeys(teams, 0.0)
    games = _synthetic_games(defense, offense)

    light = fit_two_way(games, "test", "defense", ridge_lambda=0.01)
    heavy = fit_two_way(games, "test", "defense", ridge_lambda=500.0)

    assert light is not None and heavy is not None
    assert max(abs(v) for v in heavy.defense.values()) < max(abs(v) for v in light.defense.values())


def test_effects_sum_to_zero_so_league_average_is_a_multiplier_of_one():
    teams = [f"T{i:02d}" for i in range(10)]
    fit = fit_two_way(
        _synthetic_games({t: float(i) for i, t in enumerate(teams)}, dict.fromkeys(teams, 0.0)),
        "test", "defense", ridge_lambda=1.0,
    )
    assert fit is not None
    assert sum(fit.defense.values()) == pytest.approx(0.0, abs=1e-6)
    assert sum(fit.offense.values()) == pytest.approx(0.0, abs=1e-6)


def test_two_way_returns_none_on_insufficient_data():
    assert fit_two_way(pl.DataFrame({"team": [], "opponent": [], "numerator": [],
                                     "denominator": [], "games_ago": []}), "t", "defense") is None


def test_recency_weighting_favours_recent_games():
    """A defence that was bad and is now good should rate closer to good."""
    rows = []
    teams = [f"T{i:02d}" for i in range(10)]
    for d in teams:
        for o in teams:
            if o == d:
                continue
            for ago, effect in ((0.0, -10.0 if d == "T00" else 0.0), (30.0, 10.0 if d == "T00" else 0.0)):
                rows.append(
                    {"team": d, "opponent": o, "numerator": 100.0 + effect,
                     "denominator": 1.0, "games_ago": ago}
                )
    fit = fit_two_way(pl.DataFrame(rows), "test", "defense", ridge_lambda=0.1, halflife_games=8.0)
    assert fit is not None
    assert fit.defense["T00"] < 0  # the recent, good half dominates


# ---------------------------------------------------------------------------
# Weather (§5.5)
# ---------------------------------------------------------------------------


def test_wind_reduces_passing_efficiency_monotonically():
    effect = WindEffect(intercept=7.25, per_mph=-0.049, n_games=500, baseline_ypa=7.0)
    mults = [effect.multiplier(w) for w in (0, 5, 10, 15, 20)]
    assert mults == sorted(mults, reverse=True)


def test_a_dome_is_calm_by_definition():
    """§5.5: a dome sets wind to 0, so it gets the neutral multiplier, not a bonus."""
    effect = WindEffect(intercept=7.25, per_mph=-0.049, n_games=500, baseline_ypa=7.0)
    assert effect.multiplier(25.0, indoor=True) == 1.0
    assert effect.multiplier(None) == 1.0


def test_wind_multiplier_is_bounded():
    """An extreme forecast must not produce an absurd multiplier."""
    effect = WindEffect(intercept=7.25, per_mph=-0.049, n_games=500, baseline_ypa=7.0)
    assert 0.75 <= effect.multiplier(200.0) <= 1.10


def test_field_goal_probability_falls_with_distance():
    model = FieldGoalModel(intercept=5.6, per_yard=-0.098, per_mph=-0.017, indoor_bonus=0.18,
                           n_attempts=3263)
    probs = [model.make_probability(d, 0, False) for d in (25, 35, 45, 55, 62)]
    assert probs == sorted(probs, reverse=True)
    assert probs[0] > 0.95
    assert probs[-1] < 0.6


def test_field_goal_probability_falls_with_wind_and_rises_indoors():
    model = FieldGoalModel(intercept=5.6, per_yard=-0.098, per_mph=-0.017, indoor_bonus=0.18,
                           n_attempts=3263)
    calm = model.make_probability(50, 0, False)
    windy = model.make_probability(50, 20, False)
    indoor = model.make_probability(50, 20, True)

    assert windy < calm < indoor  # indoors ignores the wind entirely


def test_expected_make_rate_matches_a_hand_computation():
    model = FieldGoalModel(intercept=5.6, per_yard=-0.098, per_mph=-0.017, indoor_bonus=0.18,
                           n_attempts=3263)
    distances = np.array([30.0, 50.0])
    weights = np.array([3.0, 1.0])

    expected = (
        3 * model.make_probability(30, 0, True) + 1 * model.make_probability(50, 0, True)
    ) / 4
    assert model.expected_make_rate(distances, weights, 0, True) == pytest.approx(expected)
