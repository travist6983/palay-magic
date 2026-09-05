"""Tests for the projection distributions (§5.6).

These check the math against analytic values and against the published base rates in
``docs/reference/``. If a change here breaks a documented base rate, the model is wrong, not the
test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from backend.models.distributions import (
    fit_anytime_td,
    fit_count,
    fit_deterministic,
    fit_longest,
    fit_negative_binomial,
    fit_poisson,
)
from backend.models.stats import Family

# ---------------------------------------------------------------------------
# Negative binomial (yardage stats)
# ---------------------------------------------------------------------------


def test_negative_binomial_moments_match_the_inputs():
    """Moment matching must actually reproduce the mean and variance it was given."""
    mean, variance = 72.0, 900.0
    d = fit_negative_binomial(mean, variance)

    r, p = d.params["r"], d.params["p"]
    assert r * (1 - p) / p == pytest.approx(mean, rel=1e-9)
    assert r * (1 - p) / p**2 == pytest.approx(variance, rel=1e-9)


def test_negative_binomial_alpha_is_the_overdispersion_parameter():
    """var = mean + alpha * mean^2 is the standard parameterisation we store."""
    mean, variance = 60.0, 660.0
    d = fit_negative_binomial(mean, variance)
    alpha = d.params["alpha"]

    assert mean + alpha * mean**2 == pytest.approx(variance, rel=1e-9)


def test_negative_binomial_quantiles_match_scipy():
    d = fit_negative_binomial(70.0, 700.0)
    r, p = d.params["r"], d.params["p"]

    assert d.median == pytest.approx(stats.nbinom.ppf(0.5, r, p))
    assert d.p25 == pytest.approx(stats.nbinom.ppf(0.25, r, p))
    assert d.p75 == pytest.approx(stats.nbinom.ppf(0.75, r, p))


def test_negative_binomial_is_right_skewed():
    """Yardage is right-skewed, which is exactly why the line sits at the median, not the mean."""
    d = fit_negative_binomial(70.0, 900.0)
    assert d.median < d.mean


def test_underdispersed_input_is_floored_not_silently_dropped():
    """A negative binomial needs var > mean. Under-dispersed input must be flagged, not hidden."""
    d = fit_negative_binomial(50.0, 10.0)

    assert d.params["variance"] > d.mean
    assert any("floored" in n for n in d.notes)


def test_negative_binomial_prob_over_is_a_survival_function():
    d = fit_negative_binomial(70.0, 700.0)

    assert d.prob_over(0) > d.prob_over(70) > d.prob_over(200)
    assert 0.0 <= d.prob_over(70.0) <= 1.0
    assert d.prob_over(70.5) == pytest.approx(1.0 - stats.nbinom.cdf(70, d.params["r"], d.params["p"]))


# ---------------------------------------------------------------------------
# Poisson (count stats)
# ---------------------------------------------------------------------------


def test_poisson_reproduces_the_documented_passing_td_base_rates():
    """From how_books_build_lines.md: passing TDs average ~1.5/start and are near-perfectly Poisson.

    The doc reports Over 1.5 hitting 45.9% and 21.1% of starts finishing with zero. A pure
    Poisson(1.5) gives 44.2% and 22.3% -- within a point and a half of the empirical sample,
    which is the agreement the doc's 'almost perfectly Poisson' claim implies.
    """
    d = fit_poisson(1.5)

    assert d.prob_over(1.5) == pytest.approx(0.4422, abs=0.02)
    assert d.prob_exact(0) == pytest.approx(0.2231, abs=0.02)


def test_poisson_variance_equals_mean():
    d = fit_poisson(2.4)
    assert d.params["variance"] == pytest.approx(d.params["lam"])


def test_fit_count_uses_poisson_when_not_overdispersed():
    d = fit_count(mean=5.0, variance=5.5, overdispersion_threshold=1.3)
    assert d.family is Family.POISSON


def test_fit_count_switches_to_negative_binomial_when_overdispersed():
    """§5.6: negative binomial when variance/mean > 1.3."""
    d = fit_count(mean=5.0, variance=9.0, overdispersion_threshold=1.3)

    assert d.family is Family.NEGATIVE_BINOMIAL
    assert d.params["variance_mean_ratio"] == pytest.approx(1.8)


def test_fit_count_threshold_is_exclusive():
    """Exactly at the threshold stays Poisson -- the extra parameter buys nothing there."""
    assert fit_count(mean=10.0, variance=13.0, overdispersion_threshold=1.3).family is Family.POISSON


# ---------------------------------------------------------------------------
# Touchdowns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lam", [0.1, 0.35, 0.7, 1.2, 2.0])
def test_anytime_td_is_one_minus_exp_neg_lambda(lam):
    """§5.6: P(anytime TD) = 1 - e^(-lambda)."""
    assert fit_anytime_td(lam).params["p"] == pytest.approx(1 - math.exp(-lam))


def test_anytime_td_at_the_documented_workhorse_rate():
    """DataStreak: workhorse RBs (10+ carries) scored in 49.6% of games.

    That implies lambda ~= -ln(1 - 0.496) = 0.685.
    """
    lam = -math.log(1 - 0.496)
    assert fit_anytime_td(lam).params["p"] == pytest.approx(0.496, abs=1e-6)


def test_anytime_td_probability_is_the_mean_of_the_bernoulli():
    d = fit_anytime_td(0.8)
    assert d.mean == pytest.approx(d.params["p"])
    assert d.prob_over(0.5) == pytest.approx(d.params["p"])


def test_zero_lambda_gives_zero_td_probability():
    assert fit_anytime_td(0.0).params["p"] == 0.0


# ---------------------------------------------------------------------------
# Longest X
# ---------------------------------------------------------------------------


def test_longest_carries_a_point_mass_at_zero():
    """A longest-X prop settles Under when the player records no such play (D9)."""
    d = fit_longest(n_plays=1.0, explosive_rate=0.1, yards_mean=6.0, yards_scale=8.0, seed=7)

    assert d.params["p_zero"] > 0.0
    assert d.params["p_zero"] == pytest.approx(math.exp(-1.0), abs=0.05)


def test_longest_grows_with_more_opportunities():
    low = fit_longest(n_plays=3, explosive_rate=0.1, yards_mean=7.0, yards_scale=9.0, seed=1)
    high = fit_longest(n_plays=15, explosive_rate=0.1, yards_mean=7.0, yards_scale=9.0, seed=1)

    assert high.median > low.median


def test_longest_grows_with_a_heavier_explosive_tail():
    dull = fit_longest(n_plays=8, explosive_rate=0.02, yards_mean=7.0, yards_scale=9.0, seed=3)
    boom = fit_longest(n_plays=8, explosive_rate=0.30, yards_mean=7.0, yards_scale=9.0, seed=3)

    assert boom.median > dull.median


def test_longest_is_reproducible():
    """'Show math' is only meaningful if the same inputs give the same number every time (§1.4)."""
    kwargs = dict(n_plays=9, explosive_rate=0.12, yards_mean=7.5, yards_scale=9.0, seed=42)
    assert fit_longest(**kwargs).median == fit_longest(**kwargs).median


def test_longest_prob_over_falls_monotonically():
    d = fit_longest(n_plays=10, explosive_rate=0.12, yards_mean=7.5, yards_scale=9.0, seed=5)
    probs = [d.prob_over(x) for x in (5, 15, 25, 40, 60)]

    assert probs == sorted(probs, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probs)


# ---------------------------------------------------------------------------
# Deterministic combinations
# ---------------------------------------------------------------------------


def test_kicking_points_mean_is_three_fgm_plus_one_xpm():
    """§5.6 / the prop reference: kicking points = 3 x FGM + 1 x XPM."""
    d = fit_deterministic(
        [("fg_made", 3.0, fit_poisson(1.6)), ("xp_made", 1.0, fit_poisson(2.3))]
    )

    assert d.mean == pytest.approx(3 * 1.6 + 1 * 2.3, rel=1e-3)


def test_kicking_points_support_is_reachable_combinations_only():
    """3*FGM + 1*XPM can never produce a value no (FGM, XPM) pair reaches."""
    d = fit_deterministic(
        [("fg_made", 3.0, fit_poisson(1.0)), ("xp_made", 1.0, fit_poisson(1.0))]
    )
    support = set(d.params["support"])
    reachable = {3 * f + x for f in range(0, 12) for x in range(0, 12)}

    assert support <= reachable


def test_deterministic_probabilities_sum_to_one():
    d = fit_deterministic(
        [("fg_made", 3.0, fit_poisson(1.5)), ("xp_made", 1.0, fit_poisson(2.0))]
    )
    assert sum(d.params["probs"]) == pytest.approx(1.0, abs=1e-6)


def test_deterministic_prob_over_matches_a_direct_simulation():
    rng = np.random.default_rng(11)
    fgm = rng.poisson(1.5, 200_000)
    xpm = rng.poisson(2.0, 200_000)
    empirical = float(((3 * fgm + xpm) > 7.5).mean())

    d = fit_deterministic(
        [("fg_made", 3.0, fit_poisson(1.5)), ("xp_made", 1.0, fit_poisson(2.0))]
    )

    assert d.prob_over(7.5) == pytest.approx(empirical, abs=0.01)


def test_empty_terms_is_a_zero_distribution():
    d = fit_deterministic([])
    assert d.mean == 0.0


# ---------------------------------------------------------------------------
# Shared contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dist",
    [
        fit_negative_binomial(70.0, 700.0),
        fit_poisson(1.5),
        fit_anytime_td(0.7),
        fit_longest(n_plays=8, explosive_rate=0.1, yards_mean=7.0, yards_scale=9.0, seed=2),
        fit_deterministic([("fg_made", 3.0, fit_poisson(1.5)), ("xp_made", 1.0, fit_poisson(2.0))]),
    ],
)
def test_every_family_reports_ordered_quantiles(dist):
    assert dist.p25 <= dist.median <= dist.p75


@pytest.mark.parametrize(
    "dist",
    [
        fit_negative_binomial(70.0, 700.0),
        fit_poisson(1.5),
        fit_anytime_td(0.7),
        fit_longest(n_plays=8, explosive_rate=0.1, yards_mean=7.0, yards_scale=9.0, seed=2),
        fit_deterministic([("fg_made", 3.0, fit_poisson(1.5)), ("xp_made", 1.0, fit_poisson(2.0))]),
    ],
)
def test_every_family_serialises_everything_the_browser_needs(dist):
    """D10: the UI recomputes P(over) from these params alone -- no API call per keystroke."""
    payload = dist.to_json()

    assert set(payload) >= {"family", "params", "mean", "median", "p25", "p75", "integer_valued"}
    import json

    json.loads(json.dumps(payload))  # must be JSON-serialisable as-is


@pytest.mark.parametrize("line", [0.5, 10.5, 49.5, 99.5])
def test_over_and_under_partition_the_probability(line):
    """With a half-point line a push is impossible, so P(over) + P(under) must be exactly 1."""
    d = fit_negative_binomial(60.0, 600.0)
    assert d.prob_over(line) + d.prob_under(line) == pytest.approx(1.0, abs=1e-9)


def test_integer_line_leaves_room_for_a_push():
    """At a whole number an exact match is a push -- over and under must not sum to 1."""
    d = fit_poisson(2.0)
    assert d.prob_exact(2) > 0
    assert d.prob_over(2) + d.prob_under(2) == pytest.approx(1.0 - d.prob_exact(2))
