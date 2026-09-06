"""Tests for the injury play-probability layer (§5.7) and target redistribution (§5.4)."""

from __future__ import annotations

import math

import pytest

from backend.models.injury import (
    EXCLUDED_STATUSES,
    apply_unconditional,
    coin_flip_distance,
    inflate_variance_for_uncertainty,
    is_eligible,
    log_loss,
    practice_participation_signal,
    redistribute_target_share,
)

# ---------------------------------------------------------------------------
# Eligibility (§4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["Out", "IR", "PUP", "Doubtful", "Injured Reserve"])
def test_excluded_designations_drop_off_the_board(status):
    """§4: exclude anyone whose status is Out, IR, PUP, or Doubtful."""
    assert not is_eligible(status)
    assert status in EXCLUDED_STATUSES


def test_questionable_players_stay_on_the_board():
    """§4: Questionable players stay, with their play probability displayed."""
    assert is_eligible("Questionable")


def test_no_designation_means_healthy():
    assert is_eligible(None)
    assert is_eligible("")


# ---------------------------------------------------------------------------
# Conditional vs unconditional (§5.7)
# ---------------------------------------------------------------------------


def test_unconditional_scales_by_play_probability():
    """§5.7: E[X] = P(played) * E[X | played]."""
    assert apply_unconditional(85.0, 0.6) == pytest.approx(51.0)


def test_a_certain_starter_has_identical_conditional_and_unconditional_numbers():
    assert apply_unconditional(85.0, 1.0) == 85.0


def test_play_probability_is_clamped():
    assert apply_unconditional(85.0, 1.5) == 85.0
    assert apply_unconditional(85.0, -0.2) == 0.0


def test_uncertainty_inflates_the_variance():
    """A coin-flip Questionable must show a much wider p25-p75 band than a healthy player."""
    healthy = inflate_variance_for_uncertainty(85.0, 400.0, 1.0)
    coin_flip = inflate_variance_for_uncertainty(85.0, 400.0, 0.5)

    assert healthy == pytest.approx(400.0)
    assert coin_flip > 4 * healthy


def test_variance_formula_matches_the_mixture():
    """Var[X] = p(var + mu^2) - (p*mu)^2 for a play/don't-play mixture with a hard zero."""
    mu, var, p = 85.0, 400.0, 0.6
    expected = p * (var + mu**2) - (p * mu) ** 2

    assert inflate_variance_for_uncertainty(mu, var, p) == pytest.approx(expected)


def test_a_player_who_cannot_play_has_no_variance():
    assert inflate_variance_for_uncertainty(85.0, 400.0, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Target redistribution (§5.4)
# ---------------------------------------------------------------------------


def test_redistribution_conserves_the_absent_share():
    """The missing share has to land somewhere -- none of it may evaporate."""
    remaining = {"WR2": 0.18, "TE": 0.14, "RB": 0.10}
    out = redistribute_target_share(0.24, remaining)

    assert sum(out.values()) == pytest.approx(sum(remaining.values()) + 0.24)


def test_default_split_is_sixty_twentyfive_fifteen():
    """§5.4: absent 60/25/15 to WR2/TE/RB when no games without the absent player exist."""
    out = redistribute_target_share(0.20, {"WR2": 0.0, "TE": 0.0, "RB": 0.0})

    assert out["WR2"] == pytest.approx(0.12)
    assert out["TE"] == pytest.approx(0.05)
    assert out["RB"] == pytest.approx(0.03)


def test_historical_shares_take_priority_over_the_default_split():
    """§5.4 prefers what actually happened in the games the absent player missed."""
    remaining = {"WR2": 0.18, "TE": 0.14, "RB": 0.10}
    out = redistribute_target_share(
        0.24, remaining, historical_without={"WR2": 0.30, "TE": 0.20, "RB": 0.10}
    )

    assert out["WR2"] == pytest.approx(0.18 + 0.24 * 0.5)
    assert out["TE"] == pytest.approx(0.14 + 0.24 * (1 / 3))
    assert sum(out.values()) == pytest.approx(sum(remaining.values()) + 0.24)


def test_missing_role_keys_spread_proportionally_instead_of_being_lost():
    """A team with no receiving TE must not silently drop the TE's 25%."""
    remaining = {"WR2": 0.20, "RB": 0.10}
    out = redistribute_target_share(0.20, remaining)

    assert sum(out.values()) == pytest.approx(0.50)


def test_zero_absent_share_is_a_no_op():
    remaining = {"WR2": 0.18, "TE": 0.14}
    assert redistribute_target_share(0.0, remaining) == remaining


def test_empty_roster_does_not_crash():
    assert redistribute_target_share(0.24, {}) == {}


# ---------------------------------------------------------------------------
# Normalisation and reporting helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Full Participation in Practice", "Full"),
        ("Limited Participation in Practice", "Limited"),
        ("Did Not Participate In Practice", "DNP"),
        ("DNP", "DNP"),
        ("full", "Full"),
        (None, "Unknown"),
        ("", "Unknown"),
    ],
)
def test_practice_status_normalisation(raw, expected):
    """nflverse and Sleeper spell practice participation differently; the model needs one key."""
    assert practice_participation_signal(raw) == expected


def test_coin_flip_distance_is_zero_at_a_coin_flip_and_one_at_certainty():
    assert coin_flip_distance(0.5) == pytest.approx(0.0)
    assert coin_flip_distance(1.0) == pytest.approx(1.0)
    assert coin_flip_distance(0.0) == pytest.approx(1.0)
    assert coin_flip_distance(0.95) == pytest.approx(0.9)


def test_log_loss_rewards_a_confident_correct_call():
    assert log_loss(0.95, played=True) < log_loss(0.55, played=True)
    assert log_loss(0.05, played=False) < log_loss(0.45, played=False)


def test_log_loss_is_finite_at_the_extremes():
    """A 0 or 1 prediction must not produce an infinity that poisons the backtest average."""
    assert math.isfinite(log_loss(0.0, played=True))
    assert math.isfinite(log_loss(1.0, played=False))


# ---------------------------------------------------------------------------
# Role buckets (§5.7, migration 002)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "share,expected",
    [
        (1.00, "starter"),
        (0.80, "starter"),
        (0.55, "starter"),
        (0.54, "rotational"),
        (0.20, "rotational"),
        (0.19, "fringe"),
        (0.00, "fringe"),
        (None, "no_recent_games"),
    ],
)
def test_role_bucket_thresholds(share, expected):
    """Pooling roles drags P(played | Questionable + Full) from 84.8% to 70.6% (migration 002)."""
    from backend.models.injury import role_bucket

    assert role_bucket(share) == expected


def test_a_player_with_no_recent_games_is_not_treated_as_a_starter():
    """A player back from IR has no four-week history; guessing 'starter' would over-project him."""
    from backend.models.injury import ROLE_NO_HISTORY, role_bucket

    assert role_bucket(None) == ROLE_NO_HISTORY
