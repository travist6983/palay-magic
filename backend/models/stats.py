"""The canonical stat registry (docs/DECISIONS.md D9).

One place that answers, for every position and stat PropLab projects:

* what the stat is called everywhere (DB, API, UI),
* which nflverse column it is derived from,
* which distribution family models it (§5.6),
* which defensive metric its opponent adjustment uses (§5.2),
* and the settlement rule that decides whether we are even measuring the right thing
  (``sharp_bettor_prop_reference.md``).

Nothing downstream may invent a stat key. If it is not here, it is not projected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Position(StrEnum):
    """The six prop-relevant positions PropLab ranks (§1)."""

    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    K = "K"
    LB = "LB"


class Family(StrEnum):
    """Distribution families (§5.6). The UI recomputes P(over) from stored params (D10)."""

    NEGATIVE_BINOMIAL = "negative_binomial"
    """Right-skewed, over-dispersed yardage. params: {mean, dispersion (alpha)}."""

    POISSON = "poisson"
    """Counts with variance ~= mean. params: {lam}."""

    BERNOULLI = "bernoulli"
    """Anytime-TD style yes/no, derived from a Poisson lambda. params: {p, lam}."""

    EMPIRICAL_MAX = "empirical_max"
    """`longest_*`: Monte Carlo max over n plays, with an explicit point mass at 0.

    params: {quantiles: {...}, p_zero, n_plays, explosive_rate}. Settles Under when the
    player records no such play, so the zero mass is part of the distribution, not an
    edge case (D9).
    """

    DETERMINISTIC = "deterministic"
    """A linear combination of other stats, e.g. kicking_points = 3*FGM + 1*XPM.

    params: {terms: [{stat, coefficient}], mean}.
    """


class Aggregation(StrEnum):
    """How a stat's opponent adjustment is applied."""

    VOLUME = "volume"
    """Usage-driven: adjust by the defense's plays/pace/pass-rate effect."""

    EFFICIENCY = "efficiency"
    """Per-opportunity: adjust by the defense's yards-allowed multiplier."""

    SCORING = "scoring"
    """Red-zone/TD-rate driven."""


@dataclass(frozen=True)
class StatSpec:
    """Everything PropLab needs to know about one projectable stat."""

    key: str
    label: str
    positions: tuple[Position, ...]
    family: Family
    aggregation: Aggregation
    source_column: str | None
    """Column in the ``raw_player_stats`` view, or None when derived from pbp/other stats."""

    defense_metric: str
    """Key into ``defense_multipliers.metric`` used to adjust this stat (§5.2)."""

    integer_valued: bool = True
    high_variance: bool = False
    """Labelled 'high variance' in the UI (§5.6): sacks and interceptions."""

    settlement_note: str = ""
    derived_from: tuple[str, ...] = field(default_factory=tuple)

    projected: bool = True
    """False for a stat carried only as a DENOMINATOR (an RB's targets). It is written to the
    adjusted game log so per-target rates can be computed, but never projected or displayed.
    Without it every RB's catch rate and yards-per-target fell to the league prior with 0 games."""


# ---------------------------------------------------------------------------
# The registry (D9). Order within a position is the UI's display order.
# ---------------------------------------------------------------------------

_QB = (Position.QB,)
_RB = (Position.RB,)
_WR = (Position.WR,)
_TE = (Position.TE,)
_K = (Position.K,)
_LB = (Position.LB,)
_WRTE = (Position.WR, Position.TE)

STAT_SPECS: tuple[StatSpec, ...] = (
    # --- QB -----------------------------------------------------------------
    StatSpec("pass_attempts", "Pass attempts", _QB, Family.NEGATIVE_BINOMIAL, Aggregation.VOLUME,
             "attempts", "pass_volume_allowed",
             settlement_note="A sack is not a pass attempt; an interception is."),
    StatSpec("completions", "Completions", _QB, Family.NEGATIVE_BINOMIAL, Aggregation.VOLUME,
             "completions", "completion_rate_allowed"),
    StatSpec("passing_yards", "Passing yards", _QB, Family.NEGATIVE_BINOMIAL, Aggregation.EFFICIENCY,
             "passing_yards", "pass_yards_per_attempt_allowed", integer_valued=False,
             settlement_note="Official passing yards, overtime included."),
    StatSpec("passing_tds", "Passing TDs", _QB, Family.POISSON, Aggregation.SCORING,
             "passing_tds", "pass_td_rate_allowed",
             settlement_note="Almost perfectly Poisson; ~21% of starts are zero."),
    StatSpec("interceptions", "Interceptions", _QB, Family.POISSON, Aggregation.SCORING,
             "passing_interceptions", "int_rate_generated", high_variance=True,
             settlement_note="Near-random week to week. Small lambda."),
    StatSpec("rush_attempts", "Rush attempts", _QB, Family.NEGATIVE_BINOMIAL, Aggregation.VOLUME,
             "carries", "rush_volume_allowed"),
    StatSpec("rushing_yards", "Rushing yards", _QB, Family.NEGATIVE_BINOMIAL, Aggregation.EFFICIENCY,
             "rushing_yards", "rush_yards_per_carry_allowed", integer_valued=False,
             settlement_note="QB rushing yards count only toward QB rushing props."),
    StatSpec("longest_completion", "Longest completion", _QB, Family.EMPIRICAL_MAX,
             Aggregation.EFFICIENCY, None, "explosive_pass_allowed", integer_valued=False,
             settlement_note="Settles Under if the QB records no completion."),
    StatSpec("anytime_rush_td", "Anytime rushing TD", _QB, Family.BERNOULLI, Aggregation.SCORING,
             "rushing_tds", "rush_td_rate_allowed",
             settlement_note="Rushing scores only. A passing TD never counts."),
    # --- RB -----------------------------------------------------------------
    StatSpec("rush_attempts", "Rush attempts", _RB, Family.NEGATIVE_BINOMIAL, Aggregation.VOLUME,
             "carries", "rush_volume_allowed",
             settlement_note="Kneel-downs count as negative rushing attempts."),
    StatSpec("targets", "Targets", _RB, Family.NEGATIVE_BINOMIAL, Aggregation.VOLUME,
             "targets", "rec_volume_allowed_rb", projected=False),
    StatSpec("rushing_yards", "Rushing yards", _RB, Family.NEGATIVE_BINOMIAL, Aggregation.EFFICIENCY,
             "rushing_yards", "rush_yards_allowed_rb", integer_valued=False),
    StatSpec("receptions", "Receptions", _RB, Family.NEGATIVE_BINOMIAL, Aggregation.VOLUME,
             "receptions", "rec_volume_allowed_rb"),
    StatSpec("receiving_yards", "Receiving yards", _RB, Family.NEGATIVE_BINOMIAL,
             Aggregation.EFFICIENCY, "receiving_yards", "rec_yards_allowed_rb",
             integer_valued=False),
    StatSpec("rush_rec_yards", "Rush + rec yards", _RB, Family.NEGATIVE_BINOMIAL,
             Aggregation.EFFICIENCY, None, "rush_yards_allowed_rb", integer_valued=False,
             derived_from=("rushing_yards", "receiving_yards"),
             settlement_note="Sum of official rushing and receiving yards."),
    StatSpec("longest_rush", "Longest rush", _RB, Family.EMPIRICAL_MAX, Aggregation.EFFICIENCY,
             None, "explosive_rush_allowed", integer_valued=False,
             settlement_note="Settles Under if no carry is recorded."),
    StatSpec("anytime_td", "Anytime TD", _RB, Family.BERNOULLI, Aggregation.SCORING,
             None, "rz_td_rate_allowed",
             derived_from=("rushing_tds", "receiving_tds"),
             settlement_note="Requires possession in the end zone. Goal-line role dominates."),
    # --- WR -----------------------------------------------------------------
    StatSpec("targets", "Targets", _WR, Family.NEGATIVE_BINOMIAL, Aggregation.VOLUME,
             "targets", "target_volume_allowed_wr",
             settlement_note="The purest opportunity market: strips out efficiency variance."),
    StatSpec("receptions", "Receptions", _WR, Family.NEGATIVE_BINOMIAL, Aggregation.VOLUME,
             "receptions", "rec_volume_allowed_wr"),
    StatSpec("receiving_yards", "Receiving yards", _WR, Family.NEGATIVE_BINOMIAL,
             Aggregation.EFFICIENCY, "receiving_yards", "rec_yards_allowed_wr",
             integer_valued=False),
    StatSpec("longest_reception", "Longest reception", _WR, Family.EMPIRICAL_MAX,
             Aggregation.EFFICIENCY, None, "explosive_pass_allowed", integer_valued=False,
             settlement_note="Settles Under if no catch is recorded."),
    StatSpec("anytime_td", "Anytime TD", _WR, Family.BERNOULLI, Aggregation.SCORING,
             "receiving_tds", "rz_td_rate_allowed",
             settlement_note="Red-zone target share matters more than total yardage."),
    # --- TE -----------------------------------------------------------------
    StatSpec("targets", "Targets", _TE, Family.NEGATIVE_BINOMIAL, Aggregation.VOLUME,
             "targets", "target_volume_allowed_te"),
    StatSpec("receptions", "Receptions", _TE, Family.NEGATIVE_BINOMIAL, Aggregation.VOLUME,
             "receptions", "rec_volume_allowed_te"),
    StatSpec("receiving_yards", "Receiving yards", _TE, Family.NEGATIVE_BINOMIAL,
             Aggregation.EFFICIENCY, "receiving_yards", "rec_yards_allowed_te",
             integer_valued=False),
    StatSpec("anytime_td", "Anytime TD", _TE, Family.BERNOULLI, Aggregation.SCORING,
             "receiving_tds", "rz_td_rate_allowed",
             settlement_note="TEs see a larger share of targets inside the 20."),
    # --- K ------------------------------------------------------------------
    StatSpec("fg_attempts", "FG attempts", _K, Family.POISSON, Aggregation.SCORING,
             "fg_att", "fg_attempts_allowed",
             settlement_note="Driven by how often the offense stalls in range."),
    StatSpec("fg_made", "FG made", _K, Family.POISSON, Aggregation.SCORING,
             "fg_made", "fg_attempts_allowed",
             settlement_note="Make% falls ~5-8% per 10 yards past 40; wind is the big variable."),
    StatSpec("xp_made", "XP made", _K, Family.POISSON, Aggregation.SCORING,
             "pat_made", "rz_td_rate_allowed",
             settlement_note="A two-point conversion never credits the kicker."),
    StatSpec("kicking_points", "Kicking points", _K, Family.DETERMINISTIC, Aggregation.SCORING,
             None, "fg_attempts_allowed", integer_valued=False,
             derived_from=("fg_made", "xp_made"),
             settlement_note="3 x FGM + 1 x XPM. Bets stand if an active kicker plays no snap."),
    StatSpec("longest_fg", "Longest FG", _K, Family.EMPIRICAL_MAX, Aggregation.SCORING,
             "fg_long", "fg_attempts_allowed", integer_valued=False,
             settlement_note="Settles Under if the kicker makes no field goal."),
    # --- LB -----------------------------------------------------------------
    StatSpec("tackles_assists", "Tackles + assists", _LB, Family.NEGATIVE_BINOMIAL,
             Aggregation.VOLUME, None, "opp_plays_allowed",
             derived_from=("def_tackles_solo", "def_tackle_assists"),
             settlement_note=(
                 "DraftKings convention: DEFENSIVE plays only, special teams excluded (D2). "
                 "Settles off the official gamebook, never PFF or team charting."
             )),
    StatSpec("solo_tackles", "Solo tackles", _LB, Family.NEGATIVE_BINOMIAL, Aggregation.VOLUME,
             "def_tackles_solo", "opp_plays_allowed",
             settlement_note="Noisier than combined: the solo/assist split is scorer-dependent."),
    StatSpec("sacks", "Sacks", _LB, Family.POISSON, Aggregation.SCORING,
             "def_sacks", "pressure_allowed", integer_valued=False, high_variance=True,
             settlement_note="Half-sacks count 0.5 and settle Yes on an anytime-sack (0.5) line."),
    StatSpec("passes_defended", "Passes defended", _LB, Family.POISSON, Aggregation.VOLUME,
             "def_pass_defended", "opp_dropbacks", high_variance=True,
             settlement_note="More volume-stable than INTs, but still a longshot market."),
)


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

STATS_BY_POSITION: dict[Position, tuple[StatSpec, ...]] = {
    pos: tuple(s for s in STAT_SPECS if pos in s.positions) for pos in Position
}

_BY_KEY: dict[tuple[Position, str], StatSpec] = {
    (pos, spec.key): spec for spec in STAT_SPECS for pos in spec.positions
}


def stats_for(position: Position | str, include_denominators: bool = False) -> tuple[StatSpec, ...]:
    """Every stat PropLab projects for a position, in display order.

    ``include_denominators`` adds the denominator-only stats (an RB's targets), which the adjusted
    game log needs and nothing user-facing should show.
    """
    specs = STATS_BY_POSITION[Position(position)]
    return specs if include_denominators else tuple(s for s in specs if s.projected)


def get_spec(position: Position | str, key: str) -> StatSpec:
    """Look up one stat spec. Raises KeyError for a stat this position does not have."""
    return _BY_KEY[(Position(position), key)]


def has_stat(position: Position | str, key: str) -> bool:
    return (Position(position), key) in _BY_KEY


def all_stat_keys() -> tuple[str, ...]:
    """Every distinct stat key, deduplicated across positions."""
    seen: dict[str, None] = {}
    for spec in STAT_SPECS:
        seen.setdefault(spec.key, None)
    return tuple(seen)


def defense_metrics() -> tuple[str, ...]:
    """Every defensive metric the registry references. adjust.py must compute all of these."""
    return tuple(dict.fromkeys(s.defense_metric for s in STAT_SPECS))


# The headline projections shown on each position's ranked table (§8).
HEADLINE_STATS: dict[Position, tuple[str, ...]] = {
    Position.QB: ("passing_yards", "passing_tds", "pass_attempts"),
    Position.RB: ("rushing_yards", "rush_rec_yards", "anytime_td"),
    Position.WR: ("receiving_yards", "receptions", "anytime_td"),
    Position.TE: ("receiving_yards", "receptions", "anytime_td"),
    Position.K: ("kicking_points", "fg_made", "fg_attempts"),
    Position.LB: ("tackles_assists", "solo_tackles", "sacks"),
}
