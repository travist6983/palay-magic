"""The projection engine: usage × efficiency, opponent-adjusted, as a distribution (§5).

Every stat is built the same way:

    volume      = team expected volume (§5.3) × player share (§5.4) × opponent volume multiplier
    efficiency  = opponent-neutral per-opportunity rate (§5.1) × opponent multiplier × weather
    mean        = volume × efficiency
    dispersion  = the player's own historical over-dispersion, transported to the new mean
    projection  = a fitted distribution (§5.6), reported median / p25 / p75 / P(over line)

Two choices worth stating.

**Baselines are computed on opponent-adjusted values, not raw ones.** A player's rate history is
divided through by the defence he faced in each game before being averaged, so the baseline is
"what he does against a league-average opponent". This week's multiplier is then applied once, to
that neutral number. Averaging raw values and multiplying would apply the schedule twice.

**Dispersion is transported, not copied.** A player's variance is estimated at his historical mean;
projecting a different mean means carrying across the *shape* rather than the number. We estimate
his over-dispersion ``alpha`` from ``var = mean + alpha·mean²`` and re-apply it at the projected
mean, falling back to a position-level alpha below six games (§5.6). Copying the raw variance
would make a player projected for 40 yards carry the variance of his 90-yard games, and §6's
p25-p75 coverage would come out far too wide.

Everything is reported twice: conditional on the player being active, and unconditional (§5.7).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from backend.config import get_settings
from backend.db.connection import connect
from backend.logging_setup import get_logger
from backend.models.adjust import get_multiplier, multiplier_lookup
from backend.models.baseline import Baseline, GameObservation, compute_baseline
from backend.models.distributions import (
    Distribution,
    fit_anytime_td,
    fit_count,
    fit_deterministic,
    fit_longest,
    fit_negative_binomial,
    fit_poisson,
)
from backend.models.injury import (
    PlayProbability,
    apply_unconditional,
    inflate_variance_for_uncertainty,
    role_bucket,
)
from backend.models.injury import (
    lookup as injury_lookup,
)
from backend.models.stats import Family, Position, get_spec
from backend.models.weather import (
    FieldGoalModel,
    WindEffect,
    attempt_distance_distribution,
    fit_field_goal_model,
    fit_wind_effect,
    kicking_context,
)

log = get_logger(__name__)

# Position-level over-dispersion fallbacks, used below six games (§5.6). Fitted in
# :func:`fit_position_dispersion` and cached per process.
_DISPERSION_CACHE: dict[tuple[str, str], float] = {}
_RATE_PRIOR_CACHE: dict[tuple[str, str, str], float] = {}
_DISPERSION_SCALE_CACHE: dict[tuple[str, str], float] | None = None

# XP conversion rate net of two-point attempts, so kicking points are not inflated.
XP_PER_TD = 0.94


@dataclass
class MathStep:
    """One line of the "Show math" drawer (§8)."""

    section: str
    label: str
    value: float
    detail: str = ""


@dataclass
class StatProjection:
    """A projected distribution for one stat, plus the trace that produced it."""

    stat: str
    position: str
    distribution: Distribution
    conditional_on_playing: bool
    play_probability: float
    high_variance: bool
    steps: list[MathStep] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "stat": self.stat,
            "position": self.position,
            "conditional_on_playing": self.conditional_on_playing,
            "play_probability": self.play_probability,
            "high_variance": self.high_variance,
            **self.distribution.to_json(),
            "math": [
                {"section": s.section, "label": s.label, "value": s.value, "detail": s.detail}
                for s in self.steps
            ],
        }


@dataclass
class PlayerProjection:
    """Every stat for one player in one week, conditional and unconditional."""

    gsis_id: str
    display_name: str
    position: str
    team: str
    opponent: str
    season: int
    week: int
    play: PlayProbability
    conditional: dict[str, StatProjection] = field(default_factory=dict)
    unconditional: dict[str, StatProjection] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    flags: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "gsis_id": self.gsis_id,
            "display_name": self.display_name,
            "position": self.position,
            "team": self.team,
            "opponent": self.opponent,
            "season": self.season,
            "week": self.week,
            "injury": self.play.to_json(),
            "context": self.context,
            "flags": self.flags,
            "projections": {k: v.to_json() for k, v in self.conditional.items()},
            "unconditional": {k: v.to_json() for k, v in self.unconditional.items()},
        }


# ---------------------------------------------------------------------------
# Dispersion
# ---------------------------------------------------------------------------


def alpha_from_moments(mean: float, variance: float, floor: float = 0.02) -> float:
    """Over-dispersion ``alpha`` from ``var = mean + alpha * mean²``.

    This is the shape parameter that survives a change of mean, which is why it is what we carry
    forward rather than the variance itself.
    """
    if mean <= 0:
        return floor
    return max((variance - mean) / (mean * mean), floor)


def variance_at(mean: float, alpha: float, scale: float = 1.0) -> float:
    """Re-apply an over-dispersion to a new mean, times the calibrated width scale (§6)."""
    return max((mean + alpha * mean * mean) * scale, mean * 1.05)


def dispersion_scale(position: str, stat: str) -> float:
    """The empirically calibrated variance multiplier for this cell (§6, migration 010).

    A model's own variance estimate describes noise around a *known* mean. Ours is estimated off a
    handful of games, so the predictive interval has to be wider than the outcome interval. How
    much wider is measured, not guessed: 1.0 until `proplab calibrate-dispersion` has run.
    """
    global _DISPERSION_SCALE_CACHE
    if _DISPERSION_SCALE_CACHE is None:
        try:
            with connect() as con:
                rows = con.execute(
                    "SELECT position, stat, scale FROM dispersion_calibration"
                ).fetchall()
            _DISPERSION_SCALE_CACHE = {(p, s): float(v) for p, s, v in rows}
        except Exception:  # noqa: BLE001 - table may not exist yet
            _DISPERSION_SCALE_CACHE = {}
    return _DISPERSION_SCALE_CACHE.get((position, stat), 1.0)


def fit_position_dispersion(seasons: list[int] | None = None) -> dict[tuple[str, str], float]:
    """League-wide ``alpha`` per (position, stat), for players with too little history (§5.6)."""
    global _DISPERSION_CACHE
    if _DISPERSION_CACHE:
        return _DISPERSION_CACHE

    seasons = seasons or list(get_settings().seasons)
    season_list = ",".join(str(s) for s in seasons)
    with connect() as con:
        df = con.execute(
            f"""
            SELECT position, stat, avg(value) AS mean, var_samp(value) AS variance, count(*) AS n
            FROM player_game_stats
            WHERE season IN ({season_list}) AND position IN ('QB','RB','WR','TE','K','LB')
            GROUP BY 1, 2 HAVING count(*) >= 200
            """
        ).pl()

    _DISPERSION_CACHE = {
        (r["position"], r["stat"]): alpha_from_moments(r["mean"] or 0.0, r["variance"] or 0.0)
        for r in df.to_dicts()
    }
    log.info("position dispersion fitted for %d (position, stat) pairs", len(_DISPERSION_CACHE))
    return _DISPERSION_CACHE


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class ProjectionContext:
    """Everything shared across the week's projections. Built once per run."""

    season: int
    week: int
    environment: dict[str, dict]
    multipliers: dict[tuple[str, str], float]
    weather: dict[str, dict]
    wind_effect: WindEffect
    fg_model: FieldGoalModel
    fg_distances: tuple[np.ndarray, np.ndarray]
    dispersion: dict[tuple[str, str], float]
    league_implied: float


def build_context(season: int, week: int) -> ProjectionContext:
    """Assemble the week's shared inputs."""
    with connect() as con:
        env = con.execute(
            "SELECT * FROM team_environment WHERE season = ? AND week = ?", [season, week]
        ).pl()

    environment = {r["team"]: r for r in env.to_dicts()}
    league_implied = float(env["implied_total"].mean()) if env.height else 22.0

    return ProjectionContext(
        season=season,
        week=week,
        environment=environment,
        multipliers=multiplier_lookup(season, week),
        weather=kicking_context(season, week),
        wind_effect=fit_wind_effect(),
        fg_model=fit_field_goal_model(),
        fg_distances=attempt_distance_distribution(),
        dispersion=fit_position_dispersion(),
        league_implied=league_implied,
    )


# ---------------------------------------------------------------------------
# Player inputs
# ---------------------------------------------------------------------------


def _player_history(gsis_id: str, season: int, week: int, window: int) -> pl.DataFrame:
    """The player's recent adjusted game log, most recent first."""
    with connect() as con:
        return con.execute(
            """
            SELECT * FROM (
                SELECT a.*, row_number() OVER (
                    PARTITION BY a.stat ORDER BY a.season DESC, a.week DESC
                ) - 1 AS games_ago
                FROM adjusted_game_log a
                WHERE a.gsis_id = ? AND ((a.season < ?) OR (a.season = ? AND a.week < ?))
            ) WHERE games_ago < ?
            """,
            [gsis_id, season, season, week, window],
        ).pl()


def _player_usage(gsis_id: str, season: int, week: int, window: int) -> pl.DataFrame:
    """The player's recent usage rows, most recent first."""
    with connect() as con:
        return con.execute(
            """
            SELECT * FROM (
                SELECT u.*, row_number() OVER (ORDER BY u.season DESC, u.week DESC) - 1 AS games_ago
                FROM player_game_usage u
                WHERE u.gsis_id = ? AND ((u.season < ?) OR (u.season = ? AND u.week < ?))
            ) WHERE games_ago < ?
            """,
            [gsis_id, season, season, week, window],
        ).pl()


def _play_gains(gsis_id: str, kind: str, season: int, week: int, limit: int = 400) -> np.ndarray:
    """The player's observed per-play gains, most recent first (§5.6 longest-X bootstrap).

    ``kind`` is ``reception``, ``completion``, ``rush`` or ``fg``. Returns an empty array when the
    player has no such plays, which sends :func:`fit_longest` down its parametric fallback.
    """
    column = {
        "reception": "receiver_player_id",
        "completion": "passer_player_id",
        "rush": "rusher_player_id",
    }.get(kind)

    with connect() as con:
        if kind == "fg":
            rows = con.execute(
                """
                SELECT kick_distance FROM raw_pbp
                WHERE kicker_player_id = ? AND field_goal_result = 'made'
                  AND season_type = 'REG' AND kick_distance IS NOT NULL
                  AND ((season < ?) OR (season = ? AND week < ?))
                ORDER BY season DESC, week DESC LIMIT ?
                """,
                [gsis_id, season, season, week, limit],
            ).fetchall()
        elif column:
            condition = "rush_attempt = 1" if kind == "rush" else "complete_pass = 1"
            rows = con.execute(
                f"""
                SELECT yards_gained FROM raw_pbp
                WHERE {column} = ? AND {condition} AND season_type = 'REG'
                  AND yards_gained IS NOT NULL
                  AND ((season < ?) OR (season = ? AND week < ?))
                ORDER BY season DESC, week DESC LIMIT ?
                """,
                [gsis_id, season, season, week, limit],
            ).fetchall()
        else:
            return np.array([])

    return np.array([float(r[0]) for r in rows if r[0] is not None])


def _baseline_for(history: pl.DataFrame, stat: str, season: int, adjusted: bool = True) -> Baseline:
    """Recency-weighted USAGE baseline for one stat, on adjusted values by default (§5.1, §5.2).

    Deliberately capped at the short ``recency_window``: the adjusted log now holds the longer
    efficiency window, and a volume baseline should not reach back across a role change.
    """
    window = get_settings().recency_window
    rows = (
        history.filter((pl.col("stat") == stat) & (pl.col("games_ago") < window)).sort("games_ago")
    )
    if rows.is_empty():
        return Baseline(0.0, 0.0, 0, 0.0, insufficient_history=True)

    column = "adjusted_value" if adjusted else "raw_value"
    return compute_baseline(
        [
            GameObservation(
                season=r["season"], week=r["week"], value=float(r[column] or 0.0),
                is_prior_season=r["season"] < season, opponent=r["opponent"] or "",
            )
            for r in rows.to_dicts()
        ]
    )


def _usage_baseline(usage: pl.DataFrame, column: str) -> tuple[float, int]:
    """Snap-weighted, recency-weighted mean of a usage share. Returns (value, games used).

    Weighting by snaps is what keeps a five-snap rest game from being read as a change in role.
    """
    settings = get_settings()
    rows = usage.filter(pl.col(column).is_not_null())
    if rows.is_empty():
        return 0.0, 0

    decay, discount = settings.recency_decay, settings.prior_season_discount
    target_season = usage["season"].max()

    total_w = 0.0
    total_v = 0.0
    for r in rows.to_dicts():
        snaps = float((r.get("offense_snaps") or 0) + (r.get("defense_snaps") or 0)) or 1.0
        w = decay ** float(r["games_ago"]) * snaps
        if r["season"] < target_season:
            w *= discount
        total_w += w
        total_v += w * float(r[column])
    return (total_v / total_w if total_w > 0 else 0.0), rows.height


def _position_rate_priors(seasons: list[int] | None = None) -> dict[tuple[str, str, str], float]:
    """League rate per (position, numerator, denominator), e.g. WR yards per target.

    Used as the prior every player's efficiency rate is shrunk toward. Computed once per process.
    """
    global _RATE_PRIOR_CACHE
    if _RATE_PRIOR_CACHE:
        return _RATE_PRIOR_CACHE

    seasons = seasons or list(get_settings().seasons)
    season_list = ",".join(str(s) for s in seasons)
    pairs = [
        ("rushing_yards", "rush_attempts"),
        ("receiving_yards", "targets"),
        ("receptions", "targets"),
        ("passing_yards", "pass_attempts"),
        ("completions", "pass_attempts"),
        ("interceptions", "pass_attempts"),
        ("solo_tackles", "tackles_assists"),
    ]
    out: dict[tuple[str, str, str], float] = {}
    with connect() as con:
        for num, den in pairs:
            rows = con.execute(
                f"""
                SELECT position,
                       sum(CASE WHEN stat = ? THEN value END)
                         / nullif(sum(CASE WHEN stat = ? THEN value END), 0) AS rate
                FROM player_game_stats
                WHERE season IN ({season_list}) AND stat IN (?, ?)
                  AND position IN ('QB','RB','WR','TE','K','LB')
                GROUP BY 1
                """,
                [num, den, num, den],
            ).fetchall()
            for position, rate in rows:
                if rate is not None:
                    out[(position, num, den)] = float(rate)

    _RATE_PRIOR_CACHE = out
    log.info("efficiency rate priors fitted for %d (position, rate) pairs", len(out))
    return out


def _rate(
    history: pl.DataFrame,
    numerator: str,
    denominator: str,
    season: int,
    position: str,
    prior_games: float | None = None,
) -> tuple[float, int]:
    """Opponent-adjusted per-opportunity rate, shrunk toward the positional mean (§5.5).

    Computed as a ratio of recency-weighted sums rather than a mean of per-game ratios, so a
    two-target game does not carry the same weight as a twelve-target one, and over the longer
    ``efficiency_window`` rather than the six-game usage window.

    The shrinkage is what stops a bad stretch being read as a new true rate: mixing in
    ``prior_games`` games' worth of touches at the positional average pulls a 3.09 yards-per-carry
    six-game sample back toward reality without discarding real evidence of a decline. The prior
    is sized from the player's OWN per-game volume, so it is worth the same few games to a
    workhorse back and to a third receiver.

    Returns:
        ``(rate, games used)``.
    """
    settings = get_settings()
    if prior_games is None:
        prior_games = settings.efficiency_prior_games

    num = history.filter(pl.col("stat") == numerator).sort("games_ago")
    den = history.filter(pl.col("stat") == denominator).sort("games_ago")
    prior = _position_rate_priors().get((position, numerator, denominator))

    if num.is_empty() or den.is_empty():
        return (prior or 0.0), 0

    den_map = {(r["season"], r["week"]): float(r["adjusted_value"] or 0.0) for r in den.to_dicts()}
    decay, discount = settings.recency_decay, settings.prior_season_discount
    target_season = history["season"].max()

    wn = wd = weight_total = 0.0
    used = 0
    for r in num.to_dicts():
        d = den_map.get((r["season"], r["week"]))
        if not d or d <= 0:
            continue
        w = decay ** float(r["games_ago"])
        if r["season"] < target_season:
            w *= discount
        wn += w * float(r["adjusted_value"] or 0.0)
        wd += w * d
        weight_total += w
        used += 1

    if wd <= 0 or weight_total <= 0:
        return (prior or 0.0), 0
    if prior is None or prior_games <= 0:
        return wn / wd, used

    # Size the prior from the player's own per-game volume, measured on the SAME weighted scale as
    # wd. Stating it in raw opportunities would make it worth wildly different amounts to a
    # workhorse and to a third receiver.
    opportunities_per_game = wd / weight_total
    prior_weight = prior_games * opportunities_per_game
    return (wn + prior * prior_weight) / (wd + prior_weight), used


# ---------------------------------------------------------------------------
# Per-position projection
# ---------------------------------------------------------------------------


def _share_of_team(
    history: pl.DataFrame, usage: pl.DataFrame, stat: str, team_column: str, default: float
) -> float:
    """The player's recency-weighted share of a team total, e.g. attempts or targets.

    Both sides are summed with the same weights before dividing, so a game where the team threw 20
    times counts less than one where it threw 45. Measured against the same denominator the
    environment model produces, which is what stops the sack/scramble discount being applied twice.
    """
    if history.is_empty() or usage.is_empty():
        return default

    settings = get_settings()
    decay, discount = settings.recency_decay, settings.prior_season_discount
    target_season = usage["season"].max()

    team_totals = {
        (r["season"], r["week"]): float(r.get(team_column) or 0.0) for r in usage.to_dicts()
    }
    rows = history.filter(pl.col("stat") == stat)
    if rows.is_empty():
        return default

    wn = wd = 0.0
    for r in rows.to_dicts():
        team_total = team_totals.get((r["season"], r["week"]))
        if not team_total or team_total <= 0:
            continue
        w = decay ** float(r["games_ago"])
        if r["season"] < target_season:
            w *= discount
        wn += w * float(r["raw_value"] or 0.0)
        wd += w * team_total

    if wd <= 0:
        return default
    return float(np.clip(wn / wd, 0.0, 1.0))


def _passing_td_share(team: str, ctx: ProjectionContext, player_pass_tds: float) -> float:
    """Share of a team's touchdowns that come as passing scores, shrunk toward the league split.

    League-wide roughly 58% of offensive touchdowns are thrown. A team's own rate over the trailing
    window is noisy on ~2.4 touchdowns a game, so it is shrunk toward that league value.
    """
    with connect() as con:
        row = con.execute(
            """
            WITH recent AS (
                SELECT season, week, sum(value) AS v, stat FROM (
                    SELECT g.season, g.week, g.stat, g.value,
                           dense_rank() OVER (ORDER BY g.season DESC, g.week DESC) AS ago
                    FROM player_game_stats g
                    WHERE g.team = ? AND g.stat IN ('passing_tds', 'rushing_tds', 'receiving_tds')
                      AND ((g.season < ?) OR (g.season = ? AND g.week < ?))
                ) WHERE ago <= 8 GROUP BY 1, 2, 4
            )
            SELECT
                sum(CASE WHEN stat = 'passing_tds' THEN v ELSE 0 END) AS pass_tds,
                sum(CASE WHEN stat IN ('passing_tds', 'rushing_tds') THEN v ELSE 0 END) AS off_tds
            FROM recent
            """,
            [team, ctx.season, ctx.season, ctx.week],
        ).fetchone()

    league_share = 0.58
    prior_tds = 12.0
    if not row or not row[1]:
        return league_share
    pass_tds, off_tds = float(row[0] or 0.0), float(row[1] or 0.0)
    shrunk = (pass_tds + league_share * prior_tds) / (off_tds + prior_tds)
    return float(np.clip(shrunk, 0.35, 0.80))


def _scaled_poisson(position: str, stat: str, lam: float, steps: list[MathStep]) -> Distribution:
    """A Poisson whose width has been calibrated (§6). Above scale 1 it becomes a neg-binomial."""
    scale = dispersion_scale(position, stat)
    if abs(scale - 1.0) < 1e-6:
        return fit_poisson(lam)
    steps.append(MathStep("5.6 dispersion", "calibrated width scale", scale,
                          "fitted on a held-out season"))
    return fit_count(lam, max(lam * scale, lam * 1.0001), overdispersion_threshold=1.05)


def _make(
    stat: str,
    position: str,
    dist: Distribution,
    steps: list[MathStep],
    play: PlayProbability,
) -> StatProjection:
    spec = get_spec(position, stat)
    return StatProjection(
        stat=stat,
        position=position,
        distribution=dist,
        conditional_on_playing=True,
        play_probability=play.p_played,
        high_variance=spec.high_variance,
        steps=steps,
    )


def _count_projection(
    stat: str,
    position: str,
    mean: float,
    history: pl.DataFrame,
    ctx: ProjectionContext,
    steps: list[MathStep],
    play: PlayProbability,
) -> StatProjection:
    """Fit a count stat, transporting the player's own over-dispersion to the projected mean."""
    base = _baseline_for(history, stat, ctx.season)
    if base.n_games >= get_settings().recency_window and base.mean > 0:
        alpha = alpha_from_moments(base.mean, base.variance)
        source = f"player, {base.n_games} games"
    else:
        alpha = ctx.dispersion.get((position, stat), 0.15)
        source = f"position prior ({position})"

    scale = dispersion_scale(position, stat)
    variance = variance_at(mean, alpha, scale)
    steps.append(MathStep("5.6 dispersion", "alpha (over-dispersion)", alpha, source))
    steps.append(MathStep("5.6 dispersion", "calibrated width scale", scale, "fitted on a held-out season"))
    steps.append(MathStep("5.6 dispersion", "projected variance", variance,
                          "var = (mean + alpha*mean^2) * scale"))

    dist = fit_count(mean, variance, get_settings().overdispersion_threshold)
    return _make(stat, position, dist, steps, play)


def _yardage_projection(
    stat: str,
    position: str,
    mean: float,
    history: pl.DataFrame,
    ctx: ProjectionContext,
    steps: list[MathStep],
    play: PlayProbability,
) -> StatProjection:
    """Fit a yardage stat as a negative binomial (§5.6)."""
    base = _baseline_for(history, stat, ctx.season)
    if base.n_games >= get_settings().recency_window and base.mean > 0:
        alpha = alpha_from_moments(base.mean, base.variance)
        source = f"player, {base.n_games} games"
    else:
        alpha = ctx.dispersion.get((position, stat), 0.35)
        source = f"position prior ({position})"

    scale = dispersion_scale(position, stat)
    variance = variance_at(mean, alpha, scale)
    steps.append(MathStep("5.6 dispersion", "alpha (over-dispersion)", alpha, source))
    steps.append(MathStep("5.6 dispersion", "calibrated width scale", scale, "fitted on a held-out season"))
    steps.append(MathStep("5.6 dispersion", "projected variance", variance,
                          "var = (mean + alpha*mean^2) * scale"))

    return _make(stat, position, fit_negative_binomial(mean, variance), steps, play)


def _longest_projection(
    stat: str,
    position: str,
    opportunities: float,
    gsis_id: str,
    kind: str,
    ctx: ProjectionContext,
    steps: list[MathStep],
    play: PlayProbability,
    fallback_yards: float = 8.0,
) -> StatProjection:
    """Monte Carlo the longest single play, bootstrapping the player's own gains (§5.6)."""
    settings = get_settings()
    gains = _play_gains(gsis_id, kind, ctx.season, ctx.week)

    steps.append(MathStep("5.6 longest", "expected opportunities", opportunities))
    steps.append(
        MathStep("5.6 longest", "observed plays resampled", float(gains.size),
                 f"{kind} gains; parametric fallback below 8")
    )
    if gains.size:
        steps.append(MathStep("5.6 longest", "median observed gain", float(np.median(gains))))
        steps.append(MathStep("5.6 longest", "best observed gain", float(gains.max())))

    dist = fit_longest(
        n_plays=opportunities,
        play_gains=gains if gains.size else None,
        yards_mean=fallback_yards,
        yards_scale=max(fallback_yards * 1.5, 5.0),
        explosive_rate=0.1,
        n_sims=settings.longest_sims,
        seed=abs(hash((stat, gsis_id, ctx.season, ctx.week))) % (2**31),
    )
    steps.append(
        MathStep("5.6 longest", "P(no qualifying play)", dist.params["p_zero"],
                 "settles Under when the player records none")
    )
    return _make(stat, position, dist, steps, play)


def project_player(gsis_id: str, ctx: ProjectionContext) -> PlayerProjection | None:
    """Project every stat for one player (§5). Returns None when the player cannot be projected."""
    settings = get_settings()
    window = settings.recency_window

    with connect() as con:
        row = con.execute(
            "SELECT display_name, position, position_group FROM players WHERE gsis_id = ?",
            [gsis_id],
        ).fetchone()
        rank_row = con.execute(
            "SELECT team, opponent, injury_status, play_probability, changed_team, changed_coach, "
            "       pass_rate_shift, insufficient_history, components "
            "FROM rankings WHERE season = ? AND week = ? AND gsis_id = ?",
            [ctx.season, ctx.week, gsis_id],
        ).fetchone()

    if not row:
        return None
    display_name, position_str, _pg = row
    if position_str not in {p.value for p in Position}:
        return None
    position = Position(position_str)

    history = _player_history(gsis_id, ctx.season, ctx.week, settings.efficiency_window)
    usage = _player_usage(gsis_id, ctx.season, ctx.week, window)

    team = (rank_row[0] if rank_row else None) or (
        usage["team"][0] if usage.height else None
    )
    env = ctx.environment.get(team)
    if not env:
        return None
    opponent = env["opponent"]

    prior_snap = None
    if usage.height:
        prior_snap = float(
            usage.select(
                pl.max_horizontal(
                    pl.col("offense_pct").fill_null(0), pl.col("defense_pct").fill_null(0)
                ).mean()
            ).item()
            or 0.0
        )

    with connect() as con:
        inj = con.execute(
            "SELECT injury_status, practice_participation FROM injury_status "
            "WHERE gsis_id = ? ORDER BY CASE WHEN source='sleeper' THEN 0 ELSE 1 END LIMIT 1",
            [gsis_id],
        ).fetchone()

    play = injury_lookup(
        report_status=inj[0] if inj else None,
        practice_status=inj[1] if inj else None,
        prior_snap_share=prior_snap,
        gsis_id=gsis_id,
    )

    weather = ctx.weather.get(team, {"indoor": False, "wind": None})
    wind_mult = ctx.wind_effect.multiplier(weather.get("wind"), weather.get("indoor", False))

    proj = PlayerProjection(
        gsis_id=gsis_id,
        display_name=display_name,
        position=position.value,
        team=team,
        opponent=opponent,
        season=ctx.season,
        week=ctx.week,
        play=play,
        context={
            "implied_total": env["implied_total"],
            "spread": env["spread"],
            "expected_plays": env["expected_plays"],
            "expected_pass_rate": env["expected_pass_rate"],
            "expected_pass_attempts": env["expected_pass_attempts"],
            "expected_rush_attempts": env["expected_rush_attempts"],
            "expected_team_tds": env["expected_team_tds"],
            "roof": weather.get("roof"),
            "wind": weather.get("wind"),
            "indoor": weather.get("indoor"),
            "wind_multiplier": wind_mult,
            "is_home": env["is_home"],
            "n_games_used": int(history["games_ago"].n_unique()) if history.height else 0,
            "role": role_bucket(prior_snap),
            "prior_snap_share": prior_snap,
        },
        flags={
            "changed_team": bool(rank_row[4]) if rank_row else False,
            "changed_coach": bool(rank_row[5]) if rank_row else False,
            "pass_rate_shift": float(rank_row[6]) if rank_row else 0.0,
            "insufficient_history": bool(rank_row[7]) if rank_row else False,
            "all_history_prior_season": bool(
                history.height and int(history["season"].max()) < ctx.season
            ),
        },
    )

    builder = {
        Position.QB: _project_qb,
        Position.RB: _project_rb,
        Position.WR: _project_receiver,
        Position.TE: _project_receiver,
        Position.K: _project_kicker,
        Position.LB: _project_lb,
    }[position]

    builder(proj, history, usage, env, ctx, wind_mult)
    _add_unconditional(proj)
    return proj


def _mult(ctx: ProjectionContext, opponent: str, metric: str) -> float:
    return get_multiplier(ctx.multipliers, opponent, metric)


def _project_qb(proj, history, usage, env, ctx, wind_mult) -> None:
    """QB: volume from team dropbacks, efficiency from adjusted per-attempt rates."""
    opp = proj.opponent
    pos = "QB"

    # env["expected_pass_attempts"] is already net of sacks and scrambles, so the share applied to
    # it must be the player's share of team ATTEMPTS. dropback_share is attempts/dropbacks, which
    # is the same ~0.885 discount again -- using it here knocked a starter from 34 attempts to 28.
    share = _share_of_team(history, usage, "pass_attempts", "team_pass_attempts", 0.92)
    vol_mult = _mult(ctx, opp, "pass_volume_allowed")
    attempts = env["expected_pass_attempts"] * share * vol_mult

    steps = [
        MathStep("5.3 environment", "team expected pass attempts", env["expected_pass_attempts"]),
        MathStep("5.4 usage", "player share of team pass attempts", share),
        MathStep("5.2 opponent", "pass volume allowed multiplier", vol_mult, opp),
        MathStep("5.4 usage", "projected pass attempts", attempts),
    ]
    proj.conditional["pass_attempts"] = _count_projection(
        "pass_attempts", pos, attempts, history, ctx, list(steps), proj.play
    )

    comp_rate, n = _rate(history, "completions", "pass_attempts", ctx.season, pos)
    comp_rate = comp_rate if comp_rate > 0 else 0.64
    comp_mult = _mult(ctx, opp, "completion_rate_allowed")
    completions = attempts * comp_rate * comp_mult
    proj.conditional["completions"] = _count_projection(
        "completions", pos, completions, history, ctx,
        [*steps,
         MathStep("5.5 efficiency", "completion rate (opponent-adjusted)", comp_rate, f"{n} games"),
         MathStep("5.2 opponent", "completion rate allowed multiplier", comp_mult, opp),
         MathStep("5.5 efficiency", "projected completions", completions)],
        proj.play,
    )

    ypa, n = _rate(history, "passing_yards", "pass_attempts", ctx.season, pos)
    ypa = ypa if ypa > 0 else 7.0
    yds_mult = _mult(ctx, opp, "pass_yards_allowed")
    passing_yards = attempts * ypa * yds_mult * wind_mult
    proj.conditional["passing_yards"] = _yardage_projection(
        "passing_yards", pos, passing_yards, history, ctx,
        [*steps,
         MathStep("5.5 efficiency", "yards per attempt (opponent-adjusted)", ypa, f"{n} games"),
         MathStep("5.2 opponent", "pass yards allowed multiplier", yds_mult, opp),
         MathStep("5.5 efficiency", "wind multiplier", wind_mult,
                  f"{proj.context['wind']} mph" if proj.context["wind"] is not None else "dome/unknown"),
         MathStep("5.5 efficiency", "projected passing yards", passing_yards)],
        proj.play,
    )

    # Passing TDs: the team's expected touchdowns times the share thrown BY THIS QB. The share
    # must be measured against team touchdowns, not against the quarterback's own pass/rush split
    # -- that ratio is ~0.97 for a pocket passer and turned 3.3 expected team TDs into a lambda of
    # 3.0, more than double the league-average 1.5 the reference doc reports.
    td_base = _baseline_for(history, "passing_tds", ctx.season)
    rush_td_base = _baseline_for(history, "rushing_tds", ctx.season)
    team_tds = env["expected_team_tds"]
    pass_share = _passing_td_share(proj.team, ctx, td_base.mean)
    td_mult = _mult(ctx, opp, "pass_td_rate_allowed")
    lam_pass_td = team_tds * pass_share * td_mult
    proj.conditional["passing_tds"] = _make(
        "passing_tds", pos, _scaled_poisson(pos, "passing_tds", lam_pass_td, []),
        [MathStep("5.3 environment", "team expected TDs", team_tds),
         MathStep("5.6 touchdowns", "share of team TDs thrown by this QB", pass_share),
         MathStep("5.2 opponent", "passing TD rate allowed multiplier", td_mult, opp),
         MathStep("5.6 touchdowns", "lambda", lam_pass_td, "Poisson")],
        proj.play,
    )

    int_rate, n = _rate(history, "interceptions", "pass_attempts", ctx.season, pos)
    int_rate = int_rate if int_rate > 0 else 0.023
    int_mult = _mult(ctx, opp, "int_rate_generated")
    lam_int = attempts * int_rate * int_mult
    proj.conditional["interceptions"] = _make(
        "interceptions", pos, _scaled_poisson(pos, "interceptions", lam_int, []),
        [MathStep("5.5 efficiency", "interception rate per attempt", int_rate, f"{n} games"),
         MathStep("5.2 opponent", "INTs forced multiplier", int_mult, opp),
         MathStep("5.6 touchdowns", "lambda", lam_int, "high variance: near-random week to week")],
        proj.play,
    )

    rush_base = _baseline_for(history, "rush_attempts", ctx.season)
    rush_mult = _mult(ctx, opp, "rush_volume_allowed")
    rush_attempts = rush_base.mean * rush_mult
    proj.conditional["rush_attempts"] = _count_projection(
        "rush_attempts", pos, rush_attempts, history, ctx,
        [MathStep("5.1 baseline", "recency-weighted rush attempts", rush_base.mean,
                  f"{rush_base.n_games} games"),
         MathStep("5.2 opponent", "rush volume allowed multiplier", rush_mult, opp)],
        proj.play,
    )

    ypc, n = _rate(history, "rushing_yards", "rush_attempts", ctx.season, pos)
    ypc = ypc if ypc > 0 else 4.5
    ry_mult = _mult(ctx, opp, "rush_yards_allowed")
    rushing_yards = rush_attempts * ypc * ry_mult
    proj.conditional["rushing_yards"] = _yardage_projection(
        "rushing_yards", pos, rushing_yards, history, ctx,
        [MathStep("5.5 efficiency", "yards per carry (opponent-adjusted)", ypc, f"{n} games"),
         MathStep("5.2 opponent", "rush yards allowed multiplier", ry_mult, opp),
         MathStep("5.5 efficiency", "projected rushing yards", rushing_yards)],
        proj.play,
    )

    proj.conditional["longest_completion"] = _longest_projection(
        "longest_completion", pos, completions, proj.gsis_id, "completion", ctx, [], proj.play,
        fallback_yards=ypa / max(comp_rate, 0.1),
    )

    gl_share, _ = _usage_baseline(usage, "gl_carry_share")
    lam_rush_td = team_tds * float(np.clip(gl_share if gl_share > 0 else rush_td_base.mean / max(team_tds, 1e-6), 0.0, 0.6)) * _mult(ctx, opp, "rush_td_rate_allowed")
    proj.conditional["anytime_rush_td"] = _make(
        "anytime_rush_td", pos, fit_anytime_td(lam_rush_td),
        [MathStep("5.6 touchdowns", "goal-line carry share", gl_share),
         MathStep("5.6 touchdowns", "lambda", lam_rush_td,
                  "rushing scores only; a passing TD never counts"),
         MathStep("5.6 touchdowns", "P(anytime rushing TD)", 1 - np.exp(-lam_rush_td), "1 - e^-lambda")],
        proj.play,
    )


def _project_rb(proj, history, usage, env, ctx, wind_mult) -> None:
    """RB: carries from team rush volume × carry share; receiving from team targets × target share."""
    opp = proj.opponent
    pos = "RB"

    carry_share, _ = _usage_baseline(usage, "carry_share")
    vol_mult = _mult(ctx, opp, "rush_volume_allowed")
    carries = env["expected_rush_attempts"] * carry_share * vol_mult
    carry_steps = [
        MathStep("5.3 environment", "team expected rush attempts", env["expected_rush_attempts"]),
        MathStep("5.4 usage", "carry share", carry_share),
        MathStep("5.2 opponent", "rush volume allowed multiplier", vol_mult, opp),
        MathStep("5.4 usage", "projected carries", carries),
    ]
    proj.conditional["rush_attempts"] = _count_projection(
        "rush_attempts", pos, carries, history, ctx, list(carry_steps), proj.play
    )

    ypc, n = _rate(history, "rushing_yards", "rush_attempts", ctx.season, pos)
    ypc = ypc if ypc > 0 else 4.3
    ypc_mult = _mult(ctx, opp, "rush_yards_allowed_rb")
    rushing_yards = carries * ypc * ypc_mult
    proj.conditional["rushing_yards"] = _yardage_projection(
        "rushing_yards", pos, rushing_yards, history, ctx,
        [*carry_steps,
         MathStep("5.5 efficiency", "yards per carry (opponent-adjusted)", ypc, f"{n} games"),
         MathStep("5.2 opponent", "YPC allowed to RBs multiplier", ypc_mult, opp),
         MathStep("5.5 efficiency", "projected rushing yards", rushing_yards)],
        proj.play,
    )

    target_share, _ = _usage_baseline(usage, "target_share")
    team_targets = env["expected_pass_attempts"]
    tgt_mult = _mult(ctx, opp, "rec_volume_allowed_rb")
    targets = team_targets * target_share * tgt_mult

    catch_rate, n = _rate(history, "receptions", "targets", ctx.season, pos)
    catch_rate = catch_rate if catch_rate > 0 else 0.75
    receptions = targets * catch_rate
    rec_steps = [
        MathStep("5.3 environment", "team expected pass attempts", team_targets),
        MathStep("5.4 usage", "target share", target_share),
        MathStep("5.2 opponent", "RB receptions allowed multiplier", tgt_mult, opp),
        MathStep("5.5 efficiency", "catch rate", catch_rate, f"{n} games"),
        MathStep("5.4 usage", "projected receptions", receptions),
    ]
    proj.conditional["receptions"] = _count_projection(
        "receptions", pos, receptions, history, ctx, list(rec_steps), proj.play
    )

    ypt, n = _rate(history, "receiving_yards", "targets", ctx.season, pos)
    ypt = ypt if ypt > 0 else 6.2
    ypt_mult = _mult(ctx, opp, "rec_yards_allowed_rb")
    receiving_yards = targets * ypt * ypt_mult * wind_mult
    proj.conditional["receiving_yards"] = _yardage_projection(
        "receiving_yards", pos, receiving_yards, history, ctx,
        [*rec_steps,
         MathStep("5.5 efficiency", "yards per target (opponent-adjusted)", ypt, f"{n} games"),
         MathStep("5.2 opponent", "yards per target allowed to RBs", ypt_mult, opp),
         MathStep("5.5 efficiency", "projected receiving yards", receiving_yards)],
        proj.play,
    )

    combined = rushing_yards + receiving_yards
    proj.conditional["rush_rec_yards"] = _yardage_projection(
        "rush_rec_yards", pos, combined, history, ctx,
        [MathStep("5.6 combined", "projected rushing yards", rushing_yards),
         MathStep("5.6 combined", "projected receiving yards", receiving_yards),
         MathStep("5.6 combined", "projected rush + rec yards", combined)],
        proj.play,
    )

    proj.conditional["longest_rush"] = _longest_projection(
        "longest_rush", pos, carries, proj.gsis_id, "rush", ctx, [], proj.play, fallback_yards=ypc,
    )

    gl_share, _ = _usage_baseline(usage, "gl_carry_share")
    rz_tgt_share, _ = _usage_baseline(usage, "rz_target_share")
    # Goal-line role dominates: 86.6% of rushing TDs since 2010 came from inside the red zone and
    # 57.4% from inside the 5, so total touches are the wrong basis (§5.6).
    td_share = 0.75 * gl_share + 0.25 * rz_tgt_share
    td_mult = _mult(ctx, opp, "rz_td_rate_allowed")
    lam_td = env["expected_team_tds"] * float(np.clip(td_share, 0.0, 0.85)) * td_mult
    proj.conditional["anytime_td"] = _make(
        "anytime_td", pos, fit_anytime_td(lam_td),
        [MathStep("5.6 touchdowns", "goal-line carry share", gl_share),
         MathStep("5.6 touchdowns", "red-zone target share", rz_tgt_share),
         MathStep("5.6 touchdowns", "blended TD share", td_share, "0.75 goal line + 0.25 red-zone targets"),
         MathStep("5.3 environment", "team expected TDs", env["expected_team_tds"]),
         MathStep("5.2 opponent", "red-zone TD rate allowed", td_mult, opp),
         MathStep("5.6 touchdowns", "lambda", lam_td),
         MathStep("5.6 touchdowns", "P(anytime TD)", 1 - np.exp(-lam_td), "1 - e^-lambda")],
        proj.play,
    )


def _project_receiver(proj, history, usage, env, ctx, wind_mult) -> None:
    """WR / TE: targets from team pass volume × target share, then catch rate and yards per target."""
    opp = proj.opponent
    pos = proj.position
    suffix = "wr" if pos == "WR" else "te"

    target_share, _ = _usage_baseline(usage, "target_share")
    tgt_mult = _mult(ctx, opp, f"target_volume_allowed_{suffix}")
    targets = env["expected_pass_attempts"] * target_share * tgt_mult
    steps = [
        MathStep("5.3 environment", "team expected pass attempts", env["expected_pass_attempts"]),
        MathStep("5.4 usage", "target share", target_share),
        MathStep("5.2 opponent", f"targets allowed to {pos}s multiplier", tgt_mult, opp),
        MathStep("5.4 usage", "projected targets", targets),
    ]
    proj.conditional["targets"] = _count_projection(
        "targets", pos, targets, history, ctx, list(steps), proj.play
    )

    catch_rate, n = _rate(history, "receptions", "targets", ctx.season, pos)
    catch_rate = catch_rate if catch_rate > 0 else 0.65
    rec_mult = _mult(ctx, opp, f"rec_volume_allowed_{suffix}")
    receptions = targets * catch_rate * rec_mult / max(tgt_mult, 1e-6)
    proj.conditional["receptions"] = _count_projection(
        "receptions", pos, receptions, history, ctx,
        [*steps,
         MathStep("5.5 efficiency", "catch rate (opponent-adjusted)", catch_rate, f"{n} games"),
         MathStep("5.2 opponent", f"receptions allowed to {pos}s multiplier", rec_mult, opp),
         MathStep("5.4 usage", "projected receptions", receptions)],
        proj.play,
    )

    ypt, n = _rate(history, "receiving_yards", "targets", ctx.season, pos)
    ypt = ypt if ypt > 0 else 8.0
    ypt_mult = _mult(ctx, opp, f"rec_yards_allowed_{suffix}")
    receiving_yards = targets * ypt * ypt_mult * wind_mult
    proj.conditional["receiving_yards"] = _yardage_projection(
        "receiving_yards", pos, receiving_yards, history, ctx,
        [*steps,
         MathStep("5.5 efficiency", "yards per target (opponent-adjusted)", ypt, f"{n} games"),
         MathStep("5.2 opponent", f"yards per target allowed to {pos}s", ypt_mult, opp),
         MathStep("5.5 efficiency", "wind multiplier", wind_mult),
         MathStep("5.5 efficiency", "projected receiving yards", receiving_yards)],
        proj.play,
    )

    if pos == "WR":
        proj.conditional["longest_reception"] = _longest_projection(
            "longest_reception", pos, receptions, proj.gsis_id, "reception", ctx, [], proj.play,
            fallback_yards=ypt / max(catch_rate, 0.1),
        )

    rz_share, _ = _usage_baseline(usage, "rz_target_share")
    td_mult = _mult(ctx, opp, "rz_td_rate_allowed")
    # Red-zone target share, not total yardage: 25%+ is the reliability threshold in the reference.
    lam_td = env["expected_team_tds"] * float(np.clip(rz_share, 0.0, 0.6)) * td_mult * 0.62
    proj.conditional["anytime_td"] = _make(
        "anytime_td", pos, fit_anytime_td(lam_td),
        [MathStep("5.6 touchdowns", "red-zone target share", rz_share),
         MathStep("5.3 environment", "team expected TDs", env["expected_team_tds"]),
         MathStep("5.2 opponent", "red-zone TD rate allowed", td_mult, opp),
         MathStep("5.6 touchdowns", "share of team TDs that are receiving", 0.62,
                  "league split of receiving vs rushing scores"),
         MathStep("5.6 touchdowns", "lambda", lam_td),
         MathStep("5.6 touchdowns", "P(anytime TD)", 1 - np.exp(-lam_td), "1 - e^-lambda")],
        proj.play,
    )


def _project_kicker(proj, history, usage, env, ctx, wind_mult) -> None:
    """K: attempts from the offence stalling in range, makes from the fitted distance model."""
    opp = proj.opponent
    pos = "K"
    weather = ctx.weather.get(proj.team, {})
    indoor = bool(weather.get("indoor"))
    wind = weather.get("wind")

    fga_base = _baseline_for(history, "fg_attempts", ctx.season)
    fga_mult = _mult(ctx, opp, "fg_attempts_allowed")
    team_fga = env["expected_team_fgs"]
    # Blend the team's structural expectation with the kicker's own recent volume; the kicker is
    # the only one taking his team's attempts, so the two are measuring the same thing.
    fg_attempts = 0.5 * team_fga * fga_mult + 0.5 * fga_base.mean * fga_mult
    attempt_steps = [
        MathStep("5.3 environment", "team expected FG attempts", team_fga),
        MathStep("5.1 baseline", "kicker recent FG attempts", fga_base.mean, f"{fga_base.n_games} games"),
        MathStep("5.2 opponent", "FG attempts allowed multiplier", fga_mult, opp),
        MathStep("5.6 kicking", "projected FG attempts", fg_attempts),
    ]
    proj.conditional["fg_attempts"] = _make(
        "fg_attempts", pos, _scaled_poisson(pos, "fg_attempts", fg_attempts, attempt_steps),
        list(attempt_steps), proj.play
    )

    distances, weights = ctx.fg_distances
    make_rate = ctx.fg_model.expected_make_rate(distances, weights, wind, indoor)
    fg_made = fg_attempts * make_rate
    made_steps = [
        *attempt_steps,
        MathStep("5.6 kicking", "expected make rate", make_rate,
                 f"{'dome' if indoor else f'{wind} mph'}, league attempt-distance mix"),
        MathStep("5.6 kicking", "projected FG made", fg_made),
    ]
    proj.conditional["fg_made"] = _make(
        "fg_made", pos, _scaled_poisson(pos, "fg_made", fg_made, made_steps), made_steps, proj.play
    )

    xp = env["expected_team_tds"] * XP_PER_TD
    proj.conditional["xp_made"] = _make(
        "xp_made", pos, _scaled_poisson(pos, "xp_made", xp, []),
        [MathStep("5.3 environment", "team expected TDs", env["expected_team_tds"]),
         MathStep("5.6 kicking", "XP conversion net of two-point tries", XP_PER_TD),
         MathStep("5.6 kicking", "projected XP made", xp)],
        proj.play,
    )

    points = fit_deterministic(
        [("fg_made", 3.0, fit_poisson(fg_made)), ("xp_made", 1.0, fit_poisson(xp))]
    )
    proj.conditional["kicking_points"] = _make(
        "kicking_points", pos, points,
        [*made_steps,
         MathStep("5.6 kicking", "projected XP made", xp),
         MathStep("5.6 kicking", "kicking points", points.mean, "3 x FGM + 1 x XPM")],
        proj.play,
    )

    proj.conditional["longest_fg"] = _longest_projection(
        "longest_fg", pos, fg_made, proj.gsis_id, "fg", ctx,
        [MathStep("5.6 kicking", "expected makes to draw from", fg_made)], proj.play,
        fallback_yards=38.0,
    )


def _project_lb(proj, history, usage, env, ctx, wind_mult) -> None:
    """LB: tackle volume is the opposing offence's play count times the player's tackle share."""
    opp = proj.opponent
    pos = "LB"
    opp_env = ctx.environment.get(opp, env)

    tackle_share, _ = _usage_baseline(usage, "tackle_share")
    snap_share, _ = _usage_baseline(usage, "defense_pct")
    plays_mult = _mult(ctx, opp, "opp_plays_allowed")
    opp_plays = opp_env["expected_plays"] * plays_mult

    # Tackle opportunities scale with the opponent's RUN rate as well as its play count: a
    # pass-heavy opponent is bearish for box linebackers even at the same volume.
    rush_rate = 1.0 - opp_env["expected_pass_rate"]
    league_rush_rate = 0.43
    rush_factor = float(np.clip(rush_rate / league_rush_rate, 0.75, 1.3))

    base = _baseline_for(history, "tackles_assists", ctx.season)
    tackles = base.mean * plays_mult * rush_factor
    steps = [
        MathStep("5.1 baseline", "recency-weighted tackles + assists", base.mean,
                 f"{base.n_games} games, DraftKings settlement (defence only)"),
        MathStep("5.3 environment", "opponent expected plays", opp_plays),
        MathStep("5.2 opponent", "opponent plays multiplier", plays_mult, opp),
        MathStep("5.3 environment", "opponent rush rate", rush_rate),
        MathStep("5.5 efficiency", "rush-rate factor", rush_factor,
                 "a run-heavy opponent means more box tackles"),
        MathStep("5.4 usage", "team tackle share", tackle_share),
        MathStep("5.4 usage", "defensive snap share", snap_share),
        MathStep("5.4 usage", "projected tackles + assists", tackles),
    ]
    proj.conditional["tackles_assists"] = _count_projection(
        "tackles_assists", pos, tackles, history, ctx, list(steps), proj.play
    )

    solo_ratio, n = _rate(history, "solo_tackles", "tackles_assists", ctx.season, pos)
    solo_ratio = solo_ratio if solo_ratio > 0 else 0.62
    solo = tackles * solo_ratio
    proj.conditional["solo_tackles"] = _count_projection(
        "solo_tackles", pos, solo, history, ctx,
        [*steps,
         MathStep("5.5 efficiency", "solo share of combined tackles", solo_ratio, f"{n} games"),
         MathStep("5.4 usage", "projected solo tackles", solo)],
        proj.play,
    )

    sack_base = _baseline_for(history, "sacks", ctx.season)
    pressure_mult = _mult(ctx, opp, "pressure_allowed")
    lam_sacks = max(sack_base.mean, 0.01) * pressure_mult
    proj.conditional["sacks"] = _make(
        "sacks", pos, _scaled_poisson(pos, "sacks", lam_sacks, []),
        [MathStep("5.1 baseline", "recency-weighted sacks", sack_base.mean, f"{sack_base.n_games} games"),
         MathStep("5.2 opponent", "sacks allowed by opponent multiplier", pressure_mult, opp),
         MathStep("5.6 touchdowns", "lambda", lam_sacks,
                  "high variance: even elite rushers post frequent zero-sack games")],
        proj.play,
    )

    pd_base = _baseline_for(history, "passes_defended", ctx.season)
    pass_mult = _mult(ctx, opp, "pass_volume_allowed")
    lam_pd = max(pd_base.mean, 0.01) * pass_mult
    proj.conditional["passes_defended"] = _make(
        "passes_defended", pos, _scaled_poisson(pos, "passes_defended", lam_pd, []),
        [MathStep("5.1 baseline", "recency-weighted passes defended", pd_base.mean,
                  f"{pd_base.n_games} games"),
         MathStep("5.2 opponent", "opponent pass volume multiplier", pass_mult, opp),
         MathStep("5.6 touchdowns", "lambda", lam_pd, "high variance")],
        proj.play,
    )


def _add_unconditional(proj: PlayerProjection) -> None:
    """Mirror every conditional projection into an unconditional one (§5.7).

    ``E[X] = P(played) · E[X | played]``, and the variance picks up the play/don't-play mixture, so
    a coin-flip Questionable shows the wide band that uncertainty actually implies.
    """
    p = proj.play.p_played
    # E[snap share | played] is an ABSOLUTE share, so it must be compared to the player's own
    # normal share rather than applied on top of a baseline that already reflects it. A 90%-snap
    # receiver expected at 76% loses 16% of his usage, not 24%.
    normal = float(proj.context.get("prior_snap_share") or 0.0)
    expected = float(proj.play.expected_snap_share or 0.0)
    if normal > 0.05 and expected > 0:
        snap_scale = float(np.clip(expected / normal, 0.5, 1.0))
    else:
        snap_scale = 1.0
    proj.context["snap_scale"] = snap_scale

    for stat, sp in proj.conditional.items():
        dist = sp.distribution
        if dist.family is Family.BERNOULLI:
            lam = dist.params["lam"] * p * snap_scale
            new = fit_anytime_td(lam)
        elif dist.family is Family.POISSON:
            new = fit_poisson(apply_unconditional(dist.params["lam"], p) * snap_scale)
        elif dist.family is Family.NEGATIVE_BINOMIAL:
            mean = apply_unconditional(dist.mean, p) * snap_scale
            var = inflate_variance_for_uncertainty(dist.mean * snap_scale, dist.params["variance"], p)
            new = fit_negative_binomial(max(mean, 1e-6), max(var, mean * 1.05))
        else:
            # Monte Carlo and deterministic families are already distributions over outcomes;
            # scaling their parameters is not meaningful, so they carry through unchanged and are
            # labelled as conditional.
            new = dist

        steps = [
            *sp.steps,
            MathStep("5.7 injury", "P(played)", p, proj.play.source),
            MathStep("5.7 injury", "E[snap share | played]", snap_scale, proj.play.role),
        ]
        proj.unconditional[stat] = StatProjection(
            stat=stat, position=sp.position, distribution=new,
            conditional_on_playing=False, play_probability=p,
            high_variance=sp.high_variance, steps=steps,
        )


# ---------------------------------------------------------------------------
# Persistence and CLI
# ---------------------------------------------------------------------------


def project_week(season: int, week: int, gsis_ids: list[str] | None = None) -> int:
    """Project every ranked player for the week and store the results.

    Returns:
        Number of projection rows written (conditional and unconditional combined).
    """
    ctx = build_context(season, week)
    if not ctx.environment:
        log.warning("no team environment for %s week %s; run the odds stage first", season, week)
        return 0

    if gsis_ids is None:
        with connect() as con:
            gsis_ids = [
                r[0]
                for r in con.execute(
                    "SELECT DISTINCT gsis_id FROM rankings WHERE season = ? AND week = ?",
                    [season, week],
                ).fetchall()
            ]
    if not gsis_ids:
        log.warning("no ranked players for %s week %s", season, week)
        return 0

    proj_rows: list[dict[str, Any]] = []
    math_rows: list[dict[str, Any]] = []

    for gsis in gsis_ids:
        try:
            p = project_player(gsis, ctx)
        except Exception:  # noqa: BLE001 - one bad player must not lose the whole week
            log.exception("projection failed for %s", gsis)
            continue
        if p is None:
            continue

        for conditional, bucket in ((True, p.conditional), (False, p.unconditional)):
            for stat, sp in bucket.items():
                d = sp.distribution
                proj_rows.append(
                    {
                        "season": season, "week": week, "gsis_id": gsis, "position": p.position,
                        "stat": stat, "dist_family": str(d.family),
                        "params": json.dumps(d.params, default=float),
                        "mean": d.mean, "median": d.median, "p25": d.p25, "p75": d.p75,
                        "conditional_on_playing": conditional,
                        "play_probability": sp.play_probability,
                        "high_variance": sp.high_variance,
                    }
                )

        for stat, sp in p.conditional.items():
            for i, step in enumerate(sp.steps):
                math_rows.append(
                    {
                        "season": season, "week": week, "gsis_id": gsis, "stat": stat,
                        "step": i, "section": step.section, "label": step.label,
                        "value": float(step.value) if step.value is not None else None,
                        "detail": step.detail,
                    }
                )

    if not proj_rows:
        return 0

    with connect() as con:
        con.register("proj_df", pl.DataFrame(proj_rows))
        con.register("math_df", pl.DataFrame(math_rows))
        try:
            con.execute("BEGIN TRANSACTION")
            con.execute("DELETE FROM projections WHERE season = ? AND week = ?", [season, week])
            con.execute("DELETE FROM projection_math WHERE season = ? AND week = ?", [season, week])
            con.execute(
                "INSERT INTO projections (season, week, gsis_id, position, stat, dist_family, "
                " params, mean, median, p25, p75, conditional_on_playing, play_probability, "
                " high_variance, computed_at) "
                "SELECT season, week, gsis_id, position, stat, dist_family, params, mean, median, "
                "       p25, p75, conditional_on_playing, play_probability, high_variance, now() "
                "FROM proj_df"
            )
            con.execute(
                "INSERT INTO projection_math "
                "(season, week, gsis_id, stat, step, section, label, value, detail) "
                "SELECT season, week, gsis_id, stat, step, section, label, value, detail FROM math_df"
            )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            con.unregister("proj_df")
            con.unregister("math_df")

    log.info("projections: %s week %s -> %d rows for %d players",
             season, week, len(proj_rows), len(gsis_ids))
    return len(proj_rows)


def resolve_player(name_or_id: str) -> str | None:
    """Resolve a player name or gsis_id to a gsis_id. Case-insensitive, prefers an exact match."""
    with connect() as con:
        row = con.execute("SELECT gsis_id FROM players WHERE gsis_id = ?", [name_or_id]).fetchone()
        if row:
            return row[0]
        rows = con.execute(
            "SELECT gsis_id, display_name, last_season FROM players "
            "WHERE lower(display_name) = lower(?) ORDER BY last_season DESC NULLS LAST",
            [name_or_id],
        ).fetchall()
        if not rows:
            rows = con.execute(
                "SELECT gsis_id, display_name, last_season FROM players "
                "WHERE lower(display_name) LIKE lower(?) ORDER BY last_season DESC NULLS LAST LIMIT 5",
                [f"%{name_or_id}%"],
            ).fetchall()
    return rows[0][0] if rows else None


def project_player_json(name_or_id: str, season: int, week: int) -> dict[str, Any]:
    """Full deep-dive for one player as JSON (§9 milestone-4 checkpoint)."""
    gsis = resolve_player(name_or_id)
    if not gsis:
        return {"error": f"no player matching {name_or_id!r}"}

    ctx = build_context(season, week)
    p = project_player(gsis, ctx)
    if p is None:
        return {"error": f"{name_or_id!r} could not be projected for {season} week {week}"}

    payload = p.to_json()
    payload["game_log"] = _game_log_json(gsis, season, week)
    return payload


def _game_log_json(gsis_id: str, season: int, week: int) -> list[dict[str, Any]]:
    """The adjusted game log rows behind a projection, for the deep-dive table (§8)."""
    with connect() as con:
        rows = con.execute(
            """
            SELECT season, week, opponent, stat, raw_value, opponent_multiplier, adjusted_value,
                   opponent_rank, metric
            FROM adjusted_game_log
            WHERE gsis_id = ? AND ((season < ?) OR (season = ? AND week < ?))
            ORDER BY season DESC, week DESC, stat
            """,
            [gsis_id, season, season, week],
        ).pl()
    return rows.to_dicts()


def print_projection(name_or_id: str, season: int, week: int) -> None:
    """Human-readable deep-dive for one player."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    payload = project_player_json(name_or_id, season, week)
    if "error" in payload:
        console.print(f"[red]{payload['error']}[/red]")
        return

    ctx = payload["context"]
    console.print(
        f"\n[bold]{payload['display_name']}[/bold]  {payload['position']} · "
        f"{payload['team']} vs {payload['opponent']} · {season} week {week}"
    )
    conditions = "dome" if ctx["indoor"] else (
        f"{ctx['wind']:.0f} mph wind" if ctx["wind"] is not None else "wind unknown"
    )
    console.print(
        f"  implied total {ctx['implied_total']:.1f} · spread {ctx['spread']:+.1f} · "
        f"{conditions} · wind multiplier {ctx['wind_multiplier']:.3f}"
    )
    inj = payload["injury"]
    console.print(
        f"  status {inj['report_status'] or 'healthy'} · P(play) {inj['p_played']:.0%} "
        f"({inj['source']}, role {inj['role']})"
    )
    flags = [k for k, v in payload["flags"].items() if v is True]
    if flags:
        console.print(f"  [yellow]flags: {', '.join(flags)}[/yellow]")

    table = Table("stat", "median", "p25", "p75", "mean", "family", "uncond. median")
    for stat, sp in payload["projections"].items():
        u = payload["unconditional"].get(stat, {})
        label = stat + ("  ⚡" if sp["high_variance"] else "")
        table.add_row(
            label, f"{sp['median']:.2f}", f"{sp['p25']:.2f}", f"{sp['p75']:.2f}",
            f"{sp['mean']:.2f}", sp["family"].replace("_", " "),
            f"{u.get('median', float('nan')):.2f}",
        )
    console.print(table)
