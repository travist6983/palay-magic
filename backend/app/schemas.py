"""Pydantic response models — the contract the frontend codes against.

Everything is precomputed on refresh, so the API is a thin read layer over DuckDB (§8). The one
thing worth noting is :class:`DistributionParams`: the browser recomputes ``P(over line)`` locally
on every keystroke from these parameters rather than calling back per keystroke (D10), so they
must be complete enough to evaluate the distribution client-side.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Freshness = Literal["green", "yellow", "red"]


class SourceStatus(BaseModel):
    """One data source's freshness badge (§8 header)."""

    source: str
    status: Freshness
    last_success_at: str | None = None
    last_attempt_at: str | None = None
    detail: str | None = None
    rows: int | None = None
    age_hours: float | None = None


class Meta(BaseModel):
    """Header state: what week we are showing and how fresh each source is."""

    season: int
    week: int
    season_type: str
    state_source: str
    games_played_this_season: int
    season_start_date: str | None = None
    last_refresh_at: str | None = None
    last_refresh_seconds: float | None = None
    failed_stages: list[str] = Field(default_factory=list)
    sources: list[SourceStatus] = Field(default_factory=list)
    positions: list[str] = Field(default_factory=list)
    all_history_prior_season: bool = False
    odds_budget_remaining: int | None = None
    llm_available: bool = False


class DistributionParams(BaseModel):
    """A stored distribution, complete enough to evaluate in the browser (D10)."""

    family: str
    params: dict[str, Any]
    mean: float
    median: float
    p25: float
    p75: float
    integer_valued: bool = True


class ProjectionOut(BaseModel):
    """One projected stat."""

    stat: str
    label: str
    distribution: DistributionParams
    unconditional: DistributionParams | None = None
    high_variance: bool = False
    play_probability: float = 1.0
    settlement_note: str = ""


class MatchupOut(BaseModel):
    """How the opponent grades against this position."""

    metric: str
    label: str
    multiplier: float
    raw_multiplier: float
    rank: int
    percentile: float
    z_score: float
    predictive_weight: float | None = None
    mse_reduction_pct: float | None = None


class BoardRow(BaseModel):
    """One row of a position board (§8)."""

    rank: int
    gsis_id: str
    display_name: str
    team: str | None
    opponent: str | None
    headshot_url: str | None = None
    score: float
    injury_status: str | None = None
    play_probability: float = 1.0
    insufficient_history: bool = False
    changed_team: bool = False
    changed_coach: bool = False
    pass_rate_shift: float = 0.0
    opponent_multiplier: float = 1.0
    opponent_rank: int | None = None
    headline: list[ProjectionOut] = Field(default_factory=list)


class Board(BaseModel):
    """A whole position board."""

    position: str
    season: int
    week: int
    rows: list[BoardRow] = Field(default_factory=list)


class GameLogCell(BaseModel):
    """One stat in one past game, raw and opponent-adjusted."""

    stat: str
    raw_value: float
    adjusted_value: float
    opponent_multiplier: float
    opponent_rank: int | None = None


class GameLogRow(BaseModel):
    """One past game in the deep-dive table (§8)."""

    season: int
    week: int
    opponent: str | None
    prior_season: bool = False
    cells: dict[str, GameLogCell] = Field(default_factory=dict)


class MathStepOut(BaseModel):
    """One line of the "Show math" drawer (§8)."""

    section: str
    label: str
    value: float | None
    detail: str = ""


class EnvironmentOut(BaseModel):
    """The game environment card (§8)."""

    implied_total: float | None = None
    spread: float | None = None
    total_line: float | None = None
    expected_plays: float | None = None
    expected_pass_rate: float | None = None
    expected_pass_attempts: float | None = None
    expected_rush_attempts: float | None = None
    expected_team_tds: float | None = None
    roof: str | None = None
    wind: float | None = None
    temp: float | None = None
    indoor: bool = False
    wind_multiplier: float = 1.0
    is_home: bool | None = None
    odds_source: str | None = None


class InjuryOut(BaseModel):
    """Injury designation and what it means for usage (§5.7)."""

    report_status: str | None = None
    practice_status: str | None = None
    p_played: float = 1.0
    expected_snap_share: float = 1.0
    role: str = "starter"
    source: str = "healthy"
    n_observations: int = 0
    note: str | None = None
    usage_risk: str | None = None
    teammates_affected: list[str] = Field(default_factory=list)


class PlayerDetail(BaseModel):
    """The whole deep-dive page (§8)."""

    gsis_id: str
    display_name: str
    position: str
    team: str | None
    opponent: str | None
    season: int
    week: int
    headshot_url: str | None = None
    height: int | None = None
    weight: int | None = None
    years_exp: int | None = None
    environment: EnvironmentOut
    injury: InjuryOut
    flags: dict[str, Any] = Field(default_factory=dict)
    projections: list[ProjectionOut] = Field(default_factory=list)
    game_log: list[GameLogRow] = Field(default_factory=list)
    matchup: list[MatchupOut] = Field(default_factory=list)
    math: dict[str, list[MathStepOut]] = Field(default_factory=dict)
    narrative: str | None = None
    stat_order: list[str] = Field(default_factory=list)


class DefenseRow(BaseModel):
    """One defence's grade against a position, for the matchup page."""

    team: str
    multiplier: float
    raw_value: float
    league_avg: float
    rank: int
    percentile: float
    n_games: int


class DefenseTable(BaseModel):
    position: str
    metric: str
    label: str
    season: int
    week: int
    mse_reduction_pct: float | None = None
    rows: list[DefenseRow] = Field(default_factory=list)


class CalibrationRow(BaseModel):
    position: str
    stat: str
    n: int
    mae: float
    bias: float
    coverage: float
    pit_central: float
    brier: float
    degenerate: bool = False


class CalibrationOut(BaseModel):
    run_id: str
    season: int
    week_start: int
    week_end: int
    coverage: float | None = None
    pit_central: float | None = None
    rows: list[CalibrationRow] = Field(default_factory=list)
