// Mirrors backend/app/schemas.py. Keep the two in step.

export type Freshness = 'green' | 'yellow' | 'red'

export interface SourceStatus {
  source: string
  status: Freshness
  last_success_at: string | null
  last_attempt_at: string | null
  detail: string | null
  rows: number | null
  age_hours: number | null
}

export interface Meta {
  season: number
  week: number
  season_type: string
  state_source: string
  games_played_this_season: number
  season_start_date: string | null
  last_refresh_at: string | null
  last_refresh_seconds: number | null
  failed_stages: string[]
  sources: SourceStatus[]
  positions: string[]
  all_history_prior_season: boolean
  odds_budget_remaining: number | null
  llm_available: boolean
}

export type Family =
  | 'negative_binomial'
  | 'poisson'
  | 'bernoulli'
  | 'empirical_max'
  | 'deterministic'

export interface DistributionParams {
  family: Family
  params: Record<string, any>
  mean: number
  median: number
  p25: number
  p75: number
  integer_valued: boolean
}

export interface Projection {
  stat: string
  label: string
  distribution: DistributionParams
  unconditional: DistributionParams | null
  high_variance: boolean
  play_probability: number
  settlement_note: string
}

export interface BoardRow {
  rank: number
  gsis_id: string
  display_name: string
  team: string | null
  opponent: string | null
  headshot_url: string | null
  score: number
  injury_status: string | null
  play_probability: number
  insufficient_history: boolean
  changed_team: boolean
  changed_coach: boolean
  pass_rate_shift: number
  opponent_multiplier: number
  opponent_rank: number | null
  headline: Projection[]
}

export interface Board {
  position: string
  season: number
  week: number
  rows: BoardRow[]
}

export interface GameLogCell {
  stat: string
  raw_value: number
  adjusted_value: number
  opponent_multiplier: number
  opponent_rank: number | null
}

export interface GameLogRow {
  season: number
  week: number
  opponent: string | null
  prior_season: boolean
  cells: Record<string, GameLogCell>
}

export interface MathStep {
  section: string
  label: string
  value: number | null
  detail: string
}

export interface Matchup {
  metric: string
  label: string
  multiplier: number
  raw_multiplier: number
  rank: number
  percentile: number
  z_score: number
  predictive_weight: number | null
  mse_reduction_pct: number | null
}

export interface Environment {
  implied_total: number | null
  spread: number | null
  total_line: number | null
  expected_plays: number | null
  expected_pass_rate: number | null
  expected_pass_attempts: number | null
  expected_rush_attempts: number | null
  expected_team_tds: number | null
  roof: string | null
  wind: number | null
  temp: number | null
  indoor: boolean
  wind_multiplier: number
  is_home: boolean | null
  odds_source: string | null
}

export interface Injury {
  report_status: string | null
  practice_status: string | null
  p_played: number
  expected_snap_share: number
  role: string
  source: string
  n_observations: number
  note: string | null
  usage_risk: string | null
  teammates_affected: string[]
}

export interface PlayerDetail {
  gsis_id: string
  display_name: string
  position: string
  team: string | null
  opponent: string | null
  season: number
  week: number
  headshot_url: string | null
  height: number | null
  weight: number | null
  years_exp: number | null
  environment: Environment
  injury: Injury
  flags: Record<string, any>
  projections: Projection[]
  game_log: GameLogRow[]
  matchup: Matchup[]
  math: Record<string, MathStep[]>
  narrative: string | null
  stat_order: string[]
}

export interface DefenseRow {
  team: string
  multiplier: number
  raw_value: number
  league_avg: number
  rank: number
  percentile: number
  n_games: number
}

export interface DefenseTable {
  position: string
  metric: string
  label: string
  season: number
  week: number
  mse_reduction_pct: number | null
  rows: DefenseRow[]
}

export interface CalibrationRow {
  position: string
  stat: string
  n: number
  mae: number
  bias: number
  coverage: number
  pit_central: number
  brier: number
  degenerate: boolean
}

export interface Calibration {
  run_id: string
  season: number
  week_start: number
  week_end: number
  coverage: number | null
  pit_central: number | null
  rows: CalibrationRow[]
}
