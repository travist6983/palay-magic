-- PropLab DuckDB schema (derived tables only).
--
-- Raw nflverse datasets live as Parquet in data/raw/ and are exposed as VIEWs by
-- backend/db/views.py (docs/DECISIONS.md D5). Everything below is computed by PropLab.
--
-- Migrations are plain SQL, applied in filename order by backend/db/migrate.py.
-- This file is migration 001 and is idempotent.

-- ---------------------------------------------------------------------------
-- Identity
-- ---------------------------------------------------------------------------

-- The crosswalk. Everything joins through gsis_id (D6).
CREATE TABLE IF NOT EXISTS players (
    gsis_id           VARCHAR PRIMARY KEY,
    display_name      VARCHAR,
    first_name        VARCHAR,
    last_name         VARCHAR,
    position          VARCHAR,
    position_group    VARCHAR,
    team              VARCHAR,          -- latest known team
    espn_id           VARCHAR,
    pfr_id            VARCHAR,
    pff_id            VARCHAR,
    sleeper_id        VARCHAR,
    sportradar_id     VARCHAR,
    birth_date        DATE,
    height            INTEGER,
    weight            INTEGER,
    headshot_url      VARCHAR,
    rookie_season     INTEGER,
    last_season       INTEGER,
    years_exp         INTEGER,
    draft_year        INTEGER,
    draft_round       INTEGER,
    draft_pick        INTEGER,
    status            VARCHAR,
    updated_at        TIMESTAMP DEFAULT current_timestamp
);
CREATE INDEX IF NOT EXISTS players_sleeper_idx ON players (sleeper_id);
CREATE INDEX IF NOT EXISTS players_pfr_idx     ON players (pfr_id);
CREATE INDEX IF NOT EXISTS players_espn_idx    ON players (espn_id);
CREATE INDEX IF NOT EXISTS players_pos_idx     ON players (position);

-- ---------------------------------------------------------------------------
-- Live injury state (Sleeper + ESPN). nflverse's official weekly report is a raw view.
-- ---------------------------------------------------------------------------

-- Latest known status per player, from whichever source last succeeded.
CREATE TABLE IF NOT EXISTS injury_status (
    gsis_id                 VARCHAR,
    sleeper_id              VARCHAR,
    source                  VARCHAR,   -- 'sleeper' | 'espn'
    observed_at             TIMESTAMP,
    team                    VARCHAR,
    position                VARCHAR,
    injury_status           VARCHAR,   -- Questionable | Doubtful | Out | IR | NULL
    roster_status           VARCHAR,   -- Active | Inactive | Injured Reserve | PUP | ...
    practice_participation  VARCHAR,   -- DNP | Limited | Full
    injury_body_part        VARCHAR,
    injury_notes            VARCHAR,
    depth_chart_position    VARCHAR,
    depth_chart_order       INTEGER,
    PRIMARY KEY (gsis_id, source)
);

-- Every observed change, from diffing consecutive Sleeper snapshots (§3).
CREATE TABLE IF NOT EXISTS injury_events (
    event_id     VARCHAR PRIMARY KEY,  -- md5(gsis_id|field|observed_at)
    gsis_id      VARCHAR,
    sleeper_id   VARCHAR,
    player_name  VARCHAR,
    team         VARCHAR,
    position     VARCHAR,
    field        VARCHAR,              -- injury_status | practice_participation | depth_chart_order | status
    old_value    VARCHAR,
    new_value    VARCHAR,
    observed_at  TIMESTAMP
);
CREATE INDEX IF NOT EXISTS injury_events_player_idx ON injury_events (gsis_id, observed_at);

-- Empirical P(played) and E[snap share | played] per (designation, practice status). §5.7
CREATE TABLE IF NOT EXISTS injury_play_rates (
    report_status     VARCHAR,
    practice_status   VARCHAR,
    position_group    VARCHAR,
    n_observations    INTEGER,
    n_played          INTEGER,
    p_played          DOUBLE,
    mean_snap_share   DOUBLE,   -- conditional on playing
    sd_snap_share     DOUBLE,
    computed_at       TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (report_status, practice_status, position_group)
);

-- ---------------------------------------------------------------------------
-- Game environment (§5.3)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS game_environment (
    game_id              VARCHAR PRIMARY KEY,
    season               INTEGER,
    week                 INTEGER,
    home_team            VARCHAR,
    away_team            VARCHAR,
    kickoff              TIMESTAMP,
    spread_line          DOUBLE,   -- positive = home favored, nflverse convention
    total_line           DOUBLE,
    home_implied_total   DOUBLE,
    away_implied_total   DOUBLE,
    roof                 VARCHAR,
    surface              VARCHAR,
    temp                 DOUBLE,
    wind                 DOUBLE,   -- mph; forced to 0 for dome/closed
    odds_source          VARCHAR,  -- 'the-odds-api' | 'nflverse-schedules'
    observed_at          TIMESTAMP DEFAULT current_timestamp
);

-- Team-level pace / pass-rate expectations feeding usage (§5.3).
CREATE TABLE IF NOT EXISTS team_environment (
    season                 INTEGER,
    week                   INTEGER,
    team                   VARCHAR,
    opponent               VARCHAR,
    game_id                VARCHAR,
    is_home                BOOLEAN,
    implied_total          DOUBLE,
    spread                 DOUBLE,   -- team's own spread; negative = favored
    sec_per_play           DOUBLE,
    expected_plays         DOUBLE,
    base_pass_rate         DOUBLE,   -- recency-weighted actual
    proe                   DOUBLE,   -- pass rate over expectation, from pbp xpass
    script_adjustment      DOUBLE,   -- shift applied for spread (§5.3)
    expected_pass_rate     DOUBLE,
    expected_pass_attempts DOUBLE,
    expected_rush_attempts DOUBLE,
    expected_dropbacks     DOUBLE,
    expected_drives        DOUBLE,
    expected_rz_trips      DOUBLE,
    expected_team_tds      DOUBLE,
    expected_team_fgs      DOUBLE,
    computed_at            TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (season, week, team)
);

-- ---------------------------------------------------------------------------
-- Opponent adjustment (§5.2)
-- ---------------------------------------------------------------------------

-- Position-specific defensive strength as a multiplier relative to league average.
-- 1.12 = this defense allows 12% more than average. Shrunk toward 1.0 (D8/§5.2).
CREATE TABLE IF NOT EXISTS defense_multipliers (
    season          INTEGER,
    week            INTEGER,   -- "as of" week: uses games strictly BEFORE this week
    team            VARCHAR,   -- the defense
    position        VARCHAR,   -- QB | RB | WR | TE | K | LB | ALL
    metric          VARCHAR,   -- see backend/models/adjust.py METRICS
    raw_value       DOUBLE,
    league_avg      DOUBLE,
    multiplier      DOUBLE,    -- shrunk
    raw_multiplier  DOUBLE,    -- unshrunk, for "show math"
    n_games         INTEGER,
    rank            INTEGER,   -- 1 = allows the most (softest)
    computed_at     TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (season, week, team, position, metric)
);

-- Opponent-adjusted player game logs: raw and adjusted, side by side (§5.2).
CREATE TABLE IF NOT EXISTS adjusted_game_log (
    gsis_id            VARCHAR,
    season             INTEGER,
    week               INTEGER,
    game_id            VARCHAR,
    team               VARCHAR,
    opponent           VARCHAR,
    position           VARCHAR,
    stat               VARCHAR,
    raw_value          DOUBLE,
    opponent_multiplier DOUBLE,
    adjusted_value     DOUBLE,   -- raw / multiplier
    opponent_rank      INTEGER,
    metric             VARCHAR,  -- which defense_multipliers metric was applied
    computed_at        TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (gsis_id, season, week, stat)
);

-- ---------------------------------------------------------------------------
-- Ranking (§4)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS rankings (
    season            INTEGER,
    week              INTEGER,
    position          VARCHAR,
    rank              INTEGER,
    gsis_id           VARCHAR,
    score             DOUBLE,
    tiebreaker        DOUBLE,
    team              VARCHAR,
    opponent          VARCHAR,
    n_games_used      INTEGER,
    insufficient_history BOOLEAN,
    changed_team      BOOLEAN,
    changed_coach     BOOLEAN,
    pass_rate_shift   DOUBLE,
    injury_status     VARCHAR,
    play_probability  DOUBLE,
    components        JSON,     -- every input to the score, for "show math"
    computed_at       TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (season, week, position, gsis_id)
);

-- ---------------------------------------------------------------------------
-- Projections (§5.6)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS projections (
    season            INTEGER,
    week              INTEGER,
    gsis_id           VARCHAR,
    position          VARCHAR,
    stat              VARCHAR,   -- docs/DECISIONS.md D9
    dist_family       VARCHAR,   -- negative_binomial | poisson | bernoulli | empirical_max | deterministic
    params            JSON,      -- family-specific; the browser recomputes P(over) from these (D10)
    mean              DOUBLE,
    median            DOUBLE,
    p25               DOUBLE,
    p75               DOUBLE,
    conditional_on_playing BOOLEAN,  -- TRUE = assumes the player is active (§5.7)
    play_probability  DOUBLE,
    high_variance     BOOLEAN,   -- UI label for sacks/INTs (§5.6)
    computed_at       TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (season, week, gsis_id, stat, conditional_on_playing)
);

-- Every intermediate value from §5, keyed to the projection it produced ("Show math", §8).
CREATE TABLE IF NOT EXISTS projection_math (
    season      INTEGER,
    week        INTEGER,
    gsis_id     VARCHAR,
    stat        VARCHAR,
    step        INTEGER,   -- ordering
    section     VARCHAR,   -- '5.1 baseline' | '5.2 opponent' | ...
    label       VARCHAR,
    value       DOUBLE,
    detail      VARCHAR,
    PRIMARY KEY (season, week, gsis_id, stat, step)
);

-- ---------------------------------------------------------------------------
-- Backtest (§6)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS backtest_results (
    run_id        VARCHAR,
    season        INTEGER,
    week          INTEGER,
    gsis_id       VARCHAR,
    position      VARCHAR,
    stat          VARCHAR,
    projected_median DOUBLE,
    projected_p25 DOUBLE,
    projected_p75 DOUBLE,
    dist_family   VARCHAR,
    params        JSON,
    actual        DOUBLE,
    abs_error     DOUBLE,
    in_interval   BOOLEAN,
    p_over_median DOUBLE,   -- model P(X > median), for the Brier score
    outcome_over  BOOLEAN,
    PRIMARY KEY (run_id, season, week, gsis_id, stat)
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id       VARCHAR PRIMARY KEY,
    season       INTEGER,
    week_start   INTEGER,
    week_end     INTEGER,
    n_projections INTEGER,
    started_at   TIMESTAMP,
    finished_at  TIMESTAMP,
    config       JSON
);

-- ---------------------------------------------------------------------------
-- Operational: budgets, freshness, caches (§3, §7, §8)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS api_budget (
    api          VARCHAR,   -- 'the-odds-api' | 'anthropic' | 'sleeper'
    period       VARCHAR,   -- 'YYYY-MM' for monthly, 'YYYY-MM-DD' for daily
    n_calls      INTEGER DEFAULT 0,
    budget       INTEGER,
    last_call_at TIMESTAMP,
    PRIMARY KEY (api, period)
);

CREATE TABLE IF NOT EXISTS source_freshness (
    source           VARCHAR PRIMARY KEY,  -- 'nflverse' | 'sleeper' | 'espn' | 'the-odds-api' | 'anthropic'
    last_attempt_at  TIMESTAMP,
    last_success_at  TIMESTAMP,
    status           VARCHAR,   -- 'green' | 'yellow' | 'red'
    detail           VARCHAR,
    n_rows           BIGINT
);

CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key    VARCHAR PRIMARY KEY,  -- sha256(task|model|canonical input)
    task         VARCHAR,
    model        VARCHAR,
    input_json   JSON,
    output_json  JSON,
    input_tokens INTEGER,
    output_tokens INTEGER,
    created_at   TIMESTAMP DEFAULT current_timestamp
);

-- What week the app is currently showing, and when each stage last ran.
CREATE TABLE IF NOT EXISTS refresh_state (
    key         VARCHAR PRIMARY KEY,
    value       VARCHAR,
    updated_at  TIMESTAMP DEFAULT current_timestamp
);
