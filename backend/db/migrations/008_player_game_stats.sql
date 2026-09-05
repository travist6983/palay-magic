-- One row per (player, game, stat): the single source every downstream stage reads.
--
-- Materialised rather than derived on the fly because ranking, adjusted game logs, projections and
-- the backtest all need the same numbers, and three of the four would otherwise re-derive
-- longest-play stats from 148k play-by-play rows each time.
--
-- Stat keys are exactly the canonical ones in backend/models/stats.py (D9). Derived stats
-- (rush_rec_yards, kicking_points, anytime_td, the longest_* family) are computed here so no
-- consumer has to know how they are assembled.

CREATE TABLE IF NOT EXISTS player_game_stats (
    gsis_id   VARCHAR,
    season    INTEGER,
    week      INTEGER,
    game_id   VARCHAR,
    team      VARCHAR,
    opponent  VARCHAR,
    position  VARCHAR,
    stat      VARCHAR,
    value     DOUBLE,
    PRIMARY KEY (gsis_id, season, week, stat)
);

CREATE INDEX IF NOT EXISTS pgs_lookup_idx ON player_game_stats (gsis_id, stat, season, week);
CREATE INDEX IF NOT EXISTS pgs_position_idx ON player_game_stats (position, stat, season, week);

-- Per-player, per-game usage context: the share numbers that drive §5.4, plus the snap and
-- route participation that §4's ranking and §5.7's role bucket need.
CREATE TABLE IF NOT EXISTS player_game_usage (
    gsis_id            VARCHAR,
    season             INTEGER,
    week               INTEGER,
    game_id            VARCHAR,
    team               VARCHAR,
    opponent           VARCHAR,
    position           VARCHAR,
    offense_snaps      INTEGER,
    offense_pct        DOUBLE,
    defense_snaps      INTEGER,
    defense_pct        DOUBLE,
    st_snaps           INTEGER,
    team_pass_attempts DOUBLE,
    team_rush_attempts DOUBLE,
    team_targets       DOUBLE,
    team_plays         DOUBLE,
    target_share       DOUBLE,
    carry_share        DOUBLE,
    dropback_share     DOUBLE,
    air_yards_share    DOUBLE,
    rz_targets         DOUBLE,
    rz_target_share    DOUBLE,
    rz_carries         DOUBLE,
    gl_carries         DOUBLE,   -- inside the 5
    gl_carry_share     DOUBLE,
    team_tackles       DOUBLE,
    tackle_share       DOUBLE,
    route_participation DOUBLE,  -- routes run / team dropbacks, from NGS where available
    PRIMARY KEY (gsis_id, season, week)
);

CREATE INDEX IF NOT EXISTS pgu_lookup_idx ON player_game_usage (gsis_id, season, week);
