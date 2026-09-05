-- Add the `unit` dimension to defense_multipliers (§5.2).
--
-- Most props are adjusted for the opposing DEFENSE: how many receiving yards does this team allow
-- to WRs. But LB props are adjusted for the opposing OFFENSE: how many plays does it run (which
-- drives tackle volume) and how many sacks does it allow. Both are "the opponent unit's effect on
-- this stat", so they live in one table with a `unit` discriminator rather than two near-identical
-- tables.
--
-- unit = 'defense' -> what this team's defense allows
-- unit = 'offense' -> what this team's offense runs into / gives up

DROP TABLE IF EXISTS defense_multipliers;

CREATE TABLE defense_multipliers (
    season          INTEGER,
    week            INTEGER,   -- "as of": uses games strictly BEFORE this week
    team            VARCHAR,
    unit            VARCHAR,   -- 'defense' | 'offense'
    position        VARCHAR,   -- QB | RB | WR | TE | K | LB | ALL
    metric          VARCHAR,
    numerator       DOUBLE,
    denominator     DOUBLE,
    raw_value       DOUBLE,    -- numerator / denominator
    league_avg      DOUBLE,
    multiplier      DOUBLE,    -- shrunk toward 1.0 with k games of league-average prior
    raw_multiplier  DOUBLE,    -- unshrunk, shown in "Show math"
    n_games         INTEGER,
    rank            INTEGER,   -- 1 = allows the most (softest matchup)
    computed_at     TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (season, week, team, unit, metric)
);

CREATE INDEX IF NOT EXISTS defmult_lookup_idx ON defense_multipliers (season, week, team, metric);

-- Per-team, per-game facts that every multiplier is aggregated from. Materialised once so the
-- backtest can replay any week without re-scanning 148k play-by-play rows each time.
CREATE TABLE IF NOT EXISTS unit_game_facts (
    season      INTEGER,
    week        INTEGER,
    game_id     VARCHAR,
    team        VARCHAR,
    unit        VARCHAR,
    opponent    VARCHAR,
    kickoff     TIMESTAMP,
    metric      VARCHAR,
    numerator   DOUBLE,
    denominator DOUBLE,
    PRIMARY KEY (season, week, team, unit, metric)
);

CREATE INDEX IF NOT EXISTS ugf_team_idx ON unit_game_facts (team, unit, metric, season, week);
