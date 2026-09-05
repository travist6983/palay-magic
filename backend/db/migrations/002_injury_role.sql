-- Add the role dimension to injury_play_rates (§5.7).
--
-- Why: computed without it, P(played | Questionable + Full practice) came out at 70.6%, which
-- contradicts free_nfl_data_sources.md's "very high". The cause is population, not a join bug:
-- the weekly injury report is dominated by fringe players who often do not dress at all, and
-- pooling them with starters drags every cell down. Split by the player's role over the prior
-- four weeks, a Questionable + Full STARTER plays 84.8% of the time and a Questionable + DNP
-- starter 53.7% -- which is the "roughly coin flip" the doc describes.
--
-- Applying the pooled number to a starting WR would have under-projected him by ~15 points of
-- play probability, so this dimension is load-bearing, not cosmetic.

DROP TABLE IF EXISTS injury_play_rates;

CREATE TABLE injury_play_rates (
    report_status     VARCHAR,
    practice_status   VARCHAR,
    role_bucket       VARCHAR,   -- 'starter' | 'rotational' | 'fringe' | 'no_recent_games'
    position_group    VARCHAR,   -- 'ALL' for the marginal row
    n_observations    INTEGER,
    n_played          INTEGER,
    p_played          DOUBLE,
    mean_snap_share   DOUBLE,    -- conditional on playing
    sd_snap_share     DOUBLE,
    computed_at       TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (report_status, practice_status, role_bucket, position_group)
);
