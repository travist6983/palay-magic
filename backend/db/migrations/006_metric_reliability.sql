-- How much of each metric's apparent edge actually survives into the next game (§5.2).
--
-- Fitted out of sample: pair each team's trailing-window rate with the next game's rate, both as
-- deviations from league average, and regress through the origin weighted by the next game's
-- denominator. The slope IS the optimal shrinkage weight, and unlike a flat k it is measured
-- rather than assumed.

CREATE TABLE IF NOT EXISTS metric_reliability (
    unit        VARCHAR,
    metric      VARCHAR,
    beta        DOUBLE,   -- clamped to [0, 1]; the shrinkage weight actually applied
    beta_raw    DOUBLE,   -- unclamped fit, so a negative slope is visible rather than hidden
    r_squared   DOUBLE,
    n_pairs     BIGINT,
    implied_k   DOUBLE,   -- n(1-beta)/beta, comparable to the spec's k = 6
    computed_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (unit, metric)
);

ALTER TABLE defense_multipliers ADD COLUMN IF NOT EXISTS beta DOUBLE;
