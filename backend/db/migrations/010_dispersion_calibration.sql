-- Empirically calibrated dispersion scales (§6).
--
-- The projection's own variance estimate is not the predictive variance. It captures game-to-game
-- noise around a KNOWN mean, but the mean is itself estimated off a handful of games, and a
-- six-game variance estimate is biased low on top of that. Measured on a 2025 replay, yardage
-- intervals covered only 34% of outcomes instead of 50%; the variance needed roughly 2.2x.
--
-- So the width is calibrated rather than derived: solve for the scale that puts 50% of outcomes
-- inside p25-p75, per (position, stat), fitted on one season and validated on another. Counts came
-- out slightly OVER-covered, so the same procedure shrinks them.

CREATE TABLE IF NOT EXISTS dispersion_calibration (
    position    VARCHAR,
    stat        VARCHAR,
    scale       DOUBLE,   -- multiplies the model's variance before the distribution is fitted
    n           INTEGER,
    coverage_before DOUBLE,
    coverage_after  DOUBLE,
    fitted_on   VARCHAR,  -- e.g. '2024 weeks 5-18'
    computed_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (position, stat)
);
