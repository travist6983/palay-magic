-- Held-out mean-bias calibration per (position, stat) (§6).
--
-- After the equation fixes the remaining errors are LEVEL errors of a few percent -- rushing
-- yards +2.2, passing yards -7.1, solo tackles +0.6 on a 2025 replay -- the residue of many
-- small measured constants and shrinkage choices. A multiplicative correction fitted on one
-- season's replay and validated on another absorbs them the same way dispersion_calibration
-- absorbs the interval width. Fitted on 2024, applied to 2025 and 2026.
CREATE TABLE IF NOT EXISTS bias_calibration (
    position    VARCHAR,
    stat        VARCHAR,
    ratio       DOUBLE,   -- actual mean / projected mean on the fit season, shrunk toward 1
    n           INTEGER,
    projected_mean DOUBLE,
    actual_mean DOUBLE,
    fitted_on   VARCHAR,
    computed_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (position, stat)
);
