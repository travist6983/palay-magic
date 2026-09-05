-- Fitted coefficients for the game-environment models (§5.3).
--
-- §5.3 describes the shape (implied team total, pace, PROE, a game-script shift for big spreads)
-- but not the magnitudes. Assuming them would put unvalidated constants underneath every usage
-- projection, so each relationship is fitted on 2023-2025 team-games and the coefficients stored
-- here, where "Show math" can display them and the backtest can re-fit them.

CREATE TABLE IF NOT EXISTS environment_models (
    model       VARCHAR,   -- 'plays' | 'pass_rate' | 'team_tds' | 'team_fgs' | 'sec_per_play'
    term        VARCHAR,   -- 'intercept' | a feature name
    coefficient DOUBLE,
    std_error   DOUBLE,
    n_observations INTEGER,
    r_squared   DOUBLE,
    rmse        DOUBLE,
    computed_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (model, term)
);
