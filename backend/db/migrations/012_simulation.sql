-- Monte Carlo joint simulation output (§5.8).
--
-- The analytic projections in `projections` are the source of truth for every MARGINAL -- they are
-- the ones validated to 56.8% coverage on held-out 2025. The simulation exists to supply what a
-- marginal cannot: the CORRELATION between outcomes in the same game. A quarterback's yards and
-- his receiver's yards move together; a running back's carries and his own team trailing move
-- apart. Books price that correlation with full-game Monte Carlo and charge 20-35% hold on same
-- game parlays precisely because bettors cannot.
--
-- So the sim is transformed to preserve each player's analytic marginal exactly (rank-preserving),
-- and only its dependence structure is kept. It can add information; it cannot degrade calibration.

CREATE TABLE IF NOT EXISTS simulation_runs (
    run_id      VARCHAR PRIMARY KEY,
    season      INTEGER,
    week        INTEGER,
    n_sims      INTEGER,
    n_games     INTEGER,
    n_players   INTEGER,
    seconds     DOUBLE,
    seed        INTEGER,
    created_at  TIMESTAMP DEFAULT current_timestamp
);

-- Pairwise correlation between two player-stats in the same game.
CREATE TABLE IF NOT EXISTS simulation_correlations (
    run_id      VARCHAR,
    season      INTEGER,
    week        INTEGER,
    game_id     VARCHAR,
    gsis_id_a   VARCHAR,
    stat_a      VARCHAR,
    gsis_id_b   VARCHAR,
    stat_b      VARCHAR,
    correlation DOUBLE,
    same_team   BOOLEAN,
    PRIMARY KEY (run_id, gsis_id_a, stat_a, gsis_id_b, stat_b)
);

CREATE INDEX IF NOT EXISTS simcorr_game_idx ON simulation_correlations (season, week, game_id);

-- The simulated draws themselves, thinned, so the UI can price an arbitrary parlay without
-- re-running the simulation.
CREATE TABLE IF NOT EXISTS simulation_draws (
    run_id    VARCHAR,
    season    INTEGER,
    week      INTEGER,
    game_id   VARCHAR,
    gsis_id   VARCHAR,
    stat      VARCHAR,
    sim_index INTEGER,
    value     DOUBLE,
    PRIMARY KEY (run_id, gsis_id, stat, sim_index)
);

CREATE INDEX IF NOT EXISTS simdraws_lookup_idx ON simulation_draws (season, week, gsis_id, stat);
