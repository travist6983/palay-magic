-- PIT and the degenerate-interval flag on backtest rows (§6).
--
-- p25-p75 coverage is not a usable calibration metric for a low-count stat: a Poisson(0.13) has
-- p25 = p75 = 0, so "inside the interval" catches every zero and coverage reads 85%+ however good
-- the model is. The randomised probability integral transform is uniform under a correct model
-- for discrete and continuous families alike, so it is stored alongside.
ALTER TABLE backtest_results ADD COLUMN IF NOT EXISTS pit DOUBLE;
ALTER TABLE backtest_results ADD COLUMN IF NOT EXISTS degenerate_interval BOOLEAN;
