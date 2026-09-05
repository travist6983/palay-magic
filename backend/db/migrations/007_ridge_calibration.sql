-- Store the calibrated ridge penalty per metric (§5.2).
--
-- Walk-forward evaluation over 2023-2025 (fit on everything before week w, predict week w) shows
-- the two-way offence x defence ridge model beats a shrunk raw trailing-8 rate on all 29 metrics,
-- usually by 5-10x, and beats the unshrunk raw rate -- which is itself WORSE than just predicting
-- league average on every single metric -- by even more.
--
--   metric                     raw + measured beta      two-way ridge
--   success_rate_allowed                     3.22%             13.35%
--   target_volume_allowed_wr                 1.76%             10.19%
--   rec_yards_allowed_wr                     0.05%              1.98%
--   (% weighted-MSE reduction against a league-average prediction)
--
-- The penalty is tuned per metric because the optimum ranges from 5 (success rate, high signal)
-- to 250 (FG attempts allowed, almost none).

ALTER TABLE metric_reliability ADD COLUMN IF NOT EXISTS ridge_lambda DOUBLE;
ALTER TABLE metric_reliability ADD COLUMN IF NOT EXISTS mse_reduction_pct DOUBLE;
ALTER TABLE metric_reliability ADD COLUMN IF NOT EXISTS raw_mse_reduction_pct DOUBLE;

ALTER TABLE defense_multipliers ADD COLUMN IF NOT EXISTS ridge_lambda DOUBLE;
ALTER TABLE defense_multipliers ADD COLUMN IF NOT EXISTS model VARCHAR DEFAULT 'two_way_ridge';
