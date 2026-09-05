-- Separate ratio-scale metrics from additive-scale ones (§5.2).
--
-- A multiplier is a ratio to league average, which only means something when the metric lives on
-- a ratio scale with a meaningful zero: yards, targets, carries, rates bounded at zero. EPA does
-- not. League-average EPA per dropback in the 2025 trailing-8 window was +0.035, so dividing by
-- it produced multipliers of 6.22 for Washington and -2.59 for Philadelphia -- numbers that would
-- have multiplied a projection into nonsense had anything used them as a divisor.
--
-- Fix: every metric declares a scale. Additive-scale metrics are pinned to multiplier = 1.0 so
-- they can never adjust a stat, and carry their signal in z_score instead, which is what the UI's
-- defensive-rank bar and the LLM narrative ("bottom-8 run defense") actually want.

ALTER TABLE defense_multipliers ADD COLUMN IF NOT EXISTS scale VARCHAR DEFAULT 'ratio';
ALTER TABLE defense_multipliers ADD COLUMN IF NOT EXISTS z_score DOUBLE;
ALTER TABLE defense_multipliers ADD COLUMN IF NOT EXISTS percentile DOUBLE;
