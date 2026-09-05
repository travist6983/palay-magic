-- Record the shrinkage actually applied to each multiplier (§5.2).
--
-- k is now estimated per metric by empirical Bayes rather than fixed at 6, so "Show math" has to
-- be able to say which k produced a given number.
ALTER TABLE defense_multipliers ADD COLUMN IF NOT EXISTS effective_k DOUBLE;
