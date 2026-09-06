-- E[current snap share / normal snap share | played] per cell (§5.7).
--
-- The unconditional projection scaled usage by E[snap share | played] divided by the player's
-- own normal share. That population mean (0.762) is averaged over starters whose normal share is
-- ~0.82, so dividing it by a 95%-snap player's own share cut his usage 20-23% -- while the data
-- says a Questionable starter who plays keeps 93% of his usual snaps (median 100%). The ratio is
-- what should be stored, and it is now.
ALTER TABLE injury_play_rates ADD COLUMN IF NOT EXISTS mean_snap_ratio DOUBLE;
