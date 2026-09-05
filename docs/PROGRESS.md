# PropLab — Build Progress

Status per milestone (§9 of the build spec). Each entry records what landed, how it was verified,
and what is still open.

---

## Milestone 1 — Scaffold + ingest  ·  DONE

**Landed**

- Repo scaffold, `uv` env on Python 3.12, `pyproject.toml`, `Makefile`, `.env.example`, `README.md`
- `backend/config.py` — pydantic-settings; every model knob (§5) and budget (§3) is configurable
- `backend/logging_setup.py` — Rich logging, one setup for CLI and API
- `backend/db/` — connection, plain-SQL migrations, and the raw-Parquet view registry (D5)
- `backend/db/migrations/001_initial.sql` — 17 derived tables covering identity, injuries, game
  environment, defensive multipliers, adjusted game logs, rankings, projections, the "show math"
  trace, backtest results, and the operational tables (budgets, freshness, LLM cache)
- `backend/ingest/base.py` — atomic Parquet writes, retry, `@resilient` (a dead source degrades,
  never raises), and per-source freshness badges
- `backend/ingest/budget.py` — persisted request counters so the Odds API 500/month cap is
  enforced in code across process restarts, not by convention (D8)
- `backend/ingest/{nflverse,sleeper,espn,odds,crosswalk}.py` — one module per source (§3)
- `backend/state.py` — current season/week from Sleeper `/v1/state/nfl`, with a schedules-derived
  fallback. Never hardcoded (§10)
- `backend/pipeline.py` — the ordered refresh stages; model stages self-skip until they exist
- `backend/cli.py` — typer CLI: `backfill refresh counts state freshness budget defense rank
  project backtest config`
- `docs/DECISIONS.md` — the contract (D1–D10), including the two resolved ambiguities below
- `docs/nflverse_schema.json` — verified column lists for all 17 nflverse datasets, so no module
  has to guess a column name

**Landed early (needed by later milestones, written while ingest ran)**

- `backend/models/stats.py` — the canonical stat registry (D9): 25 stat keys across six positions,
  each with its distribution family, defensive metric, and settlement rule
- `backend/models/distributions.py` — §5.6 in full: negative binomial, Poisson, over-dispersion
  test, anytime-TD Poisson, Monte Carlo `longest_*` with its point mass at zero, and exact
  numerical convolution for kicking points

**Verified**

- 8 infrastructure tests + 44 distribution tests pass
- The distributions reproduce the published base rates in `docs/reference/`:
  Poisson(1.5) gives P(over 1.5 passing TDs) = 44.2% against the doc's 45.9%, and
  P(zero passing TDs) = 22.3% against the doc's 21.1%
- `3 × E[FGM] + 1 × E[XPM]` convolution matches a 200k-draw simulation to within 1 point

**Resolved ambiguities** (full reasoning in `docs/DECISIONS.md`)

1. **`nfl_data_py` cannot install on Python 3.12** — it pins `pandas<2, numpy<2`, neither of which
   ships cp312 wheels. Switched to nflverse's official successor `nflreadpy` (D1).
2. **Tackles + assists** — FanDuel includes special-teams tackles, DraftKings excludes them.
   Measured: 9.9% of 2025 solo tackles were on special teams. We project the DraftKings number,
   which maps 1:1 onto the PFR gamebook defense table (D2). Verified that
   `def_tackles_solo + def_tackle_assists` reproduces PFR `def_tackles_combined` on 89.2% of 2025
   LB game-weeks exactly, while `def_tackles_with_assist` matches only 19.0% — that column is not
   the assist count and must never be used for tackle props.

**Checkpoint — row counts after `make backfill` + `make refresh`**

| View / table | Rows |
|---|---|
| raw_pbp (2023-25) | 147,928 |
| raw_player_stats | 57,048 |
| raw_snap_counts | 79,767 |
| raw_injuries | 17,882 |
| raw_depth_charts | 1,127,746 |
| raw_rosters (2023-26) | 12,389 |
| raw_pfr_pass / rush / rec / def | 2,081 / 7,094 / 13,580 / 23,820 |
| raw_players | 24,828 |
| raw_schedules | 7,548 |
| raw_ngs_passing / rushing / receiving | 5,933 / 6,059 / 14,731 |
| raw_sleeper_players | 12,226 |
| raw_espn_injuries | 1,899 |
| raw_espn_weather | 16 |
| **players** (crosswalk) | **25,040** |
| injury_status | 5,782 |
| injury_play_rates | 376 |
| game_environment | 30 |

`make refresh` end to end: **91.7s** — well inside the five-minute budget (§1.5), though 86.9s of
it is the ESPN injury hydrate (~1,930 requests at 6 in flight). Worth optimising in milestone 7;
the model stages still have ~3.5 minutes of headroom.

**Defects found and fixed during verification**

1. **Sleeper `gsis_id` coverage** — Sleeper supplies it for only 3,893 of 12,226 players, so ~20%
   of rostered players at prop positions got no injury data at all. Added unique-match recovery by
   `espn_id` then `normalised name | team | position` (D12). TE coverage 78.0% → 88.3%, K 82.5% →
   95.0%.
2. **Injury play rates were pooling roles** — gave a Questionable + Full player 70.6%, against the
   reference doc's "very high". Split by prior-four-week snap share, a *starter* in that cell
   plays 84.8%, Questionable + DNP 53.9% ("roughly coin flip", as documented). Migration 002 (D11).
   The first attempt at this was itself wrong: deriving the role window from the same snap join
   that decides `played` made every cell read exactly 1.000.
3. **`build_game_environment` was not atomic** — it deleted a week then inserted, outside a
   transaction and with no empty-frame guard, so a transient failure silently wiped that week's
   rows. Observed live: 2026 Week 1 lost all 16 rows. Now one transaction, and an empty frame
   leaves the existing week untouched.
4. **Empty odds Parquet** — a zero-row `odds_{season}_{week}.parquet` made `raw_odds` look present
   but useless. `ingest_odds` now skips the write and falls back to schedules.

**Open**

- `injury_events` is empty because there is only one Sleeper snapshot so far. It populates on the
  second daily pull, which is the intended behaviour, not a gap.

---

## Milestone 2 — Defensive multipliers + adjusted game logs  ·  DONE

The spec's estimator does not work, and the backtest is how we know. Walk-forward over 2023–2025:

| | raw trailing-8 | + measured shrinkage | two-way ridge |
|---|---|---|---|
| success rate allowed | −6.1% | +3.2% | **+13.4%** |
| targets allowed to WRs | −6.1% | +1.8% | **+10.2%** |
| yards/target allowed to WRs | −11.3% | +0.1% | **+2.0%** |

(% weighted-MSE reduction against predicting league average.) The prescribed trailing-8 rate is
**worse than league average on all 29 metrics**, because it confounds how good a defence is with
how good the offences it faced were. Production uses a two-way offence × defence ridge model with a
per-metric penalty tuned by walk-forward search.

Two things the numbers forced: EPA metrics are pinned to a multiplier of 1.0 (league average EPA
per dropback was +0.035, so a ratio gave Washington 6.22× and Philadelphia −2.59×), and the honest
headline is that **matchup effects are small** — 1–3% MSE reduction for most receiving stats. The
UI shows that number rather than implying the matchup decides the game.

## Milestone 3 — Ranking  ·  DONE

Usage-first boards for all six positions, games weighted by snaps so a Week 18 rest game stops
reading as a role change. Depth-chart gate for QB and K — without it, a backup's injury-stretch
volume put Easton Stick above Patrick Mahomes. Thin histories shrink toward the position median.

## Milestone 4 — Projections v1  ·  DONE

`usage × efficiency`, opponent-adjusted, as a distribution. Baselines are computed on
opponent-adjusted values so the schedule is not counted twice. Usage and efficiency use different
windows: six games for role, seventeen shrunk toward a positional prior for efficiency, because
Gibbs' last six 2025 games ran 3.09 yards a carry against a true rate near 4.5.

Wind and field goals are fitted, not assumed: −0.049 Y/A per mph (−5.0% at 15 mph), and a logistic
FG model reproducing the league's 85.3% make rate. `longest_*` bootstraps the player's own play
gains; the parametric tail it replaced put a QB's median longest completion at 75 yards.

## Milestone 5 — Backtest  ·  DONE

**Held-out 2025, weeks 5–18: p25–p75 coverage 56.8%, PIT central mass 50.8%.** §6's gate is
40–60%; both clear it. Hyperparameters are refit on prior seasons only, so the test season is not
in its own training set, and the Odds API is never called.

The backtest changed the model three times: it exposed that yardage dispersion needed 1.7–2.7×
(fitted on 2024, validated on 2025), that coverage is meaningless for low-count stats where
p25 = p75, and that anytime-TD Brier was being scored against its own median.

## Milestone 6 — API + frontend  ·  DONE

`make dev` opens the app. FastAPI read layer over a published snapshot; React + Vite + Tailwind
front end with the six boards, the deep-dive page, and a model-health page.

The client recomputes `P(over)` in the browser from the returned distribution parameters, with no
request per keystroke (D10). 17 vitest cases pin the TypeScript against fixtures generated by the
Python to 1e-6.

## Milestone 7 — LLM notes, Odds API, weather  ·  DONE

Three narrow tasks behind a cached, budget-capped wrapper. Every prompt forbids inventing numbers
and carries the settlement rules. No key means no notes and no other change.

## Milestone 8 — Monte Carlo joint sim  ·  DONE

Drive-level, both teams against a shared clock and score. Each series is transformed to match its
analytic marginal rank for rank, so the simulation contributes only dependence and cannot degrade
the calibration §6 validated. `proplab parlay` prices a multi-leg ticket against the naive
independent product.

---

## Where the model is weak

Stated plainly, because these are the things that would mislead someone reading a number:

1. **Matchup effects are small.** 1–3% MSE reduction for most receiving stats. The two-way ridge is
   a real improvement over the alternatives, and it is still a small signal.
2. **Play-count and pass-rate models are weak** (R² 0.013 and 0.099). Single-game pace is close to
   unpredictable; the models mostly return the league mean with a real but modest spread effect.
3. **Coverage on low-count stats cannot be judged.** Sacks, interceptions, anytime TD: the interval
   is a point mass and the metric degenerates. PIT and Brier are the honest read.
4. **QB rushing yards pool mobile and pocket quarterbacks**, and no interval width fixes that. It
   is the worst-calibrated cell in the table.
5. **Week 1 2026 rests entirely on 2025.** Every ranking, projection and multiplier. The team,
   coach and pass-rate flags are the only current-season signal, and the app says so on every page.
