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

## Milestone 2 — Defensive multipliers + adjusted game logs  ·  NOT STARTED
## Milestone 3 — Ranking  ·  NOT STARTED
## Milestone 4 — Projections v1  ·  NOT STARTED
## Milestone 5 — Backtest  ·  NOT STARTED
## Milestone 6 — API + frontend  ·  NOT STARTED
## Milestone 7 — LLM notes, Odds API, weather  ·  NOT STARTED
## Milestone 8 — Monte Carlo joint sim  ·  NOT STARTED
