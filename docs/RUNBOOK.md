# PropLab — Runbook

## Weekly rhythm

nflverse publishes the completed previous week by Tuesday morning ET, and the NFL's final injury
designations land Friday. So the useful cadence is two refreshes a week:

```cron
# Tuesday 09:00 — the previous week's games have landed with EPA attached
0 9 * * 2  cd /path/to/proplab && make refresh >> data/refresh.log 2>&1

# Saturday 09:00 — after Friday's final injury designations
0 9 * * 6  cd /path/to/proplab && make refresh >> data/refresh.log 2>&1
```

A refresh takes about **40 seconds**. The Odds API is called at most once per refresh, so two a
week is 8–9 of the 500 monthly requests.

## What runs when

| Command | Cadence | Time | Notes |
|---|---|---|---|
| `make backfill` | once | ~2 min | Downloads ~50 MB of Parquet for 2023–2026 |
| `make calibrate` | once, then after a model change | ~4 min | Walk-forward tuning; the penalties are stable |
| `make refresh` | twice weekly | ~40 s | Everything the app serves |
| `make backtest` | after a model change | ~60 s | The gate before trusting a change |
| `proplab simulate` | on demand | ~30 s | Correlations for parlay pricing |

## Running the app while refreshing

The API serves a **published snapshot** (`data/proplab-serve.duckdb`), not the live database.
DuckDB locks per file and a read-only reader still blocks a writer, so without the snapshot a
running `make dev` would make `make refresh` fail outright. The refresh publishes a fresh copy when
it finishes and the app picks it up on its next request — no restart needed.

If the snapshot is ever missing, `make dev` creates one, or run `proplab publish`.

## When something breaks

Every source degrades rather than failing the refresh. Check what is stale:

```bash
uv run proplab freshness   # per-source green/yellow/red
uv run proplab counts      # row counts for every view and table
uv run proplab budget      # Odds API requests used this month
uv run proplab llm         # Anthropic token usage
```

| Symptom | Cause | Fix |
|---|---|---|
| `the-odds-api` yellow, detail "ODDS_API_KEY not set" | Expected without a key | Nothing — nflverse `schedules` supplies the lines |
| `sleeper` skipped, "cache is Nh old" | The 1-call-per-day guard (D8) | Expected. `--force` overrides, but respect Sleeper's ask |
| `espn` yellow | The hidden API changed or rate-limited | Injury designations fall back to Sleeper; no action needed |
| A 2026 nflverse dataset "not published yet" | No games played | Expected before Week 1 |
| Rankings empty | No `team_environment` for the week | `make refresh`; check `proplab counts` |
| Board looks stale in the browser | Snapshot not republished | `uv run proplab publish` |

## Rate limits, enforced in code

| Source | Limit | Where |
|---|---|---|
| Sleeper `/v1/players/nfl` | 1/day | `backend/ingest/sleeper.py`, refuses under 20h |
| The Odds API | 500/month | `backend/ingest/budget.py`, persisted counter |
| Anthropic | 150/refresh | `backend/llm/client.py` |
| nflverse | none | — |

## Reading the numbers

- **Lines sit at the median, not the mean.** Yardage is right-skewed; the median is where a book
  would post.
- **Matchup effects are small.** The deep-dive page shows each matchup signal's measured MSE
  reduction. For most receiving metrics it is 1–3%. A red bar is information, not an edge.
- **Coverage on low-count stats is meaningless.** For sacks or anytime TD the p25 and p75 land on
  the same integer, so "inside the interval" catches every zero. The model page greys those out and
  judges them on PIT and Brier.
- **Everything is conditional on playing** unless you toggle to the unconditional column. A
  Questionable starter plays about 71% of the time.
