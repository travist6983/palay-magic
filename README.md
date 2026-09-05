# PropLab

A single-user, locally-run NFL player-prop research app.

PropLab ranks the top 10 players at each prop-relevant position (QB · RB · WR · TE · K · LB),
pulls their recent game logs, adjusts every stat for the quality of the defense they faced, and
produces a **probabilistic projection** — median, p25, p75, and P(over X) — for each major stat in
the current week.

Nothing here is shared or deployed. No auth, no multi-tenancy, no cloud.

## Quick start

```bash
cp .env.example .env      # optional: add ANTHROPIC_API_KEY and ODDS_API_KEY
make setup                # uv sync + npm install
make backfill             # one-time nflverse pull (2023-2026). Takes a few minutes.
make refresh              # recompute rankings + projections for the current week
make dev                  # http://localhost:5173
```

## Commands

| Command | What it does |
|---|---|
| `make setup` | Install Python and JS dependencies |
| `make backfill` | One-time historical ingest of all seasons |
| `make refresh` | Pull current-season data, recompute defenses, rankings, projections |
| `make dev` | Run the FastAPI backend and the Vite frontend together |
| `make test` | pytest (model math) + vitest (frontend utils) |
| `make backtest` | Score last season's projections against actuals |

The CLI underneath is `uv run proplab <command>` — see `uv run proplab --help`.

## Weekly cron

nflverse publishes the completed previous week by Tuesday morning ET. Refresh Tuesday and again
Saturday (after the Friday injury designations land):

```cron
0 9 * * 2   cd /path/to/proplab && make refresh >> data/refresh.log 2>&1
0 9 * * 6   cd /path/to/proplab && make refresh >> data/refresh.log 2>&1
```

## Documentation

- `docs/reference/` — the three source-of-truth research docs (data sources, prop settlement
  rules, how books build lines)
- `docs/PROGRESS.md` — build status per milestone
- `docs/DECISIONS.md` — the contract: naming, conventions, and resolved ambiguities

## Responsible gambling

This is a research tool, not a guarantee. Bet only what you can afford to lose.
US problem gambling helpline: **1-800-GAMBLER** (1-800-426-2537), or call/text **1-800-522-4700**.
