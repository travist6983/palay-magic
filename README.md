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
make calibrate            # one-time: tune the opponent model and interval widths
make refresh              # recompute rankings + projections for the current week
make dev                  # http://localhost:5173
```

Neither API key is required. Without `ODDS_API_KEY` the implied team totals come from nflverse
`schedules`, which already carries closing spreads and totals. Without `ANTHROPIC_API_KEY` the
narrative notes are simply omitted; every number is unaffected.

## Commands

| Command | What it does |
|---|---|
| `make setup` | Install Python and JS dependencies |
| `make backfill` | One-time historical ingest of all seasons |
| `make refresh` | Pull current-season data, recompute defenses, rankings, projections |
| `make dev` | Run the FastAPI backend and the Vite frontend together |
| `make calibrate` | Re-tune the opponent-adjustment model and the interval widths |
| `make test` | pytest (model math) + vitest (frontend utils) |
| `make backtest` | Score last season's projections against actuals |
| `make notes` | Generate the week's LLM narratives (needs a key) |

The CLI underneath is `uv run proplab <command>` — see `uv run proplab --help`.

## Weekly cron

nflverse publishes the completed previous week by Tuesday morning ET. Refresh Tuesday and again
Saturday (after the Friday injury designations land):

```cron
0 9 * * 2   cd /path/to/proplab && make refresh >> data/refresh.log 2>&1
0 9 * * 6   cd /path/to/proplab && make refresh >> data/refresh.log 2>&1
```

## How it works

Every projection is `usage × efficiency`, adjusted for the opponent and the game environment, and
reported as a **distribution** rather than a point estimate — because a posted line sits at the
median of a skewed distribution, not at the mean.

- **Usage** comes from the market: the spread and total give an implied team total, which drives
  expected plays, pass rate and touchdowns. The player's share of that volume is recency-weighted
  and snap-weighted, so a Week 18 rest game does not read as a change in role.
- **Efficiency** is the player's opponent-neutral per-opportunity rate, shrunk toward a positional
  prior. Efficiency is far less predictable than usage, so it gets a longer window and more
  shrinkage.
- **The opponent adjustment** is a two-way offence × defence ridge model, not a raw trailing
  average. Walk-forward testing showed the raw average is *worse than predicting league average*
  on every metric, because it confounds how good a defence is with how good the offences it faced
  were.
- **The intervals are calibrated, not derived.** Fitted on 2024 and validated on held-out 2025:
  p25–p75 coverage 56.8%, PIT central mass 50.8%, against a 50% target.

An honest caveat the app surfaces rather than hides: opponent matchup effects are **small**. Over
2023–2025 the matchup signal removes only 1–3% of squared error for most receiving stats. The
deep-dive page shows that number next to every matchup bar.

## Documentation

- `docs/reference/` — the three source-of-truth research docs (data sources, prop settlement
  rules, how books build lines)
- `docs/PROGRESS.md` — build status per milestone
- `docs/DECISIONS.md` — the contract: naming, conventions, and resolved ambiguities

## Responsible gambling

This is a research tool, not a guarantee. Bet only what you can afford to lose.
US problem gambling helpline: **1-800-GAMBLER** (1-800-426-2537), or call/text **1-800-522-4700**.
