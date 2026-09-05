# PropLab — Decisions & Conventions

The contract every module codes against. Resolved ambiguities live here, with the reason.

---

## D1. nflverse loader: `nflreadpy`, not `nfl_data_py`

`nfl_data_py==0.3.3` pins `pandas<2.0, numpy<2.0`. Neither publishes cp312 wheels, so it cannot
install on Python 3.12 (verified: source build fails on `pkg_resources`). nflverse's official
successor `nflreadpy` (polars-native, `requires-python >=3.10`) loads every dataset we need.

Decided with the user 2026-09-05. Direct nflverse-release Parquet URLs are the fallback if the
package breaks (`backend/ingest/nflverse.py::_download_release_asset`).

## D2. Tackles + Assists = **defensive plays only** (DraftKings convention)

FanDuel includes special-teams tackles in Tackles+Assists; DraftKings excludes them. Measured on
2025 pbp: **9.9%** of all solo tackles occurred on special teams — material, not noise.

We project the **DraftKings** number. Rationale: it maps 1:1 onto the PFR/nflverse gamebook
defense table, which is what we validate and backtest against.

**Verified stat identity** (2025, LB game-weeks, n=2067):
`def_tackles_solo + def_tackle_assists` reproduces PFR `def_tackles_combined` exactly on 1843/2067
(89.2%) of rows. The alternative `def_tackles_solo + def_tackles_with_assist` matches only 392
(19.0%). **`def_tackles_with_assist` is NOT the assist count — never use it for tackle props.**

Per `sharp_bettor_prop_reference.md`: both books settle off the **official NFL gamebook**, never
PFF or team charting. Projecting off charted counts systematically overestimates.

## D3. Current week comes from Sleeper `/v1/state/nfl`. Never hardcoded.

As of 2026-09-05 it returns `{week: 1, season: "2026", season_type: "regular",
season_start_date: "2026-09-09"}`. **Zero 2026 regular-season games have been played.**

Consequences, accepted by the user:
- Every Week 1 ranking and projection is built on 2025 (and earlier) game logs.
- §4's 0.85 prior-season discount applies to *every* player, so it is a no-op for relative
  ranking. It stays in the code because it becomes live the moment 2026 games exist.
- Team / head-coach / OC / pass-rate change flags carry the real Week 1 signal. They are computed
  from 2026 rosters + depth charts vs. 2025 game logs and surfaced prominently in the UI.

## D4. Odds: The Odds API for the current week, nflverse `schedules` everywhere else

nflverse `schedules` already carries `spread_line` and `total_line` for 2026 Week 1 (verified),
plus `roof`, `temp`, `wind`. It is free, unlimited, and is the **only** source used by the
backtest (§6) so historical replays cost zero API requests.

The Odds API is called **at most once per refresh**, never from a request handler, guarded by a
counter persisted in `api_budget` (`backend/ingest/odds.py`). Monthly cap 500, configurable.
If the key is absent or the budget is exhausted, we fall back to `schedules` and mark the source
freshness badge yellow.

## D5. Storage: Parquet is the source of truth, DuckDB views read it

Raw nflverse datasets land in `data/raw/{dataset}_{season}.parquet` and are exposed to DuckDB as
**views** over `read_parquet(...)`. Derived artifacts (defense multipliers, rankings,
projections, caches) are **tables** in `data/proplab.duckdb`.

Why: re-ingesting a season is an atomic file replace, the DB file stays small, and DuckDB pushes
projection/predicate down into the Parquet so the 372-column pbp scan stays fast.

## D6. Identity: `gsis_id` is the primary key for everything

`players` is keyed on `gsis_id`. nflverse `players.parquet` already supplies `espn_id`, `pfr_id`,
`pff_id`; `rosters_{season}.parquet` supplies `sleeper_id` and `sportradar_id`. Sleeper's dump
adds live `injury_status`, `practice_participation`, and `depth_chart_order`, and is the
authority for `sleeper_id` when the roster file lacks one.

Joins that cannot use `gsis_id`:
- `snap_counts` keys on `pfr_player_id` → join via `players.pfr_id`
- `pfr_advstats` keys on `pfr_player_id` → same
- NGS keys on `player_gsis_id` → rename to `gsis_id`

## D7. 2025+ depth charts have no `week` column

Each row carries an ISO8601 `dt` timestamp and appends to history (`dt, team, player_name,
espn_id, gsis_id, pos_grp, pos_abb, pos_slot, pos_rank`). **Never join depth charts on week.**
Take the latest snapshot at or before the target timestamp.

## D8. Rate limits are enforced in code, not by convention

| Source | Limit | Enforcement |
|---|---|---|
| Sleeper `/v1/players/nfl` | 1 call/day | Refuses to re-fetch if cache < 20h old |
| The Odds API | 500/month | Persisted counter in `api_budget`, checked before every call |
| Anthropic | 150 calls/refresh | Counter in the `llm/client.py` wrapper + disk cache |
| nflverse | none | — |
| ESPN hidden API | undocumented | Polite delay, retries, degrades silently |

Every external call is wrapped in retry + try/except. A failure logs, records a red badge in
`source_freshness`, and **never** crashes the app — stale data still renders.

## D9. Stat naming

Canonical stat keys used across DB, API, and UI. Position-scoped so `receiving_yards` means the
same thing everywhere.

| Position | Stat keys |
|---|---|
| QB | `pass_attempts` `completions` `passing_yards` `passing_tds` `interceptions` `rush_attempts` `rushing_yards` `longest_completion` `anytime_rush_td` |
| RB | `rush_attempts` `rushing_yards` `receptions` `receiving_yards` `rush_rec_yards` `longest_rush` `anytime_td` |
| WR | `targets` `receptions` `receiving_yards` `longest_reception` `anytime_td` |
| TE | `targets` `receptions` `receiving_yards` `anytime_td` |
| K | `fg_attempts` `fg_made` `xp_made` `kicking_points` `longest_fg` |
| LB | `tackles_assists` `solo_tackles` `sacks` `passes_defended` |

Settlement rules baked into the projection (from `sharp_bettor_prop_reference.md`):
- **`longest_*` settles Under if the player records no such play** → the distribution carries an
  explicit point mass at 0, it is not a truncated positive distribution.
- **A QB passing TD never counts toward `anytime_rush_td`** → rushing scores only.
- **Half-sacks count as 0.5**, and settle "Yes" on an anytime-sack (0.5) market.
- **Full-game props include OT.** We project full-game totals.
- **`kicking_points` = 3 × FGM + 1 × XPM.** Two-point conversions never credit the kicker.
- **`tackles_assists` excludes special teams** (D2).

## D10. Distributions are stored as parameters, not as samples

`projections` stores `dist_family` + a JSON `params` blob. The API returns those parameters, and
the browser recomputes `P(over line)` locally on every keystroke — no API call per keystroke (§8).
The same parameters drive the "Show math" drawer, so what you see is literally what produced the
number.
