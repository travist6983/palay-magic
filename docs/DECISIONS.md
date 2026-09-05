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

## D11. Injury play rates are conditioned on role, not just designation

Computed the obvious way — `P(played)` per (designation, practice status) — the 2023–2025 data
says a **Questionable + Full** player plays only **70.6%** of the time. That contradicts
`free_nfl_data_sources.md`, which calls that cell "very high" (~95%).

It is not a join bug. It is population: the weekly injury report is dominated by fringe players
who often do not dress at all, and pooling them with starters drags every cell down. Splitting by
the player's mean snap share over the **previous four weeks**:

| Designation | Final practice | Starter (≥55% snaps) | n |
|---|---|---|---|
| Questionable | Full | **84.8%** | 302 |
| Questionable | Limited | **73.7%** | 1,299 |
| Questionable | DNP | **53.9%** | 401 |
| Doubtful | DNP | 1.4% | 146 |
| Out | any | 0.0% | 1,400 |

Which reproduces the doc's qualitative table ("very high / moderate / roughly coin flip / low /
zero") — the DNP cell in particular lands within a point of "coin flip".

Applying the pooled 70.6% to a Questionable starting WR would have under-projected him by ~15
points of play probability, so this dimension is load-bearing. `role_bucket` is part of the
`injury_play_rates` primary key (migration 002).

**The trap that produced a wrong answer first:** deriving the role window from the same
snap-counts join that decides `played` makes every player *with* a role bucket one who played by
construction — every cell reads exactly 1.000. The role window must be computed off the
injury-report spine with a left join to prior weeks, never off the current week's snap row.

Second output of the same table: **E[snap share | played] ≈ 0.76 for a Questionable starter.** A
banged-up starter who suits up is not a full-snap player, and usage must be scaled accordingly
(§5.4).

## D12. Sleeper's `gsis_id` covers only a third of its dump — recover the rest by name

Sleeper supplies `gsis_id` for 3,893 of 12,226 players. CeeDee Lamb, Bucky Irving and Brandon
Aubrey all lack one. Taken at face value, ~20% of 2026-rostered players at prop positions get no
injury designation or practice status at all, which guts §5.7.

`backend/ingest/crosswalk.py::_resolve_sleeper_by_name` recovers them with two fallback keys,
applied only when the match is **unique on both sides**: `espn_id`, then
`normalised name | team | position` against the current roster. Ambiguous keys are dropped rather
than guessed — a wrong crosswalk row silently attaches one player's injury to another.

Coverage among 2026-rostered players at the six prop positions, before → after:

| Pos | Before | After |
|---|---|---|
| QB | 95.0% | 96.6% |
| RB | 89.3% | 92.2% |
| WR | 82.8% | 89.4% |
| TE | 78.0% | 88.3% |
| K | 82.5% | 95.0% |
| LB | 78.8% | 86.0% |
