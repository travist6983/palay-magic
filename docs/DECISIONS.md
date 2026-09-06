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

## D13. Timestamps cross the API as naive UTC, and the client must stamp them

The API serialises `datetime` values from the backend's `utcnow()` with no zone marker —
`"2026-09-05 16:30:40.543348"`. JavaScript reads a zone-less string as **local** time, so every
age in the header shifted by the viewer's UTC offset: a four-hour-old refresh rendered as
"23m ago" in Detroit, beside the backend's own "age: 4.1h" in the same tooltip.

East of UTC it is worse than an error of degree. The offset makes the computed age negative,
`relativeTime` returns "just now", and the §8 freshness badges become structurally incapable of
ever reporting staleness — the one thing they exist to do.

`frontend/src/lib/format.ts::utcIso` normalises at the single point every consumer goes through,
and is a no-op on a value that already carries a zone. `format.test.ts` pins it.

## D14. A board's matchup multiplier is not always the projected stat

`queries.board` picks the opponent multiplier from `get_spec(position, headline_stats[0])`, and on
two of the six boards that metric is not the stat in the headline column:

| Board | Headline stat | Matchup metric |
|---|---|---|
| K | Kicking points | **FG attempts allowed / game** |
| LB | Tackles + assists | **Offensive plays run / game** |

The UI had been naming the metric from the projection label, which made the kicker and linebacker
tooltips state something false. The API now ships `opponent_metric` and `opponent_metric_label`
so the UI names what the number actually measures instead of inferring it.

## D15. Non-designations are normalised once, at the ranking layer

Sleeper emits `"NA"` and `"-"` where a healthy player simply has no designation (D11 covers the
play-probability side). The raw string was still being written to `rankings.injury_status`, so the
board rendered an injury chip and an empirical-play-rate tooltip on 19 players who are fine —
several of them rank 1 or 2.

`build_rankings` now normalises through `injury.NON_DESIGNATIONS` before storing. Doing it at the
source means no consumer has to keep its own copy of the list in step.

---

# Review pass (2026-09-06): what the equations got wrong

A full review of the modelling code — five module reviewers executing the maths against the
database, plus an independent check of the backtest tables — found 19 defects in the equations
and predictions. All are fixed; the ones that changed a number materially are recorded here.

## D16. The anytime-TD share is fitted, not a raw usage fraction

§5.6 says `λ = expected_team_TDs × player_TD_share` from red-zone target share and goal-line
carry share. That structure was kept; the hand-set weights (`0.75·gl + 0.25·rz` against all team
touchdowns, `0.62` for receivers) were not a share and were badly off on a 2025 replay:

| Position | Predicted P(TD) | Realised | Bias |
|---|---|---|---|
| QB rushing | 28.7% | 14.1% | +14.6pp |
| RB | 69.9% | 58.5% | +11.5pp |
| TE | 27.2% | 40.5% | −13.3pp |

A constant at the positional base rate beat every raw-share λ on log-loss. The share is now a
fitted linear function of the usage signals — separately for rushing and receiving scores, each
against that side's expected team touchdowns — with the fit weighted by trailing usage so the
players the app projects dominate it (`backend/models/touchdowns.py`). Held-out 2025, top-20%
usage: RB +0.8pp, WR +0.4pp, TE −7.9pp, QB +7.7pp. TE and QB remain the weakest cells.

## D17. Team targets, not team attempts

`target_share` is defined against team *targets* (`gamelog.py`) but was multiplied by expected
pass *attempts*. Official targets are 95.4% of attempts (throwaways, spikes, batted balls), so
every target, reception and receiving-yard projection ran ~4.9% high. `expected_targets =
attempts × measured targets-per-attempt`, stored with the other measured constants.

## D18. Ridge calibration scores the rated side alone

`tune_ridge` and `_walk_forward_scores` scored the full two-way prediction — offence effect plus
defence effect — but the projection only ever applies the defence effect. The credited MSE
reduction was mostly the *offence's* predictability: `rec_volume_allowed_te` read 9.8% of which
9.3% was the faced side and 0.7% the defence.

Scored honestly, the best defensive signal removes 5.9% of MSE (`opp_pass_rate`) and three
metrics are slightly negative out of sample — yards-per-target allowed to WRs and RBs, and FG
attempts allowed. Those are pinned to a multiplier of 1.0 (`model = 'pinned:no_signal'`). The
matchup signals the public treats as most decisive are the ones with no signal.

## D19. The unconditional projection is a mixture, and it was not being built as one

Three related errors in `_add_unconditional`:

- A Poisson refit at `λ·p` had the right mean and the wrong shape: no absence mass at zero, half
  the variance. McCaffrey receptions P(over 5): 0.14 stored vs 0.26 correct. Now the play /
  don't-play mixture moments, carried by a negative binomial.
- Anytime-TD used `1 − e^(−λp)`; because `1 − e^(−x)` is concave that is always high, by 9 points
  at McCaffrey's λ. It is `p · (1 − e^(−λ))`.
- The usage cut for an active-but-limited player divided a *population* mean snap share (0.762)
  by the player's own share, cutting a 95%-snap starter by a fifth. The data says a Questionable
  starter who plays keeps 93% of his usual snaps (median 100%); the measured ratio is now stored
  per cell (`mean_snap_ratio`, migration 013) and used directly.

## D20. Injury table: three fit/apply mismatches

- Sleeper emits `Full` / `Limited` / `DNP`; the table stored nflverse's long strings. Live practice
  status could never find its cell. Both sides now use the three tokens.
- The role bucket was fitted on same-season weeks w−4..w−1 and applied over the last four games
  of any season — so every Week-1 starter was fitted as `no_recent_games` (53%) and applied as
  `starter` (71%) when his true rate was 63–68%. The fit now uses the cross-season window it is
  applied with.
- A `(status, role, position)` level was unreachable. Questionable QB starters play 44.6% of the
  time; Mahomes was shown 71%. He now reads 39% via the QB cell.

## D21. Replays use the official report for their week

`injury_status` holds only the live snapshot with no date on it, and the backtest was stamping
September-2026 designations onto 2024 and 2025 weeks. `_injury_state(season, week)` now reads
`raw_injuries` for a historical week. Relatedly, `backtest --refit` overwrote the production
ridge penalties and environment coefficients with 2023–24-only fits and never restored them; the
live 2026 Week 1 had been built on those. Every replay now ends by refitting on the full history.

## D22. Smaller corrections, all measured rather than assumed

| Was | Is | Why |
|---|---|---|
| serve-time pace = all-time mean of every play | rolling mean of last-6 per-game medians, one helper for fit and serve | fit/serve statistics differed by ~5s; expected plays +2.3/game for every team |
| `XP_PER_TD = 0.94` | 0.909 measured | 0.94 was *attempts*; omitted the make rate |
| `expected_rush_attempts = plays − dropbacks` | fitted team rush-attempts model | dropped scrambles that `carry_share`'s denominator includes; RB carries −7% |
| red-zone trip = any row inside the 20 | scrimmage snap or FG try inside the 20 | the XP row from the 15 made every long TD drive a red-zone conversion (2168 flagged vs 1794 real) |
| `passes_defended → pass_volume_allowed` (a defence metric) | `opp_dropbacks` (opponent offence) | wrong unit |
| integer Poisson for sacks | Poisson on half-sacks (`unit = 0.5`) | 17.5% of sack credits are exactly 0.5 |
| `_pmf = 0` for kicking points, `integer_valued = False` | enumerated support, integer | P(under) at 9 was wrong by the 7% point mass |
| kick-return TDs credited to the offence | excluded | nflverse sets posteam to the receiving team on kickoffs |
| kneels and scrambles in the explosive-rush denominator | excluded | kneel share of carries faced ranges 0.75–7.1% across defences |
| `redistribute_target_share` defined, never called | called; the position's slice split among healthy peers | an Out WR1 moved nobody; the first wiring handed the whole slice to *each* WR and put targets +2.6 |

## D23. A held-out mean-bias correction was tried and rejected

After the review fixes, the remaining errors were level errors of a few percent. The obvious
remedy is a multiplicative correction per (position, stat) fitted on one season's replay, the
same mechanism the interval widths use. It was built (`proplab calibrate-bias`, migration 014)
and it fails validation:

| Stat | 2025 bias before | with 2024-fitted ratio |
|---|---|---|
| targets | +1.05 | +0.53 |
| rushing yards | −0.05 | **+4.61** |
| passing yards | +2.74 | **+4.85** |
| coverage | 58.6% | 60.7% |

The ratios that hurt (QB rushing yards ×1.07, passing yards ×1.05) were fitting 2024's noise, not
a structural bias. The command is kept as a diagnostic; the table is empty in production and
`make calibrate` does not run it. The honest position is that a few percent of level error is
the model's floor with this data, and the intervals already price it.
