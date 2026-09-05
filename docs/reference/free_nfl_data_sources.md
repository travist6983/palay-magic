# Free NFL Data Sources & Injury Pipeline

**Compiled:** September 5, 2026 (Week 1, 2026 season)
**Purpose:** Free/low-cost data sources to feed a prop and game-line handicapping model.
**Verify before building:** free tiers change often. Every pricing claim below should be re-checked at the source link.

---

## TL;DR — The Recommended Stack

| Layer | Source | Cost | Key needed |
|---|---|---|---|
| Historical + weekly stats, EPA, play-by-play | **nflverse** | Free | No |
| Fast injury designations + depth chart | **Sleeper API** | Free | No |
| Near-live injuries, scores, news | **ESPN hidden API** | Free | No |
| Official injury report (source of truth) | **NFL.com injury report** | Free | No (scrape) |
| Odds / devig baseline | **The Odds API** | Free tier | Yes |
| Breaking injury news | **RSS + X/Twitter beat reporters** | Free | No |

Everything except odds and PFF-grade data is genuinely free at usable volume.

---

## 1. nflverse / nflfastR — The Backbone

**The single most important free source.** This is the public analytics infrastructure that Ben Baldwin, rbsdm.com, and most independent NFL modelers run on.

- Site: https://nflreadr.nflverse.com/
- Data repo: https://github.com/nflverse/nflverse-data/releases
- Update schedule: https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html

### What it is
Not a REST API. It's versioned Parquet / CSV / RDS files published to GitHub Releases, refreshed automatically. You pull files, not endpoints. **No API key. No rate limit.**

### Access
```bash
pip install nfl_data_py     # Python
```
```r
install.packages("nflreadr")  # R
```

Or hit the raw release assets directly:
```
https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2026.parquet
```

### What you get
| Dataset | Contents |
|---|---|
| `load_pbp()` | Play-by-play with **EPA, WP, CPOE, success rate** pre-computed |
| `load_player_stats()` | Weekly player box-score stats |
| `load_snap_counts()` | Snap share (from PFR) |
| `load_depth_charts()` | Depth charts |
| `load_injuries()` | Official weekly injury reports |
| `load_nextgen_stats()` | NGS: aDOT, separation, time-to-throw, rush yards over expected |
| `load_pfr_advstats()` | PFR advanced: pressure rate, YAC, broken tackles |
| `load_ff_opportunity()` | Expected fantasy points / usage modeling |
| `load_schedules()` | Schedule + closing lines (Lee Sharpe's nfldata) |

### Update cadence (critical for "last week's game" freshness)
- **Play-by-play:** nightly after each game day, plus additional pulls during game days
- **Next Gen Stats:** nightly, roughly 3–5am ET during the season
- **Snap counts:** every 6 hours (0, 6, 12, 18 UTC)
- **PFR advanced stats:** daily, 7am UTC
- **Rosters:** daily, 7am UTC
- **Depth charts:** daily, 7am UTC year-round

**Practical implication:** Tuesday morning you have the complete previous week with EPA attached. That is fast enough for a weekly model. It is *not* fast enough for in-week injury reaction.

> **2025+ change:** depth charts are no longer assigned a week number. Each update carries an ISO8601 timestamp and appends to history. Adjust any joins that assumed a `week` column.

---

## 2. Sleeper API — Best Free Injury Designations

- Docs: https://docs.sleeper.com/
- **No authentication. No API key. No registration.** Read-only.

### The endpoint that matters
```
GET https://api.sleeper.app/v1/players/nfl
```

Returns every NFL player keyed by Sleeper player ID with:

| Field | Value |
|---|---|
| `injury_status` | `Questionable` / `Doubtful` / `Out` / `IR` / `null` when healthy |
| `injury_body_part` | e.g. `Hamstring` |
| `injury_start_date` | Date injury logged |
| `injury_notes` | Free-text detail |
| `practice_participation` | DNP / Limited / Full |
| `status` | `Active`, `Inactive`, `Injured Reserve`, `PUP` |
| `depth_chart_position` | Depth chart order |
| `depth_chart_order` | Integer rank at position |
| `team`, `position`, `age`, `years_exp` | Bio |
| `espn_id`, `gsis_id`, `sportradar_id`, `pfr_id`, `rotowire_id` | **Cross-platform join keys** |

### Why this is the sleeper pick (sorry)
1. Fantasy platforms update injury designations aggressively — roster lock depends on it.
2. It carries **practice participation**, which is the actual predictive signal. A "Questionable" who was Full Friday plays ~95% of the time. A "Questionable" who was DNP Friday is a coin flip.
3. `gsis_id` joins directly to nflverse. `espn_id` joins to ESPN. This is your ID crosswalk for free.

### Constraints
- The full dump is roughly 5 MB. Sleeper explicitly asks you **not to call it more than once per day**.
- Cache it locally. Diff against yesterday's snapshot to detect status changes.

### Also useful
```
GET https://api.sleeper.app/v1/state/nfl          # current week, season, season_type
GET https://api.sleeper.app/v1/players/nfl/trending/add?lookback_hours=24&limit=25
```
The trending endpoint is a crude but real proxy for breaking news — a spike in adds for a backup RB usually means the starter just got hurt.

---

## 3. ESPN Hidden API — Near-Live Injuries and Scores

Undocumented, unsupported, free, no key. Widely used. Can break without notice — wrap in try/except and monitor.

### Injury endpoints
```
# Per team (team ID 1–34)
https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{TEAM_ID}/injuries

# Team list to resolve IDs
https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams
```

The core API returns `$ref` links rather than inline objects — you follow references to hydrate each injury record. Slightly annoying, but each record includes status, type, detail, side, and return-date estimate where available.

### Other useful endpoints
```
# Scoreboard (live scores, odds, weather)
https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates=20260907

# News
https://site.api.espn.com/apis/site/v2/sports/football/nfl/news

# Game summary — box score, drives, win probability
https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={EVENT_ID}

# Play-by-play
https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{ID}/competitions/{ID}/plays?limit=300

# Team roster / depth chart / stats
https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{ID}?enable=roster,projection,stats
```

Reference endpoint lists:
- https://gist.github.com/nntrn/ee26cb2a0716de0947a0a4e9a157bc1c
- https://gist.github.com/jeffmaxey/2a4dd2b43f1150c8a16804fb7c63efa8

---

## 4. Odds Data — Where Free Runs Out

### The Odds API — https://the-odds-api.com/
- Permanent free tier: **500 requests/month**, no credit card
- Covers 40+ books, moneyline / spreads / totals
- **Player props are a paid feature.** Free tier gets you core markets only.

> ⚠️ **Name collision warning:** `the-odds-api.com` and `theoddsapi.com` are different products with different pricing and different free tiers. Confirm which one you signed up for. The 500/month free tier belongs to **the-odds-api.com**.

### Other options
| Provider | Free tier | Props? |
|---|---|---|
| Sports Game Odds — sportsgameodds.com | "Amateur" tier, slower refresh | Paid tiers for fast props |
| OddsPapi — oddspapi.io | Free tier includes historical odds | Props on free tier (verify) |
| Highlightly — highlightly.net | 100 req/day free | Odds **excluded** from free plan |

**Bottom line:** you can devig sides and totals for free. NFL player props will cost you roughly $30–100/month from any credible provider. Budget for it if props are the target market.

---

## 5. Convenience Wrappers (Optional)

### Big Balls Sports Data — https://bigballsdata.com/nfl-api
A hosted REST wrapper over nflverse data. Free tier: **100 req/min, 1,000 req/day**.
- Endpoints for PBP with EPA/WP, weekly player stats, game logs, rosters 2020–2026, standings, **weekly injuries**
- Written post-game, not a live feed. Last ingest timestamps published at `GET /v1/coverage`
- **Use case:** worth it if you want HTTP endpoints instead of managing Parquet files. Otherwise go direct to nflverse and skip the dependency.

### API-Sports (American Football) — https://api-sports.io/documentation/nfl/v1
- Free tier: teams, players, games only, 5 req/min
- Injuries, standings, and season stats require the $9.99/mo tier
- Play-by-play, advanced stats, props require $39.99/mo
- **Verdict:** the free tier is too thin to be useful here

### MySportsFeeds — https://www.mysportsfeeds.com/
- Free or near-free for **strictly non-commercial personal use** — you must apply
- Includes injuries, play-by-play, DFS, odds
- **Verdict:** worth applying if this stays a personal project

---

## 6. The Injury Problem — Full Solution

Injury information exists at four distinct latencies. You need all four, and you need to understand that only the first three are free.

### Tier 1 — Official Weekly Injury Report (source of truth, scheduled)
The NFL mandates practice reports Wed/Thu/Fri and a final game status designation Friday (or day-before for non-Sunday games).

- **NFL.com:** https://www.nfl.com/injuries/ — official, week-by-week, per game
- **nflverse `load_injuries()`** — same data, already parsed, historical back to 2009
- **Sleeper** — carries designation + practice participation together

**Model this, don't just read it.** Build the historical base rates from nflverse:

| Designation + Friday practice | Approx. play rate |
|---|---|
| Questionable + Full | Very high |
| Questionable + Limited | Moderate |
| Questionable + DNP | Roughly coin flip |
| Doubtful (any) | Low |
| Out | Zero |

Compute these yourself from `load_injuries()` joined to `load_snap_counts()` — did the player actually take snaps the following Sunday? That join gives you a calibrated probability of playing, plus expected snap share if active. **This is the single highest-value thing you can build with free data**, because most public bettors read the designation as a label rather than a probability.

### Tier 2 — Daily Status Polling (hours of latency)
Cron job, twice daily:
```
06:00 ET  →  Sleeper /v1/players/nfl   (cache + diff vs yesterday)
18:00 ET  →  ESPN /teams/{id}/injuries (all 34 team IDs)
```
Diff the snapshots. Any change in `injury_status`, `practice_participation`, or `depth_chart_order` is an alert. Depth chart movement often precedes the official designation.

### Tier 3 — Breaking News (minutes of latency, free)
No free API delivers beat-reporter news. Build an RSS/feed aggregator instead:

- **Team beat blogs** — Feedspot maintains a categorized directory: https://rss.feedspot.com/nfl_rss_feeds/
- **ESPN news endpoint** — `site.api.espn.com/apis/site/v2/sports/football/nfl/news`
- **RotoWire injury page** — https://www.rotowire.com/football/injury-report.php (scrape; feed access is a paid partnership)
- **RotoBaller feeds** — https://www.rotoballer.com/fantasy-sports-player-news-feeds-and-apis/335167 — XML/RSS/JSON player news, paid but genuinely cheap, 50–150 items/day
- **X/Twitter lists** — Schefter, Rapoport, Pelissero, Garafolo plus your 32 team beat writers. API access is now expensive; a scraper or a third-party bridge is the practical route.
- **Sleeper trending adds** — free, crude, surprisingly fast leading indicator

### Tier 4 — What You Cannot Get Free
Sportradar's NFL API is the paid gold standard. It now exposes an `estimated_return_date` attribute on weekly injuries alongside practice status and body part. Books use feeds like this. If you ever need injury data at professional latency, this is the tier, and it is not cheap.

### The uncomfortable truth
By the time an injury reaches a free feed, the line has already moved. Sportsbooks buy low-latency official league data specifically so this never happens to them.

**So don't compete on speed. Compete on interpretation.** Your edge from free data is a calibrated model of *what a designation means* — probability of playing, expected snap share if active, and the downstream redistribution of targets and carries to teammates. That work is done Tuesday through Friday and doesn't require beating anyone to a tweet.

---

## 7. What Stays Paid

| Source | Why you might want it | Cost |
|---|---|---|
| **PFF** | Player grades, YPRR, coverage grades, pressure rate | Paid |
| **FTN / DVOA** | Aaron Schatz's DVOA, DYAR, Adjusted Line Yards | Paid |
| **SumerSports / TruMedia** | Tracking-derived route and coverage data | Paid |
| **Establish The Run** | Props/DFS projections, PROE content | Paid |
| **Sportradar / Genius** | Low-latency official league feeds | Enterprise |
| **Prop odds (any provider)** | Line shopping, devig, CLV tracking | ~$30–100/mo |

Free substitutes that get you most of the way: **Next Gen Stats** (free, in nflverse) covers separation and aDOT. **PFR advanced stats** (free, in nflverse) covers pressure rate and broken tackles. **rbsdm.com** gives you EPA/success-rate dashboards without writing code.

---

## 8. Suggested Build Order

1. **Ingest nflverse** — pbp, weekly stats, snap counts, injuries, NGS. Store locally in Parquet or DuckDB. One backfill, then nightly deltas.
2. **Build the ID crosswalk** — pull Sleeper's player dump once, keep the `gsis_id` / `espn_id` / `pfr_id` / `rotowire_id` mapping. Everything downstream depends on this.
3. **Build the injury probability model** — historical designation + practice status → actual snaps played. Calibrate. This is the differentiated piece.
4. **Set up daily polling** — Sleeper + ESPN, snapshot and diff, alert on change.
5. **Add RSS aggregation** — team beats + ESPN news, keyword-filtered.
6. **Add odds last** — free tier for sides/totals to validate the model against closing lines before spending on props.

Track **closing line value** from step 1. If your projections don't beat the closing line on paper over a meaningful sample, no amount of data plumbing fixes that.

---

## Quick Reference — Links

| Source | URL |
|---|---|
| nflverse data schedule | https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html |
| nflverse releases | https://github.com/nflverse/nflverse-data/releases |
| nfl_data_py | https://github.com/nflverse/nfl_data_py |
| Sleeper docs | https://docs.sleeper.com/ |
| ESPN endpoint gist | https://gist.github.com/nntrn/ee26cb2a0716de0947a0a4e9a157bc1c |
| NFL official injuries | https://www.nfl.com/injuries/ |
| The Odds API | https://the-odds-api.com/ |
| Big Balls Sports Data | https://bigballsdata.com/nfl-api |
| MySportsFeeds | https://www.mysportsfeeds.com/ |
| rbsdm.com (free EPA dashboards) | https://rbsdm.com/stats/stats/ |
| NFL RSS directory | https://rss.feedspot.com/nfl_rss_feeds/ |

---

*If any of this is being used for wagering: bet only what you can afford to lose, and treat the model as a discipline tool rather than a guarantee. US problem gambling helpline: 1-800-GAMBLER.*
