"""Top-10 boards: who is worth projecting at each position (§4).

Ranked by a **usage-first** score over the last six games played, not by name recognition. Usage
is the part of a projection that is actually predictable — the reference doc's whole argument for
where the edge lives is opportunity, not efficiency — so the board is ordered by projected
opportunity in this week's game environment.

Two details that decide whether the ranking is honest:

**Games are weighted by snaps, not equally.** A star who takes five snaps in a Week 18 rest game
produced a real 1-target line, and an unweighted mean treats that as evidence he is a
one-target player. Weighting each game's share observation by the snaps behind it makes a rest
game or a first-quarter injury exit contribute almost nothing, without any hand-written rule about
which games to throw away.

**The season boundary is flagged, not hidden.** Right now every game in every window is from a
prior season, so §4's 0.85 discount multiplies all weights equally and cancels out of the mean
(D3). What actually carries Week 1 signal is whether the player changed teams, whether his offence
changed coordinator, and whether its projected pass rate moved — all three are computed and
surfaced rather than folded silently into a score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from backend.config import get_settings
from backend.db.connection import connect
from backend.logging_setup import get_logger
from backend.models.injury import (
    EXCLUDED_STATUSES,
    NON_DESIGNATIONS,
)
from backend.models.injury import (
    lookup as injury_lookup,
)
from backend.models.stats import Position

log = get_logger(__name__)

TOP_N = 10

# Minimum share of defensive snaps for an LB to be eligible (§4). Tackle props are only bettable
# on genuine every-down defenders; a rotational LB's line is lower and far noisier.
EVERY_DOWN_THRESHOLD = 0.80


@dataclass
class RankedPlayer:
    """One row of a position board."""

    gsis_id: str
    display_name: str
    position: str
    team: str | None
    opponent: str | None
    score: float
    tiebreaker: float
    n_games_used: int
    insufficient_history: bool
    changed_team: bool
    changed_coach: bool
    pass_rate_shift: float
    injury_status: str | None
    play_probability: float
    components: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _recent_usage(season: int, week: int, window: int) -> pl.DataFrame:
    """Each player's last ``window`` games of usage, most recent first, with a games_ago index."""
    with connect() as con:
        return con.execute(
            """
            SELECT * FROM (
                SELECT u.*,
                       row_number() OVER (
                           PARTITION BY u.gsis_id ORDER BY u.season DESC, u.week DESC
                       ) - 1 AS games_ago
                FROM player_game_usage u
                WHERE (u.season < ?) OR (u.season = ? AND u.week < ?)
            ) WHERE games_ago < ?
            """,
            [season, season, week, window],
        ).pl()


def _recent_stats(season: int, week: int, window: int, stats: tuple[str, ...]) -> pl.DataFrame:
    """Each player's last ``window`` games for the given stat keys."""
    placeholders = ",".join("?" * len(stats))
    with connect() as con:
        return con.execute(
            f"""
            SELECT * FROM (
                SELECT g.gsis_id, g.season, g.week, g.stat, g.value, g.position, g.team, g.opponent,
                       row_number() OVER (
                           PARTITION BY g.gsis_id, g.stat ORDER BY g.season DESC, g.week DESC
                       ) - 1 AS games_ago
                FROM player_game_stats g
                WHERE ((g.season < ?) OR (g.season = ? AND g.week < ?))
                  AND g.stat IN ({placeholders})
            ) WHERE games_ago < ?
            """,
            [season, season, week, *stats, window],
        ).pl()


def _weighted(
    df: pl.DataFrame,
    value_col: str,
    weight_col: str | None = None,
    group: str = "gsis_id",
) -> pl.DataFrame:
    """Recency-weighted mean of ``value_col``, optionally precision-weighted by ``weight_col``.

    The recency weight is ``decay ** games_ago`` with the prior-season discount applied per game
    (§4, §5.1). ``weight_col`` multiplies it, which is how a five-snap rest game stops counting as
    a full observation.
    """
    settings = get_settings()
    decay, discount = settings.recency_decay, settings.prior_season_discount
    target_season = df["season"].max()

    work = df.with_columns(
        (pl.lit(decay) ** pl.col("games_ago")).alias("_w_recency"),
        (pl.col("season") < target_season).alias("_prior"),
    ).with_columns(
        pl.when(pl.col("_prior"))
        .then(pl.col("_w_recency") * discount)
        .otherwise(pl.col("_w_recency"))
        .alias("_w")
    )

    if weight_col:
        work = work.with_columns(
            (pl.col("_w") * pl.col(weight_col).fill_null(0.0).clip(0.0, None)).alias("_w")
        )

    return (
        work.filter(pl.col(value_col).is_not_null() & (pl.col("_w") > 0))
        .group_by(group)
        .agg(
            ((pl.col(value_col) * pl.col("_w")).sum() / pl.col("_w").sum()).alias(value_col),
            pl.col("_w").sum().alias("weight_total"),
            pl.len().alias("n_games"),
            pl.col("_prior").sum().alias("n_prior_season"),
        )
    )


def _context(season: int, week: int) -> tuple[dict[str, dict], float]:
    """This week's team environment, keyed by team, plus the league-average implied total."""
    with connect() as con:
        env = con.execute(
            "SELECT * FROM team_environment WHERE season = ? AND week = ?", [season, week]
        ).pl()
    if env.is_empty():
        return {}, 22.0
    league = float(env["implied_total"].mean())
    return {r["team"]: r for r in env.to_dicts()}, league


def _current_teams(season: int) -> dict[str, str]:
    """gsis_id -> team on the current roster. Determines who is even playing this week."""
    with connect() as con:
        rows = con.execute(
            "SELECT gsis_id, team FROM raw_rosters WHERE season = ? AND gsis_id IS NOT NULL",
            [season],
        ).fetchall()
    return {g: t for g, t in rows if t}


def _depth_chart(position: Position) -> tuple[dict[str, int], set[str]]:
    """Latest depth-chart rank per player, and the set of players listed at this position.

    2025+ depth charts carry an ISO8601 ``dt`` and append to history rather than being keyed on a
    week (D7), so "current" means the newest snapshot, not this week's row.

    The chart uses scheme-specific abbreviations (``WLB``/``SLB``/``MLB`` rather than ``LB``,
    ``RCB``/``LCB`` rather than ``CB``), so matching is on the position *group* with the
    abbreviation as a fallback.
    """
    group = {
        Position.QB: "QB",
        Position.RB: "RB",
        Position.WR: "WR",
        Position.TE: "TE",
        Position.K: "K",
        Position.LB: "LB",
    }[position]

    with connect() as con:
        rows = con.execute(
            """
            WITH latest AS (SELECT max(dt) AS dt FROM raw_depth_charts)
            SELECT d.gsis_id, d.pos_rank, d.pos_abb, d.pos_grp
            FROM raw_depth_charts d, latest
            WHERE d.dt = latest.dt AND d.gsis_id IS NOT NULL
              AND (upper(coalesce(d.pos_grp, '')) = ? OR upper(coalesce(d.pos_abb, '')) LIKE ?)
            """,
            [group, f"%{group}%"],
        ).fetchall()

    ranks: dict[str, int] = {}
    listed: set[str] = set()
    for gsis, rank, _abb, _grp in rows:
        listed.add(gsis)
        if rank is not None:
            ranks[gsis] = min(rank, ranks.get(gsis, rank))
    return ranks, listed


# Positions where only the starter has a bettable prop. A backup QB or kicker who started a few
# games in an injury stretch still carries a full starter's volume baseline, which is how Easton
# Stick outranked Patrick Mahomes before this gate existed.
STARTER_ONLY: frozenset[Position] = frozenset({Position.QB, Position.K})


def _injury_state() -> dict[str, dict]:
    """gsis_id -> the freshest live injury row across sources (Sleeper preferred over ESPN)."""
    with connect() as con:
        rows = con.execute(
            """
            SELECT gsis_id, injury_status, practice_participation, roster_status, source
            FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY gsis_id
                    ORDER BY CASE WHEN source = 'sleeper' THEN 0 ELSE 1 END, observed_at DESC
                ) AS rn
                FROM injury_status
            ) WHERE rn = 1
            """
        ).fetchall()
    return {
        r[0]: {
            "injury_status": r[1],
            "practice_participation": r[2],
            "roster_status": r[3],
            "source": r[4],
        }
        for r in rows
        if r[0]
    }


def _continuity(season: int, week: int) -> dict[str, dict]:
    """Per-team head coach and projected pass-rate change vs. the prior season (§4)."""
    with connect() as con:
        coaches = con.execute(
            """
            SELECT season, team, any_value(coach) AS coach FROM (
                SELECT season, home_team AS team, home_coach AS coach FROM raw_schedules
                WHERE game_type = 'REG'
                UNION ALL
                SELECT season, away_team, away_coach FROM raw_schedules WHERE game_type = 'REG'
            ) WHERE coach IS NOT NULL GROUP BY 1, 2
            """
        ).pl()
        pass_rates = con.execute(
            """
            SELECT season, posteam AS team,
                   sum(qb_dropback)::DOUBLE
                     / nullif(sum(CASE WHEN qb_dropback = 1 OR rush_attempt = 1 THEN 1 ELSE 0 END), 0)
                     AS pass_rate
            FROM raw_pbp WHERE season_type = 'REG' AND posteam IS NOT NULL
            GROUP BY 1, 2
            """
        ).pl()

    out: dict[str, dict] = {}
    prior = season - 1
    coach_now = {r["team"]: r["coach"] for r in coaches.filter(pl.col("season") == season).to_dicts()}
    coach_prev = {r["team"]: r["coach"] for r in coaches.filter(pl.col("season") == prior).to_dicts()}
    pr_now = {r["team"]: r["pass_rate"] for r in pass_rates.filter(pl.col("season") == season).to_dicts()}
    pr_prev = {r["team"]: r["pass_rate"] for r in pass_rates.filter(pl.col("season") == prior).to_dicts()}

    for team in set(coach_prev) | set(coach_now) | set(pr_prev):
        now, prev = coach_now.get(team), coach_prev.get(team)
        out[team] = {
            "changed_coach": bool(now and prev and now != prev),
            "coach": now or prev,
            "prior_coach": prev,
            "pass_rate_shift": (pr_now.get(team, pr_prev.get(team, 0.0)) or 0.0)
            - (pr_prev.get(team, 0.0) or 0.0),
        }
    return out


# ---------------------------------------------------------------------------
# Position scores (§4)
# ---------------------------------------------------------------------------


def _score_frame(season: int, week: int, position: Position, window: int) -> pl.DataFrame:
    """Per-player usage score and tiebreaker for one position, before context is applied."""
    usage = _recent_usage(season, week, window)
    if usage.is_empty():
        return pl.DataFrame()

    usage = usage.filter(pl.col("position") == position.value)
    if usage.is_empty():
        return pl.DataFrame()

    # player_game_stats is league-wide, so every stat frame below must be restricted to the
    # players who actually play this position. Without this the TE board fills up with wide
    # receivers, who out-target every tight end in the league.
    eligible = set(usage["gsis_id"].to_list())

    def _restrict(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.filter(pl.col("gsis_id").is_in(list(eligible)))

    # Snap weight: how much evidence each game carries. Falls back to 1 when snap data is missing
    # so a player is never dropped for want of a PFR row.
    off = pl.col("offense_snaps").fill_null(0).cast(pl.Float64)
    dfn = pl.col("defense_snaps").fill_null(0).cast(pl.Float64)
    usage = usage.with_columns(
        pl.when((off + dfn) > 0).then(off + dfn).otherwise(1.0).alias("snap_weight")
    )

    match position:
        case Position.QB:
            stats = _restrict(_recent_stats(season, week, window, ("pass_attempts", "rush_attempts")))
            attempts = stats.filter(pl.col("stat") == "pass_attempts")
            rushes = stats.filter(pl.col("stat") == "rush_attempts")
            base = _weighted(attempts, "value").rename({"value": "pass_attempts"})
            rush = _weighted(rushes, "value").rename({"value": "rush_attempts"}).select(
                "gsis_id", "rush_attempts"
            )
            frame = base.join(rush, on="gsis_id", how="left").with_columns(
                # Dropbacks: attempts plus sacks and scrambles, which the box score folds elsewhere.
                (pl.col("pass_attempts") * 1.13).alias("score_base"),
                pl.col("rush_attempts").fill_null(0.0).alias("tiebreaker"),
            )
            frame = frame.join(
                usage.select("gsis_id", "team", "position").unique(subset=["gsis_id"]),
                on="gsis_id",
                how="left",
            )

        case Position.RB:
            stats = _restrict(_recent_stats(season, week, window, ("rush_attempts", "targets")))
            carries = _weighted(stats.filter(pl.col("stat") == "rush_attempts"), "value").rename(
                {"value": "carries"}
            )
            tgts = _weighted(stats.filter(pl.col("stat") == "targets"), "value").rename(
                {"value": "targets"}
            ).select("gsis_id", "targets")
            gl = _weighted(usage, "gl_carry_share", "snap_weight").select(
                "gsis_id", "gl_carry_share"
            )
            frame = (
                carries.join(tgts, on="gsis_id", how="left")
                .join(gl, on="gsis_id", how="left")
                .with_columns(
                    (pl.col("carries") + pl.col("targets").fill_null(0.0)).alias("score_base"),
                    pl.col("gl_carry_share").fill_null(0.0).alias("tiebreaker"),
                )
                .join(
                    usage.select("gsis_id", "team", "position").unique(subset=["gsis_id"]),
                    on="gsis_id",
                    how="left",
                )
            )

        case Position.WR | Position.TE:
            stats = _restrict(_recent_stats(season, week, window, ("targets",)))
            tgts = _weighted(stats, "value").rename({"value": "targets"})
            part = _weighted(usage, "offense_pct", "snap_weight").select("gsis_id", "offense_pct")
            rz = _weighted(usage, "rz_target_share", "snap_weight").select(
                "gsis_id", "rz_target_share"
            )
            frame = (
                tgts.join(part, on="gsis_id", how="left")
                .join(rz, on="gsis_id", how="left")
                .with_columns(
                    # Route participation, proxied by offensive snap share: a receiver who is on
                    # the field for 85% of snaps has a far more repeatable target line than one
                    # posting the same targets on 40%.
                    (pl.col("targets") * pl.col("offense_pct").fill_null(0.5).clip(0.1, 1.0))
                    .alias("score_base"),
                    pl.col("rz_target_share").fill_null(0.0).alias("tiebreaker"),
                )
                .join(
                    usage.select("gsis_id", "team", "position").unique(subset=["gsis_id"]),
                    on="gsis_id",
                    how="left",
                )
            )

        case Position.K:
            stats = _restrict(_recent_stats(season, week, window, ("fg_attempts", "xp_made")))
            fga = _weighted(stats.filter(pl.col("stat") == "fg_attempts"), "value").rename(
                {"value": "fg_attempts"}
            )
            xp = _weighted(stats.filter(pl.col("stat") == "xp_made"), "value").rename(
                {"value": "xp_made"}
            ).select("gsis_id", "xp_made")
            frame = (
                fga.join(xp, on="gsis_id", how="left")
                .with_columns(
                    pl.col("fg_attempts").alias("score_base"),
                    pl.col("xp_made").fill_null(0.0).alias("tiebreaker"),
                )
                .join(
                    usage.select("gsis_id", "team", "position").unique(subset=["gsis_id"]),
                    on="gsis_id",
                    how="left",
                )
            )

        case Position.LB:
            snap = _weighted(usage, "defense_pct", "snap_weight").select("gsis_id", "defense_pct")
            tackle = _weighted(usage, "tackle_share", "snap_weight").select(
                "gsis_id", "tackle_share"
            )
            frame = (
                snap.join(tackle, on="gsis_id", how="left")
                .join(
                    _weighted(usage, "defense_pct", "snap_weight").select(
                        "gsis_id", "n_games", "n_prior_season"
                    ),
                    on="gsis_id",
                    how="left",
                )
                .with_columns(
                    (pl.col("defense_pct").fill_null(0.0) * pl.col("tackle_share").fill_null(0.0))
                    .alias("score_base"),
                    pl.col("defense_pct").fill_null(0.0).alias("tiebreaker"),
                )
                .join(
                    usage.select("gsis_id", "team", "position").unique(subset=["gsis_id"]),
                    on="gsis_id",
                    how="left",
                )
            )
            # §4: only every-down defenders are eligible for tackle props.
            frame = frame.filter(pl.col("defense_pct") >= EVERY_DOWN_THRESHOLD)

        case _:  # pragma: no cover - Position is exhaustive
            return pl.DataFrame()

    if "n_games" not in frame.columns:
        frame = frame.with_columns(pl.lit(0).alias("n_games"), pl.lit(0).alias("n_prior_season"))
    return frame


def build_rankings(season: int, week: int, window: int | None = None) -> int:
    """Build and persist the top-10 board for every position (§4).

    Returns:
        Number of ranked rows written across all positions.
    """
    settings = get_settings()
    window = settings.recency_window if window is None else window

    env, league_implied = _context(season, week)
    rosters = _current_teams(season)
    injuries = _injury_state()
    continuity = _continuity(season, week)

    with connect() as con:
        names = {
            g: (n, p, pg)
            for g, n, p, pg in con.execute(
                "SELECT gsis_id, display_name, position, position_group FROM players"
            ).fetchall()
        }
        prior_team = {
            g: t
            for g, t in con.execute(
                "SELECT gsis_id, any_value(team) FROM raw_rosters WHERE season = ? GROUP BY 1",
                [season - 1],
            ).fetchall()
        }
        snap_history = {
            g: s
            for g, s in con.execute(
                """
                SELECT gsis_id, avg(greatest(coalesce(offense_pct,0), coalesce(defense_pct,0)))
                FROM (
                    SELECT gsis_id, offense_pct, defense_pct,
                           row_number() OVER (PARTITION BY gsis_id ORDER BY season DESC, week DESC) AS rn
                    FROM player_game_usage
                    WHERE (season < ?) OR (season = ? AND week < ?)
                ) WHERE rn <= 4 GROUP BY 1
                """,
                [season, season, week],
            ).fetchall()
        }

    all_rows: list[dict[str, Any]] = []

    for position in Position:
        frame = _score_frame(season, week, position, window)
        if frame.is_empty():
            log.warning("no candidates for %s in %s week %s", position, season, week)
            continue

        depth_ranks, depth_listed = _depth_chart(position)
        if position in STARTER_ONLY and not depth_ranks:
            log.warning(
                "no current depth chart for %s - falling back to usage alone, which will let "
                "backups onto the board",
                position,
            )

        ranked: list[RankedPlayer] = []
        for r in frame.to_dicts():
            gsis = r["gsis_id"]

            depth_rank = depth_ranks.get(gsis)
            if position in STARTER_ONLY and depth_ranks and depth_rank != 1:
                continue
            name, pos, _pg = names.get(gsis, (gsis, position.value, None))
            team = rosters.get(gsis) or r.get("team")
            if not team or team not in env:
                continue  # not on a roster, or the team is not playing this week

            inj = injuries.get(gsis, {})
            status = inj.get("injury_status")
            # Sleeper emits "NA" and "-" where a healthy player simply has no designation. The
            # play-probability lookup already treats those as healthy, but the raw string was still
            # being stored on the ranking row, so the UI rendered an injury chip and an
            # empirical-play-rate tooltip on rank-1 players who are fine. Normalise once, here,
            # rather than asking every consumer to keep its own copy of the list.
            if status and status.strip() in NON_DESIGNATIONS:
                status = None
            roster_status = inj.get("roster_status")
            if status in EXCLUDED_STATUSES or roster_status in {"Injured Reserve", "PUP", "Inactive"}:
                continue

            play = injury_lookup(
                report_status=status,
                practice_status=inj.get("practice_participation"),
                prior_snap_share=snap_history.get(gsis),
                gsis_id=gsis,
            )

            ctx = env[team]
            cont = continuity.get(team, {})
            implied_factor = (ctx["implied_total"] or league_implied) / max(league_implied, 1e-6)

            base = float(r.get("score_base") or 0.0)
            # §4 applies the team's scoring environment to every position's usage score. K is the
            # one where implied total is additive rather than multiplicative -- a kicker's volume
            # comes from the offence stalling, not from how many plays he is on the field for.
            score = base + implied_factor if position is Position.K else base * implied_factor

            n_games = int(r.get("n_games") or 0)
            ranked.append(
                RankedPlayer(
                    gsis_id=gsis,
                    display_name=name,
                    position=position.value,
                    team=team,
                    opponent=ctx["opponent"],
                    score=score,
                    tiebreaker=float(r.get("tiebreaker") or 0.0),
                    n_games_used=n_games,
                    insufficient_history=n_games < settings.min_games_for_history,
                    changed_team=bool(prior_team.get(gsis) and prior_team[gsis] != team),
                    changed_coach=bool(cont.get("changed_coach")),
                    pass_rate_shift=float(cont.get("pass_rate_shift") or 0.0),
                    injury_status=status,
                    play_probability=play.p_played,
                    components={
                        "usage_base": base,
                        "implied_total": ctx["implied_total"],
                        "implied_factor": implied_factor,
                        "expected_plays": ctx["expected_plays"],
                        "expected_pass_rate": ctx["expected_pass_rate"],
                        "tiebreaker": float(r.get("tiebreaker") or 0.0),
                        "n_prior_season_games": int(r.get("n_prior_season") or 0),
                        "play_probability": play.p_played,
                        "play_probability_source": play.source,
                        "role": play.role,
                        "prior_snap_share": snap_history.get(gsis),
                        "prior_coach": cont.get("prior_coach"),
                        "coach": cont.get("coach"),
                        "depth_chart_rank": depth_rank,
                        "on_depth_chart": gsis in depth_listed,
                    },
                )
            )

        # §4: a player with almost no history is projected from priors rather than taken at face
        # value. Shrinking the score toward the position's median by sample size is the ranking-side
        # version of that -- otherwise one big game from a fill-in starter outranks a season of
        # evidence, which is exactly how a one-game linebacker landed 4th on the board.
        if ranked:
            median_score = sorted(p.score for p in ranked)[len(ranked) // 2]
            prior_games = 2.0
            for p in ranked:
                n = float(p.n_games_used)
                w = n / (n + prior_games)
                p.components["score_before_shrinkage"] = p.score
                p.components["shrinkage_weight"] = w
                p.components["position_median_score"] = median_score
                p.score = w * p.score + (1.0 - w) * median_score

        ranked.sort(
            key=lambda p: (
                p.score,
                p.tiebreaker,
                -(p.components.get("depth_chart_rank") or 99),
            ),
            reverse=True,
        )
        for i, p in enumerate(ranked[:TOP_N], start=1):
            all_rows.append(
                {
                    "season": season, "week": week, "position": p.position, "rank": i,
                    "gsis_id": p.gsis_id, "score": p.score, "tiebreaker": p.tiebreaker,
                    "team": p.team, "opponent": p.opponent, "n_games_used": p.n_games_used,
                    "insufficient_history": p.insufficient_history,
                    "changed_team": p.changed_team, "changed_coach": p.changed_coach,
                    "pass_rate_shift": p.pass_rate_shift, "injury_status": p.injury_status,
                    "play_probability": p.play_probability,
                    "components": json.dumps(p.components, default=str),
                }
            )

    if not all_rows:
        log.warning("no rankings produced for %s week %s", season, week)
        return 0

    frame = pl.DataFrame(all_rows)
    with connect() as con:
        con.register("rank_df", frame)
        try:
            con.execute("BEGIN TRANSACTION")
            con.execute("DELETE FROM rankings WHERE season = ? AND week = ?", [season, week])
            con.execute(
                """
                INSERT INTO rankings
                    (season, week, position, rank, gsis_id, score, tiebreaker, team, opponent,
                     n_games_used, insufficient_history, changed_team, changed_coach,
                     pass_rate_shift, injury_status, play_probability, components, computed_at)
                SELECT season, week, position, rank, gsis_id, score, tiebreaker, team, opponent,
                       n_games_used, insufficient_history, changed_team, changed_coach,
                       pass_rate_shift, injury_status, play_probability, components, now()
                FROM rank_df
                """
            )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            con.unregister("rank_df")

    log.info("rankings: %s week %s -> %d rows", season, week, frame.height)
    return frame.height


def get_rankings(season: int, week: int, position: str | None = None) -> pl.DataFrame:
    """Read a stored board."""
    where = "WHERE r.season = ? AND r.week = ?"
    params: list[Any] = [season, week]
    if position:
        where += " AND r.position = ?"
        params.append(position.upper())

    with connect() as con:
        return con.execute(
            f"""
            SELECT r.*, p.display_name
            FROM rankings r LEFT JOIN players p USING (gsis_id)
            {where}
            ORDER BY r.position, r.rank
            """,
            params,
        ).pl()


def print_rankings(
    season: int, week: int, position: str | None = None, recompute: bool = False
) -> None:
    """Print the six boards (§9 milestone-3 checkpoint)."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    if recompute:
        build_rankings(season, week)

    df = get_rankings(season, week, position)
    if df.is_empty():
        console.print("[yellow]no rankings stored — run `proplab refresh` or pass --recompute[/yellow]")
        return

    for pos in df["position"].unique(maintain_order=True):
        rows = df.filter(pl.col("position") == pos).sort("rank")
        table = Table(title=f"{pos} · {season} week {week}", title_justify="left")
        table.add_column("#", justify="right")
        table.add_column("player")
        table.add_column("team")
        table.add_column("opp")
        table.add_column("score", justify="right")
        table.add_column("tiebreak", justify="right")
        table.add_column("gp", justify="right")
        table.add_column("status")
        table.add_column("flags", style="yellow")

        for r in rows.to_dicts():
            flags = []
            if r["changed_team"]:
                flags.append("new team")
            if r["changed_coach"]:
                flags.append("new HC")
            if abs(r["pass_rate_shift"] or 0) > 0.03:
                flags.append(f"pass rate {r['pass_rate_shift']:+.0%}")
            if r["insufficient_history"]:
                flags.append("thin history")

            status = r["injury_status"] or "—"
            if r["injury_status"]:
                status += f" ({r['play_probability']:.0%})"

            table.add_row(
                str(r["rank"]), r["display_name"] or r["gsis_id"], r["team"] or "—",
                r["opponent"] or "—", f"{r['score']:.2f}", f"{r['tiebreaker']:.3f}",
                str(r["n_games_used"]), status, ", ".join(flags),
            )
        console.print(table)
        console.print()
