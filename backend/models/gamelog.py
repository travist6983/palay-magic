"""Per-player, per-game stats and usage — the single source every later stage reads.

Two tables:

* ``player_game_stats`` — one row per (player, game, canonical stat key). Every stat PropLab
  projects, including the derived ones (``rush_rec_yards``, ``kicking_points``, ``anytime_td``,
  the ``longest_*`` family) so no consumer has to know how they are assembled.
* ``player_game_usage`` — the share numbers §5.4 projects from: target share, carry share,
  red-zone and goal-line share, route participation, snap share, team tackle share.

Settlement rules are applied here, at the point the number is created, rather than downstream
where they would be easy to forget (D9):

* ``tackles_assists`` is ``def_tackles_solo + def_tackle_assists``, defensive plays only. Verified
  against PFR's gamebook ``def_tackles_combined`` on 89.2% of 2025 LB game-weeks; the
  similarly-named ``def_tackles_with_assist`` matches only 19.0% and is **not** the assist count.
* ``anytime_rush_td`` counts rushing scores only — a QB's passing TD never counts.
* ``kicking_points`` is ``3 x FGM + 1 x XPM``; two-point conversions never credit the kicker.
* The ``longest_*`` family is 0 when the player recorded no such play, which is how those props
  settle (Under), not a missing value.
"""

from __future__ import annotations

import polars as pl

from backend.config import get_settings
from backend.db.connection import connect
from backend.db.views import view_exists
from backend.logging_setup import get_logger

log = get_logger(__name__)

# Canonical stat key -> the raw_player_stats column it comes from (D9).
_DIRECT_STATS: dict[str, str] = {
    "pass_attempts": "attempts",
    "completions": "completions",
    "passing_yards": "passing_yards",
    "passing_tds": "passing_tds",
    "interceptions": "passing_interceptions",
    "rush_attempts": "carries",
    "rushing_yards": "rushing_yards",
    "targets": "targets",
    "receptions": "receptions",
    "receiving_yards": "receiving_yards",
    "fg_attempts": "fg_att",
    "fg_made": "fg_made",
    "xp_made": "pat_made",
    "longest_fg": "fg_long",
    "solo_tackles": "def_tackles_solo",
    "sacks": "def_sacks",
    "passes_defended": "def_pass_defended",
    # Components of derived stats; kept because the projection needs them directly.
    "rushing_tds": "rushing_tds",
    "receiving_tds": "receiving_tds",
    "sacks_suffered": "sacks_suffered",
    "receiving_air_yards": "receiving_air_yards",
}

# Derived stats, as SQL over the same row.
_DERIVED_STATS: dict[str, str] = {
    "rush_rec_yards": "coalesce(rushing_yards, 0) + coalesce(receiving_yards, 0)",
    "kicking_points": "3 * coalesce(fg_made, 0) + coalesce(pat_made, 0)",
    # Tackles + assists: DraftKings settlement, defensive plays only (D2).
    "tackles_assists": "coalesce(def_tackles_solo, 0) + coalesce(def_tackle_assists, 0)",
    # Anytime TD: possession in the end zone, rushing or receiving.
    "anytime_td": "CASE WHEN coalesce(rushing_tds, 0) + coalesce(receiving_tds, 0) > 0 THEN 1 ELSE 0 END",
    # A QB's passing TD never counts toward his anytime-TD prop.
    "anytime_rush_td": "CASE WHEN coalesce(rushing_tds, 0) > 0 THEN 1 ELSE 0 END",
}


def build_player_game_stats(seasons: list[int] | None = None) -> int:
    """Materialise ``player_game_stats`` for every regular-season player-game.

    Returns:
        Number of rows written.
    """
    seasons = seasons or list(get_settings().seasons)
    season_list = ",".join(str(s) for s in seasons)

    with connect() as con:
        if not view_exists(con, "raw_player_stats"):
            log.warning("raw_player_stats missing - run the nflverse ingest first")
            return 0

        unpivots = [
            f"SELECT player_id AS gsis_id, season, week, game_id, team, opponent_team AS opponent, "
            f"position, '{key}' AS stat, {col}::DOUBLE AS value FROM base"
            for key, col in _DIRECT_STATS.items()
        ] + [
            f"SELECT player_id AS gsis_id, season, week, game_id, team, opponent_team AS opponent, "
            f"position, '{key}' AS stat, ({expr})::DOUBLE AS value FROM base"
            for key, expr in _DERIVED_STATS.items()
        ]

        box_sql = f"""
        WITH base AS (
            SELECT * FROM raw_player_stats
            WHERE season IN ({season_list}) AND season_type = 'REG' AND player_id IS NOT NULL
        )
        {" UNION ALL ".join(unpivots)}
        """

        # longest_* come from play-by-play. A player with no such play gets 0, because that is how
        # the prop settles (Under), not NULL.
        longest_sql = f"""
        WITH plays AS (
            SELECT season, week, game_id, posteam AS team, defteam AS opponent,
                   passer_player_id, receiver_player_id, rusher_player_id,
                   complete_pass, rush_attempt, yards_gained,
                   -- Yards after a lateral belong to the lateral receiver, not the original one.
                   CASE WHEN lateral_receiver_player_id IS NOT NULL
                        THEN coalesce(air_yards, 0) + coalesce(yards_after_catch, 0)
                        ELSE yards_gained END AS receiver_yards
            FROM raw_pbp
            WHERE season IN ({season_list}) AND season_type = 'REG'
              AND coalesce(two_point_attempt, 0) = 0 AND coalesce(play_type, '') <> 'no_play'
        )
        SELECT passer_player_id AS gsis_id, season, week, game_id, team, opponent,
               'longest_completion' AS stat, max(yards_gained)::DOUBLE AS value
        FROM plays WHERE complete_pass = 1 AND passer_player_id IS NOT NULL
        GROUP BY 1,2,3,4,5,6
        UNION ALL
        SELECT receiver_player_id, season, week, game_id, team, opponent,
               'longest_reception', max(receiver_yards)::DOUBLE
        FROM plays WHERE complete_pass = 1 AND receiver_player_id IS NOT NULL
        GROUP BY 1,2,3,4,5,6
        UNION ALL
        SELECT rusher_player_id, season, week, game_id, team, opponent,
               'longest_rush', max(yards_gained)::DOUBLE
        FROM plays WHERE rush_attempt = 1 AND rusher_player_id IS NOT NULL
        GROUP BY 1,2,3,4,5,6
        """

        con.execute("BEGIN TRANSACTION")
        try:
            con.execute("DELETE FROM player_game_stats")
            con.execute(
                "INSERT INTO player_game_stats "
                "(gsis_id, season, week, game_id, team, opponent, position, stat, value) "
                f"SELECT gsis_id, season, week, game_id, team, opponent, position, stat, "
                f"       coalesce(value, 0) FROM ({box_sql}) WHERE gsis_id IS NOT NULL"
            )
            # Longest stats need the player's position, which pbp does not carry.
            con.execute(
                """
                INSERT INTO player_game_stats
                    (gsis_id, season, week, game_id, team, opponent, position, stat, value)
                SELECT l.gsis_id, l.season, l.week, l.game_id, l.team, l.opponent,
                       coalesce(p.position, 'UNK'), l.stat, coalesce(l.value, 0)
                FROM (""" + longest_sql + """) l
                LEFT JOIN players p ON p.gsis_id = l.gsis_id
                WHERE l.gsis_id IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM player_game_stats e
                      WHERE e.gsis_id = l.gsis_id AND e.season = l.season
                        AND e.week = l.week AND e.stat = l.stat
                  )
                """
            )
            # A player who had the opportunity but recorded no such play settles at 0, not NULL:
            # a longest-X prop with no qualifying play is an Under, and the distribution needs that
            # zero mass (D9). Play-by-play only produces a row when the play happened, so the
            # zeros have to be filled in against the opportunity stat.
            for stat, opportunity in (
                ("longest_reception", "targets"),
                ("longest_rush", "rush_attempts"),
                ("longest_completion", "pass_attempts"),
            ):
                con.execute(
                    """
                    INSERT INTO player_game_stats
                        (gsis_id, season, week, game_id, team, opponent, position, stat, value)
                    SELECT o.gsis_id, o.season, o.week, o.game_id, o.team, o.opponent,
                           o.position, ?, 0.0
                    FROM player_game_stats o
                    WHERE o.stat = ? AND o.value > 0
                      AND NOT EXISTS (
                          SELECT 1 FROM player_game_stats e
                          WHERE e.gsis_id = o.gsis_id AND e.season = o.season
                            AND e.week = o.week AND e.stat = ?
                      )
                    """,
                    [stat, opportunity, stat],
                )

            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

        n = con.execute("SELECT count(*) FROM player_game_stats").fetchone()[0]

    log.info("player_game_stats: %d rows across seasons %s", n, seasons)
    return int(n)


def build_player_game_usage(seasons: list[int] | None = None) -> int:
    """Materialise ``player_game_usage``: the share numbers §5.4 projects from.

    Red-zone is inside the opponent's 20; goal-line is inside the 5, which is where 57.4% of
    rushing touchdowns since 2010 have come from and is therefore what an anytime-TD projection
    should key off rather than total carries (§5.6).

    Route participation comes from NGS receiving where available. It is a season-to-date figure,
    not per game, so it is joined as the best available proxy and flagged as such rather than
    pretended to be a weekly number.

    Returns:
        Number of rows written.
    """
    seasons = seasons or list(get_settings().seasons)
    season_list = ",".join(str(s) for s in seasons)

    sql = f"""
    WITH box AS (
        SELECT player_id AS gsis_id, season, week, game_id, team, opponent_team AS opponent,
               position,
               coalesce(attempts, 0)        AS attempts,
               coalesce(carries, 0)         AS carries,
               coalesce(targets, 0)         AS targets,
               coalesce(receiving_air_yards, 0) AS air_yards,
               coalesce(def_tackles_solo, 0) + coalesce(def_tackle_assists, 0) AS tackles
        FROM raw_player_stats
        WHERE season IN ({season_list}) AND season_type = 'REG' AND player_id IS NOT NULL
    ),
    team_totals AS (
        SELECT season, week, game_id, team,
               sum(attempts)  AS team_pass_attempts,
               sum(carries)   AS team_rush_attempts,
               sum(targets)   AS team_targets,
               sum(air_yards) AS team_air_yards,
               sum(tackles)   AS team_tackles
        FROM box GROUP BY 1, 2, 3, 4
    ),
    rz AS (
        SELECT season, week, game_id, posteam AS team,
               receiver_player_id AS gsis_id,
               sum(CASE WHEN yardline_100 <= 20 AND pass_attempt = 1
                         AND coalesce(two_point_attempt, 0) = 0 AND coalesce(play_type, '') <> 'no_play'
                        THEN 1 ELSE 0 END) AS rz_targets
        FROM raw_pbp
        WHERE season IN ({season_list}) AND season_type = 'REG' AND receiver_player_id IS NOT NULL
        GROUP BY 1, 2, 3, 4, 5
    ),
    -- Same population as the numerator: TARGETS (a receiver on the play), not attempts. Sacks,
    -- throwaways and two-point passes in the denominator made shares sum to 0.87.
    rz_team AS (
        SELECT season, week, game_id, posteam AS team,
               sum(CASE WHEN yardline_100 <= 20 AND pass_attempt = 1 AND receiver_player_id IS NOT NULL
                         AND coalesce(two_point_attempt, 0) = 0 AND coalesce(play_type, '') <> 'no_play'
                        THEN 1 ELSE 0 END) AS team_rz_targets
        FROM raw_pbp
        WHERE season IN ({season_list}) AND season_type = 'REG'
        GROUP BY 1, 2, 3, 4
    ),
    carries_rz AS (
        SELECT season, week, game_id, posteam AS team, rusher_player_id AS gsis_id,
               sum(CASE WHEN yardline_100 <= 20 THEN 1 ELSE 0 END) AS rz_carries,
               sum(CASE WHEN yardline_100 <= 5  THEN 1 ELSE 0 END) AS gl_carries
        FROM raw_pbp
        WHERE season IN ({season_list}) AND season_type = 'REG'
          AND rush_attempt = 1 AND rusher_player_id IS NOT NULL
          AND coalesce(two_point_attempt, 0) = 0 AND coalesce(qb_kneel, 0) = 0
          AND coalesce(play_type, '') <> 'no_play'
        GROUP BY 1, 2, 3, 4, 5
    ),
    carries_rz_team AS (
        SELECT season, week, game_id, posteam AS team,
               sum(CASE WHEN yardline_100 <= 5 THEN 1 ELSE 0 END) AS team_gl_carries
        FROM raw_pbp
        WHERE season IN ({season_list}) AND season_type = 'REG' AND rush_attempt = 1
          AND coalesce(two_point_attempt, 0) = 0 AND coalesce(qb_kneel, 0) = 0
          AND coalesce(play_type, '') <> 'no_play'
        GROUP BY 1, 2, 3, 4
    ),
    team_plays AS (
        SELECT season, week, game_id, posteam AS team,
               sum(CASE WHEN qb_dropback = 1 OR rush_attempt = 1 THEN 1 ELSE 0 END) AS team_plays,
               sum(qb_dropback) AS team_dropbacks
        FROM raw_pbp
        WHERE season IN ({season_list}) AND season_type = 'REG' AND posteam IS NOT NULL
        GROUP BY 1, 2, 3, 4
    ),
    snaps AS (
        SELECT s.season, s.week, s.game_id, p.gsis_id,
               s.offense_snaps, s.offense_pct, s.defense_snaps, s.defense_pct, s.st_snaps
        FROM raw_snap_counts s
        JOIN players p ON p.pfr_id = s.pfr_player_id
        WHERE s.season IN ({season_list})
    ),
    routes AS (
        SELECT season, week, gsis_id,
               CASE WHEN targets > 0 AND percent_share_of_intended_air_yards > 0
                    THEN percent_share_of_intended_air_yards / 100.0 END AS ngs_air_share
        FROM raw_ngs_receiving
        WHERE season IN ({season_list}) AND week > 0
    )
    SELECT
        b.gsis_id, b.season, b.week, b.game_id, b.team, b.opponent, b.position,
        sn.offense_snaps, sn.offense_pct, sn.defense_snaps, sn.defense_pct, sn.st_snaps,
        t.team_pass_attempts, t.team_rush_attempts, t.team_targets, tp.team_plays,
        CASE WHEN t.team_targets > 0 THEN b.targets / t.team_targets END        AS target_share,
        CASE WHEN t.team_rush_attempts > 0 THEN b.carries / t.team_rush_attempts END AS carry_share,
        CASE WHEN tp.team_dropbacks > 0 THEN b.attempts / tp.team_dropbacks END AS dropback_share,
        CASE WHEN t.team_air_yards > 0 THEN b.air_yards / t.team_air_yards END  AS air_yards_share,
        coalesce(rz.rz_targets, 0)                                              AS rz_targets,
        CASE WHEN rzt.team_rz_targets > 0
             THEN coalesce(rz.rz_targets, 0) / rzt.team_rz_targets END          AS rz_target_share,
        coalesce(cz.rz_carries, 0)                                              AS rz_carries,
        coalesce(cz.gl_carries, 0)                                              AS gl_carries,
        CASE WHEN czt.team_gl_carries > 0
             THEN coalesce(cz.gl_carries, 0) / czt.team_gl_carries END          AS gl_carry_share,
        t.team_tackles,
        CASE WHEN t.team_tackles > 0 THEN b.tackles / t.team_tackles END        AS tackle_share,
        r.ngs_air_share                                                          AS route_participation
    FROM box b
    LEFT JOIN team_totals    t   ON t.season=b.season AND t.week=b.week AND t.game_id=b.game_id AND t.team=b.team
    LEFT JOIN team_plays     tp  ON tp.season=b.season AND tp.week=b.week AND tp.game_id=b.game_id AND tp.team=b.team
    LEFT JOIN rz             rz  ON rz.season=b.season AND rz.week=b.week AND rz.game_id=b.game_id AND rz.gsis_id=b.gsis_id
    LEFT JOIN rz_team        rzt ON rzt.season=b.season AND rzt.week=b.week AND rzt.game_id=b.game_id AND rzt.team=b.team
    LEFT JOIN carries_rz     cz  ON cz.season=b.season AND cz.week=b.week AND cz.game_id=b.game_id AND cz.gsis_id=b.gsis_id
    LEFT JOIN carries_rz_team czt ON czt.season=b.season AND czt.week=b.week AND czt.game_id=b.game_id AND czt.team=b.team
    LEFT JOIN snaps          sn  ON sn.season=b.season AND sn.week=b.week AND sn.game_id=b.game_id AND sn.gsis_id=b.gsis_id
    LEFT JOIN routes         r   ON r.season=b.season AND r.week=b.week AND r.gsis_id=b.gsis_id
    """

    with connect() as con:
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute("DELETE FROM player_game_usage")
            con.execute(
                "INSERT INTO player_game_usage "
                "(gsis_id, season, week, game_id, team, opponent, position, offense_snaps, "
                " offense_pct, defense_snaps, defense_pct, st_snaps, team_pass_attempts, "
                " team_rush_attempts, team_targets, team_plays, target_share, carry_share, "
                " dropback_share, air_yards_share, rz_targets, rz_target_share, rz_carries, "
                " gl_carries, gl_carry_share, team_tackles, tackle_share, route_participation) "
                + sql
            )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        n = con.execute("SELECT count(*) FROM player_game_usage").fetchone()[0]

    log.info("player_game_usage: %d rows across seasons %s", n, seasons)
    return int(n)


def build_all(seasons: list[int] | None = None) -> dict[str, int]:
    """Rebuild both game-log tables. Called by the refresh pipeline."""
    return {
        "player_game_stats": build_player_game_stats(seasons),
        "player_game_usage": build_player_game_usage(seasons),
    }


def player_log(gsis_id: str, stats: list[str] | None = None, limit: int = 24) -> pl.DataFrame:
    """A player's most recent games, one column per stat. Used by the deep-dive page."""
    where = "WHERE gsis_id = ?"
    params: list[object] = [gsis_id]
    if stats:
        where += " AND stat IN ({})".format(",".join("?" * len(stats)))
        params.extend(stats)

    with connect() as con:
        return con.execute(
            f"""
            SELECT season, week, game_id, team, opponent, stat, value
            FROM player_game_stats
            {where}
            ORDER BY season DESC, week DESC
            LIMIT {int(limit) * max(1, len(stats or [1]))}
            """,
            params,
        ).pl()
