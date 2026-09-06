"""Injury play probability (§5.7).

The reference doc calls this "the single highest-value thing you can build with free data",
because most bettors read a designation as a label rather than a probability. We compute the
empirical ``P(played)`` and ``E[snap share | played]`` for every (report status, Friday practice
status) pair by joining nflverse's official weekly injury report to the following week's snap
counts, then apply it to every ``Questionable`` player.

Every projection is reported twice: **conditional on playing** and **unconditional** (§5.7).

Join note (D6): ``raw_injuries`` keys on ``gsis_id`` but ``raw_snap_counts`` keys on
``pfr_player_id``, so the join goes through the ``players`` crosswalk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

from backend.config import get_settings
from backend.db.connection import connect
from backend.logging_setup import get_logger

log = get_logger(__name__)

# Designations that remove a player from the board entirely (§4).
EXCLUDED_STATUSES: frozenset[str] = frozenset(
    {
        "Out", "IR", "PUP", "Doubtful", "Injured Reserve",
        # Sleeper's roster vocabulary: suspended, non-football injury, did not report, COVID.
        # None of these players take a snap, so none belongs on a board (§4).
        "Sus", "NFI", "DNR", "COV", "Physically Unable to Perform",
    }
)

NON_DESIGNATIONS: frozenset[str] = frozenset({"NA", "N/A", "-", ""})
"""Sleeper emits these where a healthy player has no designation. They are not injury statuses,
and treating "NA" as one would push 95 healthy players off the board."""

# Designations that keep a player on the board with a play probability shown (§4).
UNCERTAIN_STATUSES: frozenset[str] = frozenset({"Questionable"})

# Fallback rates used only when a cell has too little history. Directionally from
# free_nfl_data_sources.md Tier 1; the computed table supersedes these everywhere it has data.
FALLBACK_PLAY_RATES: dict[tuple[str, str], float] = {
    ("Questionable", "Full"): 0.85,
    ("Questionable", "Limited"): 0.74,
    ("Questionable", "DNP"): 0.54,
    ("Doubtful", "Full"): 0.35,
    ("Doubtful", "Limited"): 0.25,
    ("Doubtful", "DNP"): 0.10,
    ("Out", "Full"): 0.0,
    ("Out", "Limited"): 0.0,
    ("Out", "DNP"): 0.0,
}

MIN_CELL_OBSERVATIONS = 30
"""Below this many observations a cell falls back rather than trusting a tiny sample."""

# Role buckets, by mean snap share over the four weeks before the game (§5.7, migration 002).
# Pooling these together is the trap: it drags P(played | Questionable + Full) from 84.8% down to
# 70.6%, because the weekly injury report is dominated by fringe players who never dress.
ROLE_STARTER = "starter"
ROLE_ROTATIONAL = "rotational"
ROLE_FRINGE = "fringe"
ROLE_NO_HISTORY = "no_recent_games"

ROLE_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.55, ROLE_STARTER),
    (0.20, ROLE_ROTATIONAL),
    (0.00, ROLE_FRINGE),
)


def role_bucket(prior_snap_share: float | None) -> str:
    """Classify a player's role from mean snap share over the previous four weeks.

    Args:
        prior_snap_share: mean of ``max(offense_pct, defense_pct)`` over weeks w-4..w-1, or None
            when the player has no games in that window.

    Returns:
        One of ``starter`` (>= 55%), ``rotational`` (>= 20%), ``fringe`` (< 20%), or
        ``no_recent_games``.
    """
    if prior_snap_share is None:
        return ROLE_NO_HISTORY
    for threshold, name in ROLE_THRESHOLDS:
        if prior_snap_share >= threshold:
            return name
    return ROLE_FRINGE


@dataclass(frozen=True)
class PlayProbability:
    """What a designation actually means for one player this week."""

    gsis_id: str
    report_status: str | None
    practice_status: str | None
    p_played: float
    expected_snap_share: float
    """E[snap share | played]. Multiplies usage for a banged-up player who plays reduced snaps."""

    n_observations: int
    source: str
    """'empirical' when computed from history, 'fallback' when the cell was too thin."""

    excluded: bool
    """True for Out / IR / PUP / Doubtful, which drop off the board entirely (§4)."""

    role: str = "starter"
    """Which role bucket the rate came from. See :func:`role_bucket`."""

    snap_ratio: float = 1.0
    """E[snap share / normal share | played]. The usage multiplier for an active-but-limited
    player: a Questionable starter who suits up keeps ~93% of his usual snaps."""

    def to_json(self) -> dict[str, object]:
        return {
            "report_status": self.report_status,
            "practice_status": self.practice_status,
            "p_played": self.p_played,
            "expected_snap_share": self.expected_snap_share,
            "n_observations": self.n_observations,
            "source": self.source,
            "excluded": self.excluded,
            "role": self.role,
            "snap_ratio": self.snap_ratio,
        }


def compute_play_rates(seasons: list[int] | None = None) -> pl.DataFrame:
    """Build the empirical designation -> play-rate table and persist it (§5.7).

    For every row of the official weekly injury report, ask whether that player took a snap that
    week. Three dimensions, in decreasing order of how much they move the answer:

    1. ``report_status`` -- the game designation (Out / Doubtful / Questionable).
    2. ``practice_status`` -- the final practice participation, which the reference doc calls the
       actual predictive signal.
    3. ``role_bucket`` -- mean snap share over the previous four weeks. **Load-bearing.** Without
       it a Questionable + Full player looks like a 70.6% bet; split out, a *starter* in that cell
       plays 84.8% of the time while the fringe players dragging the pooled number down are ones
       we would never project anyway.

    The role window is computed off the injury-report spine, never off the current week's snap
    row. Deriving it from the same join that decides ``played`` would make every player with a
    role bucket one who played by construction, and every cell would read 1.000.

    Snap share is ``max(offense_pct, defense_pct)`` so a two-way contributor is not recorded at
    zero, and "played" means at least one snap of any kind, special teams included.

    Args:
        seasons: seasons to learn from. Defaults to every season in the cache.

    Returns:
        One row per (report_status, practice_status, role_bucket, position_group), plus an 'ALL'
        position_group row for the marginal, with ``p_played`` and ``mean_snap_share``.
    """
    seasons = seasons or list(get_settings().seasons)
    season_list = ",".join(str(s) for s in seasons)

    sql = f"""
    WITH snaps AS (
        SELECT
            s.season,
            s.week,
            p.gsis_id,
            greatest(coalesce(s.offense_pct, 0), coalesce(s.defense_pct, 0)) AS snap_share,
            (coalesce(s.offense_snaps, 0) + coalesce(s.defense_snaps, 0)
                 + coalesce(s.st_snaps, 0)) AS total_snaps,
            -- Absolute game order across seasons, so "the last four games" can cross a season
            -- boundary the same way ranking.py's application of this table does.
            s.season * 100 + s.week AS game_key
        FROM raw_snap_counts s
        JOIN players p ON p.pfr_id = s.pfr_player_id
        WHERE s.season IN ({season_list})
    ),
    spine AS (
        SELECT
            i.season,
            i.week,
            i.gsis_id,
            i.report_status,
            -- Normalised to the same three tokens Sleeper emits ('Full' / 'Limited' / 'DNP'), so a
            -- live designation can actually find its cell. The long nflverse strings never matched.
            CASE
                WHEN lower(coalesce(i.practice_status, '')) LIKE 'full%' THEN 'Full'
                WHEN lower(coalesce(i.practice_status, '')) LIKE 'limited%' THEN 'Limited'
                WHEN lower(coalesce(i.practice_status, '')) LIKE 'did not%' THEN 'DNP'
                WHEN coalesce(i.practice_status, '') = '' THEN 'Unknown'
                ELSE i.practice_status
            END AS practice_status,
            coalesce(pl.position_group, i.position, 'UNK') AS position_group,
            i.season * 100 + i.week AS game_key
        FROM raw_injuries i
        LEFT JOIN players pl ON pl.gsis_id = i.gsis_id
        WHERE i.gsis_id IS NOT NULL
          AND i.season IN ({season_list})
          AND i.report_status IS NOT NULL
          AND i.report_status <> 'Note'
    ),
    -- Role = mean snap share over the player's last FOUR GAMES before this one, any season.
    -- This is exactly how ranking.py buckets a live player, so the table is fitted on the
    -- definition it is applied with. Fitting on same-season weeks w-4..w-1 put every Week-1
    -- starter in 'no_recent_games' at 53% when his true rate was 63-68%.
    role AS (
        SELECT
            sp.season, sp.week, sp.gsis_id,
            avg(prior.snap_share) AS prior_share,
            count(prior.snap_share) AS n_prior_games
        FROM spine sp
        LEFT JOIN (
            SELECT a.gsis_id, a.game_key AS target_key, b.snap_share,
                   row_number() OVER (PARTITION BY a.gsis_id, a.game_key ORDER BY b.game_key DESC) AS rn
            FROM (SELECT DISTINCT gsis_id, game_key FROM spine) a
            JOIN snaps b ON b.gsis_id = a.gsis_id AND b.game_key < a.game_key
        ) prior ON prior.gsis_id = sp.gsis_id AND prior.target_key = sp.game_key AND prior.rn <= 4
        GROUP BY 1, 2, 3
    ),
    joined AS (
        SELECT
            sp.report_status,
            sp.practice_status,
            sp.position_group,
            CASE
                WHEN r.n_prior_games = 0 OR r.prior_share IS NULL THEN 'no_recent_games'
                WHEN r.prior_share >= 0.55 THEN 'starter'
                WHEN r.prior_share >= 0.20 THEN 'rotational'
                ELSE 'fringe'
            END AS role_bucket,
            CASE WHEN coalesce(cur.total_snaps, 0) > 0 THEN 1 ELSE 0 END AS played,
            cur.snap_share,
            CASE WHEN coalesce(cur.total_snaps, 0) > 0 AND r.prior_share > 0.05
                 THEN least(cur.snap_share / r.prior_share, 1.5) END AS snap_ratio
        FROM spine sp
        LEFT JOIN role  r   ON r.gsis_id = sp.gsis_id AND r.season = sp.season AND r.week = sp.week
        LEFT JOIN snaps cur ON cur.gsis_id = sp.gsis_id AND cur.season = sp.season AND cur.week = sp.week
    ),
    by_position AS (
        SELECT report_status, practice_status, role_bucket, position_group,
               count(*) AS n_observations, sum(played) AS n_played,
               sum(played)::DOUBLE / count(*) AS p_played,
               avg(CASE WHEN played = 1 THEN snap_share END) AS mean_snap_share,
               stddev_samp(CASE WHEN played = 1 THEN snap_share END) AS sd_snap_share,
               avg(snap_ratio) AS mean_snap_ratio
        FROM joined GROUP BY 1, 2, 3, 4
    ),
    marginal AS (
        SELECT report_status, practice_status, role_bucket, 'ALL' AS position_group,
               count(*) AS n_observations, sum(played) AS n_played,
               sum(played)::DOUBLE / count(*) AS p_played,
               avg(CASE WHEN played = 1 THEN snap_share END) AS mean_snap_share,
               stddev_samp(CASE WHEN played = 1 THEN snap_share END) AS sd_snap_share,
               avg(snap_ratio) AS mean_snap_ratio
        FROM joined GROUP BY 1, 2, 3
    )
    SELECT * FROM by_position
    UNION ALL
    SELECT * FROM marginal
    ORDER BY n_observations DESC
    """

    with connect() as con:
        df = con.execute(sql).pl()
        con.execute("DELETE FROM injury_play_rates")
        con.register("tmp_rates", df)
        con.execute(
            """
            INSERT INTO injury_play_rates
                (report_status, practice_status, role_bucket, position_group, n_observations,
                 n_played, p_played, mean_snap_share, sd_snap_share, mean_snap_ratio, computed_at)
            SELECT report_status, practice_status, role_bucket, position_group, n_observations,
                   n_played, p_played, mean_snap_share, sd_snap_share, mean_snap_ratio, now()
            FROM tmp_rates
            """
        )
        con.unregister("tmp_rates")

    log.info("computed %d injury play-rate cells from seasons %s", df.height, seasons)
    return df


def marginal_play_rates(role: str | None = None) -> pl.DataFrame:
    """The headline (designation, practice, role) table, collapsed across position groups.

    Args:
        role: restrict to one role bucket. ``'starter'`` is the population PropLab actually
            projects, so it is the row that matters for the top-10 boards.
    """
    where = "WHERE position_group = 'ALL'"
    params: list[object] = []
    if role:
        where += " AND role_bucket = ?"
        params.append(role)

    with connect() as con:
        return con.execute(
            f"""
            SELECT report_status, practice_status, role_bucket, n_observations, n_played,
                   p_played, mean_snap_share
            FROM injury_play_rates
            {where}
            ORDER BY report_status, practice_status, role_bucket
            """,
            params,
        ).pl()


def lookup(
    report_status: str | None,
    practice_status: str | None,
    prior_snap_share: float | None = None,
    position_group: str | None = None,
    gsis_id: str = "",
) -> PlayProbability:
    """Resolve one player's play probability from the learned table (§5.7).

    Resolution order, most specific first. Each level is used only when it has at least
    ``MIN_CELL_OBSERVATIONS`` observations behind it:

    1. (status, practice, role, position group)
    2. (status, practice, role) collapsed across position groups
    3. (status, role) collapsed across practice statuses. This is the level that carries the
       common case: before Wednesday of game week no practice report exists at all.
    4. (status, practice) collapsed across roles -- the pooled number, which under-states a
       starter, so it is a last resort rather than the default
    5. the documented fallback constants
    6. healthy (1.0)

    A player with no designation at all is healthy. ``Out``, ``IR`` and ``PUP`` short-circuit to
    zero by rule (§4); ``Doubtful`` still reads its empirical cell, which lands near 1%.

    Args:
        report_status: the game designation.
        practice_status: final practice participation (any spelling; normalised internally).
        prior_snap_share: mean snap share over the previous four weeks. Supplying this is what
            separates a starter's ~85% from the pooled ~71%.
        position_group: nflverse position group, for the most specific cell.
        gsis_id: carried through onto the result for logging.
    """
    if report_status and report_status.strip() in NON_DESIGNATIONS:
        report_status = None

    excluded = bool(report_status) and report_status in EXCLUDED_STATUSES

    if not report_status:
        return PlayProbability(gsis_id, None, practice_status, 1.0, 1.0, 0, "healthy", False, ROLE_STARTER)

    if report_status in EXCLUDED_STATUSES and report_status != "Doubtful":
        return PlayProbability(gsis_id, report_status, practice_status, 0.0, 0.0, 0, "rule", True, ROLE_STARTER)

    practice = practice_participation_signal(practice_status)
    role = role_bucket(prior_snap_share)
    cols = "sum(n_observations), sum(n_played)::DOUBLE / nullif(sum(n_observations), 0), " \
           "sum(mean_snap_share * n_played) / nullif(sum(n_played), 0), " \
           "sum(mean_snap_ratio * n_played) / nullif(sum(n_played), 0)"

    def _result(row, source: str) -> PlayProbability:
        return PlayProbability(
            gsis_id=gsis_id,
            report_status=report_status,
            practice_status=practice,
            p_played=float(row[1]),
            expected_snap_share=float(row[2]) if row[2] is not None else 1.0,
            n_observations=int(row[0]),
            source=source,
            excluded=excluded,
            role=role,
            snap_ratio=float(row[3]) if row[3] is not None else 1.0,
        )

    with connect() as con:
        if position_group:
            row = con.execute(
                f"SELECT {cols} FROM injury_play_rates "
                "WHERE report_status = ? AND practice_status = ? AND role_bucket = ? "
                "  AND position_group = ?",
                [report_status, practice, role, position_group],
            ).fetchone()
            if row and row[0] and row[0] >= MIN_CELL_OBSERVATIONS:
                return _result(row, "empirical:position")

        row = con.execute(
            f"SELECT {cols} FROM injury_play_rates "
            "WHERE report_status = ? AND practice_status = ? AND role_bucket = ? "
            "  AND position_group = 'ALL'",
            [report_status, practice, role],
        ).fetchone()
        if row and row[0] and row[0] >= MIN_CELL_OBSERVATIONS:
            return _result(row, "empirical:role")

        # (status, role, POSITION) across practice statuses. Quarterbacks are the case that
        # matters: a Questionable QB starter plays 44.6% of the time against 71% for starters at
        # large, and without this level that cell was unreachable.
        if position_group:
            row = con.execute(
                f"SELECT {cols} FROM injury_play_rates "
                "WHERE report_status = ? AND role_bucket = ? AND position_group = ?",
                [report_status, role, position_group],
            ).fetchone()
            if row and row[0] and row[0] >= MIN_CELL_OBSERVATIONS:
                return _result(row, "empirical:position_marginal")

        # Marginalise over practice status, KEEPING the role. This is the case that matters most
        # in practice: before Wednesday of game week there is no practice report at all, and the
        # tiny "Unknown" practice cell is worse than no split. Collapsing across practice while
        # holding role fixed keeps the dimension that actually moves the number (D11) -- a
        # Questionable starter lands near 72% instead of the cross-role 61%.
        row = con.execute(
            f"SELECT {cols} FROM injury_play_rates "
            "WHERE report_status = ? AND role_bucket = ? AND position_group = 'ALL'",
            [report_status, role],
        ).fetchone()
        if row and row[0] and row[0] >= MIN_CELL_OBSERVATIONS:
            return _result(row, "empirical:role_marginal")

        row = con.execute(
            f"SELECT {cols} FROM injury_play_rates "
            "WHERE report_status = ? AND practice_status = ? AND position_group = 'ALL'",
            [report_status, practice],
        ).fetchone()

    if row and row[0] and row[0] >= MIN_CELL_OBSERVATIONS:
        log.debug(
            "play rate for %s/%s/%s fell back to the pooled cell, which under-states a starter",
            report_status, practice, role,
        )
        return _result(row, "empirical:pooled")

    fallback = FALLBACK_PLAY_RATES.get((report_status, practice))
    if fallback is None:
        fallback = 0.0 if excluded else 1.0
    return PlayProbability(
        gsis_id, report_status, practice, fallback, 1.0,
        int(row[0]) if row and row[0] else 0, "fallback", excluded, role,
    )


def apply_unconditional(
    conditional_mean: float,
    play_probability: float,
    zero_if_absent: bool = True,
) -> float:
    """Convert a conditional-on-playing projection into an unconditional one (§5.7).

        E[X] = P(played) * E[X | played]

    A player who does not play records zero for every counting stat, so the unconditional mean is
    a straight scaling. ``zero_if_absent=False`` returns the conditional value unchanged, which is
    what a stat that voids rather than settling at zero would need.
    """
    if not zero_if_absent:
        return conditional_mean
    return conditional_mean * max(0.0, min(1.0, play_probability))


def inflate_variance_for_uncertainty(
    conditional_mean: float,
    conditional_variance: float,
    play_probability: float,
) -> float:
    """Variance of the unconditional outcome, treating absence as a hard zero.

    With ``p = P(played)`` and a mixture of (play, don't play):

        Var[X] = p * (var + mu^2) - (p * mu)^2

    A coin-flip Questionable therefore carries far more variance than a healthy player with the
    same median, which is exactly the uncertainty the p25-p75 band should show.
    """
    p = max(0.0, min(1.0, play_probability))
    second_moment = p * (conditional_variance + conditional_mean**2)
    return max(0.0, second_moment - (p * conditional_mean) ** 2)


def is_eligible(report_status: str | None) -> bool:
    """§4: exclude Out / IR / PUP / Doubtful. Questionable players stay on the board."""
    if not report_status:
        return True
    return report_status not in EXCLUDED_STATUSES


def redistribute_target_share(
    absent_share: float,
    remaining: dict[str, float],
    historical_without: dict[str, float] | None = None,
    default_split: tuple[float, float, float] = (0.60, 0.25, 0.15),
    default_order: tuple[str, str, str] = ("WR2", "TE", "RB"),
) -> dict[str, float]:
    """Redistribute an absent teammate's target share (§5.4).

    Preferred: use the remaining players' historical shares **in the games the absent player
    missed**, which captures the actual role change rather than an assumption. When no such games
    exist, fall back to the documented 60/25/15 split to WR2/TE/RB.

    Args:
        absent_share: the missing player's target share, e.g. 0.24.
        remaining: current share by role key for the players who are playing.
        historical_without: shares observed in games the absent player missed, if any.
        default_split: the fallback proportions (§5.4).
        default_order: which role keys the fallback proportions apply to.

    Returns:
        Updated shares by role key. The total added equals ``absent_share``.
    """
    if absent_share <= 0 or not remaining:
        return dict(remaining)

    out = dict(remaining)

    if historical_without:
        total = sum(v for v in historical_without.values() if v > 0)
        if total > 0:
            for key, share in historical_without.items():
                if key in out and share > 0:
                    out[key] += absent_share * (share / total)
            return out

    for key, weight in zip(default_order, default_split, strict=True):
        if key in out:
            out[key] += absent_share * weight

    # Any portion whose role key is absent from the roster spreads proportionally over who is left.
    assigned = sum(
        absent_share * w for k, w in zip(default_order, default_split, strict=True) if k in out
    )
    leftover = absent_share - assigned
    if leftover > 1e-9:
        total_current = sum(remaining.values())
        if total_current > 0:
            for key, share in remaining.items():
                out[key] += leftover * (share / total_current)

    return out


def practice_participation_signal(practice_status: str | None) -> str:
    """Normalise the several spellings nflverse and Sleeper use for practice participation."""
    if not practice_status:
        return "Unknown"
    s = practice_status.strip().lower()
    if s.startswith("full"):
        return "Full"
    if s.startswith("limited"):
        return "Limited"
    if "did not" in s or s in {"dnp", "out"}:
        return "DNP"
    return practice_status.strip()


def coin_flip_distance(p: float) -> float:
    """How informative a designation is: 0 at a coin flip, 1 at certainty.

    Used to sort the UI's 'most uncertain' list -- a Questionable at 0.50 deserves attention in a
    way a Questionable at 0.95 does not.
    """
    return abs(2.0 * max(0.0, min(1.0, p)) - 1.0)


def log_loss(p: float, played: bool) -> float:
    """Log loss of a single play-probability prediction. Lower is better. Used by the backtest."""
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -math.log(p) if played else -math.log(1 - p)
