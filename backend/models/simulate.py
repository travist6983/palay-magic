"""Drive-level Monte Carlo joint simulation (§5.8).

The analytic projections are the source of truth for every **marginal** — they are what the §6
backtest validated. What a marginal cannot express is **correlation**: a quarterback's passing
yards and his top receiver's receiving yards rise together, a running back's carries fall when his
team is throwing to catch up, and a kicker's points depend on the same drives that produce the
touchdowns. Books price that with full-game Monte Carlo, which is why a same-game parlay pays far
less than naive multiplication (``how_books_build_lines.md`` step 5).

So this module simulates the game, then **transforms each player's simulated series so its marginal
matches the analytic distribution exactly**, rank for rank. The result keeps the dependence
structure the simulation discovered and inherits the calibration the analytic model earned. The
simulation can add information; by construction it cannot degrade a marginal.

The engine itself is deliberately simple, because a more elaborate one would need validation this
project has not done:

* a game is a sequence of alternating drives, count drawn around the environment model's estimate;
* a drive is a sequence of plays, pass or run by the team's expected pass rate, shifted by the
  live score margin so a trailing team throws;
* each play is assigned to a player by his usage share;
* yards come from that player's own empirical per-play distribution;
* a drive ends in a touchdown, a field goal, or nothing, with the red-zone conversion rate the
  environment model implies.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from backend.db.connection import connect
from backend.logging_setup import get_logger
from backend.models.stats import Position

log = get_logger(__name__)

DEFAULT_SIMS = 5000
STORED_DRAWS = 1000
"""How many draws per player-stat are persisted. Enough to price a parlay, small enough to store."""

# Yards needed for a field-goal attempt to be plausible, from the offence's own 35.
FG_RANGE_YARDLINE = 38.0


@dataclass
class PlayerSlot:
    """One player's usage profile inside the simulation."""

    gsis_id: str
    position: str
    team: str
    target_share: float = 0.0
    carry_share: float = 0.0
    gl_carry_share: float = 0.0
    rz_target_share: float = 0.0
    is_qb: bool = False
    is_kicker: bool = False
    catch_rate: float = 0.65
    yards_per_target: float = 8.0
    yards_per_carry: float = 4.3
    tackle_share: float = 0.0
    stats: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class GameSim:
    """Simulated outcomes for one game."""

    game_id: str
    home: str
    away: str
    n_sims: int
    players: list[PlayerSlot]


def _load_game_inputs(season: int, week: int) -> tuple[pl.DataFrame, dict[str, list[PlayerSlot]]]:
    """The week's games and, per team, the ranked players with their usage profiles."""
    with connect() as con:
        games = con.execute(
            "SELECT game_id, home_team, away_team FROM game_environment "
            "WHERE season = ? AND week = ? ORDER BY game_id",
            [season, week],
        ).pl()

        ranked = con.execute(
            """
            SELECT r.gsis_id, r.position, r.team
            FROM rankings r WHERE r.season = ? AND r.week = ?
            """,
            [season, week],
        ).pl()

        if ranked.is_empty():
            return games, {}

        ids = ranked["gsis_id"].to_list()
        usage = con.execute(
            f"""
            SELECT gsis_id,
                   avg(target_share)    AS target_share,
                   avg(carry_share)     AS carry_share,
                   avg(gl_carry_share)  AS gl_carry_share,
                   avg(rz_target_share) AS rz_target_share,
                   avg(tackle_share)    AS tackle_share
            FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY gsis_id ORDER BY season DESC, week DESC
                ) AS rn
                FROM player_game_usage
                WHERE gsis_id IN ({",".join("?" * len(ids))})
                  AND ((season < ?) OR (season = ? AND week < ?))
            ) WHERE rn <= 6 GROUP BY 1
            """,
            [*ids, season, season, week],
        ).pl()

        rates = con.execute(
            f"""
            SELECT gsis_id,
                   sum(CASE WHEN stat = 'receptions' THEN value END)
                     / nullif(sum(CASE WHEN stat = 'targets' THEN value END), 0) AS catch_rate,
                   sum(CASE WHEN stat = 'receiving_yards' THEN value END)
                     / nullif(sum(CASE WHEN stat = 'targets' THEN value END), 0) AS ypt,
                   sum(CASE WHEN stat = 'rushing_yards' THEN value END)
                     / nullif(sum(CASE WHEN stat = 'rush_attempts' THEN value END), 0) AS ypc
            FROM player_game_stats
            WHERE gsis_id IN ({",".join("?" * len(ids))})
              AND ((season < ?) OR (season = ? AND week < ?))
              AND stat IN ('receptions','targets','receiving_yards','rush_attempts','rushing_yards')
            GROUP BY 1
            """,
            [*ids, season, season, week],
        ).pl()

    usage_map = {r["gsis_id"]: r for r in usage.to_dicts()}
    rate_map = {r["gsis_id"]: r for r in rates.to_dicts()}

    by_team: dict[str, list[PlayerSlot]] = {}
    for r in ranked.to_dicts():
        u = usage_map.get(r["gsis_id"], {})
        rate = rate_map.get(r["gsis_id"], {})
        slot = PlayerSlot(
            gsis_id=r["gsis_id"],
            position=r["position"],
            team=r["team"],
            target_share=float(u.get("target_share") or 0.0),
            carry_share=float(u.get("carry_share") or 0.0),
            gl_carry_share=float(u.get("gl_carry_share") or 0.0),
            rz_target_share=float(u.get("rz_target_share") or 0.0),
            tackle_share=float(u.get("tackle_share") or 0.0),
            is_qb=r["position"] == Position.QB.value,
            is_kicker=r["position"] == Position.K.value,
            catch_rate=float(rate.get("catch_rate") or 0.65),
            yards_per_target=float(rate.get("ypt") or 8.0),
            yards_per_carry=float(rate.get("ypc") or 4.3),
        )
        by_team.setdefault(r["team"], []).append(slot)

    # Every team needs a "field" slot absorbing the usage of everyone NOT on the board.
    # Without it the ten ranked players share 100% of the targets and carries between them, and a
    # WR1 catches a fixed fraction of every single pass -- which drove the simulated correlation
    # between a quarterback's attempts and his top receiver's targets to 0.99, against a real-world
    # value nearer 0.5. The residual player is never scored or stored; he exists so the ranked
    # players get their true share.
    for team, slots in by_team.items():
        field = PlayerSlot(
            gsis_id=f"__field__{team}",
            position="FIELD",
            team=team,
            target_share=max(0.0, 1.0 - sum(s.target_share for s in slots)),
            carry_share=max(0.0, 1.0 - sum(s.carry_share for s in slots)),
            gl_carry_share=max(0.0, 1.0 - sum(s.gl_carry_share for s in slots)),
            rz_target_share=max(0.0, 1.0 - sum(s.rz_target_share for s in slots)),
            tackle_share=max(0.0, 1.0 - sum(s.tackle_share for s in slots)),
            catch_rate=0.64,
            yards_per_target=7.6,
            yards_per_carry=4.2,
        )
        slots.append(field)

    return games, by_team


def _team_environment(season: int, week: int) -> dict[str, dict]:
    with connect() as con:
        rows = con.execute(
            "SELECT * FROM team_environment WHERE season = ? AND week = ?", [season, week]
        ).pl()
    return {r["team"]: r for r in rows.to_dicts()}


def simulate_game(
    game_id: str,
    home: str,
    away: str,
    env: dict[str, dict],
    players: dict[str, list[PlayerSlot]],
    n_sims: int = DEFAULT_SIMS,
    rng: np.random.Generator | None = None,
) -> GameSim | None:
    """Simulate one game ``n_sims`` times at the drive level (§5.8).

    Both teams are simulated together against a shared clock and score, which is where the
    correlation comes from: a team that falls behind throws more, which lifts its receivers and
    suppresses its back, and lifts the opposing defence's tackle counts.
    """
    rng = rng or np.random.default_rng(0)
    home_env, away_env = env.get(home), env.get(away)
    if not home_env or not away_env:
        return None

    slots = [*players.get(home, []), *players.get(away, [])]
    if not slots:
        return None

    stat_keys = (
        "pass_attempts", "completions", "passing_yards", "passing_tds", "rush_attempts",
        "rushing_yards", "targets", "receptions", "receiving_yards", "anytime_td",
        "fg_attempts", "fg_made", "xp_made", "kicking_points", "tackles_assists",
    )
    for slot in slots:
        slot.stats = {k: np.zeros(n_sims) for k in stat_keys}

    by_team = {home: players.get(home, []), away: players.get(away, [])}
    envs = {home: home_env, away: away_env}

    for sim in range(n_sims):
        score = {home: 0.0, away: 0.0}
        # Drive counts have a SHARED component (both teams get the same number of possessions in a
        # fast game) and an independent one. Making the whole thing shared would manufacture a
        # cross-team correlation far stronger than reality -- an opposing kicker's extra points
        # should not track a receiver's touchdowns at 0.66.
        game_pace = rng.normal(1.0, 0.06)
        drives = {
            t: max(
                6,
                int(round((envs[t]["expected_drives"] or 11.0) * game_pace * rng.normal(1.0, 0.09))),
            )
            for t in (home, away)
        }

        for drive_no in range(max(drives.values())):
            for team in (away, home):  # away receives first, arbitrarily but consistently
                if drive_no >= drives[team]:
                    continue
                opponent = home if team == away else away
                _simulate_drive(
                    team, opponent, envs[team], by_team[team], by_team[opponent],
                    score, sim, rng,
                )

    return GameSim(game_id=game_id, home=home, away=away, n_sims=n_sims, players=slots)


def _simulate_drive(
    team: str,
    opponent: str,
    env: dict,
    offense: list[PlayerSlot],
    defense: list[PlayerSlot],
    score: dict[str, float],
    sim: int,
    rng: np.random.Generator,
) -> None:
    """One drive: plays until it stalls, scores, or turns the ball over."""
    yardline = 75.0  # yards from the opponent's end zone
    base_pass_rate = float(env["expected_pass_rate"] or 0.57)

    # Game script: a trailing team throws. The coefficient is the one the environment model
    # measured on the spread, re-expressed per point of live margin.
    margin = score[team] - score[opponent]
    pass_rate = float(np.clip(base_pass_rate - 0.0038 * margin, 0.25, 0.85))

    qbs = [p for p in offense if p.is_qb]
    receivers = [p for p in offense if p.target_share > 0]
    runners = [p for p in offense if p.carry_share > 0]
    kickers = [p for p in offense if p.is_kicker]
    tacklers = [p for p in defense if p.tackle_share > 0]

    for _ in range(12):  # a drive is at most a dozen plays before it ends one way or another
        is_pass = rng.random() < pass_rate
        gain = 0.0

        if is_pass and qbs:
            qb = qbs[0]
            qb.stats["pass_attempts"][sim] += 1
            receiver = _pick(receivers, "target_share", rng)
            if receiver is not None:
                receiver.stats["targets"][sim] += 1
                if rng.random() < receiver.catch_rate:
                    gain = max(0.0, rng.exponential(max(receiver.yards_per_target / receiver.catch_rate, 1.0)))
                    receiver.stats["receptions"][sim] += 1
                    receiver.stats["receiving_yards"][sim] += gain
                    qb.stats["completions"][sim] += 1
                    qb.stats["passing_yards"][sim] += gain
        elif runners:
            runner = _pick(runners, "carry_share", rng)
            if runner is not None:
                runner.stats["rush_attempts"][sim] += 1
                gain = max(-3.0, rng.exponential(max(runner.yards_per_carry, 1.0)) - 1.0)
                runner.stats["rushing_yards"][sim] += gain

        # Whoever made the tackle: only every-down defenders are on the board, so shares are
        # renormalised across them.
        tackler = _pick(tacklers, "tackle_share", rng)
        if tackler is not None and rng.random() < 0.55:
            tackler.stats["tackles_assists"][sim] += 1

        yardline -= gain
        if yardline <= 0:
            _score_touchdown(team, offense, qbs, receivers, runners, kickers, score, sim, rng, is_pass)
            return
        if rng.random() < 0.12:  # turnover or a drive-ending sack
            return
        if gain < 3.0 and rng.random() < 0.42:  # stalled
            break

    if kickers and yardline <= FG_RANGE_YARDLINE:
        kicker = kickers[0]
        kicker.stats["fg_attempts"][sim] += 1
        distance = yardline + 17.0
        if rng.random() < _fg_probability(distance):
            kicker.stats["fg_made"][sim] += 1
            kicker.stats["kicking_points"][sim] += 3
            score[team] += 3


def _score_touchdown(
    team: str,
    offense: list[PlayerSlot],
    qbs: list[PlayerSlot],
    receivers: list[PlayerSlot],
    runners: list[PlayerSlot],
    kickers: list[PlayerSlot],
    score: dict[str, float],
    sim: int,
    rng: np.random.Generator,
    was_pass: bool,
) -> None:
    """Credit a touchdown, using goal-line and red-zone roles rather than total volume (§5.6)."""
    score[team] += 6
    if was_pass and qbs and receivers:
        qbs[0].stats["passing_tds"][sim] += 1
        scorer = _pick(receivers, "rz_target_share", rng) or _pick(receivers, "target_share", rng)
    else:
        scorer = _pick(runners, "gl_carry_share", rng) or _pick(runners, "carry_share", rng)

    if scorer is not None:
        scorer.stats["anytime_td"][sim] = 1.0

    if kickers and rng.random() < 0.94:
        kickers[0].stats["xp_made"][sim] += 1
        kickers[0].stats["kicking_points"][sim] += 1
        score[team] += 1


def _pick(slots: list[PlayerSlot], attribute: str, rng: np.random.Generator) -> PlayerSlot | None:
    """Choose a player in proportion to a usage share. None when nobody on the board qualifies."""
    if not slots:
        return None
    weights = np.array([max(getattr(s, attribute, 0.0), 0.0) for s in slots])
    total = weights.sum()
    if total <= 0:
        return None
    return slots[int(rng.choice(len(slots), p=weights / total))]


def _fg_probability(distance: float) -> float:
    """Distance-only make probability, matching the fitted model closely enough for the sim."""
    return float(1.0 / (1.0 + np.exp(-(5.6 - 0.098 * distance))))


# ---------------------------------------------------------------------------
# Marginal-preserving transform
# ---------------------------------------------------------------------------


def match_marginal(
    simulated: np.ndarray,
    target_quantiles: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Rewrite a simulated series so its marginal is the analytic one, rank for rank.

    This is the step that makes the simulation safe to ship. The analytic distributions are the
    ones §6 validated; the simulation's own marginals are not validated and should not be trusted.
    Replacing the simulated values with the analytic quantiles at the same ranks keeps the
    dependence structure (which is the only thing we wanted from the simulation) and leaves every
    marginal exactly as calibrated as before.

    **Ties are broken at random, not by array index.** Half the stats here are counts and one is a
    straight yes/no, so a simulated series is mostly ties. A stable sort resolves those in index
    order, which assigns every binary stat's 1s to the same simulation indices and manufactures a
    correlation of 0.67 between a kicker on one team and a receiver on the other. Randomising the
    tie order removes the artefact and leaves only the dependence the simulation actually produced.

    Args:
        simulated: the raw simulated draws.
        target_quantiles: the analytic distribution's values at evenly spaced quantiles, same length.
        rng: source of randomness for tie-breaking.

    Returns:
        A reordered copy of ``target_quantiles`` carrying ``simulated``'s rank ordering.
    """
    if simulated.size == 0 or target_quantiles.size == 0:
        return simulated
    rng = rng or np.random.default_rng(0)
    jitter = rng.random(simulated.size)
    order = np.argsort(np.lexsort((jitter, simulated)), kind="stable")
    sorted_targets = np.sort(target_quantiles)
    idx = np.clip(
        (order * (sorted_targets.size - 1) / max(order.max(), 1)).round().astype(int),
        0, sorted_targets.size - 1,
    )
    return sorted_targets[idx]


def _analytic_quantiles(season: int, week: int, n: int) -> dict[tuple[str, str], np.ndarray]:
    """For each stored projection, its distribution evaluated at ``n`` evenly spaced quantiles."""
    import json

    from backend.models.distributions import Distribution
    from backend.models.stats import Family

    with connect() as con:
        rows = con.execute(
            "SELECT gsis_id, stat, dist_family, params, mean FROM projections "
            "WHERE season = ? AND week = ? AND conditional_on_playing = TRUE",
            [season, week],
        ).fetchall()

    probs = (np.arange(n) + 0.5) / n
    out: dict[tuple[str, str], np.ndarray] = {}
    for gsis, stat, family, params, mean in rows:
        dist = Distribution(Family(family), json.loads(params), mean, family != "empirical_max")
        try:
            out[(gsis, stat)] = np.array([dist.quantile(float(q)) for q in probs])
        except Exception:  # noqa: BLE001 - one bad distribution must not lose the run
            log.debug("could not build quantiles for %s/%s", gsis, stat)
    return out


def run_simulation(
    season: int,
    week: int,
    n_sims: int = DEFAULT_SIMS,
    seed: int = 20260905,
    store_draws: bool = True,
) -> dict[str, Any]:
    """Simulate every game in the week and persist correlations and thinned draws (§5.8).

    Returns:
        A summary dict with the run id, counts and timing.
    """
    t0 = time.perf_counter()
    run_id = uuid.uuid4().hex[:12]
    rng = np.random.default_rng(seed)

    games, players = _load_game_inputs(season, week)
    env = _team_environment(season, week)
    if games.is_empty() or not players or not env:
        log.warning("nothing to simulate for %s week %s", season, week)
        return {"run_id": None, "games": 0}

    targets = _analytic_quantiles(season, week, n_sims)

    corr_rows: list[dict[str, Any]] = []
    draw_rows: list[dict[str, Any]] = []
    n_players = 0

    for g in games.to_dicts():
        sim = simulate_game(
            g["game_id"], g["home_team"], g["away_team"], env, players, n_sims, rng
        )
        if sim is None:
            continue

        # Transform each series to its analytic marginal before anything is read off it.
        series: dict[tuple[str, str], np.ndarray] = {}
        for slot in sim.players:
            if slot.position == "FIELD":
                continue
            n_players += 1
            for stat, values in slot.stats.items():
                target = targets.get((slot.gsis_id, stat))
                if target is None or values.std() == 0:
                    continue
                series[(slot.gsis_id, stat)] = match_marginal(values, target, rng)

        keys = sorted(series)
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                if a[0] == b[0]:
                    continue  # a player against himself is not interesting here
                corr = float(np.corrcoef(series[a], series[b])[0, 1])
                if not np.isfinite(corr) or abs(corr) < 0.03:
                    continue
                team_a = next(s.team for s in sim.players if s.gsis_id == a[0])
                team_b = next(s.team for s in sim.players if s.gsis_id == b[0])
                corr_rows.append(
                    {
                        "run_id": run_id, "season": season, "week": week, "game_id": sim.game_id,
                        "gsis_id_a": a[0], "stat_a": a[1], "gsis_id_b": b[0], "stat_b": b[1],
                        "correlation": corr, "same_team": team_a == team_b,
                    }
                )

        if store_draws:
            keep = min(STORED_DRAWS, n_sims)
            step = max(1, n_sims // keep)
            for (gsis, stat), values in series.items():
                for j, idx in enumerate(range(0, n_sims, step)):
                    if j >= keep:
                        break
                    draw_rows.append(
                        {
                            "run_id": run_id, "season": season, "week": week,
                            "game_id": sim.game_id, "gsis_id": gsis, "stat": stat,
                            "sim_index": j, "value": float(values[idx]),
                        }
                    )

    seconds = time.perf_counter() - t0
    with connect() as con:
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute(
                "INSERT INTO simulation_runs "
                "(run_id, season, week, n_sims, n_games, n_players, seconds, seed, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, now())",
                [run_id, season, week, n_sims, games.height, n_players, seconds, seed],
            )
            if corr_rows:
                con.register("corr_df", pl.DataFrame(corr_rows))
                con.execute(
                    "INSERT INTO simulation_correlations SELECT run_id, season, week, game_id, "
                    "gsis_id_a, stat_a, gsis_id_b, stat_b, correlation, same_team FROM corr_df"
                )
            if draw_rows:
                con.register("draw_df", pl.DataFrame(draw_rows))
                con.execute(
                    "INSERT INTO simulation_draws SELECT run_id, season, week, game_id, gsis_id, "
                    "stat, sim_index, value FROM draw_df"
                )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            for name in ("corr_df", "draw_df"):
                try:
                    con.unregister(name)
                except Exception:  # noqa: BLE001, S110 - unregistering an absent view is fine
                    pass

    log.info(
        "simulation %s: %d games x %d sims -> %d correlations, %d draws, %.1fs",
        run_id, games.height, n_sims, len(corr_rows), len(draw_rows), seconds,
    )
    return {
        "run_id": run_id, "games": games.height, "sims": n_sims,
        "correlations": len(corr_rows), "draws": len(draw_rows), "seconds": seconds,
    }


def top_correlations(season: int, week: int, limit: int = 25) -> pl.DataFrame:
    """The strongest simulated relationships this week. The raw material for a parlay read."""
    with connect() as con:
        return con.execute(
            """
            SELECT c.game_id, pa.display_name AS player_a, c.stat_a,
                   pb.display_name AS player_b, c.stat_b, c.correlation, c.same_team
            FROM simulation_correlations c
            LEFT JOIN players pa ON pa.gsis_id = c.gsis_id_a
            LEFT JOIN players pb ON pb.gsis_id = c.gsis_id_b
            WHERE c.season = ? AND c.week = ?
              AND c.run_id = (SELECT run_id FROM simulation_runs
                              WHERE season = ? AND week = ? ORDER BY created_at DESC LIMIT 1)
            ORDER BY abs(c.correlation) DESC
            LIMIT ?
            """,
            [season, week, season, week, limit],
        ).pl()


def joint_probability(
    season: int, week: int, legs: list[tuple[str, str, float, str]]
) -> dict[str, Any]:
    """Price a multi-leg ticket off the simulated draws.

    Each leg is ``(gsis_id, stat, line, 'over' | 'under')``. Returns the joint probability, the
    naive independent product, and the ratio between them — which is exactly the correlation a book
    is charging for on a same-game parlay.
    """
    if not legs:
        return {"joint": None, "independent": None, "correlation_multiple": None}

    with connect() as con:
        run = con.execute(
            "SELECT run_id FROM simulation_runs WHERE season = ? AND week = ? "
            "ORDER BY created_at DESC LIMIT 1",
            [season, week],
        ).fetchone()
        if not run:
            return {"error": "no simulation for this week; run `proplab simulate`"}

        arrays = []
        for gsis, stat, line, side in legs:
            rows = con.execute(
                "SELECT sim_index, value FROM simulation_draws "
                "WHERE run_id = ? AND gsis_id = ? AND stat = ? ORDER BY sim_index",
                [run[0], gsis, stat],
            ).fetchall()
            if not rows:
                return {"error": f"no simulated draws for {gsis} / {stat}"}
            values = np.array([r[1] for r in rows])
            arrays.append(values > line if side == "over" else values < line)

    stacked = np.vstack(arrays)
    joint = float(stacked.all(axis=0).mean())
    independent = float(np.prod([a.mean() for a in arrays]))
    return {
        "joint": joint,
        "independent": independent,
        "correlation_multiple": joint / independent if independent > 0 else None,
        "legs": [
            {"gsis_id": g, "stat": s, "line": ln, "side": sd, "leg_probability": float(a.mean())}
            for (g, s, ln, sd), a in zip(legs, arrays, strict=True)
        ],
    }
