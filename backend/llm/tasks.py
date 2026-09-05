"""The three LLM tasks (§7): injury context, deep-dive narrative, and the math explainer.

Each builds a compact payload from data PropLab already computed, calls
:func:`backend.llm.client.complete`, and returns parsed JSON or ``None``. Nothing here feeds a
number back into the model.
"""

from __future__ import annotations

import json
from typing import Any

from backend.db.connection import connect
from backend.llm.client import complete, reset_refresh_budget
from backend.llm.prompts import DEEP_DIVE_SYSTEM, INJURY_SYSTEM, SHOW_MATH_SYSTEM
from backend.logging_setup import get_logger

log = get_logger(__name__)


def injury_note(gsis_id: str, season: int, week: int) -> dict[str, Any] | None:
    """Two or three sentences on what a designation means for usage (§7 task 1)."""
    with connect() as con:
        row = con.execute(
            """
            SELECT p.display_name, p.position, r.team, r.opponent, r.play_probability,
                   r.components, i.injury_status, i.practice_participation, i.injury_body_part,
                   i.injury_notes, i.depth_chart_order
            FROM rankings r
            JOIN players p USING (gsis_id)
            LEFT JOIN injury_status i
                   ON i.gsis_id = r.gsis_id AND i.source = 'sleeper'
            WHERE r.gsis_id = ? AND r.season = ? AND r.week = ?
            """,
            [gsis_id, season, week],
        ).fetchone()
        if not row or not row[6]:
            return None  # healthy players get no note

        teammates = con.execute(
            """
            SELECT p.display_name, r.rank, r.injury_status
            FROM rankings r JOIN players p USING (gsis_id)
            WHERE r.season = ? AND r.week = ? AND r.team = ? AND r.gsis_id <> ?
            ORDER BY r.position, r.rank LIMIT 12
            """,
            [season, week, row[2], gsis_id],
        ).fetchall()

    components = json.loads(row[5]) if row[5] else {}
    payload = {
        "player": row[0],
        "position": row[1],
        "team": row[2],
        "opponent": row[3],
        "designation": row[6],
        "practice_participation": row[7] or "no practice report published yet",
        "body_part": row[8],
        "notes": row[9],
        "depth_chart_order": row[10],
        "play_probability": round(float(row[4] or 1.0), 3),
        "expected_snap_share_if_plays": components.get("prior_snap_share"),
        "role": components.get("role"),
        "play_probability_source": components.get("play_probability_source"),
        "teammates_on_the_board": [
            {"name": t[0], "rank": t[1], "designation": t[2]} for t in teammates
        ],
    }

    prompt = (
        "Here is everything known about this player's status and his team's board.\n\n"
        + json.dumps(payload, indent=2)
        + "\n\nWrite the injury note."
    )
    result = complete("injury", INJURY_SYSTEM, prompt, payload, max_tokens=600)
    return result.data if result else None


def deep_dive_note(gsis_id: str, season: int, week: int) -> dict[str, Any] | None:
    """Why the projection looks the way it does (§7 task 2). Cached per (player, week)."""
    from backend.app import queries

    detail = queries.player_detail(gsis_id, season, week)
    if not detail:
        return None

    payload = {
        "player": detail["display_name"],
        "position": detail["position"],
        "team": detail["team"],
        "opponent": detail["opponent"],
        "week": f"{season} week {week}",
        "environment": {
            k: _round(v)
            for k, v in detail["environment"].items()
            if v is not None and k not in {"odds_source"}
        },
        "flags": detail["flags"],
        "injury": {
            "designation": detail["injury"]["report_status"],
            "play_probability": _round(detail["injury"]["p_played"]),
            "role": detail["injury"]["role"],
        },
        "projections": [
            {
                "stat": p["label"],
                "median": _round(p["distribution"]["median"]),
                "p25": _round(p["distribution"]["p25"]),
                "p75": _round(p["distribution"]["p75"]),
                "high_variance": p["high_variance"],
            }
            for p in detail["projections"]
        ],
        "recent_games": [
            {
                "when": f"{g['season']} wk {g['week']}",
                "opponent": g["opponent"],
                "prior_season": g["prior_season"],
                "stats": {
                    c["stat"]: {
                        "raw": _round(c["raw_value"]),
                        "opponent_adjusted": _round(c["adjusted_value"]),
                        "opponent_rank_1_is_softest": c["opponent_rank"],
                    }
                    for c in g["cells"].values()
                },
            }
            for g in detail["game_log"]
        ],
        "matchup": [
            {
                "metric": m["label"],
                "multiplier": _round(m["multiplier"]),
                "rank_1_is_softest": m["rank"],
                "measured_mse_reduction_pct": _round(m["mse_reduction_pct"]),
            }
            for m in detail["matchup"]
        ],
    }

    prompt = (
        "Here is the full projection and the game log behind it.\n\n"
        + json.dumps(payload, indent=2)
        + "\n\nWrite the deep-dive note."
    )
    result = complete("deep_dive", DEEP_DIVE_SYSTEM, prompt, payload, max_tokens=900)
    if not result:
        return None

    # Store under a stable key too, so the API can find it without recomputing the payload hash.
    _store_alias(f"deep_dive:{gsis_id}:{season}:{week}", "deep_dive", payload, result.data)
    return result.data


def show_math_note(gsis_id: str, stat: str, season: int, week: int) -> dict[str, Any] | None:
    """Turn the stored intermediate values into a readable step list (§7 task 3)."""
    with connect() as con:
        steps = con.execute(
            "SELECT section, label, value, detail FROM projection_math "
            "WHERE gsis_id = ? AND stat = ? AND season = ? AND week = ? ORDER BY step",
            [gsis_id, stat, season, week],
        ).fetchall()
        name = con.execute("SELECT display_name FROM players WHERE gsis_id = ?", [gsis_id]).fetchone()

    if not steps:
        return None

    payload = {
        "player": name[0] if name else gsis_id,
        "stat": stat,
        "steps": [
            {"section": s[0], "label": s[1], "value": _round(s[2]), "detail": s[3]} for s in steps
        ],
    }
    prompt = (
        "Here are the intermediate values that produced this projection, in order.\n\n"
        + json.dumps(payload, indent=2)
        + "\n\nNarrate them."
    )
    result = complete("show_math", SHOW_MATH_SYSTEM, prompt, payload, max_tokens=1200)
    if not result:
        return None
    _store_alias(f"show_math:{gsis_id}:{stat}:{season}:{week}", "show_math", payload, result.data)
    return result.data


def generate_week_notes(
    season: int, week: int, include_deep_dives: bool = True
) -> dict[str, int]:
    """Generate every note for the week, inside the per-refresh cap (§7).

    Injury notes come first because they are the ones that change a decision; deep dives fill the
    remaining budget. Returns counts of what was produced.
    """
    from backend.config import get_settings

    reset_refresh_budget()
    settings = get_settings()
    if not settings.has_anthropic:
        log.info("ANTHROPIC_API_KEY not set; skipping narrative generation")
        return {"injury": 0, "deep_dive": 0, "skipped": 1}

    with connect() as con:
        flagged = [
            r[0]
            for r in con.execute(
                "SELECT gsis_id FROM rankings WHERE season = ? AND week = ? "
                "AND injury_status IS NOT NULL ORDER BY position, rank",
                [season, week],
            ).fetchall()
        ]
        ranked = [
            r[0]
            for r in con.execute(
                "SELECT gsis_id FROM rankings WHERE season = ? AND week = ? ORDER BY position, rank",
                [season, week],
            ).fetchall()
        ]

    counts = {"injury": 0, "deep_dive": 0, "skipped": 0}
    for gsis in flagged:
        if injury_note(gsis, season, week):
            counts["injury"] += 1

    if include_deep_dives:
        for gsis in ranked:
            if deep_dive_note(gsis, season, week):
                counts["deep_dive"] += 1

    log.info("LLM notes for %s week %s: %s", season, week, counts)
    return counts


def _round(value: Any, digits: int = 3) -> Any:
    """Round floats for the payload so the cache key is stable across trivial recomputation."""
    if isinstance(value, float):
        return round(value, digits)
    return value


def _store_alias(key: str, task: str, payload: dict[str, Any], data: dict[str, Any]) -> None:
    """Store a second cache row under a human-readable key, for the API to read directly."""
    from backend.config import get_settings
    from backend.llm.client import write_cache

    write_cache(key, task, get_settings().anthropic_model, payload, data, 0, 0)
