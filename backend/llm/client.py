"""Anthropic client wrapper (§7).

Claude is used only where language understanding is genuinely the job — reading an injury note,
explaining why a projection looks the way it does, turning intermediate values into prose.
**No number the LLM produces ever feeds a projection.** Every task returns JSON that is either
displayed as text or discarded; the modelling layer never reads it.

The wrapper enforces the three things §7 asks for and D8 requires:

* a **disk cache** keyed by a hash of the task and its canonical inputs, so a re-run costs nothing;
* a **per-refresh cap** of 150 calls, counted in DuckDB so it survives a restart;
* retries and graceful degradation — no key, no budget, or an API error all return ``None`` and the
  UI simply omits the note.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from typing import Any

from backend.config import get_settings
from backend.db.connection import connect
from backend.ingest.base import record_freshness
from backend.logging_setup import get_logger

log = get_logger(__name__)

_lock = threading.Lock()
_calls_this_refresh = 0


class LLMUnavailable(RuntimeError):
    """Raised internally when a call cannot be made. Callers see ``None`` instead."""


@dataclass(frozen=True)
class LLMResult:
    """One completed (or cached) LLM call."""

    task: str
    data: dict[str, Any]
    cached: bool
    input_tokens: int = 0
    output_tokens: int = 0


def cache_key(task: str, payload: dict[str, Any], model: str) -> str:
    """Stable key over the task, model and canonicalised input."""
    canonical = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(f"{task}|{model}|{canonical}".encode()).hexdigest()


def read_cache(key: str) -> dict[str, Any] | None:
    """Return a cached response, or None."""
    try:
        with connect() as con:
            row = con.execute(
                "SELECT output_json FROM llm_cache WHERE cache_key = ?", [key]
            ).fetchone()
    except Exception:  # noqa: BLE001 - a cache miss must never break a refresh
        log.exception("llm cache read failed")
        return None
    if not row or not row[0]:
        return None
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return None


def write_cache(
    key: str, task: str, model: str, payload: dict[str, Any], output: dict[str, Any],
    input_tokens: int, output_tokens: int,
) -> None:
    """Persist a response. Failures are logged and ignored."""
    try:
        with connect() as con:
            con.execute(
                "INSERT INTO llm_cache "
                "(cache_key, task, model, input_json, output_json, input_tokens, output_tokens, "
                " created_at) VALUES (?, ?, ?, ?, ?, ?, ?, now()) "
                "ON CONFLICT (cache_key) DO UPDATE SET output_json = excluded.output_json, "
                "  input_tokens = excluded.input_tokens, output_tokens = excluded.output_tokens, "
                "  created_at = now()",
                [
                    key, task, model,
                    json.dumps(payload, default=str),
                    json.dumps(output, default=str),
                    input_tokens, output_tokens,
                ],
            )
    except Exception:  # noqa: BLE001
        log.exception("llm cache write failed")


def reset_refresh_budget() -> None:
    """Zero the per-refresh call counter. Called at the start of every refresh (§7)."""
    global _calls_this_refresh
    with _lock:
        _calls_this_refresh = 0


def calls_used() -> int:
    return _calls_this_refresh


def _take_budget() -> None:
    """Reserve one call against the per-refresh cap, or raise."""
    global _calls_this_refresh
    cap = get_settings().llm_calls_per_refresh
    with _lock:
        if _calls_this_refresh >= cap:
            raise LLMUnavailable(f"per-refresh LLM cap of {cap} calls reached")
        _calls_this_refresh += 1


def _extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response.

    We ask for bare JSON and usually get it, but a fenced block or a leading sentence is a normal
    failure mode and is cheaper to parse than to retry.
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


def complete(
    task: str,
    system: str,
    prompt: str,
    payload: dict[str, Any],
    max_tokens: int | None = None,
    use_cache: bool = True,
) -> LLMResult | None:
    """Run one LLM task and return parsed JSON, or None if it could not run.

    Args:
        task: task name, used in the cache key and for token accounting.
        system: system prompt for this task.
        prompt: the user message.
        payload: the canonical inputs, hashed for the cache key. Must be JSON-serialisable.
        max_tokens: response cap; defaults to the configured value.
        use_cache: set False to force a fresh call.

    Returns:
        An :class:`LLMResult`, or None when there is no key, no budget, or the call failed.
    """
    settings = get_settings()
    model = settings.anthropic_model
    key = cache_key(task, payload, model)

    if use_cache:
        cached = read_cache(key)
        if cached is not None:
            return LLMResult(task=task, data=cached, cached=True)

    if not settings.has_anthropic:
        log.debug("ANTHROPIC_API_KEY not set; skipping %s", task)
        return None

    try:
        _take_budget()
    except LLMUnavailable as exc:
        log.warning("%s", exc)
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key, max_retries=3)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens or settings.anthropic_max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:  # noqa: BLE001 - the app must render without narratives
        log.warning("LLM task %s failed: %s: %s", task, type(exc).__name__, exc)
        record_freshness("anthropic", ok=False, detail=f"{type(exc).__name__}: {exc}")
        return None

    text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
    try:
        data = _extract_json(text)
    except json.JSONDecodeError:
        log.warning("LLM task %s returned unparseable JSON: %.200s", task, text)
        return None

    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

    write_cache(key, task, model, payload, data, input_tokens, output_tokens)
    record_freshness("anthropic", ok=True, detail=f"{task} ok")
    log.info(
        "LLM %s: %d in / %d out tokens (%d/%d calls this refresh)",
        task, input_tokens, output_tokens, _calls_this_refresh,
        settings.llm_calls_per_refresh,
    )
    return LLMResult(task, data, cached=False, input_tokens=input_tokens, output_tokens=output_tokens)


def token_usage() -> dict[str, Any]:
    """Cumulative token usage from the cache, for the CLI's ``llm`` report."""
    with connect() as con:
        rows = con.execute(
            "SELECT task, count(*), sum(input_tokens), sum(output_tokens) "
            "FROM llm_cache GROUP BY 1 ORDER BY 2 DESC"
        ).fetchall()
    return {
        "by_task": [
            {"task": t, "calls": int(n), "input_tokens": int(i or 0), "output_tokens": int(o or 0)}
            for t, n, i, o in rows
        ],
        "calls_this_refresh": _calls_this_refresh,
        "cap": get_settings().llm_calls_per_refresh,
    }
