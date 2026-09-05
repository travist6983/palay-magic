"""Tests for the LLM wrapper (§7): caching, the per-refresh cap, and graceful degradation."""

from __future__ import annotations

import json

import pytest

from backend.llm import client


@pytest.fixture(autouse=True)
def _reset_budget():
    client.reset_refresh_budget()
    yield
    client.reset_refresh_budget()


def test_cache_key_is_stable_under_key_order():
    a = client.cache_key("deep_dive", {"a": 1, "b": [2, 3]}, "m")
    b = client.cache_key("deep_dive", {"b": [2, 3], "a": 1}, "m")
    assert a == b


def test_cache_key_changes_with_task_model_and_input():
    base = client.cache_key("deep_dive", {"a": 1}, "m")
    assert client.cache_key("injury", {"a": 1}, "m") != base
    assert client.cache_key("deep_dive", {"a": 1}, "other") != base
    assert client.cache_key("deep_dive", {"a": 2}, "m") != base


def test_a_cached_response_needs_no_api_key(migrated_db, monkeypatch):
    """§7 caches aggressively: a repeated task must not spend a call or need a key."""
    payload = {"player": "Someone"}
    key = client.cache_key("deep_dive", payload, "claude-sonnet-4-6")
    client.write_cache(key, "deep_dive", "claude-sonnet-4-6", payload, {"summary": "hi"}, 10, 20)

    monkeypatch.setattr(client.get_settings(), "anthropic_api_key", None, raising=False)
    result = client.complete("deep_dive", "sys", "prompt", payload)

    assert result is not None
    assert result.cached
    assert result.data == {"summary": "hi"}
    assert client.calls_used() == 0


def test_missing_key_returns_none_rather_than_raising(migrated_db):
    """No key means no narrative, not a broken refresh (§10)."""
    assert client.complete("deep_dive", "sys", "prompt", {"x": 1}) is None


def test_per_refresh_cap_is_enforced(migrated_db, monkeypatch):
    """§7 caps calls per refresh; the counter must actually refuse."""
    settings = client.get_settings()
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key", raising=False)
    monkeypatch.setattr(settings, "llm_calls_per_refresh", 2, raising=False)

    calls = {"n": 0}

    class _FakeMessages:
        def create(self, **_kwargs):
            calls["n"] += 1
            raise RuntimeError("network down")

    class _FakeClient:
        def __init__(self, **_kwargs):
            self.messages = _FakeMessages()

    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)

    for i in range(5):
        client.complete("deep_dive", "sys", "prompt", {"i": i}, use_cache=False)

    # Two calls attempted, then the cap refuses without ever reaching the API.
    assert calls["n"] == 2
    assert client.calls_used() == 2


def test_reset_clears_the_counter(migrated_db):
    client._calls_this_refresh = 7
    client.reset_refresh_budget()
    assert client.calls_used() == 0


@pytest.mark.parametrize(
    "text",
    [
        '{"summary": "plain"}',
        '```json\n{"summary": "plain"}\n```',
        '```\n{"summary": "plain"}\n```',
        'Here you go:\n{"summary": "plain"}\nHope that helps.',
    ],
)
def test_json_is_extracted_from_the_usual_wrappers(text):
    """Asking for bare JSON usually works; a fence or a preamble is cheaper to parse than retry."""
    assert client._extract_json(text) == {"summary": "plain"}


def test_unparseable_response_returns_none(migrated_db, monkeypatch):
    settings = client.get_settings()
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key", raising=False)

    class _Block:
        type = "text"
        text = "I am afraid I cannot do that."

    class _Response:
        content = [_Block()]
        usage = None

    class _FakeMessages:
        def create(self, **_kwargs):
            return _Response()

    class _FakeClient:
        def __init__(self, **_kwargs):
            self.messages = _FakeMessages()

    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)
    assert client.complete("deep_dive", "sys", "prompt", {"x": 1}, use_cache=False) is None


def test_successful_call_is_cached_and_counted(migrated_db, monkeypatch):
    settings = client.get_settings()
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key", raising=False)

    class _Block:
        type = "text"
        text = '{"summary": "generated", "usage_risk": "low", "teammates_affected": []}'

    class _Usage:
        input_tokens = 120
        output_tokens = 45

    class _Response:
        content = [_Block()]
        usage = _Usage()

    class _FakeMessages:
        def create(self, **_kwargs):
            return _Response()

    class _FakeClient:
        def __init__(self, **_kwargs):
            self.messages = _FakeMessages()

    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)

    payload = {"player": "X"}
    first = client.complete("injury", "sys", "prompt", payload, use_cache=False)
    assert first is not None
    assert first.data["summary"] == "generated"
    assert first.input_tokens == 120
    assert not first.cached

    # Second call with the cache on must be free.
    second = client.complete("injury", "sys", "prompt", payload)
    assert second is not None
    assert second.cached
    assert client.calls_used() == 1


def test_prompts_forbid_inventing_numbers():
    """§7: the LLM must never produce a number that feeds a projection."""
    from backend.llm import prompts

    for system in (prompts.INJURY_SYSTEM, prompts.DEEP_DIVE_SYSTEM, prompts.SHOW_MATH_SYSTEM):
        assert "Never invent" in system
        assert "single JSON object" in system


def test_prompts_carry_the_settlement_rules():
    """A narrative that gets a settlement rule wrong is worse than no narrative."""
    from backend.llm import prompts

    for system in (prompts.INJURY_SYSTEM, prompts.DEEP_DIVE_SYSTEM, prompts.SHOW_MATH_SYSTEM):
        assert "NEVER counts toward his anytime-touchdown prop" in system
        assert "special teams excluded" in system


def test_write_cache_round_trips(migrated_db):
    key = "alias:test"
    client.write_cache(key, "deep_dive", "m", {"a": 1}, {"summary": "s"}, 1, 2)
    assert client.read_cache(key) == {"summary": "s"}
    assert json.dumps(client.read_cache(key))
