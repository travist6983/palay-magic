"""Tests for the ingest plumbing: budgets, atomic writes, and graceful degradation (D8)."""

from __future__ import annotations

import polars as pl
import pytest

from backend.ingest.base import IngestResult, resilient, write_parquet_atomic
from backend.ingest.budget import BudgetExceeded, check, consume, get_state


def test_budget_starts_empty_and_counts_up(migrated_db):
    state = get_state("test-api", budget=3)
    assert state.n_calls == 0
    assert state.remaining == 3

    consume("test-api", budget=3)
    consume("test-api", budget=3)
    assert get_state("test-api", budget=3).n_calls == 2
    assert get_state("test-api", budget=3).remaining == 1


def test_budget_blocks_the_call_that_would_exceed_it(migrated_db):
    """The Odds API guard has to actually refuse, not just log (D8)."""
    for _ in range(3):
        consume("test-api", budget=3)

    with pytest.raises(BudgetExceeded):
        check("test-api", budget=3)

    assert get_state("test-api", budget=3).exhausted


def test_budget_survives_a_new_connection(migrated_db):
    """A counter that reset every process would silently blow the monthly allowance."""
    consume("test-api", budget=10, n=4)

    from backend.db.connection import close_connection

    close_connection()

    assert get_state("test-api", budget=10).n_calls == 4


def test_write_parquet_atomic_leaves_no_temp_file(tmp_settings):
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_settings.raw_dir / "sample.parquet"

    write_parquet_atomic(df, path)

    assert path.exists()
    assert pl.read_parquet(path).equals(df)
    assert list(path.parent.glob("*.tmp*")) == []


def test_write_parquet_atomic_replaces_in_place(tmp_settings):
    path = tmp_settings.raw_dir / "sample.parquet"
    write_parquet_atomic(pl.DataFrame({"a": [1]}), path)
    write_parquet_atomic(pl.DataFrame({"a": [1, 2]}), path)

    assert pl.read_parquet(path).height == 2


def test_resilient_converts_an_exception_into_a_failed_result(migrated_db):
    """A source being down must never propagate out of ingest (D8)."""

    @resilient("test-source", "boom")
    def explode() -> IngestResult:
        raise ConnectionError("network is down")

    result = explode()

    assert isinstance(result, IngestResult)
    assert result.ok is False
    assert not result
    assert "network is down" in result.detail


def test_resilient_passes_success_through(migrated_db):
    @resilient("test-source", "fine")
    def works() -> IngestResult:
        return IngestResult(source="test-source", dataset="fine", n_rows=7)

    result = works()

    assert result
    assert result.n_rows == 7


def test_resilient_records_a_red_badge_on_failure(migrated_db):
    from backend.db.connection import connect

    @resilient("flaky-source", "boom")
    def explode() -> IngestResult:
        raise TimeoutError("too slow")

    explode()

    with connect() as con:
        row = con.execute(
            "SELECT status, detail FROM source_freshness WHERE source = 'flaky-source'"
        ).fetchone()

    assert row is not None
    assert row[0] == "red"
    assert "too slow" in row[1]
