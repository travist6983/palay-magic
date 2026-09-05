"""Persisted request budgets for metered APIs (docs/DECISIONS.md D8).

The Odds API free tier is 500 requests/month and the Anthropic wrapper is capped per refresh.
Both counters live in DuckDB so they survive process restarts — an in-memory counter would reset
every ``make refresh`` and quietly blow the monthly allowance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from backend.db.connection import connect
from backend.logging_setup import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class BudgetState:
    api: str
    period: str
    n_calls: int
    budget: int

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.n_calls)

    @property
    def exhausted(self) -> bool:
        return self.n_calls >= self.budget


class BudgetExceeded(RuntimeError):
    """Raised when a call would exceed the persisted allowance."""


def _period(monthly: bool, now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    return now.strftime("%Y-%m") if monthly else now.strftime("%Y-%m-%d")


def get_state(api: str, budget: int, monthly: bool = True) -> BudgetState:
    """Read the current period's counter, creating the row if it does not exist."""
    period = _period(monthly)
    with connect() as con:
        con.execute(
            "INSERT INTO api_budget (api, period, n_calls, budget) VALUES (?, ?, 0, ?) "
            "ON CONFLICT (api, period) DO UPDATE SET budget = excluded.budget",
            [api, period, budget],
        )
        row = con.execute(
            "SELECT n_calls, budget FROM api_budget WHERE api = ? AND period = ?", [api, period]
        ).fetchone()
    return BudgetState(api=api, period=period, n_calls=int(row[0]), budget=int(row[1]))


def check(api: str, budget: int, monthly: bool = True, n: int = 1) -> BudgetState:
    """Raise :class:`BudgetExceeded` if ``n`` more calls would exceed the allowance."""
    state = get_state(api, budget, monthly)
    if state.n_calls + n > state.budget:
        raise BudgetExceeded(
            f"{api}: {state.n_calls}/{state.budget} calls used in {state.period}; "
            f"{n} more would exceed the budget"
        )
    return state


def consume(api: str, budget: int, monthly: bool = True, n: int = 1) -> BudgetState:
    """Record ``n`` calls against the allowance. Call this *after* a successful request."""
    period = _period(monthly)
    with connect() as con:
        con.execute(
            "INSERT INTO api_budget (api, period, n_calls, budget, last_call_at) "
            "VALUES (?, ?, ?, ?, now()) "
            "ON CONFLICT (api, period) DO UPDATE SET "
            "  n_calls = api_budget.n_calls + excluded.n_calls, "
            "  budget = excluded.budget, "
            "  last_call_at = now()",
            [api, period, n, budget],
        )
        row = con.execute(
            "SELECT n_calls, budget FROM api_budget WHERE api = ? AND period = ?", [api, period]
        ).fetchone()
    state = BudgetState(api=api, period=period, n_calls=int(row[0]), budget=int(row[1]))
    log.info("%s budget: %d/%d used this %s", api, state.n_calls, state.budget, state.period)
    return state
