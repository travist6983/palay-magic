"""Plain-SQL migrations, applied in filename order.

Each file in ``backend/db/migrations/*.sql`` runs once; applied names are recorded in
``schema_migrations``. Files must be idempotent (``CREATE TABLE IF NOT EXISTS``) so a partially
applied migration can be replayed safely.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from backend.db.connection import connect
from backend.logging_setup import get_logger

log = get_logger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    name       VARCHAR PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT current_timestamp
);
"""


def migrate(con: duckdb.DuckDBPyConnection | None = None) -> list[str]:
    """Apply every pending migration. Returns the names that were applied this run."""
    if con is not None:
        return _apply(con)
    with connect() as owned:
        return _apply(owned)


def _apply(con: duckdb.DuckDBPyConnection) -> list[str]:
    con.execute(_BOOTSTRAP)
    already = {r[0] for r in con.execute("SELECT name FROM schema_migrations").fetchall()}
    applied: list[str] = []

    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        if path.name in already:
            continue
        log.info("applying migration %s", path.name)
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute(path.read_text())
            con.execute("INSERT INTO schema_migrations (name) VALUES (?)", [path.name])
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            log.exception("migration %s failed", path.name)
            raise
        applied.append(path.name)

    if not applied:
        log.debug("schema up to date (%d migrations already applied)", len(already))
    return applied
