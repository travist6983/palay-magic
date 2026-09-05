"""Shared pytest fixtures.

Tests must never touch the real ``data/proplab.duckdb`` or hit the network. Every fixture here
points PropLab at a temporary directory instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def tmp_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[object]:
    """Redirect the settings singleton at a throwaway data directory."""
    from backend import config as config_module

    config_module.get_settings.cache_clear()
    monkeypatch.setenv("PROPLAB_DB_PATH", str(tmp_path / "test.duckdb"))
    monkeypatch.setenv("PROPLAB_RAW_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("PROPLAB_CACHE_DIR", str(tmp_path / "cache"))

    from backend.db import connection as connection_module

    connection_module.close_connection()

    settings = config_module.get_settings()
    yield settings

    connection_module.close_connection()
    config_module.get_settings.cache_clear()


@pytest.fixture
def migrated_db(tmp_settings):
    """A migrated, empty PropLab database in a temp directory."""
    from backend.db.migrate import migrate

    migrate()
    return tmp_settings
