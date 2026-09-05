"""DuckDB access layer: connection, migrations, and views over the raw Parquet cache."""

from backend.db.connection import connect, get_connection, query, query_df
from backend.db.migrate import migrate
from backend.db.views import RAW_VIEWS, refresh_views

__all__ = [
    "RAW_VIEWS",
    "connect",
    "get_connection",
    "migrate",
    "query",
    "query_df",
    "refresh_views",
]
