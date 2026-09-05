"""Data ingestion. One module per external source (D8: every call is retried and never fatal)."""

from backend.ingest.base import (
    IngestResult,
    record_freshness,
    resilient,
    write_parquet_atomic,
)

__all__ = ["IngestResult", "record_freshness", "resilient", "write_parquet_atomic"]
