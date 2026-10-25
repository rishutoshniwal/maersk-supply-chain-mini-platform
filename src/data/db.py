"""
DuckDB database connection manager and query helpers.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

import duckdb

DEFAULT_DB_PATH = Path(os.getenv("DATABASE_PATH", "./data/curated/supply_chain.duckdb"))


def get_db_path() -> Path:
    db_env = os.getenv("DATABASE_PATH")
    if db_env:
        return Path(db_env)
    return DEFAULT_DB_PATH


def init_db(db_path: Optional[Path] = None) -> duckdb.DuckDBPyConnection:
    """Initialize connection to DuckDB database."""
    target_path = db_path or get_db_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(target_path), read_only=False)
    return conn


@contextmanager
def get_read_connection(db_path: Optional[Path] = None) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Provide a read-only DuckDB connection context."""
    target_path = db_path or get_db_path()
    if not target_path.exists():
        # Fallback to in-memory if DB does not yet exist
        conn = duckdb.connect(":memory:")
    else:
        conn = duckdb.connect(str(target_path), read_only=True)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def get_write_connection(db_path: Optional[Path] = None) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Provide a write DuckDB connection context."""
    target_path = db_path or get_db_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(target_path), read_only=False)
    try:
        yield conn
    finally:
        conn.close()
