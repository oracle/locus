"""Checkpoint backends for Tulip.

Available backends:
- MemoryCheckpointer: In-memory storage for testing/development
- FileCheckpointer: Local file-based storage
- HTTPCheckpointer: Remote HTTP API storage
- RedisBackend: Redis key-value store
- PostgreSQLBackend: PostgreSQL with JSONB
- MySQLBackend: MySQL with JSON columns
- OpenSearchBackend: OpenSearch with full-text search
- S3Backend: S3-compatible object storage (AWS S3, MinIO, Cloudflare R2)

Usage:
    ```python
    from tulip.memory.backends import (
        MemoryCheckpointer,
        MySQLBackend,
        RedisBackend,
        PostgreSQLBackend,
    )

    # For testing
    checkpointer = MemoryCheckpointer()

    # For production (choose based on your infrastructure)
    checkpointer = RedisBackend("redis://localhost:6379")
    checkpointer = PostgreSQLBackend(host="localhost", database="myapp")
    checkpointer = MySQLBackend(host="localhost", database="myapp")
    checkpointer = OpenSearchBackend(hosts=["localhost:9200"])
    checkpointer = S3Backend(
        bucket="checkpoints", endpoint_url="http://localhost:9000"
    )
    ```
"""

from typing import Any

from tulip.memory.backends.adapters import (
    StorageBackendAdapter,
    mysql_checkpointer,
    opensearch_checkpointer,
    postgresql_checkpointer,
    redis_checkpointer,
    s3_checkpointer,
)
from tulip.memory.backends.file import FileCheckpointer
from tulip.memory.backends.http import HTTPCheckpointer
from tulip.memory.backends.memory import MemoryCheckpointer
from tulip.memory.backends.mysql import MySQLBackend
from tulip.memory.backends.opensearch import OpenSearchBackend
from tulip.memory.backends.postgresql import PostgreSQLBackend
from tulip.memory.backends.redis import RedisBackend


__all__ = [
    # Full checkpointers (BaseCheckpointer interface)
    "FileCheckpointer",
    "HTTPCheckpointer",
    "MemoryCheckpointer",
    # Storage backends (simple dict interface)
    "MySQLBackend",
    "OpenSearchBackend",
    "PostgreSQLBackend",
    "RedisBackend",
    "S3Backend",
    # Adapter and factory functions
    "StorageBackendAdapter",
    "mysql_checkpointer",
    "opensearch_checkpointer",
    "postgresql_checkpointer",
    "redis_checkpointer",
    "s3_checkpointer",
]


def __getattr__(name: str) -> Any:
    """Lazy import backends that pull optional runtime dependencies."""
    if name == "S3Backend":
        from tulip.memory.backends.s3 import S3Backend

        return S3Backend

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
