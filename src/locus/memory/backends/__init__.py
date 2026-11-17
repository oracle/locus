"""Checkpoint backends for Locus.

Available backends:
- MemoryCheckpointer: In-memory storage for testing/development
- FileCheckpointer: Local file-based storage
- HTTPCheckpointer: Remote HTTP API storage
- RedisBackend: Redis key-value store
- PostgreSQLBackend: PostgreSQL with JSONB
- PgCheckpointer: PostgreSQL, one tenant's threads apart from another's (RLS),
  one statement per save, per-message retention
- MySQLBackend: MySQL with JSON columns
- OpenSearchBackend: OpenSearch with full-text search
- S3Backend: S3-compatible object storage (AWS S3, MinIO, Cloudflare R2)
- OCIBucketBackend: OCI Object Storage for cloud deployments
- OracleBackend: Oracle Database 26ai with JSON support (plus the
  versioned ``OracleCheckpointSaver`` and sync wrappers)

Usage:
    ```python
    from locus.memory.backends import (
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
    checkpointer = OCIBucketBackend(bucket_name="checkpoints", namespace="myns")
    checkpointer = OracleBackend(dsn="mydb_high", user="admin", password="secret")
    ```
"""

from typing import Any

from locus.memory.backends.adapters import (
    StorageBackendAdapter,
    mysql_checkpointer,
    oci_bucket_checkpointer,
    opensearch_checkpointer,
    oracle_checkpointer,
    postgresql_checkpointer,
    redis_checkpointer,
    s3_checkpointer,
)
from locus.memory.backends.checkpoint_index import CheckpointIndexStore
from locus.memory.backends.file import FileCheckpointer
from locus.memory.backends.http import HTTPCheckpointer
from locus.memory.backends.memory import MemoryCheckpointer
from locus.memory.backends.mysql import MySQLBackend
from locus.memory.backends.opensearch import OpenSearchBackend
from locus.memory.backends.postgresql import PostgreSQLBackend
from locus.memory.backends.redis import RedisBackend


__all__ = [
    # Full checkpointers (BaseCheckpointer interface)
    "FileCheckpointer",
    "HTTPCheckpointer",
    "MemoryCheckpointer",
    # Storage backends (simple dict interface)
    "MySQLBackend",
    "OCIBucketBackend",
    "OpenSearchBackend",
    "OracleBackend",
    "OracleCheckpointSaver",
    "OracleSyncBackend",
    "OracleSyncCheckpointSaver",
    "PgCheckpointer",
    "PostgreSQLBackend",
    "RedisBackend",
    "S3Backend",
    # Adapter and factory functions
    "CheckpointIndexStore",
    "StorageBackendAdapter",
    "mysql_checkpointer",
    "oci_bucket_checkpointer",
    "opensearch_checkpointer",
    "oracle_checkpointer",
    "postgresql_checkpointer",
    "redis_checkpointer",
    "s3_checkpointer",
]


def __getattr__(name: str) -> Any:
    """Lazy import backends that pull optional runtime dependencies."""
    if name == "PgCheckpointer":
        from locus.memory.backends.pg_checkpointer import PgCheckpointer

        return PgCheckpointer
    if name == "S3Backend":
        from locus.memory.backends.s3 import S3Backend

        return S3Backend

    if name == "OCIBucketBackend":
        from locus.memory.backends.oci_bucket import OCIBucketBackend

        return OCIBucketBackend
    if name == "OracleBackend":
        from locus.memory.backends.oracle import OracleBackend

        return OracleBackend
    if name == "OracleCheckpointSaver":
        from locus.memory.backends.oracle_versioned import OracleCheckpointSaver

        return OracleCheckpointSaver
    if name in {"OracleSyncBackend", "OracleSyncCheckpointSaver"}:
        from locus.memory.backends import oracle_sync

        return getattr(oracle_sync, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
