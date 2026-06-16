# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Checkpoint backends for Locus.

Available backends:
- MemoryCheckpointer: In-memory storage for testing/development
- FileCheckpointer: Local file-based storage
- HTTPCheckpointer: Remote HTTP API storage
- RedisBackend: Redis key-value store (runs on OCI Cache with Redis)
- PostgreSQLBackend: PostgreSQL with JSONB (runs on OCI Database with PostgreSQL)
- MySQLBackend: MySQL with JSON columns
- OpenSearchBackend: OpenSearch with full-text search (runs on OCI Search with OpenSearch)
- OCIBucketBackend: OCI Object Storage for cloud deployments
- OracleBackend: Oracle Database with JSON support

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
    checkpointer = OCIBucketBackend(bucket_name="checkpoints", namespace="myns")
    checkpointer = OracleBackend(dsn="mydb_high", user="admin", password="secret")
    ```
"""

from locus.memory.backends.adapters import (
    StorageBackendAdapter,
    mysql_checkpointer,
    oci_bucket_checkpointer,
    opensearch_checkpointer,
    oracle_checkpointer,
    postgresql_checkpointer,
    redis_checkpointer,
)
from locus.memory.backends.file import FileCheckpointer
from locus.memory.backends.http import HTTPCheckpointer
from locus.memory.backends.memory import MemoryCheckpointer
from locus.memory.backends.mysql import MySQLBackend
from locus.memory.backends.oci_bucket import OCIBucketBackend
from locus.memory.backends.opensearch import OpenSearchBackend
from locus.memory.backends.oracle import OracleBackend
from locus.memory.backends.oracle_sync import OracleSyncBackend, OracleSyncCheckpointSaver
from locus.memory.backends.oracle_versioned import OracleCheckpointSaver
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
    # Versioned (LangGraph-shape) saver
    "OracleCheckpointSaver",
    # Sync companions (langgraph-oracledb-style split)
    "OracleSyncBackend",
    "OracleSyncCheckpointSaver",
    "PostgreSQLBackend",
    "RedisBackend",
    # Adapter and factory functions
    "StorageBackendAdapter",
    "mysql_checkpointer",
    "oci_bucket_checkpointer",
    "opensearch_checkpointer",
    "oracle_checkpointer",
    "postgresql_checkpointer",
    "redis_checkpointer",
]
