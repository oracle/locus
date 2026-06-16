# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Vector stores for RAG.

Available stores:
- OracleVectorStore: Oracle 26ai with native VECTOR type
- OpenSearchVectorStore: OpenSearch with k-NN plugin
- PgVectorStore: PostgreSQL with pgvector extension
- InMemoryVectorStore: In-memory store (testing)
"""

from typing import Any

from locus.rag.stores.base import (
    BaseVectorStore,
    Document,
    SearchResult,
    VectorStore,
    VectorStoreConfig,
)


__all__ = [
    # Base
    "BaseVectorStore",
    "Document",
    "SearchResult",
    "VectorStore",
    "VectorStoreConfig",
    # Stores (lazy imports)
    "OracleVectorStore",
    "OracleSyncVectorStore",
    "OpenSearchVectorStore",
    "PgVectorStore",
    "InMemoryVectorStore",
]


def __getattr__(name: str) -> Any:
    """Lazy import stores to avoid requiring all dependencies."""
    if name == "OracleVectorStore":
        from locus.rag.stores.oracle import OracleVectorStore

        return OracleVectorStore

    if name == "OracleSyncVectorStore":
        from locus.rag.stores.oracle_sync import OracleSyncVectorStore

        return OracleSyncVectorStore

    if name == "OpenSearchVectorStore":
        from locus.rag.stores.opensearch import OpenSearchVectorStore

        return OpenSearchVectorStore

    if name == "PgVectorStore":
        from locus.rag.stores.pgvector import PgVectorStore

        return PgVectorStore

    if name == "InMemoryVectorStore":
        from locus.rag.stores.memory import InMemoryVectorStore

        return InMemoryVectorStore

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
