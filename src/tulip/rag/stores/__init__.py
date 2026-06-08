"""Vector stores for RAG.

Available stores:
- InMemoryVectorStore: In-memory store (testing / small corpora)
- PgVectorStore: PostgreSQL with pgvector extension
- OpenSearchVectorStore: OpenSearch with k-NN plugin
- QdrantVectorStore: Qdrant (local in-memory or server)
- ChromaVectorStore: Chroma (embedded or server)
"""

from typing import Any

from tulip.rag.stores.base import (
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
    "InMemoryVectorStore",
    "PgVectorStore",
    "OpenSearchVectorStore",
    "QdrantVectorStore",
    "ChromaVectorStore",
]


def __getattr__(name: str) -> Any:
    """Lazy import stores to avoid requiring all dependencies."""
    if name == "InMemoryVectorStore":
        from tulip.rag.stores.memory import InMemoryVectorStore

        return InMemoryVectorStore

    if name == "PgVectorStore":
        from tulip.rag.stores.pgvector import PgVectorStore

        return PgVectorStore

    if name == "OpenSearchVectorStore":
        from tulip.rag.stores.opensearch import OpenSearchVectorStore

        return OpenSearchVectorStore

    if name == "QdrantVectorStore":
        from tulip.rag.stores.qdrant import QdrantVectorStore

        return QdrantVectorStore

    if name == "ChromaVectorStore":
        from tulip.rag.stores.chroma import ChromaVectorStore

        return ChromaVectorStore

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
