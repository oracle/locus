"""Vector stores for RAG.

Available stores:
- InMemoryVectorStore: In-memory store (testing / small corpora)
- PgVectorStore: PostgreSQL with pgvector extension
- OpenSearchVectorStore: OpenSearch with k-NN plugin
- QdrantVectorStore: Qdrant (local in-memory or server)
- ChromaVectorStore: Chroma (embedded or server)
- OracleVectorStore: Oracle Database 26ai with the native VECTOR type
- MilvusVectorStore: Milvus server, Zilliz Cloud or embedded Milvus Lite
- WeaviateVectorStore: Weaviate (local, cloud or embedded)
- ElasticsearchVectorStore: Elasticsearch dense_vector k-NN
- RedisVectorStore: Redis Query Engine (Redis Stack / Redis 8) VECTOR fields
- MongoDBAtlasVectorStore: MongoDB Atlas Vector Search
- PineconeVectorStore: Pinecone serverless indexes

Stores that blend vector similarity with lexical relevance implement
``HybridSearchable``; ``supports_hybrid_search(store)`` tells a caller which.
"""

from typing import Any

from locus.rag.stores.base import (
    BaseVectorStore,
    Document,
    HybridSearchable,
    SearchResult,
    VectorStore,
    VectorStoreConfig,
    supports_hybrid_search,
)


__all__ = [
    # Base
    "BaseVectorStore",
    "Document",
    "HybridSearchable",
    "SearchResult",
    "VectorStore",
    "VectorStoreConfig",
    "supports_hybrid_search",
    # Stores (lazy imports)
    "InMemoryVectorStore",
    "PgVectorStore",
    "OpenSearchVectorStore",
    "QdrantVectorStore",
    "ChromaVectorStore",
    "OracleVectorStore",
    "OracleSyncVectorStore",
    "MilvusVectorStore",
    "WeaviateVectorStore",
    "ElasticsearchVectorStore",
    "RedisVectorStore",
    "MongoDBAtlasVectorStore",
    "PineconeVectorStore",
]


_LAZY_STORES = {
    "MilvusVectorStore": "locus.rag.stores.milvus",
    "WeaviateVectorStore": "locus.rag.stores.weaviate",
    "ElasticsearchVectorStore": "locus.rag.stores.elasticsearch",
    "RedisVectorStore": "locus.rag.stores.redis",
    "MongoDBAtlasVectorStore": "locus.rag.stores.mongodb",
    "PineconeVectorStore": "locus.rag.stores.pinecone",
}


def __getattr__(name: str) -> Any:
    """Lazy import stores to avoid requiring all dependencies."""
    if name == "InMemoryVectorStore":
        from locus.rag.stores.memory import InMemoryVectorStore

        return InMemoryVectorStore

    if name == "PgVectorStore":
        from locus.rag.stores.pgvector import PgVectorStore

        return PgVectorStore

    if name == "OpenSearchVectorStore":
        from locus.rag.stores.opensearch import OpenSearchVectorStore

        return OpenSearchVectorStore

    if name == "QdrantVectorStore":
        from locus.rag.stores.qdrant import QdrantVectorStore

        return QdrantVectorStore

    if name == "ChromaVectorStore":
        from locus.rag.stores.chroma import ChromaVectorStore

        return ChromaVectorStore

    if name == "OracleVectorStore":
        from locus.rag.stores.oracle import OracleVectorStore

        return OracleVectorStore
    if name == "OracleSyncVectorStore":
        from locus.rag.stores.oracle_sync import OracleSyncVectorStore

        return OracleSyncVectorStore
    if name in _LAZY_STORES:
        import importlib

        return getattr(importlib.import_module(_LAZY_STORES[name]), name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
