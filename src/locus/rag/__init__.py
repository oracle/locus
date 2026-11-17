"""RAG (Retrieval-Augmented Generation) for Locus.

This module provides components for building RAG pipelines:

Embeddings (convert text to vectors):
- OpenAIEmbeddings: OpenAI ``text-embedding-3-*`` models
- CohereEmbeddings: Cohere's direct embedding API
- OCIEmbeddings: OCI Generative AI embeddings (Cohere embed models on OCI)
- OracleInDBEmbeddings: Oracle 26ai in-database ONNX embeddings

Vector Stores (persist and search vectors):
- InMemoryVectorStore: In-memory store (testing / small corpora)
- PgVectorStore: PostgreSQL + pgvector
- OpenSearchVectorStore: OpenSearch with k-NN plugin
- QdrantVectorStore: Qdrant (local or server)
- ChromaVectorStore: Chroma (embedded or server)
- OracleVectorStore: Oracle Database 26ai with the native VECTOR type
- MilvusVectorStore, WeaviateVectorStore, ElasticsearchVectorStore,
  RedisVectorStore, MongoDBAtlasVectorStore, PineconeVectorStore,
  HeatWaveVectorStore

Rerankers (reorder candidates by relevance):
- CrossEncoderReranker: local sentence-transformers cross-encoder (offline)
- CohereReranker: Cohere's direct rerank API
- OCICohereReranker: Cohere rerank models served by OCI Generative AI
- LLMReranker: any registered chat model grades the candidates
- JinaReranker / VoyageReranker: hosted rerank APIs

Retriever (combines embedding + store):
- RAGRetriever: Unified interface for document management and retrieval

Tools (for agent integration):
- create_rag_tool: Create a search tool for agents
- create_rag_context_tool: Create a context retrieval tool
- RAGToolkit: Collection of RAG tools

Example:
    >>> from locus.rag import RAGRetriever, OpenAIEmbeddings, InMemoryVectorStore
    >>>
    >>> # Setup RAG pipeline
    >>> retriever = RAGRetriever(
    ...     embedder=OpenAIEmbeddings(model="text-embedding-3-small"),
    ...     store=InMemoryVectorStore(),
    ... )
    >>>
    >>> # Add documents
    >>> await retriever.add_documents(
    ...     [
    ...         "Python is a programming language.",
    ...         "Vector search powers retrieval-augmented generation.",
    ...     ]
    ... )
    >>>
    >>> # Retrieve relevant context
    >>> results = await retriever.retrieve("What is Python?", limit=3)
    >>> for r in results.documents:
    ...     print(f"{r.score:.2f}: {r.document.content}")

Example with agent:
    >>> from locus import Agent
    >>> from locus.rag import RAGRetriever, create_rag_tool
    >>>
    >>> agent = Agent(
    ...     model=model,
    ...     tools=[retriever.as_tool()],  # Add RAG as a tool
    ... )
"""

from typing import Any

# Ingestion — the entrance to the pipeline
from locus.rag.chunkers import (
    DEFAULT_SEPARATORS,
    MARKDOWN_SEPARATORS,
    recursive_chunks,
)

# Embeddings
from locus.rag.embeddings.base import (
    BaseEmbedding,
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingResult,
)
from locus.rag.loaders import (
    load_directory,
    load_html,
    load_markdown,
    load_pdf,
    load_text,
)

# Multimodal
from locus.rag.multimodal import (
    ContentType,
    MultimodalProcessor,
    ProcessedContent,
    process_content,
)

# Reranker (base) + retriever
from locus.rag.reranker import Reranker
from locus.rag.retriever import RAGRetriever, RetrievalResult

# Stores
from locus.rag.stores.base import (
    BaseVectorStore,
    Document,
    HybridSearchable,
    SearchResult,
    VectorStore,
    VectorStoreConfig,
    supports_hybrid_search,
)

# Tools
from locus.rag.tools import RAGToolkit, create_rag_context_tool, create_rag_tool


__all__ = [
    # Ingestion — loaders and chunkers
    "DEFAULT_SEPARATORS",
    "MARKDOWN_SEPARATORS",
    "load_directory",
    "load_html",
    "load_markdown",
    "load_pdf",
    "load_text",
    "recursive_chunks",
    # Embeddings - Base
    "BaseEmbedding",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "EmbeddingResult",
    # Embeddings - Providers (lazy)
    "OpenAIEmbeddings",
    "CohereEmbeddings",
    "OCIEmbeddings",
    "OracleInDBEmbeddings",
    "OracleSyncInDBEmbeddings",
    # Stores - Base
    "BaseVectorStore",
    "Document",
    "HybridSearchable",
    "SearchResult",
    "VectorStore",
    "VectorStoreConfig",
    "supports_hybrid_search",
    # Stores - Implementations (lazy)
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
    "HeatWaveVectorStore",
    # Oracle ingestion (lazy)
    "OracleADBLoader",
    "OracleInDBChunker",
    # Retriever
    "RAGRetriever",
    "RetrievalResult",
    # Reranker
    "Reranker",
    "CrossEncoderReranker",
    "CohereReranker",
    "OCICohereReranker",
    "LLMReranker",
    "JinaReranker",
    "VoyageReranker",
    # Multimodal
    "ContentType",
    "MultimodalProcessor",
    "ProcessedContent",
    "process_content",
    # Tools
    "RAGToolkit",
    "create_rag_context_tool",
    "create_rag_tool",
]


def __getattr__(name: str) -> Any:
    """Lazy import providers and stores."""
    # Embedding providers
    if name == "OpenAIEmbeddings":
        from locus.rag.embeddings.openai import OpenAIEmbeddings

        return OpenAIEmbeddings

    if name == "CohereEmbeddings":
        from locus.rag.embeddings.cohere import CohereEmbeddings

        return CohereEmbeddings

    # Vector stores
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

    # Rerankers
    if name == "CrossEncoderReranker":
        from locus.rag.reranker.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker

    if name == "CohereReranker":
        from locus.rag.reranker.cohere import CohereReranker

        return CohereReranker
    # Oracle Database 26ai + OCI Generative AI (need the ``oracledb`` / ``oci`` extras)
    if name in ("OCIEmbeddings", "OracleInDBEmbeddings", "OracleSyncInDBEmbeddings"):
        from locus.rag import embeddings

        return getattr(embeddings, name)
    if name in (
        "OracleVectorStore",
        "OracleSyncVectorStore",
        "MilvusVectorStore",
        "WeaviateVectorStore",
        "ElasticsearchVectorStore",
        "RedisVectorStore",
        "MongoDBAtlasVectorStore",
        "PineconeVectorStore",
        "HeatWaveVectorStore",
    ):
        from locus.rag import stores

        return getattr(stores, name)
    if name in ("LLMReranker", "JinaReranker", "VoyageReranker"):
        from locus.rag import reranker

        return getattr(reranker, name)
    if name == "OracleADBLoader":
        from locus.rag.loaders.oracle import OracleADBLoader

        return OracleADBLoader
    if name == "OracleInDBChunker":
        from locus.rag.chunkers.oracle_indb import OracleInDBChunker

        return OracleInDBChunker
    if name == "OCICohereReranker":
        from locus.rag.reranker.cohere_oci import OCICohereReranker

        return OCICohereReranker

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
