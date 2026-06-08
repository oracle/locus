"""RAG (Retrieval-Augmented Generation) for Tulip.

This module provides components for building RAG pipelines:

Embeddings (convert text to vectors):
- OpenAIEmbeddings: OpenAI ``text-embedding-3-*`` models
- CohereEmbeddings: Cohere's direct embedding API

Vector Stores (persist and search vectors):
- InMemoryVectorStore: In-memory store (testing / small corpora)
- PgVectorStore: PostgreSQL + pgvector
- OpenSearchVectorStore: OpenSearch with k-NN plugin
- QdrantVectorStore: Qdrant (local or server)
- ChromaVectorStore: Chroma (embedded or server)

Rerankers (reorder candidates by relevance):
- CrossEncoderReranker: local sentence-transformers cross-encoder (offline)
- CohereReranker: Cohere's direct rerank API

Retriever (combines embedding + store):
- RAGRetriever: Unified interface for document management and retrieval

Tools (for agent integration):
- create_rag_tool: Create a search tool for agents
- create_rag_context_tool: Create a context retrieval tool
- RAGToolkit: Collection of RAG tools

Example:
    >>> from tulip.rag import RAGRetriever, OpenAIEmbeddings, InMemoryVectorStore
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
    >>> from tulip import Agent
    >>> from tulip.rag import RAGRetriever, create_rag_tool
    >>>
    >>> agent = Agent(
    ...     model=model,
    ...     tools=[retriever.as_tool()],  # Add RAG as a tool
    ... )
"""

from typing import Any

# Embeddings
from tulip.rag.embeddings.base import (
    BaseEmbedding,
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingResult,
)

# Multimodal
from tulip.rag.multimodal import (
    ContentType,
    MultimodalProcessor,
    ProcessedContent,
    process_content,
)

# Reranker (base) + retriever
from tulip.rag.reranker import Reranker
from tulip.rag.retriever import RAGRetriever, RetrievalResult

# Stores
from tulip.rag.stores.base import (
    BaseVectorStore,
    Document,
    SearchResult,
    VectorStore,
    VectorStoreConfig,
)

# Tools
from tulip.rag.tools import RAGToolkit, create_rag_context_tool, create_rag_tool


__all__ = [
    # Embeddings - Base
    "BaseEmbedding",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "EmbeddingResult",
    # Embeddings - Providers (lazy)
    "OpenAIEmbeddings",
    "CohereEmbeddings",
    # Stores - Base
    "BaseVectorStore",
    "Document",
    "SearchResult",
    "VectorStore",
    "VectorStoreConfig",
    # Stores - Implementations (lazy)
    "InMemoryVectorStore",
    "PgVectorStore",
    "OpenSearchVectorStore",
    "QdrantVectorStore",
    "ChromaVectorStore",
    # Retriever
    "RAGRetriever",
    "RetrievalResult",
    # Reranker
    "Reranker",
    "CrossEncoderReranker",
    "CohereReranker",
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
        from tulip.rag.embeddings.openai import OpenAIEmbeddings

        return OpenAIEmbeddings

    if name == "CohereEmbeddings":
        from tulip.rag.embeddings.cohere import CohereEmbeddings

        return CohereEmbeddings

    # Vector stores
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

    # Rerankers
    if name == "CrossEncoderReranker":
        from tulip.rag.reranker.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker

    if name == "CohereReranker":
        from tulip.rag.reranker.cohere import CohereReranker

        return CohereReranker

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
