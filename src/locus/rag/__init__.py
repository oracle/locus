# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""RAG (Retrieval-Augmented Generation) for Locus.

This module provides components for building RAG pipelines:

Embeddings (convert text to vectors):
- OCIEmbeddings: OCI GenAI with Cohere models

Vector Stores (persist and search vectors):
- OracleVectorStore: Oracle 26ai with native VECTOR type
- OpenSearchVectorStore: OpenSearch with k-NN plugin
- InMemoryVectorStore: In-memory store (testing)

Retriever (combines embedding + store):
- RAGRetriever: Unified interface for document management and retrieval

Tools (for agent integration):
- create_rag_tool: Create a search tool for agents
- create_rag_context_tool: Create a context retrieval tool
- RAGToolkit: Collection of RAG tools

Example:
    >>> from locus.rag import RAGRetriever, OCIEmbeddings, OracleVectorStore
    >>>
    >>> # Setup RAG pipeline
    >>> retriever = RAGRetriever(
    ...     embedder=OCIEmbeddings(
    ...         model_id="cohere.embed-english-v3.0",
    ...         profile_name="DEFAULT",
    ...     ),
    ...     store=OracleVectorStore(
    ...         dsn="mydb_high",
    ...         user="admin",
    ...         password="secret",
    ...     ),
    ... )
    >>>
    >>> # Add documents
    >>> await retriever.add_documents(
    ...     [
    ...         "Python is a programming language.",
    ...         "Oracle Database supports native vectors.",
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

# Embeddings
from locus.rag.embeddings.base import (
    BaseEmbedding,
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingResult,
)

# Multimodal
from locus.rag.multimodal import (
    ContentType,
    MultimodalProcessor,
    ProcessedContent,
    process_content,
)

# Retriever
from locus.rag.reranker import CohereReranker, Reranker
from locus.rag.retriever import RAGRetriever, RetrievalResult

# Stores
from locus.rag.stores.base import (
    BaseVectorStore,
    Document,
    SearchResult,
    VectorStore,
    VectorStoreConfig,
)

# Tools
from locus.rag.tools import RAGToolkit, create_rag_context_tool, create_rag_tool


__all__ = [
    # Embeddings - Base
    "BaseEmbedding",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "EmbeddingResult",
    # Embeddings - Providers (lazy)
    "OCIEmbeddings",
    "OracleInDBEmbeddings",
    # Stores - Base
    "BaseVectorStore",
    "Document",
    "SearchResult",
    "VectorStore",
    "VectorStoreConfig",
    # Stores - Implementations (lazy)
    "OracleVectorStore",
    "OpenSearchVectorStore",
    "InMemoryVectorStore",
    # Loaders / chunkers (lazy)
    "OracleADBLoader",
    "OracleInDBChunker",
    # Retriever
    "RAGRetriever",
    "RetrievalResult",
    # Reranker
    "Reranker",
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
    if name == "OCIEmbeddings":
        from locus.rag.embeddings.oci import OCIEmbeddings

        return OCIEmbeddings

    # Vector stores
    if name == "OracleVectorStore":
        from locus.rag.stores.oracle import OracleVectorStore

        return OracleVectorStore

    if name == "OpenSearchVectorStore":
        from locus.rag.stores.opensearch import OpenSearchVectorStore

        return OpenSearchVectorStore

    if name == "InMemoryVectorStore":
        from locus.rag.stores.memory import InMemoryVectorStore

        return InMemoryVectorStore

    # In-DB primitives (Oracle 23ai/26ai DBMS_VECTOR_CHAIN).
    if name == "OracleInDBEmbeddings":
        from locus.rag.embeddings.oracle_indb import OracleInDBEmbeddings

        return OracleInDBEmbeddings

    if name == "OracleADBLoader":
        from locus.rag.loaders.oracle import OracleADBLoader

        return OracleADBLoader

    if name == "OracleInDBChunker":
        from locus.rag.chunkers.oracle_indb import OracleInDBChunker

        return OracleInDBChunker

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
