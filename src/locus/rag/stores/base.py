# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Vector store protocols and base classes.

Vector stores persist embeddings and enable similarity search.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable


@dataclass
class Document:
    """A document with optional embedding.

    Attributes:
        id: Unique document identifier
        content: Document text content (or extracted text for multimodal)
        embedding: Optional embedding vector
        metadata: Optional metadata for filtering
        created_at: Creation timestamp
        content_type: Type of content (text, image, pdf, audio)
        raw_content: Original binary content for multimodal documents
    """

    id: str
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    content_type: str = "text"  # text, image, pdf, audio
    raw_content: bytes | None = None  # Original binary for multimodal

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        import base64

        result = {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "content_type": self.content_type,
        }
        if self.raw_content:
            result["raw_content"] = base64.b64encode(self.raw_content).decode()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        """Create from dictionary."""
        import base64

        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(UTC)

        raw_content = data.get("raw_content")
        if isinstance(raw_content, str):
            raw_content = base64.b64decode(raw_content)

        return cls(
            id=data["id"],
            content=data["content"],
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
            created_at=created_at,
            content_type=data.get("content_type", "text"),
            raw_content=raw_content,
        )


@dataclass
class SearchResult:
    """Result from similarity search.

    Attributes:
        document: The matching document
        score: Similarity score (0.0 to 1.0, higher is more similar)
        distance: Raw distance metric (interpretation depends on distance type)
    """

    document: Document
    score: float
    distance: float | None = None


@dataclass(frozen=True)
class VectorStoreConfig:
    """Configuration for vector stores.

    Attributes:
        dimension: Expected embedding dimension
        distance_metric: Distance metric (cosine, l2, dot_product)
        index_type: Index type (flat, ivf, hnsw)
    """

    dimension: int
    distance_metric: str = "cosine"  # cosine, l2, dot_product
    index_type: str = "hnsw"  # flat, ivf, hnsw


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector stores.

    Vector stores persist documents with embeddings and enable
    fast similarity search.

    Example:
        >>> store = InMemoryVectorStore()
        >>> await store.add(doc)
        >>> results = await store.search(query_embedding, limit=5)
    """

    @property
    def config(self) -> VectorStoreConfig:
        """Get store configuration."""
        ...

    async def add(self, document: Document) -> str:
        """Add a document.

        Args:
            document: Document with embedding

        Returns:
            Document ID
        """
        ...

    async def add_batch(self, documents: list[Document]) -> list[str]:
        """Add multiple documents.

        Args:
            documents: Documents with embeddings

        Returns:
            List of document IDs
        """
        ...

    async def get(self, doc_id: str) -> Document | None:
        """Get a document by ID.

        Args:
            doc_id: Document identifier

        Returns:
            Document or None if not found
        """
        ...

    async def delete(self, doc_id: str) -> bool:
        """Delete a document.

        Args:
            doc_id: Document identifier

        Returns:
            True if deleted, False if not found
        """
        ...

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents.

        Args:
            query_embedding: Query vector
            limit: Maximum results
            threshold: Minimum similarity score (0.0-1.0)
            metadata_filter: Filter by metadata fields

        Returns:
            List of SearchResult sorted by similarity
        """
        ...

    async def count(self) -> int:
        """Count documents in store."""
        ...

    async def clear(self) -> int:
        """Delete all documents.

        Returns:
            Number of documents deleted
        """
        ...

    async def close(self) -> None:
        """Close any resources."""
        ...


class BaseVectorStore(ABC):
    """Abstract base class for vector stores.

    Provides default implementations for common methods.
    """

    @property
    @abstractmethod
    def config(self) -> VectorStoreConfig:
        """Get store configuration."""
        ...

    @abstractmethod
    async def add(self, document: Document) -> str:
        """Add a document."""
        ...

    async def add_batch(self, documents: list[Document]) -> list[str]:
        """Add multiple documents. Override for batch optimization."""
        ids = []
        for doc in documents:
            doc_id = await self.add(doc)
            ids.append(doc_id)
        return ids

    @abstractmethod
    async def get(self, doc_id: str) -> Document | None:
        """Get a document by ID."""
        ...

    @abstractmethod
    async def delete(self, doc_id: str) -> bool:
        """Delete a document."""
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""
        ...

    async def count(self) -> int:
        """Count documents. Override for efficient implementation."""
        return 0

    async def clear(self) -> int:
        """Delete all documents. Override for efficient implementation."""
        return 0

    async def close(self) -> None:
        """Close any resources."""


@runtime_checkable
class HybridSearchable(Protocol):
    """Protocol for stores that blend vector similarity with lexical relevance.

    A store that implements this protocol accepts the query *text* next to
    the query *embedding* and returns one ranked list whose ``score`` is the
    blended relevance. ``alpha`` weights the two legs the same way on every
    store: ``1.0`` is pure vector search, ``0.0`` pure lexical, ``0.5`` an
    even blend. Stores that can switch lexical scoring on and off at
    construction time also expose ``hybrid_search_available`` so a caller
    can fall back to :meth:`VectorStore.search` instead of failing.

    ``RAGRetriever(hybrid=True)`` calls :meth:`hybrid_search` when
    :func:`supports_hybrid_search` is true for its store and falls back to
    vector-only search otherwise.
    """

    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        *,
        limit: int = 10,
        alpha: float = 0.5,
        threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return documents ranked by ``alpha * vector + (1 - alpha) * lexical``."""
        ...


def supports_hybrid_search(store: object) -> bool:
    """True when ``store`` can answer :meth:`HybridSearchable.hybrid_search`.

    Checks for a callable ``hybrid_search`` and honours an optional
    ``hybrid_search_available`` flag for stores whose lexical leg is a
    construction-time option (Milvus needs a BM25 function on the
    collection, Oracle needs the text column). Used by ``RAGRetriever`` to
    decide between the blended call and the vector-only fallback.
    """
    if not callable(getattr(store, "hybrid_search", None)):
        return False
    return bool(getattr(store, "hybrid_search_available", True))
