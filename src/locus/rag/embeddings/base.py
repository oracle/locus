# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Embedding provider protocols and base classes.

Embeddings convert text into dense vectors for semantic similarity search.

Every provider inherits :class:`BaseEmbedding` and advertises its
:class:`EmbeddingCapabilities`, mirroring
:class:`locus.core.protocols.CheckpointerCapabilities`. Consumers can
check capabilities before calling optional methods::

    if embedder.capabilities.supports_query_vs_doc:
        query_vec = await embedder.embed_query(query)
    else:
        query_vec = await embedder.embed(query)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class EmbeddingCapabilities:
    """Capabilities advertised by a :class:`BaseEmbedding` provider.

    Mirrors :class:`locus.core.protocols.CheckpointerCapabilities` — every
    provider exposes this so consumers can discover optional features
    without exception-catching.
    """

    supports_query_vs_doc: bool = False
    """Provider uses a different embedding space for queries vs documents
    (Cohere ``*-v3.0`` and some OpenAI models do). When False,
    ``embed_query``/``embed_documents`` delegate to ``embed``/``embed_batch``."""

    supports_multimodal: bool = False
    """Provider can embed non-text inputs (images, audio). Implementers
    with multimodal support expose dedicated methods in addition to text
    ``embed``."""

    supports_batching: bool = True
    """Provider has a native batch endpoint (as opposed to looping
    ``embed`` internally). Consumers can use this to decide whether to
    pre-chunk their payloads."""

    max_batch_size: int = 1
    """Upper bound on inputs per batch call. 1 means no batching support."""

    max_input_tokens: int = 8192
    """Longest single input the provider accepts, in tokens."""


@dataclass(frozen=True)
class EmbeddingResult:
    """Result from embedding operation.

    Attributes:
        embedding: The embedding vector
        text: Original text that was embedded
        model: Model used for embedding
        tokens: Number of tokens used (if available)
    """

    embedding: list[float]
    text: str
    model: str
    tokens: int | None = None


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for embedding providers.

    Attributes:
        dimension: Vector dimension size
        max_tokens: Maximum tokens per request
        batch_size: Maximum texts per batch
    """

    dimension: int
    max_tokens: int = 8192
    batch_size: int = 96


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers.

    Embedding providers convert text into dense vectors that capture
    semantic meaning, enabling similarity search.

    Example:
        >>> embedder = OCIEmbeddings(model_id="cohere.embed-english-v3.0")
        >>> result = await embedder.embed("Hello world")
        >>> print(len(result.embedding))  # 1024
    """

    @property
    def config(self) -> EmbeddingConfig:
        """Get embedding configuration."""
        ...

    @property
    def capabilities(self) -> EmbeddingCapabilities:
        """Advertised capabilities. See :class:`EmbeddingCapabilities`."""
        ...

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        ...

    async def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with vector and metadata
        """
        ...

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of EmbeddingResult, one per input text
        """
        ...

    async def embed_query(self, query: str) -> EmbeddingResult:
        """Embed a query for retrieval.

        Some models use different embeddings for queries vs documents.
        Default implementation calls embed().

        Args:
            query: Query text to embed

        Returns:
            EmbeddingResult optimized for query
        """
        ...

    async def embed_documents(self, documents: list[str]) -> list[EmbeddingResult]:
        """Embed documents for storage.

        Some models use different embeddings for queries vs documents.
        Default implementation calls embed_batch().

        Args:
            documents: Document texts to embed

        Returns:
            List of EmbeddingResult optimized for storage
        """
        ...


class BaseEmbedding(ABC):
    """Abstract base class for embedding providers.

    Provides default implementations for common methods. Subclasses
    override :meth:`capabilities` to advertise optional features, and
    override :meth:`embed_query`/:meth:`embed_documents` only if
    ``supports_query_vs_doc`` is True.
    """

    @property
    @abstractmethod
    def config(self) -> EmbeddingConfig:
        """Get embedding configuration."""
        ...

    @property
    def capabilities(self) -> EmbeddingCapabilities:
        """Advertised capabilities. Default is text-only, no batching,
        and no query/doc differentiation — override in subclasses.
        """
        return EmbeddingCapabilities(
            supports_batching=True,
            max_batch_size=self.config.batch_size,
            max_input_tokens=self.config.max_tokens,
        )

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.config.dimension

    @abstractmethod
    async def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed multiple texts. Override for batch optimization."""
        results = []
        for text in texts:
            result = await self.embed(text)
            results.append(result)
        return results

    async def embed_query(self, query: str) -> EmbeddingResult:
        """Embed a query. Override if model has query-specific embeddings."""
        return await self.embed(query)

    async def embed_documents(self, documents: list[str]) -> list[EmbeddingResult]:
        """Embed documents. Override if model has document-specific embeddings."""
        return await self.embed_batch(documents)
