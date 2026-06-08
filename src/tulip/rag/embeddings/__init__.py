"""Embedding providers for RAG.

Available providers:
- OpenAIEmbeddings: OpenAI text-embedding models
- CohereEmbeddings: Cohere's direct embedding API
"""

from typing import Any

from tulip.rag.embeddings.base import (
    BaseEmbedding,
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingResult,
)


__all__ = [
    # Base
    "BaseEmbedding",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "EmbeddingResult",
    # Providers (lazy imports)
    "OpenAIEmbeddings",
    "CohereEmbeddings",
]


def __getattr__(name: str) -> Any:
    """Lazy import providers to avoid requiring all dependencies."""
    if name == "OpenAIEmbeddings":
        from tulip.rag.embeddings.openai import OpenAIEmbeddings

        return OpenAIEmbeddings

    if name == "CohereEmbeddings":
        from tulip.rag.embeddings.cohere import CohereEmbeddings

        return CohereEmbeddings

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
