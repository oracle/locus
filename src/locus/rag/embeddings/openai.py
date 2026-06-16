# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""OpenAI Embeddings provider.

Uses OpenAI's text-embedding models for semantic embeddings.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from locus.rag.embeddings.base import (
    BaseEmbedding,
    EmbeddingCapabilities,
    EmbeddingConfig,
    EmbeddingResult,
)


if TYPE_CHECKING:
    from openai import AsyncOpenAI


class OpenAIEmbeddingsConfig(BaseModel):
    """Configuration for OpenAI Embeddings."""

    model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model ID",
    )
    api_key: str | None = Field(
        default=None,
        description="OpenAI API key (defaults to OPENAI_API_KEY env var)",
    )
    dimensions: int | None = Field(
        default=None,
        description="Output dimensions (for models that support it)",
    )
    base_url: str | None = Field(
        default=None,
        description="Custom base URL for API",
    )

    # Model dimension defaults
    _model_dimensions: dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    @property
    def dimension(self) -> int:
        """Get embedding dimension for the model."""
        if self.dimensions:
            return self.dimensions
        return self._model_dimensions.get(self.model, 1536)


class OpenAIEmbeddings(BaseEmbedding):
    """OpenAI Embeddings provider.

    Uses OpenAI's text-embedding models for generating embeddings.

    Example:
        >>> embedder = OpenAIEmbeddings(
        ...     model="text-embedding-3-small",
        ...     api_key="sk-...",
        ... )
        >>> result = await embedder.embed("Hello world")
        >>> print(len(result.embedding))  # 1536
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimensions: int | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OpenAI embeddings.

        Args:
            model: OpenAI embedding model ID
            api_key: API key (defaults to OPENAI_API_KEY env var)
            dimensions: Output dimensions (for supported models)
            base_url: Custom base URL
            **kwargs: Additional configuration
        """
        self._config_model = OpenAIEmbeddingsConfig(
            model=model,
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            dimensions=dimensions,
            base_url=base_url,
        )
        self._client: AsyncOpenAI | None = None
        self._embedding_config = EmbeddingConfig(
            dimension=self._config_model.dimension,
            max_tokens=8191,
            batch_size=2048,
        )

    @property
    def config(self) -> EmbeddingConfig:
        """Get embedding configuration."""
        return self._embedding_config

    @property
    def capabilities(self) -> EmbeddingCapabilities:
        """OpenAI embeddings: text-only, native batching, no separate
        query/doc spaces (text-embedding-3-* use the same space)."""
        return EmbeddingCapabilities(
            supports_query_vs_doc=False,
            supports_multimodal=False,
            supports_batching=True,
            max_batch_size=2048,
            max_input_tokens=8192,
        )

    def _get_client(self) -> AsyncOpenAI:
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install openai"
                ) from e

            self._client = AsyncOpenAI(
                api_key=self._config_model.api_key,
                base_url=self._config_model.base_url,
            )
        return self._client

    async def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with vector and metadata
        """
        client = self._get_client()

        kwargs: dict[str, Any] = {
            "model": self._config_model.model,
            "input": text,
        }
        if self._config_model.dimensions:
            kwargs["dimensions"] = self._config_model.dimensions

        response = await client.embeddings.create(**kwargs)

        return EmbeddingResult(
            embedding=response.data[0].embedding,
            text=text,
            model=self._config_model.model,
            tokens=response.usage.total_tokens if response.usage else None,
        )

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed multiple texts in a single request.

        Args:
            texts: List of texts to embed

        Returns:
            List of EmbeddingResult, one per input text
        """
        if not texts:
            return []

        client = self._get_client()

        kwargs: dict[str, Any] = {
            "model": self._config_model.model,
            "input": texts,
        }
        if self._config_model.dimensions:
            kwargs["dimensions"] = self._config_model.dimensions

        response = await client.embeddings.create(**kwargs)

        results = []
        for i, data in enumerate(response.data):
            results.append(
                EmbeddingResult(
                    embedding=data.embedding,
                    text=texts[i],
                    model=self._config_model.model,
                    tokens=None,  # Per-text tokens not available in batch
                )
            )
        return results

    async def close(self) -> None:
        """Close the client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
