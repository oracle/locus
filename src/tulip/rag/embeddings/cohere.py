# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Cohere embeddings — calls Cohere's hosted ``embed`` API directly.

Uses the official ``cohere`` Python SDK against Cohere's own endpoint.
Cohere's ``embed-*-v3.0`` models embed queries and documents into
distinct spaces (selected via ``input_type``), so this provider
advertises ``supports_query_vs_doc=True`` and overrides
``embed_query`` / ``embed_documents`` accordingly.

Needs a ``COHERE_API_KEY``.
"""

from __future__ import annotations

import os
from typing import Any

from tulip.rag.embeddings.base import (
    BaseEmbedding,
    EmbeddingCapabilities,
    EmbeddingConfig,
    EmbeddingResult,
)


#: Default Cohere embedding model and its output dimension.
DEFAULT_COHERE_EMBED_MODEL = "embed-english-v3.0"

_MODEL_DIMENSIONS: dict[str, int] = {
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
}


class CohereEmbeddings(BaseEmbedding):
    """Embedding provider backed by Cohere's hosted ``embed`` endpoint.

    Args:
        model: Cohere embedding model id. Defaults to ``embed-english-v3.0``.
        api_key: Cohere API key. Defaults to the ``COHERE_API_KEY`` env var.
        dimension: Override the output dimension. Defaults to the known
            dimension for ``model`` (1024 for the standard v3 models).
        _client: Injection seam for tests — an object exposing an async
            ``embed(...)`` method bypasses the lazy ``cohere`` import.
    """

    def __init__(
        self,
        model: str = DEFAULT_COHERE_EMBED_MODEL,
        api_key: str | None = None,
        dimension: int | None = None,
        _client: Any = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        self._client_override = _client
        self._client: Any = None
        self._embedding_config = EmbeddingConfig(
            dimension=dimension or _MODEL_DIMENSIONS.get(model, 1024),
            max_tokens=512,
            batch_size=96,
        )

    @property
    def config(self) -> EmbeddingConfig:
        return self._embedding_config

    @property
    def capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            supports_query_vs_doc=True,
            supports_multimodal=False,
            supports_batching=True,
            max_batch_size=96,
            max_input_tokens=512,
        )

    def _get_client(self) -> Any:
        if self._client_override is not None:
            return self._client_override
        if self._client is not None:
            return self._client
        try:
            import cohere  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                'cohere is not installed. Install with: pip install "tulip-agents[cohere]"'
            ) from e
        self._client = cohere.AsyncClientV2(api_key=self.api_key)
        return self._client

    async def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        client = self._get_client()
        response = await client.embed(
            texts=texts,
            model=self.model,
            input_type=input_type,
            embedding_types=["float"],
        )
        return list(response.embeddings.float)

    async def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text as a document."""
        vectors = await self._embed([text], input_type="search_document")
        return EmbeddingResult(embedding=vectors[0], text=text, model=self.model)

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed multiple texts as documents in one request."""
        if not texts:
            return []
        vectors = await self._embed(texts, input_type="search_document")
        return [
            EmbeddingResult(embedding=v, text=t, model=self.model)
            for t, v in zip(texts, vectors, strict=True)
        ]

    async def embed_query(self, query: str) -> EmbeddingResult:
        """Embed a query in Cohere's query space (``input_type=search_query``)."""
        vectors = await self._embed([query], input_type="search_query")
        return EmbeddingResult(embedding=vectors[0], text=query, model=self.model)

    async def embed_documents(self, documents: list[str]) -> list[EmbeddingResult]:
        """Embed documents in Cohere's document space."""
        return await self.embed_batch(documents)

    async def close(self) -> None:
        """Close the underlying client if it exposes a close()."""
        client = self._client
        if client is not None:
            close = getattr(client, "close", None)
            if close is not None:
                result = close()
                if hasattr(result, "__await__"):
                    await result
            self._client = None
