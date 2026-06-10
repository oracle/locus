# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Cohere reranker — calls Cohere's hosted ``rerank`` API directly.

Uses the official ``cohere`` Python SDK against Cohere's own endpoint
(no cloud-vendor wrapper). Needs a ``COHERE_API_KEY``; for offline /
free reranking use :class:`tulip.rag.reranker.cross_encoder.CrossEncoderReranker`
instead.

Usage::

    from tulip.rag.reranker import CohereReranker

    reranker = CohereReranker(model="rerank-v3.5", top_n=5)
    top = await reranker.rerank("hepcidin in iron metabolism", candidates)
"""

from __future__ import annotations

import os
from typing import Any

from tulip.rag.reranker.base import Reranker
from tulip.rag.stores.base import SearchResult


#: Cohere's current general-purpose rerank model.
DEFAULT_COHERE_RERANK_MODEL = "rerank-v3.5"


class CohereReranker(Reranker):
    """Reranker backed by Cohere's hosted ``rerank`` endpoint.

    Args:
        model: Cohere rerank model id. Defaults to ``rerank-v3.5``.
        api_key: Cohere API key. Defaults to the ``COHERE_API_KEY`` env var.
        top_n: Trim the reranked output to the top N candidates. ``None``
            returns every candidate, reordered.
        max_tokens_per_doc: Optional per-document truncation passed to the
            Cohere API.
        _client: Injection seam for tests — an object exposing an async
            ``rerank(...)`` method bypasses the lazy ``cohere`` import and
            real network calls.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_COHERE_RERANK_MODEL,
        api_key: str | None = None,
        top_n: int | None = None,
        max_tokens_per_doc: int | None = None,
        _client: Any = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        self.top_n = top_n
        self.max_tokens_per_doc = max_tokens_per_doc
        self._client_override = _client
        self._cached_client: Any = None

    def _get_client(self) -> Any:
        if self._client_override is not None:
            return self._client_override
        if self._cached_client is not None:
            return self._cached_client
        try:
            import cohere  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                'cohere is not installed. Install with: pip install "tulip-agents[cohere]"'
            ) from e
        self._cached_client = cohere.AsyncClientV2(api_key=self.api_key)
        return self._cached_client

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        """Reorder ``candidates`` by Cohere relevance score."""
        if not candidates:
            return []

        client = self._get_client()
        documents = [c.document.content for c in candidates]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": documents,
        }
        if self.top_n is not None:
            kwargs["top_n"] = self.top_n
        if self.max_tokens_per_doc is not None:
            kwargs["max_tokens_per_doc"] = self.max_tokens_per_doc

        response = await client.rerank(**kwargs)

        ranked: list[SearchResult] = []
        for item in response.results:
            original = candidates[item.index]
            ranked.append(
                SearchResult(
                    document=original.document,
                    score=float(item.relevance_score),
                    distance=original.score,
                )
            )
        return ranked
