# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Voyage reranker — calls the Voyage AI ``rerank`` endpoint over HTTPS.

Uses ``httpx`` (a core dependency), so no extra package is needed; a
``VOYAGE_API_KEY`` is. The endpoint takes the query and the candidate
texts and returns them indexed and scored.

Usage::

    from locus.rag.reranker import VoyageReranker

    reranker = VoyageReranker(model="rerank-2.5", top_n=5)
    top = await reranker.rerank("refund SLA for digital goods", candidates)
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from locus.rag.reranker.base import Reranker
from locus.rag.stores.base import SearchResult


#: Voyage's current general-purpose rerank model.
DEFAULT_VOYAGE_RERANK_MODEL = "rerank-2.5"
DEFAULT_VOYAGE_BASE_URL = "https://api.voyageai.com/v1/rerank"


class VoyageReranker(Reranker):
    """Reranker backed by the Voyage AI rerank API.

    Args:
        model: Voyage rerank model id. Defaults to ``rerank-2.5``.
        api_key: Voyage API key. Defaults to the ``VOYAGE_API_KEY`` env var.
        top_n: Trim the reranked output to the top N candidates (sent as
            the endpoint's ``top_k``). ``None`` returns every candidate.
        truncation: Let the service truncate over-long inputs instead of
            rejecting them. Defaults to ``True``.
        base_url: Endpoint URL. Override for a proxy or a compatible server.
        timeout: Request timeout in seconds.
        _client: Injection seam for tests — an ``httpx.AsyncClient`` (for
            example one built on ``httpx.MockTransport``).
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_VOYAGE_RERANK_MODEL,
        api_key: str | None = None,
        top_n: int | None = None,
        truncation: bool = True,
        base_url: str = DEFAULT_VOYAGE_BASE_URL,
        timeout: float = 30.0,
        _client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        self.top_n = top_n
        self.truncation = truncation
        self.base_url = base_url
        self.timeout = timeout
        self._client_override = _client

    def build_request(self, query: str, documents: list[str]) -> dict[str, Any]:
        """The JSON body sent to the endpoint."""
        body: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "truncation": self.truncation,
        }
        if self.top_n is not None:
            body["top_k"] = self.top_n
        return body

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError("VoyageReranker needs an API key: pass api_key= or set VOYAGE_API_KEY")
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        if self._client_override is not None:
            response = await self._client_override.post(self.base_url, json=body, headers=headers)
        else:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.base_url, json=body, headers=headers)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return payload

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        """Reorder ``candidates`` by Voyage relevance score."""
        if not candidates:
            return []
        payload = await self._post(
            self.build_request(query, [c.document.content for c in candidates])
        )
        ranked: list[SearchResult] = []
        for item in payload.get("data", []):
            original = candidates[int(item["index"])]
            ranked.append(
                SearchResult(
                    document=original.document,
                    score=float(item["relevance_score"]),
                    distance=original.score,
                )
            )
        ranked.sort(key=lambda r: r.score, reverse=True)
        if self.top_n is not None:
            return ranked[: self.top_n]
        return ranked
