# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Unit tests for ``tulip.rag.reranker`` (closes #216).

Test surface:

  * ``RAGRetriever`` wired to a fake reranker — confirms the
    over-fetch + rerank + trim plumbing in ``retrieve()``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from tulip.rag import RAGRetriever, Reranker
from tulip.rag.stores.base import Document, SearchResult


def _doc(doc_id: str, content: str) -> Document:
    return Document(
        id=doc_id,
        content=content,
        embedding=[0.0],
        metadata={},
        created_at=datetime.now(UTC),
    )


def _hit(doc_id: str, content: str, score: float) -> SearchResult:
    return SearchResult(document=_doc(doc_id, content), score=score, distance=None)


# ============================================================================
# RAGRetriever — reranker plumbed through the retrieve() path.
# ============================================================================


class _FakeReranker(Reranker):
    """Reverses the candidate order. Tracks how many calls + what limit
    was passed so the retriever's over-fetch behaviour is observable."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_query: str | None = None
        self.last_candidates_len: int = 0

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        self.calls += 1
        self.last_query = query
        self.last_candidates_len = len(candidates)
        return list(reversed(candidates))


class _FakeStore:
    """Captures the limit the retriever asks for and returns N hits."""

    def __init__(self, n_hits: int) -> None:
        self.n_hits = n_hits
        self.last_limit: int | None = None

    @property
    def config(self) -> Any:  # for ``store_type`` logging
        return SimpleNamespace(distance_metric="cosine")

    async def search(
        self,
        query_embedding: list[float],  # noqa: ARG002
        limit: int,
        threshold: float | None = None,  # noqa: ARG002
        metadata_filter: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> list[SearchResult]:
        self.last_limit = limit
        return [_hit(str(i), f"doc-{i}", 1.0 - i * 0.01) for i in range(min(self.n_hits, limit))]


class _FakeEmbedder:
    async def embed_query(self, query: str) -> Any:  # noqa: ARG002
        return SimpleNamespace(embedding=[0.0])


@pytest.mark.asyncio
async def test_retrieve_without_reranker_uses_limit_directly() -> None:
    """Default behaviour (no reranker) — the retriever asks the store
    for exactly ``limit`` hits. No over-fetch."""
    store = _FakeStore(n_hits=100)
    retriever = RAGRetriever(embedder=_FakeEmbedder(), store=store)

    result = await retriever.retrieve("q", limit=5)

    assert store.last_limit == 5
    assert len(result.documents) == 5


@pytest.mark.asyncio
async def test_retrieve_with_reranker_overfetches_then_trims() -> None:
    """With a reranker wired in, the retriever asks the store for
    ``rerank_candidate_pool`` hits (default 50), the reranker reorders
    them, and ``retrieve()`` trims back to ``limit``."""
    store = _FakeStore(n_hits=100)
    reranker = _FakeReranker()
    retriever = RAGRetriever(
        embedder=_FakeEmbedder(),
        store=store,
        reranker=reranker,
        rerank_candidate_pool=20,
    )

    result = await retriever.retrieve("q", limit=3)

    # Over-fetched 20 from the store, then trimmed to 3.
    assert store.last_limit == 20
    assert reranker.calls == 1
    assert reranker.last_candidates_len == 20
    assert len(result.documents) == 3

    # _FakeReranker reverses order → the trimmed top-3 are the worst-
    # scoring originals (id=19, 18, 17).
    assert [r.document.id for r in result.documents] == ["19", "18", "17"]


@pytest.mark.asyncio
async def test_retrieve_with_empty_store_skips_reranker() -> None:
    """If the vector store returns zero hits, the reranker is not
    called (matches the reranker's own empty-input contract)."""
    store = _FakeStore(n_hits=0)
    reranker = _FakeReranker()
    retriever = RAGRetriever(
        embedder=_FakeEmbedder(),
        store=store,
        reranker=reranker,
        rerank_candidate_pool=20,
    )

    result = await retriever.retrieve("q", limit=5)

    assert reranker.calls == 0
    assert len(result.documents) == 0
