# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for RAGRetriever defensive type coercion.

Some model providers (notably gpt-5.x) JSON-encode tool arguments as
strings (`"min_score": "0.5"` instead of `0.5`). `RAGRetriever.retrieve`
coerces `threshold` and `limit` defensively before forwarding to the
store, so the downstream `score < threshold` comparison never sees a
string and never TypeErrors.
"""

from __future__ import annotations

from typing import Any

import pytest

from locus.rag.retriever import RAGRetriever
from locus.rag.stores.base import BaseVectorStore, Document, SearchResult


class _StubStore(BaseVectorStore):
    """Records the threshold/limit it was called with."""

    def __init__(self) -> None:
        self.last_threshold: Any = "<unset>"
        self.last_limit: Any = "<unset>"

    @property
    def config(self) -> Any:
        from locus.rag.stores.base import VectorStoreConfig

        return VectorStoreConfig(dimension=4, distance_metric="cosine", index_type="hnsw")

    async def add(self, document: Document) -> str:
        return document.id

    async def add_batch(self, documents: list[Document]) -> list[str]:
        return [d.id for d in documents]

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        self.last_threshold = threshold
        self.last_limit = limit
        return []

    async def get(self, doc_id: str) -> Document | None:
        return None

    async def delete(self, doc_id: str) -> bool:
        return True

    async def count(self) -> int:
        return 0

    async def clear(self) -> None:
        pass

    async def close(self) -> None:
        pass


class _StubEmbedder:
    """Minimal embedder for the retriever path — no real embeddings calls."""

    async def embed_query(self, query: str) -> Any:
        from locus.rag.embeddings.base import EmbeddingResult

        return EmbeddingResult(embedding=[0.0, 0.0, 0.0, 0.0], text=query, model="stub", tokens=0)


@pytest.mark.asyncio
async def test_threshold_coerces_string_to_float() -> None:
    """gpt-5.x sends `"0.5"`; retriever should hand `0.5` to the store."""
    store = _StubStore()
    retriever = RAGRetriever(embedder=_StubEmbedder(), store=store)

    await retriever.retrieve("q", threshold="0.42")  # type: ignore[arg-type]

    assert store.last_threshold == pytest.approx(0.42)
    assert isinstance(store.last_threshold, float)


@pytest.mark.asyncio
async def test_threshold_unparseable_string_becomes_none() -> None:
    """A non-numeric string should be treated as 'no threshold', not TypeError."""
    store = _StubStore()
    retriever = RAGRetriever(embedder=_StubEmbedder(), store=store)

    await retriever.retrieve("q", threshold="not-a-number")  # type: ignore[arg-type]

    assert store.last_threshold is None


@pytest.mark.asyncio
async def test_threshold_passes_float_through() -> None:
    """Floats survive the coercion path unchanged."""
    store = _StubStore()
    retriever = RAGRetriever(embedder=_StubEmbedder(), store=store)

    await retriever.retrieve("q", threshold=0.6)

    assert store.last_threshold == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_threshold_none_stays_none() -> None:
    """None means 'no filter' and must remain None."""
    store = _StubStore()
    retriever = RAGRetriever(embedder=_StubEmbedder(), store=store)

    await retriever.retrieve("q", threshold=None)

    assert store.last_threshold is None


@pytest.mark.asyncio
async def test_limit_coerces_string_to_int() -> None:
    """Some providers stringify integer args too."""
    store = _StubStore()
    retriever = RAGRetriever(embedder=_StubEmbedder(), store=store)

    await retriever.retrieve("q", limit="7")  # type: ignore[arg-type]

    assert store.last_limit == 7
    assert isinstance(store.last_limit, int)


@pytest.mark.asyncio
async def test_limit_unparseable_falls_back_to_default() -> None:
    """An unparseable limit string falls back to the retriever's default (5)."""
    store = _StubStore()
    retriever = RAGRetriever(embedder=_StubEmbedder(), store=store)

    await retriever.retrieve("q", limit="five")  # type: ignore[arg-type]

    assert store.last_limit == 5
