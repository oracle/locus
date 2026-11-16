# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``RAGRetriever(hybrid=True)`` and the client-side blend helper."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from locus.rag import RAGRetriever
from locus.rag.stores._hybrid import blend_hybrid
from locus.rag.stores.base import Document, SearchResult, supports_hybrid_search
from locus.rag.stores.memory import InMemoryVectorStore


def _hit(doc_id: str, score: float, distance: float | None = None) -> SearchResult:
    return SearchResult(
        document=Document(id=doc_id, content=doc_id), score=score, distance=distance
    )


class _FakeEmbedder:
    async def embed_query(self, text: str) -> Any:
        return SimpleNamespace(embedding=[1.0, 0.0])


class _PlainStore:
    def __init__(self) -> None:
        self.search_calls: list[dict[str, Any]] = []

    async def search(self, **kwargs: Any) -> list[SearchResult]:
        self.search_calls.append(kwargs)
        return [_hit("vec", 0.9)]


class _HybridStore(_PlainStore):
    def __init__(self) -> None:
        super().__init__()
        self.hybrid_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def hybrid_search(self, *args: Any, **kwargs: Any) -> list[SearchResult]:
        self.hybrid_calls.append((args, kwargs))
        return [_hit("blend", 0.8)]


class _UnavailableHybridStore(_HybridStore):
    hybrid_search_available = False


def _retriever(store: Any, **kwargs: Any) -> RAGRetriever:
    return RAGRetriever(embedder=_FakeEmbedder(), store=store, **kwargs)


async def test_hybrid_calls_store_hybrid_search_with_alpha_and_kwargs() -> None:
    store = _HybridStore()
    retriever = _retriever(
        store, hybrid=True, hybrid_alpha=0.3, hybrid_kwargs={"use_text_index": True}
    )
    result = await retriever.retrieve(
        "refund sla", limit=4, threshold=0.1, metadata_filter={"t": "x"}
    )

    assert retriever.hybrid_supported is True
    assert [r.document.id for r in result.documents] == ["blend"]
    assert store.search_calls == []
    ((args, kwargs),) = store.hybrid_calls
    assert args == ("refund sla", [1.0, 0.0])
    assert kwargs == {
        "limit": 4,
        "alpha": 0.3,
        "threshold": 0.1,
        "metadata_filter": {"t": "x"},
        "use_text_index": True,
    }


async def test_hybrid_off_by_default_uses_vector_search() -> None:
    store = _HybridStore()
    result = await _retriever(store).retrieve("q")
    assert [r.document.id for r in result.documents] == ["vec"]
    assert store.hybrid_calls == []


async def test_per_call_override_in_both_directions() -> None:
    store = _HybridStore()
    on = _retriever(store, hybrid=True)
    await on.retrieve("q", hybrid=False)
    assert store.hybrid_calls == []
    assert len(store.search_calls) == 1
    off = _retriever(store)
    await off.retrieve("q", hybrid=True)
    assert len(store.hybrid_calls) == 1


async def test_store_without_hybrid_falls_back_and_logs_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _PlainStore()
    retriever = _retriever(store, hybrid=True)
    with caplog.at_level(logging.WARNING, logger="locus.rag.retriever"):
        first = await retriever.retrieve("q")
        await retriever.retrieve("q")
    assert retriever.hybrid_supported is False
    assert [r.document.id for r in first.documents] == ["vec"]
    assert len(store.search_calls) == 2
    assert caplog.text.count("falling back to vector-only") == 1
    assert "_PlainStore" in caplog.text


async def test_unavailable_hybrid_flag_falls_back() -> None:
    store = _UnavailableHybridStore()
    retriever = _retriever(store, hybrid=True)
    await retriever.retrieve("q")
    assert store.hybrid_calls == []
    assert len(store.search_calls) == 1
    assert supports_hybrid_search(store) is False


async def test_hybrid_with_reranker_overfetches_then_trims() -> None:
    class _Reranker:
        async def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
            return list(reversed(candidates))

    store = _HybridStore()
    retriever = _retriever(store, hybrid=True, reranker=_Reranker(), rerank_candidate_pool=12)
    await retriever.retrieve("q", limit=2)
    assert store.hybrid_calls[0][1]["limit"] == 12


def test_hybrid_alpha_is_validated() -> None:
    with pytest.raises(ValueError):
        _retriever(_HybridStore(), hybrid_alpha=1.5)


def test_supports_hybrid_search_on_real_stores() -> None:
    assert supports_hybrid_search(InMemoryVectorStore(dimension=2)) is False


# ----------------------------------------------------------- blend helper


def test_blend_alpha_one_is_vector_order_and_keeps_distance() -> None:
    out = blend_hybrid(
        [_hit("a", 0.9, 0.1), _hit("b", 0.4, 0.6)],
        [(Document(id="b", content="b"), 7.0), (Document(id="c", content="c"), 3.5)],
        alpha=1.0,
        limit=5,
    )
    assert [r.document.id for r in out] == ["a", "b", "c"]
    assert out[0].score == pytest.approx(0.9)
    assert out[0].distance == 0.1
    assert out[2].score == 0.0
    assert out[2].distance is None


def test_blend_alpha_zero_normalises_lexical_by_max() -> None:
    out = blend_hybrid(
        [_hit("a", 0.9)],
        [(Document(id="b", content="b"), 8.0), (Document(id="a", content="a"), 2.0)],
        alpha=0.0,
        limit=5,
    )
    assert [r.document.id for r in out] == ["b", "a"]
    assert out[0].score == pytest.approx(1.0)
    assert out[1].score == pytest.approx(0.25)


def test_blend_even_mix_threshold_limit_and_clamping() -> None:
    out = blend_hybrid(
        [_hit("a", 1.4), _hit("b", -0.2), _hit("c", 0.5)],
        [(Document(id="b", content="b"), 4.0)],
        alpha=0.5,
        limit=2,
        threshold=0.3,
    )
    # a: vec clamped to 1 → 0.5; b: lex 1.0 → 0.5; c: 0.25 < threshold.
    assert {r.document.id for r in out} == {"a", "b"}
    assert all(r.score == pytest.approx(0.5) for r in out)


def test_blend_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        blend_hybrid([], [], alpha=-0.1, limit=1)
