# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``OpenSearchVectorStore.hybrid_search`` on a fake client."""

from __future__ import annotations

from typing import Any

import pytest

from locus.rag.stores.base import supports_hybrid_search
from locus.rag.stores.opensearch import OpenSearchVectorStore


class _Indices:
    async def exists(self, *, index: str) -> bool:
        return True


class _FakeOpenSearch:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.indices = _Indices()
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def search(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _hit(doc_id: str, score: float) -> dict[str, Any]:
    return {
        "_score": score,
        "_source": {
            "id": doc_id,
            "content": f"content {doc_id}",
            "embedding": [1.0, 0.0],
            "metadata": {"topic": "x"},
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    }


def _store(responses: list[dict[str, Any]]) -> tuple[OpenSearchVectorStore, _FakeOpenSearch]:
    fake = _FakeOpenSearch(responses)
    store = OpenSearchVectorStore(dimension=2, index_name="t")
    store._client = fake  # externally assigned: loop_bound leaves it alone
    return store, fake


def test_opensearch_advertises_hybrid() -> None:
    store, _ = _store([])
    assert supports_hybrid_search(store) is True


async def test_hybrid_runs_knn_then_match_and_blends() -> None:
    store, fake = _store(
        [
            {"hits": {"hits": [_hit("a", 1.0), _hit("b", 0.6)]}},  # k-NN leg
            {"hits": {"hits": [_hit("b", 9.0), _hit("a", 1.0)]}},  # BM25 leg
        ]
    )
    results = await store.hybrid_search(
        "chargeback", [1.0, 0.0], limit=2, alpha=0.0, metadata_filter={"topic": "x"}
    )
    knn_body, match_body = fake.calls[0]["body"], fake.calls[1]["body"]
    assert "knn" in str(knn_body["query"])
    assert knn_body["size"] == 20
    assert match_body["query"]["bool"]["must"] == [{"match": {"content": "chargeback"}}]
    assert match_body["query"]["bool"]["filter"] == [{"term": {"metadata.topic": "x"}}]
    assert [r.document.id for r in results] == ["b", "a"]
    assert results[0].score == pytest.approx(1.0)


async def test_hybrid_alpha_one_keeps_vector_order() -> None:
    store, _ = _store(
        [
            {"hits": {"hits": [_hit("a", 1.0), _hit("b", 0.6)]}},
            {"hits": {"hits": [_hit("b", 9.0)]}},
        ]
    )
    results = await store.hybrid_search("chargeback", [1.0, 0.0], limit=2, alpha=1.0)
    assert [r.document.id for r in results] == ["a", "b"]


async def test_hybrid_blank_query_is_vector_only() -> None:
    store, fake = _store([{"hits": {"hits": [_hit("a", 1.0)]}}])
    results = await store.hybrid_search("  ", [1.0, 0.0], limit=2)
    assert [r.document.id for r in results] == ["a"]
    assert len(fake.calls) == 1


async def test_hybrid_rejects_bad_alpha() -> None:
    store, _ = _store([])
    with pytest.raises(ValueError, match="alpha"):
        await store.hybrid_search("q", [1.0, 0.0], alpha=3.0)
