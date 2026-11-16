# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the Pinecone vector store — request shaping on a fake index."""

from __future__ import annotations

import json
from typing import Any

import pytest

from locus.rag.stores.base import Document
from locus.rag.stores.pinecone import PineconeVectorStore


class _FakeIndex:
    def __init__(self) -> None:
        self.vectors: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.query_matches: list[dict[str, Any]] = []

    def upsert(self, *, vectors: list[dict[str, Any]], namespace: str) -> None:
        self.calls.append(("upsert", {"vectors": vectors, "namespace": namespace}))
        for v in vectors:
            self.vectors[v["id"]] = v

    def fetch(self, *, ids: list[str], namespace: str) -> dict[str, Any]:
        self.calls.append(("fetch", {"ids": ids, "namespace": namespace}))
        return {"vectors": {i: self.vectors[i] for i in ids if i in self.vectors}}

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("query", kwargs))
        return {"matches": self.query_matches}

    def delete(self, **kwargs: Any) -> None:
        self.calls.append(("delete", kwargs))
        if kwargs.get("delete_all"):
            self.vectors.clear()
        for i in kwargs.get("ids") or []:
            self.vectors.pop(i, None)

    def describe_index_stats(self) -> dict[str, Any]:
        return {"namespaces": {"ns": {"vector_count": len(self.vectors)}}}


def _store(**kwargs: Any) -> tuple[PineconeVectorStore, _FakeIndex]:
    fake = _FakeIndex()
    store = PineconeVectorStore(dimension=3, namespace="ns", _index=fake, **kwargs)
    return store, fake


def _match(doc_id: str, score: float, **meta: Any) -> dict[str, Any]:
    return {
        "id": doc_id,
        "score": score,
        "values": [1.0, 0.0, 0.0],
        "metadata": {
            "_content": f"content {doc_id}",
            "_content_type": "text",
            "_created_at": "2026-01-01T00:00:00+00:00",
            "_metadata_json": json.dumps(meta),
            **meta,
        },
    }


async def test_upsert_flattens_metadata_and_keeps_json_round_trip() -> None:
    store, fake = _store()
    await store.add(
        Document(
            id="a",
            content="alpha",
            embedding=[1.0, 0.0, 0.0],
            metadata={"topic": "x", "year": 2026, "tags": ["p", "q"], "nested": {"k": 1}},
        )
    )
    name, kwargs = fake.calls[-1]
    assert name == "upsert"
    assert kwargs["namespace"] == "ns"
    vector = kwargs["vectors"][0]
    assert vector["id"] == "a"
    assert vector["values"] == [1.0, 0.0, 0.0]
    meta = vector["metadata"]
    assert meta["_content"] == "alpha"
    assert meta["topic"] == "x"
    assert meta["year"] == 2026
    assert meta["tags"] == ["p", "q"]
    assert "nested" not in meta  # not a Pinecone metadata type; only in the JSON copy
    assert json.loads(meta["_metadata_json"])["nested"] == {"k": 1}

    got = await store.get("a")
    assert got is not None
    assert got.content == "alpha"
    assert got.metadata == {"topic": "x", "year": 2026, "tags": ["p", "q"], "nested": {"k": 1}}
    assert got.embedding == [1.0, 0.0, 0.0]


async def test_add_batch_chunks_at_100() -> None:
    store, fake = _store()
    docs = [Document(id=str(i), content="c", embedding=[1.0, 0.0, 0.0]) for i in range(150)]
    ids = await store.add_batch(docs)
    assert len(ids) == 150
    upserts = [c for c in fake.calls if c[0] == "upsert"]
    assert [len(c[1]["vectors"]) for c in upserts] == [100, 50]


async def test_add_requires_embedding() -> None:
    store, _ = _store()
    with pytest.raises(ValueError, match="embedding"):
        await store.add(Document(id="a", content="x"))


async def test_query_shape_filter_and_cosine_score() -> None:
    store, fake = _store()
    fake.query_matches = [_match("a", 1.0, topic="x"), _match("b", 0.2, topic="x")]

    results = await store.search([1.0, 0.0, 0.0], limit=2, metadata_filter={"topic": "x"})

    _, kwargs = fake.calls[-1]
    assert kwargs["vector"] == [1.0, 0.0, 0.0]
    assert kwargs["top_k"] == 2
    assert kwargs["namespace"] == "ns"
    assert kwargs["filter"] == {"topic": {"$eq": "x"}}
    assert kwargs["include_metadata"] is True
    assert kwargs["include_values"] is True
    assert [r.document.id for r in results] == ["a", "b"]
    assert results[0].score == pytest.approx(1.0)
    assert results[0].distance == pytest.approx(0.0)
    assert results[0].document.metadata == {"topic": "x"}


async def test_query_threshold_and_no_filter() -> None:
    store, fake = _store()
    fake.query_matches = [_match("a", 1.0), _match("b", 0.2)]
    results = await store.search([1.0, 0.0, 0.0], limit=2, threshold=0.5)
    assert [r.document.id for r in results] == ["a"]
    assert fake.calls[-1][1]["filter"] is None


async def test_delete_count_clear() -> None:
    store, fake = _store()
    await store.add(Document(id="a", content="x", embedding=[1.0, 0.0, 0.0]))
    assert await store.count() == 1
    assert await store.delete("a") is True
    assert await store.delete("a") is False
    await store.add(Document(id="b", content="y", embedding=[0.0, 1.0, 0.0]))
    assert await store.clear() == 1
    assert ("delete", {"delete_all": True, "namespace": "ns"}) in fake.calls
    assert await store.clear() == 0  # nothing to delete, no call


def test_constructor_validation_and_metric_mapping() -> None:
    with pytest.raises(ValueError, match="distance_metric"):
        PineconeVectorStore(distance_metric="hamming", _index=object())
    l2 = PineconeVectorStore(dimension=3, distance_metric="l2", _index=object())
    assert l2._score(3.0) == (pytest.approx(0.25), 3.0)
    dot = PineconeVectorStore(dimension=3, distance_metric="dot", _index=object())
    assert dot._score(0.4) == (pytest.approx(0.4), None)
    assert PineconeVectorStore.build_filter({"a": 1, "b": "x"}) == {
        "a": {"$eq": 1},
        "b": {"$eq": "x"},
    }


def test_missing_api_key_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pinecone")
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    store = PineconeVectorStore(dimension=3)
    with pytest.raises(ValueError, match="PINECONE_API_KEY"):
        store._get_index_sync()
