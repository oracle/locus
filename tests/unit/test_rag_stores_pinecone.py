# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the Pinecone vector store — request shaping on a fake index.

A fake ``pinecone`` module is injected into ``sys.modules`` where the adapter
imports the client lazily, so the index-provisioning path runs without the
library.
"""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

from locus.rag.stores import pinecone as pinecone_module
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


class _FakePinecone:
    """``Pinecone`` control-plane client: indexes become ready after ``ready_after`` polls."""

    def __init__(self, *, api_key: str, existing: tuple[str, ...], ready_after: int) -> None:
        self.api_key = api_key
        self.indexes: dict[str, _FakeIndex] = {name: _FakeIndex() for name in existing}
        self.ready_after = ready_after
        self.calls: list[tuple[str, Any]] = []

    def has_index(self, name: str) -> bool:
        return name in self.indexes

    def create_index(self, **kwargs: Any) -> None:
        self.calls.append(("create_index", kwargs))
        self.indexes[kwargs["name"]] = _FakeIndex()

    def describe_index(self, name: str) -> dict[str, Any]:
        self.calls.append(("describe_index", name))
        self.ready_after -= 1
        return {"status": {"ready": self.ready_after < 0}}

    def Index(self, name: str) -> _FakeIndex:  # noqa: N802 — the pinecone method name
        return self.indexes[name]


class _Clock:
    def __init__(self, step: float) -> None:
        self.now = 0.0
        self.step = step
        self.slept: list[float] = []

    def monotonic(self) -> float:
        self.now += self.step
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)


def _fake_pinecone(
    monkeypatch: pytest.MonkeyPatch, *, existing: tuple[str, ...] = (), ready_after: int = 0
) -> list[_FakePinecone]:
    """Install a fake ``pinecone`` module; returns the control-plane clients built."""
    clients: list[_FakePinecone] = []
    module = types.ModuleType("pinecone")

    class Pinecone(_FakePinecone):
        def __init__(self, *, api_key: str) -> None:
            super().__init__(api_key=api_key, existing=existing, ready_after=ready_after)
            clients.append(self)

    class ServerlessSpec:
        def __init__(self, *, cloud: str, region: str) -> None:
            self.cloud = cloud
            self.region = region

    module.Pinecone = Pinecone  # type: ignore[attr-defined]
    module.ServerlessSpec = ServerlessSpec  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pinecone", module)
    return clients


def _patch_clock(monkeypatch: pytest.MonkeyPatch, step: float) -> _Clock:
    clock = _Clock(step)
    monkeypatch.setattr(
        pinecone_module, "time", SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep)
    )
    return clock


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
            metadata={
                "topic": "x",
                "year": 2026,
                "tags": ["p", "q"],
                "nested": {"k": 1},
                "_private": "never flattened",
            },
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
    assert "_private" not in meta  # ``_`` keys are reserved for the store's own fields
    assert json.loads(meta["_metadata_json"])["nested"] == {"k": 1}

    got = await store.get("a")
    assert got is not None
    assert got.content == "alpha"
    assert got.metadata == {
        "topic": "x",
        "year": 2026,
        "tags": ["p", "q"],
        "nested": {"k": 1},
        "_private": "never flattened",
    }
    assert got.embedding == [1.0, 0.0, 0.0]


async def test_add_batch_chunks_at_100() -> None:
    store, fake = _store()
    assert await store.add_batch([]) == []
    assert store.config.index_type == "hnsw"
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
    _fake_pinecone(monkeypatch)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    store = PineconeVectorStore(dimension=3)
    with pytest.raises(ValueError, match="PINECONE_API_KEY"):
        store._get_index_sync()


def test_missing_pinecone_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pinecone", None)
    with pytest.raises(ImportError, match=r"locus-sdk\[pinecone\]"):
        PineconeVectorStore(dimension=3, api_key="k")._get_index_sync()


async def test_existing_index_is_opened_without_provisioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients = _fake_pinecone(monkeypatch, existing=("locus",))
    monkeypatch.setenv("PINECONE_API_KEY", "from-env")
    store = PineconeVectorStore(dimension=3)
    index = store._get_index_sync()
    assert clients[0].api_key == "from-env"
    assert clients[0].calls == []
    assert index is clients[0].indexes["locus"]
    assert store._get_index_sync() is index  # cached
    await store.add(Document(id="a", content="x", embedding=[1.0, 0.0, 0.0]))
    assert "a" in index.vectors
    await store.close()
    assert store._index is None


def test_missing_index_is_created_and_polled_until_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    clients = _fake_pinecone(monkeypatch, ready_after=1)
    clock = _patch_clock(monkeypatch, step=1.0)
    store = PineconeVectorStore(
        dimension=3, index_name="kb", distance_metric="l2", api_key="k", cloud="oci", region="r"
    )
    index = store._get_index_sync()
    (pc,) = clients
    create = pc.calls[0]
    assert create[0] == "create_index"
    assert create[1]["name"] == "kb"
    assert create[1]["dimension"] == 3
    assert create[1]["metric"] == "euclidean"
    assert (create[1]["spec"].cloud, create[1]["spec"].region) == ("oci", "r")
    assert [c for c in pc.calls if c[0] == "describe_index"] == [("describe_index", "kb")] * 2
    assert clock.slept == [2.0]
    assert index is pc.indexes["kb"]


def test_missing_index_with_create_index_false_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_pinecone(monkeypatch)
    store = PineconeVectorStore(dimension=3, api_key="k", create_index=False)
    with pytest.raises(ValueError, match="create_index=False"):
        store._get_index_sync()


def test_index_provisioning_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_pinecone(monkeypatch, ready_after=10**6)
    clock = _patch_clock(monkeypatch, step=301.0)  # the first poll is already past the deadline
    store = PineconeVectorStore(dimension=3, api_key="k")
    with pytest.raises(TimeoutError, match="not ready after 300s"):
        store._get_index_sync()
    assert clock.slept == []


def test_to_document_tolerates_bad_metadata() -> None:
    listed = PineconeVectorStore._to_document(
        "a",
        None,
        {"_content": "c", "_metadata_json": "[1, 2]", "_created_at": "nope", "topic": "x"},
    )
    assert listed.metadata == {"topic": "x"}  # falls back to the flattened keys
    assert listed.embedding is None
    assert listed.created_at.tzinfo is not None
    broken = PineconeVectorStore._to_document("a", [], {"_metadata_json": "{broken"})
    assert broken.metadata == {}
    assert broken.content == ""
    assert broken.content_type == "text"
    assert PineconeVectorStore._to_document("a", None, None).metadata == {}


async def test_object_responses_and_missing_namespace_stats() -> None:
    """The real client returns objects, not dicts; ``_get`` reads both."""

    class _ObjectIndex:
        def fetch(self, *, ids: list[str], namespace: str) -> Any:
            record = SimpleNamespace(values=[1.0, 0.0, 0.0], metadata={"_content": "obj"})
            return SimpleNamespace(vectors={"a": record} if "a" in ids else None)

        def query(self, **kwargs: Any) -> Any:
            return SimpleNamespace(
                matches=[SimpleNamespace(id="a", score=0.5, values=None, metadata=None)]
            )

        def describe_index_stats(self) -> Any:
            return SimpleNamespace(namespaces={"other": {"vector_count": 3}})

    store = PineconeVectorStore(dimension=3, namespace="ns", _index=_ObjectIndex())
    got = await store.get("a")
    assert got is not None
    assert got.content == "obj"
    assert got.embedding == [1.0, 0.0, 0.0]
    assert await store.get("missing") is None
    (hit,) = await store.search([1.0, 0.0, 0.0], limit=1)
    assert hit.document.id == "a"
    assert hit.score == pytest.approx(0.5)
    assert hit.document.embedding is None
    assert await store.count() == 0  # namespace absent from the stats
    assert await store.clear() == 0
