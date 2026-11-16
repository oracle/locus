# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the MongoDB Atlas vector store — pipeline shaping on a fake collection.

A fake ``pymongo`` module is injected into ``sys.modules`` where the adapter
imports the client, ``ReplaceOne`` or ``SearchIndexModel`` lazily, so the
tests need no library.
"""

from __future__ import annotations

import asyncio
import sys
import types
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from locus.rag.stores import mongodb as mongodb_module
from locus.rag.stores.base import Document
from locus.rag.stores.mongodb import MongoDBAtlasVectorStore


class _Cursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = iter(rows)

    def __aiter__(self) -> _Cursor:
        return self

    async def __anext__(self) -> dict[str, Any]:
        try:
            return next(self._rows)
        except StopIteration as e:
            raise StopAsyncIteration from e


class _FakeCollection:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, Any]] = []
        self.search_rows: list[dict[str, Any]] = []
        self.indexes: list[dict[str, Any]] = []

    async def replace_one(
        self, flt: dict[str, Any], body: dict[str, Any], upsert: bool = False
    ) -> None:
        self.calls.append(("replace_one", (flt, body, upsert)))
        self.rows[flt["_id"]] = body

    async def bulk_write(self, ops: list[Any]) -> None:
        self.calls.append(("bulk_write", ops))
        for op in ops:
            doc = op._doc
            self.rows[doc["_id"]] = doc

    async def find_one(self, flt: dict[str, Any]) -> dict[str, Any] | None:
        self.calls.append(("find_one", flt))
        return self.rows.get(flt["_id"])

    async def delete_one(self, flt: dict[str, Any]) -> Any:
        self.calls.append(("delete_one", flt))
        removed = self.rows.pop(flt["_id"], None)
        return SimpleNamespace(deleted_count=1 if removed else 0)

    async def aggregate(self, pipeline: list[dict[str, Any]]) -> _Cursor:
        self.calls.append(("aggregate", pipeline))
        return _Cursor(self.search_rows)

    async def count_documents(self, flt: dict[str, Any]) -> int:
        return len(self.rows)

    async def delete_many(self, flt: dict[str, Any]) -> Any:
        n = len(self.rows)
        self.rows.clear()
        return SimpleNamespace(deleted_count=n)

    async def list_search_indexes(self, name: str | None = None) -> _Cursor:
        self.calls.append(("list_search_indexes", name))
        return _Cursor([i for i in self.indexes if name is None or i["name"] == name])

    async def create_search_index(self, model: Any) -> None:
        self.calls.append(("create_search_index", model))
        self.indexes.append({"name": model.document["name"], "queryable": True})


class _FakeDatabase:
    def __init__(self, name: str) -> None:
        self.name = name

    def __getitem__(self, collection: str) -> _FakeCollection:
        fake = _FakeCollection()
        fake.path = (self.name, collection)  # type: ignore[attr-defined]
        return fake


def _fake_pymongo(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Install a fake ``pymongo`` (+ ``pymongo.operations``); returns the clients built."""
    clients: list[Any] = []
    module = types.ModuleType("pymongo")
    operations = types.ModuleType("pymongo.operations")

    class AsyncMongoClient:
        def __init__(self, connection_string: str) -> None:
            self.connection_string = connection_string
            self.closed = False
            clients.append(self)

        def __getitem__(self, database: str) -> _FakeDatabase:
            return _FakeDatabase(database)

        async def close(self) -> None:
            self.closed = True

    class ReplaceOne:
        def __init__(self, flt: dict[str, Any], replacement: dict[str, Any], upsert: bool = False):
            self._filter = flt
            self._doc = replacement
            self._upsert = upsert

    class SearchIndexModel:
        def __init__(self, *, definition: dict[str, Any], name: str, type: str) -> None:  # noqa: A002
            self.document = {"definition": definition, "name": name, "type": type}

    module.AsyncMongoClient = AsyncMongoClient  # type: ignore[attr-defined]
    module.ReplaceOne = ReplaceOne  # type: ignore[attr-defined]
    module.operations = operations  # type: ignore[attr-defined]
    operations.SearchIndexModel = SearchIndexModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pymongo", module)
    monkeypatch.setitem(sys.modules, "pymongo.operations", operations)
    return clients


def _store(**kwargs: Any) -> tuple[MongoDBAtlasVectorStore, _FakeCollection]:
    fake = _FakeCollection()
    store = MongoDBAtlasVectorStore(dimension=3, _collection=fake, **kwargs)
    return store, fake


def _row(doc_id: str, score: float, **meta: Any) -> dict[str, Any]:
    return {
        "_id": doc_id,
        "content": f"content {doc_id}",
        "content_type": "text",
        "embedding": [1.0, 0.0, 0.0],
        "metadata": meta,
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "score": score,
    }


async def test_add_upserts_by_id_and_round_trips() -> None:
    store, fake = _store()
    await store.add(
        Document(id="a", content="alpha", embedding=[1.0, 0.0, 0.0], metadata={"k": "v"})
    )
    name, (flt, body, upsert) = fake.calls[-1]
    assert name == "replace_one"
    assert flt == {"_id": "a"}
    assert upsert is True
    assert body["embedding"] == [1.0, 0.0, 0.0]
    assert body["metadata"] == {"k": "v"}
    got = await store.get("a")
    assert got is not None
    assert got.id == "a"
    assert got.content == "alpha"
    assert got.metadata == {"k": "v"}
    assert got.created_at.tzinfo is not None


async def test_add_batch_uses_bulk_replace_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_pymongo(monkeypatch)
    store, fake = _store()
    assert await store.add_batch([]) == []
    ids = await store.add_batch(
        [
            Document(id="a", content="x", embedding=[1.0, 0.0, 0.0]),
            Document(id="b", content="y", embedding=[0.0, 1.0, 0.0]),
        ]
    )
    assert ids == ["a", "b"]
    name, ops = fake.calls[-1]
    assert name == "bulk_write"
    assert len(ops) == 2
    assert type(ops[0]).__name__ == "ReplaceOne"
    assert ops[0]._filter == {"_id": "a"}
    assert ops[0]._upsert is True
    assert await store.count() == 2


async def test_add_requires_embedding() -> None:
    store, _ = _store()
    with pytest.raises(ValueError, match="embedding"):
        await store.add(Document(id="a", content="x"))


async def test_search_pipeline_shape_and_cosine_mapping() -> None:
    store, fake = _store(filter_fields=["topic"], index_name="idx")
    fake.search_rows = [_row("a", 1.0, topic="x"), _row("b", 0.5, topic="x")]

    results = await store.search([1.0, 0.0, 0.0], limit=2, metadata_filter={"topic": "x"})

    _, pipeline = fake.calls[-1]
    stage = pipeline[0]["$vectorSearch"]
    assert stage["index"] == "idx"
    assert stage["path"] == "embedding"
    assert stage["queryVector"] == [1.0, 0.0, 0.0]
    assert stage["limit"] == 2
    assert stage["numCandidates"] == 100
    assert stage["filter"] == {"metadata.topic": {"$eq": "x"}}
    assert pipeline[1]["$project"]["score"] == {"$meta": "vectorSearchScore"}
    # Atlas reports (1 + cos) / 2.
    assert [r.document.id for r in results] == ["a", "b"]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.0)
    assert results[0].document.metadata == {"topic": "x"}


async def test_search_rejects_filter_on_undeclared_field() -> None:
    store, _ = _store(filter_fields=["topic"])
    with pytest.raises(ValueError, match="filter_fields"):
        await store.search([1.0, 0.0, 0.0], metadata_filter={"year": 2026})


async def test_search_threshold() -> None:
    store, fake = _store()
    fake.search_rows = [_row("a", 1.0), _row("b", 0.6)]
    results = await store.search([1.0, 0.0, 0.0], limit=2, threshold=0.5)
    assert [r.document.id for r in results] == ["a"]


async def test_ensure_index_creates_definition_with_filter_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_pymongo(monkeypatch)
    store, fake = _store(filter_fields=["topic", "year"], index_name="vec")
    await store.ensure_index(wait=True)
    name, model = next(c for c in fake.calls if c[0] == "create_search_index")
    assert model.document["name"] == "vec"
    assert model.document["type"] == "vectorSearch"
    definition = model.document["definition"]
    assert definition["fields"][0] == {
        "type": "vector",
        "path": "embedding",
        "numDimensions": 3,
        "similarity": "cosine",
    }
    assert {"type": "filter", "path": "metadata.topic"} in definition["fields"]
    assert {"type": "filter", "path": "metadata.year"} in definition["fields"]
    # Second call finds it and does not create again.
    await store.ensure_index()
    assert len([c for c in fake.calls if c[0] == "create_search_index"]) == 1


async def test_delete_clear() -> None:
    store, _ = _store()
    await store.add(Document(id="a", content="x", embedding=[1.0, 0.0, 0.0]))
    assert await store.delete("a") is True
    assert await store.delete("a") is False
    await store.add(Document(id="b", content="y", embedding=[0.0, 1.0, 0.0]))
    assert await store.clear() == 1
    assert await store.count() == 0


def test_constructor_validation_and_metric_mapping() -> None:
    with pytest.raises(ValueError, match="distance_metric"):
        MongoDBAtlasVectorStore(distance_metric="hamming", _collection=object())
    store = MongoDBAtlasVectorStore(dimension=3, distance_metric="euclidean", _collection=object())
    assert store.index_definition()["fields"][0]["similarity"] == "euclidean"
    assert (store.config.dimension, store.config.distance_metric) == (3, "euclidean")
    assert store.config.index_type == "hnsw"
    assert store._score(0.5) == (pytest.approx(0.5), pytest.approx(1.0))
    dot = MongoDBAtlasVectorStore(dimension=3, distance_metric="dot_product", _collection=object())
    assert dot._score(0.75) == (pytest.approx(0.5), None)


def test_missing_connection_string_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_pymongo(monkeypatch)
    monkeypatch.delenv("MONGODB_URI", raising=False)
    store = MongoDBAtlasVectorStore(dimension=3)
    with pytest.raises(ValueError, match="MONGODB_URI"):
        store._get_collection()


def test_missing_pymongo_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pymongo", None)
    with pytest.raises(ImportError, match=r"locus-sdk\[mongodb\]"):
        MongoDBAtlasVectorStore(dimension=3, connection_string="mongodb://x")._get_collection()


async def test_collection_built_from_connection_string_and_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients = _fake_pymongo(monkeypatch)
    monkeypatch.setenv("MONGODB_URI", "mongodb://from-env")
    store = MongoDBAtlasVectorStore(dimension=3, database="db", collection="c")
    await store.close()  # nothing built yet: no-op
    coll = store._get_collection()
    assert coll.path == ("db", "c")
    assert clients[0].connection_string == "mongodb://from-env"
    assert store._get_collection() is coll  # cached
    await store.close()
    assert clients[0].closed is True
    assert store._collection is None
    explicit = MongoDBAtlasVectorStore(dimension=3, connection_string="mongodb://given")
    explicit._get_collection()
    assert clients[1].connection_string == "mongodb://given"
    assert "documents" in repr(explicit)


async def test_ensure_index_waits_until_queryable_and_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_pymongo(monkeypatch)
    slept: list[float] = []

    async def no_sleep(seconds: float) -> None:
        slept.append(seconds)
        for index in fake.indexes:
            index["queryable"] = True

    monkeypatch.setattr(
        mongodb_module,
        "asyncio",
        SimpleNamespace(get_running_loop=asyncio.get_running_loop, sleep=no_sleep),
    )
    store, fake = _store(index_name="vec")
    fake.indexes.append({"name": "vec", "queryable": False})
    await store.ensure_index(wait=True)
    assert slept == [5.0]
    assert not [c for c in fake.calls if c[0] == "create_search_index"]

    fake.indexes[0]["queryable"] = False
    with pytest.raises(TimeoutError, match="'vec' was not queryable"):
        await store.ensure_index(wait=True, wait_timeout=0.0)


def test_to_document_created_at_variants_and_scores() -> None:
    naive = MongoDBAtlasVectorStore._to_document({"_id": "a", "created_at": "2026-01-01T00:00:00"})
    assert naive.created_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert naive.embedding is None
    assert naive.content == ""
    garbage = MongoDBAtlasVectorStore._to_document({"_id": "a", "created_at": "nope"})
    assert garbage.created_at.tzinfo is not None
    assert MongoDBAtlasVectorStore._to_document({}).id == ""
    euclidean = MongoDBAtlasVectorStore(dimension=3, distance_metric="l2", _collection=object())
    assert euclidean._score(0.0) == (0.0, float("inf"))


async def test_pipeline_num_candidates_override_and_missing_document() -> None:
    store, fake = _store(num_candidates=7)
    stage = store.pipeline([1.0, 0.0, 0.0], 3, None)[0]["$vectorSearch"]
    assert stage["numCandidates"] == 7
    assert "filter" not in stage
    assert store.build_filter(None) is None
    assert await store.get("missing") is None
