# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the Weaviate store.

A fake ``weaviate`` module is injected into ``sys.modules`` so the adapter's
lazy imports resolve to an in-memory client; the live store is exercised by
``tests/integration/rag/test_store_conformance_live.py``.
"""

from __future__ import annotations

import sys
import types
import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from locus.rag.stores.base import Document, supports_hybrid_search
from locus.rag.stores.weaviate import WeaviateVectorStore, meta_property


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


# ------------------------------------------------------------ fake weaviate


class _Property:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.name = kwargs["name"]
        self.data_type = kwargs["data_type"]


class _Filter:
    def __init__(self, clauses: list[tuple[str, Any]]) -> None:
        self.clauses = clauses

    def __and__(self, other: _Filter) -> _Filter:
        return _Filter(self.clauses + other.clauses)


class _ByProperty:
    def __init__(self, prop: str) -> None:
        self.prop = prop

    def equal(self, value: Any) -> _Filter:
        return _Filter([(self.prop, value)])


class _Object:
    def __init__(self, obj_uuid: str, properties: dict[str, Any], vector: list[float]) -> None:
        self.uuid = obj_uuid
        self.properties = properties
        self.vector: Any = {"default": vector}
        self.metadata = SimpleNamespace(distance=None, score=None)


class _Batch:
    def __init__(self, collection: _FakeCollection) -> None:
        self._collection = collection

    def __enter__(self) -> _Batch:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def add_object(self, *, properties: dict[str, Any], vector: list[float], uuid: str) -> None:
        self._collection.objects[uuid] = _Object(uuid, properties, vector)


class _FakeCollection:
    def __init__(self, name: str, properties: list[_Property], vector_config: Any) -> None:
        self.name = name
        self.properties = properties
        self.vector_config = vector_config
        self.objects: dict[str, _Object] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.failed_objects: list[Any] = []
        self.hybrid_scores: list[float] = []
        self.config = SimpleNamespace(get=self._config_get, add_property=self._add_property)
        self.batch = SimpleNamespace(
            fixed_size=self._fixed_size, failed_objects=self.failed_objects
        )
        self.query = SimpleNamespace(
            fetch_object_by_id=self._fetch, near_vector=self._near_vector, hybrid=self._hybrid
        )
        self.data = SimpleNamespace(delete_by_id=self._delete)
        self.aggregate = SimpleNamespace(over_all=self._over_all)

    def _config_get(self) -> Any:
        return SimpleNamespace(properties=list(self.properties))

    def _add_property(self, prop: _Property) -> None:
        self.calls.append(("add_property", prop.kwargs))
        self.properties.append(prop)

    def _fixed_size(self, *, batch_size: int) -> _Batch:
        self.calls.append(("fixed_size", {"batch_size": batch_size}))
        return _Batch(self)

    def _fetch(self, obj_uuid: str, include_vector: bool = False) -> _Object | None:
        return self.objects.get(obj_uuid)

    def _delete(self, obj_uuid: str) -> None:
        self.objects.pop(obj_uuid, None)

    def _over_all(self, *, total_count: bool) -> Any:
        return SimpleNamespace(total_count=len(self.objects))

    @staticmethod
    def _matches(obj: _Object, filters: _Filter | None) -> bool:
        return filters is None or all(obj.properties.get(p) == v for p, v in filters.clauses)

    def _near_vector(self, *, near_vector: list[float], limit: int, filters: Any, **kw: Any) -> Any:
        self.calls.append(("near_vector", {"filters": filters, "limit": limit, **kw}))
        objects = []
        for obj in self.objects.values():
            if not self._matches(obj, filters):
                continue
            hit = _Object(obj.uuid, obj.properties, obj.vector["default"])
            hit.metadata = SimpleNamespace(distance=1.0 - _dot(obj.vector["default"], near_vector))
            objects.append(hit)
        objects.sort(key=lambda o: o.metadata.distance)
        return SimpleNamespace(objects=objects[:limit])

    def _hybrid(
        self, *, query: str, vector: list[float], alpha: float, limit: int, filters: Any, **kw: Any
    ) -> Any:
        self.calls.append(
            ("hybrid", {"query": query, "alpha": alpha, "limit": limit, "filters": filters, **kw})
        )
        matching = [o for o in self.objects.values() if self._matches(o, filters)]
        scores = self.hybrid_scores or [1.0 / (i + 1) for i in range(len(matching))]
        objects = []
        for obj, score in zip(matching, scores, strict=False):
            hit = _Object(obj.uuid, obj.properties, obj.vector["default"])
            hit.metadata = SimpleNamespace(score=score)
            objects.append(hit)
        return SimpleNamespace(objects=objects[:limit])


class _FakeClient:
    def __init__(self, kind: str, store: dict[str, _FakeCollection], **kwargs: Any) -> None:
        self.kind = kind
        self.kwargs = kwargs
        self.store = store
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False
        self.collections = SimpleNamespace(
            exists=self._exists, create=self._create, get=self._get, delete=self._delete
        )

    def _exists(self, name: str) -> bool:
        return name in self.store

    def _create(self, name: str, *, vector_config: Any, properties: list[_Property]) -> None:
        self.calls.append(("create", {"name": name, "vector_config": vector_config}))
        self.store[name] = _FakeCollection(name, list(properties), vector_config)

    def _get(self, name: str) -> _FakeCollection:
        return self.store[name]

    def _delete(self, name: str) -> None:
        self.store.pop(name, None)

    def close(self) -> None:
        self.closed = True


def _self_provided(**kwargs: Any) -> tuple[str, dict[str, Any]]:
    return ("self_provided", kwargs)


def _hnsw(**kwargs: Any) -> tuple[str, dict[str, Any]]:
    return ("hnsw", kwargs)


def _metadata_query(**kwargs: Any) -> tuple[str, dict[str, Any]]:
    return ("metadata", kwargs)


def _api_key(key: str) -> tuple[str, str]:
    return ("api_key", key)


def _generate_uuid5(identifier: Any) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(identifier)))


@pytest.fixture
def fake_weaviate(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fake ``weaviate``; collections persist across clients within the test."""
    store: dict[str, _FakeCollection] = {}
    module = types.ModuleType("weaviate")
    clients: list[_FakeClient] = []

    def connect_to_embedded(**kwargs: Any) -> _FakeClient:
        clients.append(_FakeClient("embedded", store, **kwargs))
        return clients[-1]

    def connect_to_weaviate_cloud(**kwargs: Any) -> _FakeClient:
        clients.append(_FakeClient("cloud", store, **kwargs))
        return clients[-1]

    def connect_to_local(**kwargs: Any) -> _FakeClient:
        clients.append(_FakeClient("local", store, **kwargs))
        return clients[-1]

    module.clients = clients  # type: ignore[attr-defined]
    module.connect_to_embedded = connect_to_embedded  # type: ignore[attr-defined]
    module.connect_to_weaviate_cloud = connect_to_weaviate_cloud  # type: ignore[attr-defined]
    module.connect_to_local = connect_to_local  # type: ignore[attr-defined]
    module.classes = SimpleNamespace(  # type: ignore[attr-defined]
        init=SimpleNamespace(Auth=SimpleNamespace(api_key=_api_key)),
        config=SimpleNamespace(
            VectorDistances=SimpleNamespace(COSINE="cosine", L2_SQUARED="l2-squared", DOT="dot"),
            Configure=SimpleNamespace(
                Vectors=SimpleNamespace(self_provided=_self_provided),
                VectorIndex=SimpleNamespace(hnsw=_hnsw),
            ),
            Property=_Property,
            DataType=SimpleNamespace(TEXT="text", BOOL="boolean", NUMBER="number"),
            Tokenization=SimpleNamespace(FIELD="field"),
        ),
        query=SimpleNamespace(
            Filter=SimpleNamespace(by_property=_ByProperty), MetadataQuery=_metadata_query
        ),
    )
    util = types.ModuleType("weaviate.util")
    util.generate_uuid5 = _generate_uuid5  # type: ignore[attr-defined]
    module.util = util  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "weaviate", module)
    monkeypatch.setitem(sys.modules, "weaviate.util", util)
    return module


def _store(**kwargs: Any) -> WeaviateVectorStore:
    return WeaviateVectorStore(dimension=3, collection_name="t", **kwargs)


def _doc(doc_id: str, content: str, vector: list[float], **metadata: Any) -> Document:
    return Document(id=doc_id, content=content, embedding=vector, metadata=metadata)


# ------------------------------------------------------------- pure Python


def test_collection_name_is_capitalised() -> None:
    store = WeaviateVectorStore(collection_name="runbooks", _client=object())
    assert store._collection_name == "Runbooks"
    assert "Runbooks" in repr(store)


def test_metric_validation_and_config() -> None:
    with pytest.raises(ValueError, match="distance_metric"):
        WeaviateVectorStore(distance_metric="hamming", _client=object())
    store = WeaviateVectorStore(dimension=8, distance_metric="l2", _client=object())
    assert store.config.dimension == 8
    assert store.config.distance_metric == "l2"


def test_meta_property_names() -> None:
    assert meta_property("topic") == "meta_topic"
    assert meta_property("_private") == "meta__private"
    assert meta_property("bad key") is None
    assert meta_property("1abc") is None


def test_score_mapping() -> None:
    cosine = WeaviateVectorStore(_client=object())
    assert cosine._score(0.25) == pytest.approx(0.75)
    dot = WeaviateVectorStore(distance_metric="dot", _client=object())
    assert dot._score(-0.4) == pytest.approx(0.4)
    l2 = WeaviateVectorStore(distance_metric="euclidean", _client=object())
    assert l2._score(3.0) == pytest.approx(0.25)


def test_repr_for_each_target() -> None:
    assert "embedded" in repr(WeaviateVectorStore(embedded=True, _client=object()))
    assert "cluster" in repr(WeaviateVectorStore(cluster_url="https://cluster", _client=object()))
    assert "myhost" in repr(WeaviateVectorStore(host="myhost", _client=object()))


def test_to_document_tolerates_bad_objects() -> None:
    obj = SimpleNamespace(
        properties={"doc_id": "a", "metadata_json": "{broken", "created_at": "nope"},
        vector=[1.0, 0.0],
    )
    doc = WeaviateVectorStore._to_document(obj)
    assert doc.id == "a"
    assert doc.metadata == {}
    assert doc.embedding == [1.0, 0.0]
    assert doc.created_at.tzinfo is not None
    assert doc.content_type == "text"
    listed = WeaviateVectorStore._to_document(
        SimpleNamespace(properties={"metadata_json": "[1]"}, vector=None)
    )
    assert listed.metadata == {}
    assert listed.embedding is None
    assert WeaviateVectorStore._to_document(SimpleNamespace(properties=None, vector={})).id == ""


def test_missing_weaviate_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "weaviate", None)
    with pytest.raises(ImportError, match=r"locus-sdk\[weaviate\]"):
        _store()


# ------------------------------------------------------------ fake-backed


def test_connects_to_local_cloud_or_embedded(fake_weaviate: types.ModuleType) -> None:
    local = _store(host="db.internal", port=8081, grpc_port=50052, api_key="k")._get_client()
    assert local.kind == "local"
    assert local.kwargs == {
        "host": "db.internal",
        "port": 8081,
        "grpc_port": 50052,
        "auth_credentials": ("api_key", "k"),
    }
    default = _store()._get_client()
    assert default.kwargs["host"] == "localhost"
    assert default.kwargs["auth_credentials"] is None

    cloud = _store(cluster_url="https://c.weaviate.cloud", api_key="k")._get_client()
    assert cloud.kind == "cloud"
    assert cloud.kwargs == {
        "cluster_url": "https://c.weaviate.cloud",
        "auth_credentials": ("api_key", "k"),
    }

    embedded_store = _store(embedded=True, persistence_data_path="/tmp/w", embedded_port=8090)
    embedded = embedded_store._get_client()
    assert embedded.kind == "embedded"
    assert embedded.kwargs == {"port": 8090, "grpc_port": 50050, "persistence_data_path": "/tmp/w"}
    assert embedded_store._get_client() is embedded  # cached
    assert "persistence_data_path" not in _store(embedded=True)._get_client().kwargs


async def test_collection_created_with_schema_and_distance(
    fake_weaviate: types.ModuleType,
) -> None:
    store = _store(distance_metric="l2")
    assert await store.count() == 0
    client = store._get_client()
    (create,) = client.calls
    assert create[1]["name"] == "T"
    assert create[1]["vector_config"] == (
        "self_provided",
        {"vector_index_config": ("hnsw", {"distance_metric": "l2-squared"})},
    )
    collection = client.store["T"]
    assert [p.name for p in collection.properties] == [
        "doc_id",
        "content",
        "content_type",
        "metadata_json",
        "created_at",
    ]
    assert collection.properties[0].kwargs["tokenization"] == "field"
    assert collection.properties[3].kwargs["index_filterable"] is False
    assert set(store._known_props) == {
        "doc_id",
        "content",
        "content_type",
        "metadata_json",
        "created_at",
    }
    await store.count()
    assert len(client.calls) == 1  # found, not created again


async def test_add_get_delete_round_trip_with_meta_properties(
    fake_weaviate: types.ModuleType,
) -> None:
    store = _store()
    assert await store.add_batch([]) == []
    doc = _doc(
        "a",
        "alpha",
        [1.0, 0.0, 0.0],
        topic="x",
        year=2026,
        flag=True,
        ratio=0.5,
        nested={"k": 1},
        **{"bad key": "skipped"},
    )
    assert await store.add(doc) == "a"
    collection = store._get_collection()
    added = [c[1] for c in collection.calls if c[0] == "add_property"]
    assert added == [
        {
            "name": "meta_topic",
            "data_type": "text",
            "index_searchable": False,
            "tokenization": "field",
        },
        {"name": "meta_year", "data_type": "number", "index_searchable": False},
        {"name": "meta_flag", "data_type": "boolean", "index_searchable": False},
        {"name": "meta_ratio", "data_type": "number", "index_searchable": False},
    ]
    (obj,) = collection.objects.values()
    assert obj.properties["doc_id"] == "a"
    assert obj.properties["meta_topic"] == "x"
    assert obj.properties["meta_year"] == 2026.0
    assert obj.properties["meta_flag"] is True
    assert "meta_nested" not in obj.properties
    assert ("fixed_size", {"batch_size": 100}) in collection.calls

    # Known properties are not added twice.
    await store.add(_doc("b", "beta", [0.0, 1.0, 0.0], topic="y"))
    assert len([c for c in collection.calls if c[0] == "add_property"]) == 4

    got = await store.get("a")
    assert got is not None
    assert got.content == "alpha"
    assert got.metadata == {
        "topic": "x",
        "year": 2026,
        "flag": True,
        "ratio": 0.5,
        "nested": {"k": 1},
        "bad key": "skipped",
    }
    assert got.embedding == [1.0, 0.0, 0.0]
    assert await store.get("missing") is None
    assert await store.delete("a") is True
    assert await store.delete("a") is False
    assert await store.count() == 1


async def test_add_requires_embedding_and_reports_failed_objects(
    fake_weaviate: types.ModuleType,
) -> None:
    store = _store()
    with pytest.raises(ValueError, match="embedding"):
        await store.add(Document(id="a", content="x"))
    store._get_collection().failed_objects.append(SimpleNamespace(message="too big"))
    with pytest.raises(RuntimeError, match=r"rejected 1.*too big"):
        await store.add(_doc("a", "a", [1.0, 0.0, 0.0]))


async def test_search_filter_threshold_and_cosine_score(fake_weaviate: types.ModuleType) -> None:
    store = _store()
    await store.add_batch(
        [
            _doc("a", "a", [1.0, 0.0, 0.0], topic="x", year=2026),
            _doc("b", "b", [0.6, 0.8, 0.0], topic="x", year=2026),
            _doc("c", "c", [0.0, 0.0, 1.0], topic="y", year=2026),
        ]
    )
    results = await store.search(
        [1.0, 0.0, 0.0], limit=3, metadata_filter={"topic": "x", "year": 2026}
    )
    call = store._get_collection().calls[-1]
    assert call[0] == "near_vector"
    assert call[1]["filters"].clauses == [("meta_topic", "x"), ("meta_year", 2026.0)]
    assert call[1]["return_metadata"] == ("metadata", {"distance": True})
    assert call[1]["include_vector"] is True
    assert [r.document.id for r in results] == ["a", "b"]
    assert results[0].score == pytest.approx(1.0)
    assert results[0].distance == pytest.approx(0.0)
    assert results[1].score == pytest.approx(0.6)
    assert results[0].document.embedding == [1.0, 0.0, 0.0]

    kept = await store.search([1.0, 0.0, 0.0], limit=3, threshold=0.9)
    assert [r.document.id for r in kept] == ["a"]
    assert store._get_collection().calls[-1][1]["filters"] is None

    with pytest.raises(ValueError, match="not a valid Weaviate property name"):
        await store.search([1.0, 0.0, 0.0], metadata_filter={"bad key": 1})


async def test_hybrid_search_passes_alpha_and_filters(fake_weaviate: types.ModuleType) -> None:
    store = _store()
    assert supports_hybrid_search(store) is True
    await store.add_batch(
        [_doc("a", "card refunds", [1.0, 0.0, 0.0], topic="x"), _doc("b", "b", [0.0, 1.0, 0.0])]
    )
    collection = store._get_collection()
    collection.hybrid_scores = [0.8, 0.2]

    results = await store.hybrid_search(
        "card refunds", [1.0, 0.0, 0.0], limit=2, alpha=0.3, metadata_filter={"topic": "x"}
    )
    call = collection.calls[-1]
    assert call[0] == "hybrid"
    assert call[1]["query"] == "card refunds"
    assert call[1]["alpha"] == 0.3
    assert call[1]["filters"].clauses == [("meta_topic", "x")]
    assert call[1]["query_properties"] == ["content"]
    assert call[1]["return_metadata"] == ("metadata", {"score": True})
    assert [(r.document.id, r.score) for r in results] == [("a", 0.8)]

    kept = await store.hybrid_search("card refunds", [1.0, 0.0, 0.0], limit=2, threshold=0.5)
    assert [r.document.id for r in kept] == ["a"]

    fallback = await store.hybrid_search("  ", [1.0, 0.0, 0.0], limit=2)
    assert collection.calls[-1][0] == "near_vector"
    assert [r.document.id for r in fallback] == ["a", "b"]
    with pytest.raises(ValueError, match="alpha"):
        await store.hybrid_search("q", [1.0, 0.0, 0.0], alpha=-0.1)


async def test_clear_and_close(fake_weaviate: types.ModuleType) -> None:
    store = _store()
    await store.add(_doc("a", "a", [1.0, 0.0, 0.0], topic="x"))
    client = store._get_client()
    assert await store.clear() == 1
    assert "T" not in client.store
    assert store._known_props == {}
    assert await store.clear() == 0  # collection gone
    await store.add(_doc("b", "b", [0.0, 1.0, 0.0]))
    await store.close()
    assert client.closed is True
    assert store._client is None
    assert store._collection is None
    await store.close()  # no client: no-op

    # A second store opens the existing collection instead of creating it.
    reopened = _store()
    assert await reopened.count() == 1
    assert reopened._get_client().calls == []
    assert "doc_id" in reopened._known_props


async def test_injected_client_is_used_and_not_closed(fake_weaviate: types.ModuleType) -> None:
    client = _FakeClient("injected", {})
    store = WeaviateVectorStore(dimension=3, collection_name="t", _client=client)
    assert await store.count() == 0
    assert client.calls[0][0] == "create"
    await store.close()
    assert client.closed is False  # injected clients belong to the caller
