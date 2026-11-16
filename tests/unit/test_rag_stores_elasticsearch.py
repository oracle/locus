# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the Elasticsearch vector store — request shaping on a fake client."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from locus.rag.stores.base import Document
from locus.rag.stores.elasticsearch import (
    ElasticsearchVectorStore,
    build_filter_clauses,
)


class _NotFoundError(Exception):
    status_code = 404


class _Indices:
    def __init__(self) -> None:
        self.exists_result = False
        self.created: list[tuple[str, dict[str, Any]]] = []

    async def exists(self, *, index: str) -> bool:
        return self.exists_result

    async def create(self, *, index: str, mappings: dict[str, Any]) -> None:
        self.created.append((index, mappings))
        self.exists_result = True


class _FakeES:
    """Records every call; answers ``search`` from a queue of canned responses."""

    def __init__(self) -> None:
        self.indices = _Indices()
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.search_responses: list[dict[str, Any]] = []
        self.docs: dict[str, dict[str, Any]] = {}
        self.closed = False

    async def index(self, **kwargs: Any) -> None:
        self.calls.append(("index", kwargs))
        self.docs[kwargs["id"]] = kwargs["document"]

    async def bulk(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("bulk", kwargs))
        return {"errors": False, "items": []}

    async def get(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get", kwargs))
        if kwargs["id"] not in self.docs:
            raise _NotFoundError
        return {"_source": self.docs[kwargs["id"]]}

    async def delete(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("delete", kwargs))
        if kwargs["id"] not in self.docs:
            raise _NotFoundError
        del self.docs[kwargs["id"]]
        return {"result": "deleted"}

    async def search(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("search", kwargs))
        return self.search_responses.pop(0)

    async def count(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("count", kwargs))
        return {"count": len(self.docs)}

    async def delete_by_query(self, **kwargs: Any) -> None:
        self.calls.append(("delete_by_query", kwargs))
        self.docs.clear()

    async def close(self) -> None:
        self.closed = True


def _hit(doc_id: str, score: float, content: str = "", **meta: Any) -> dict[str, Any]:
    return {
        "_score": score,
        "_source": {
            "doc_id": doc_id,
            "content": content or f"content {doc_id}",
            "content_type": "text",
            "embedding": [1.0, 0.0, 0.0],
            "metadata": meta,
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    }


def _store(**kwargs: Any) -> tuple[ElasticsearchVectorStore, _FakeES]:
    fake = _FakeES()
    store = ElasticsearchVectorStore(dimension=3, index_name="t", _client=fake, **kwargs)
    return store, fake


async def test_index_created_with_dense_vector_and_flattened_metadata() -> None:
    store, fake = _store()
    await store.add(Document(id="a", content="x", embedding=[1.0, 0.0, 0.0]))
    assert fake.indices.created
    index, mappings = fake.indices.created[0]
    assert index == "t"
    embedding = mappings["properties"]["embedding"]
    assert embedding == {"type": "dense_vector", "dims": 3, "index": True, "similarity": "cosine"}
    assert mappings["properties"]["metadata"] == {"type": "flattened"}
    assert mappings["properties"]["content"] == {"type": "text"}


async def test_add_indexes_with_wait_for_refresh_and_batch_uses_bulk() -> None:
    store, fake = _store()
    await store.add(Document(id="a", content="x", embedding=[1.0, 0.0, 0.0], metadata={"k": "v"}))
    name, kwargs = fake.calls[-1]
    assert name == "index"
    assert kwargs["refresh"] == "wait_for"
    assert kwargs["document"]["metadata"] == {"k": "v"}
    assert kwargs["document"]["doc_id"] == "a"

    ids = await store.add_batch(
        [
            Document(id="b", content="y", embedding=[0.0, 1.0, 0.0]),
            Document(id="c", content="z", embedding=[0.0, 0.0, 1.0]),
        ]
    )
    assert ids == ["b", "c"]
    name, kwargs = fake.calls[-1]
    assert name == "bulk"
    assert kwargs["operations"][0] == {"index": {"_index": "t", "_id": "b"}}
    assert kwargs["operations"][1]["content"] == "y"
    assert len(kwargs["operations"]) == 4


async def test_add_requires_embedding() -> None:
    store, _ = _store()
    with pytest.raises(ValueError, match="embedding"):
        await store.add(Document(id="a", content="x"))


async def test_bulk_errors_raise() -> None:
    store, fake = _store()

    async def bulk(**kwargs: Any) -> dict[str, Any]:
        return {"errors": True, "items": [{"index": {"error": {"reason": "boom"}}}]}

    fake.bulk = bulk  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="rejected 1"):
        await store.add_batch([Document(id="a", content="x", embedding=[1.0, 0.0, 0.0])])


async def test_get_and_delete_map_not_found() -> None:
    store, fake = _store()
    assert await store.get("missing") is None
    assert await store.delete("missing") is False
    await store.add(Document(id="a", content="x", embedding=[1.0, 0.0, 0.0], metadata={"k": 1}))
    got = await store.get("a")
    assert got is not None
    assert got.id == "a"
    assert got.metadata == {"k": 1}
    assert await store.delete("a") is True


async def test_search_builds_knn_with_filter_and_maps_cosine_score() -> None:
    store, fake = _store()
    fake.indices.exists_result = True
    fake.search_responses.append({"hits": {"hits": [_hit("a", 1.0, topic="x"), _hit("b", 0.5)]}})

    results = await store.search(
        [1.0, 0.0, 0.0], limit=2, metadata_filter={"topic": "x", "year": 2026}
    )

    _, kwargs = fake.calls[-1]
    knn = kwargs["knn"]
    assert knn["field"] == "embedding"
    assert knn["k"] == 2
    assert knn["num_candidates"] == 100
    assert knn["query_vector"] == [1.0, 0.0, 0.0]
    assert knn["filter"] == [{"term": {"metadata.topic": "x"}}, {"term": {"metadata.year": 2026}}]
    assert kwargs["size"] == 2
    # ES reports (1 + cos) / 2: 1.0 → cos 1.0, 0.5 → cos 0.0.
    assert [r.document.id for r in results] == ["a", "b"]
    assert results[0].score == pytest.approx(1.0)
    assert results[0].distance == pytest.approx(0.0)
    assert results[1].score == pytest.approx(0.0)
    assert results[0].document.metadata == {"topic": "x"}


async def test_search_threshold_filters_client_side() -> None:
    store, fake = _store()
    fake.indices.exists_result = True
    fake.search_responses.append({"hits": {"hits": [_hit("a", 1.0), _hit("b", 0.6)]}})
    results = await store.search([1.0, 0.0, 0.0], limit=2, threshold=0.5)
    assert [r.document.id for r in results] == ["a"]


async def test_num_candidates_override_and_no_filter_key() -> None:
    store, fake = _store(num_candidates=37)
    fake.indices.exists_result = True
    fake.search_responses.append({"hits": {"hits": []}})
    await store.search([1.0, 0.0, 0.0], limit=5)
    _, kwargs = fake.calls[-1]
    assert kwargs["knn"]["num_candidates"] == 37
    assert "filter" not in kwargs["knn"]


async def test_hybrid_search_runs_both_legs_and_blends() -> None:
    store, fake = _store()
    fake.indices.exists_result = True
    # Vector leg: a then b. Lexical leg: b scores far above a.
    fake.search_responses.append({"hits": {"hits": [_hit("a", 1.0), _hit("b", 0.75)]}})
    fake.search_responses.append({"hits": {"hits": [_hit("b", 12.0), _hit("a", 3.0)]}})

    results = await store.hybrid_search(
        "chargeback", [1.0, 0.0, 0.0], limit=2, alpha=0.0, metadata_filter={"topic": "x"}
    )

    knn_call, match_call = fake.calls[-2], fake.calls[-1]
    assert "knn" in knn_call[1]
    assert knn_call[1]["size"] == 20  # over-fetch pool
    query = match_call[1]["query"]
    assert query["bool"]["must"] == [{"match": {"content": "chargeback"}}]
    assert query["bool"]["filter"] == [{"term": {"metadata.topic": "x"}}]
    # alpha=0 → lexical order wins.
    assert [r.document.id for r in results] == ["b", "a"]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.25)


async def test_hybrid_search_alpha_one_is_vector_order() -> None:
    store, fake = _store()
    fake.indices.exists_result = True
    fake.search_responses.append({"hits": {"hits": [_hit("a", 1.0), _hit("b", 0.75)]}})
    fake.search_responses.append({"hits": {"hits": [_hit("b", 12.0)]}})
    results = await store.hybrid_search("chargeback", [1.0, 0.0, 0.0], limit=2, alpha=1.0)
    assert [r.document.id for r in results] == ["a", "b"]


async def test_hybrid_search_empty_query_falls_back_to_vector() -> None:
    store, fake = _store()
    fake.indices.exists_result = True
    fake.search_responses.append({"hits": {"hits": [_hit("a", 1.0)]}})
    results = await store.hybrid_search("   ", [1.0, 0.0, 0.0], limit=2, alpha=0.3)
    assert [r.document.id for r in results] == ["a"]
    assert len([c for c in fake.calls if c[0] == "search"]) == 1


async def test_hybrid_search_rejects_bad_alpha() -> None:
    store, _ = _store()
    with pytest.raises(ValueError, match="alpha"):
        await store.hybrid_search("q", [1.0, 0.0, 0.0], alpha=2.0)


async def test_count_clear_close() -> None:
    store, fake = _store()
    await store.add(Document(id="a", content="x", embedding=[1.0, 0.0, 0.0]))
    assert await store.count() == 1
    assert await store.clear() == 1
    assert fake.calls[-1][0] == "delete_by_query"
    await store.close()
    assert fake.closed is True


def test_filter_clauses_and_metric_validation() -> None:
    assert build_filter_clauses(None) == []
    assert build_filter_clauses({"a": 1}) == [{"term": {"metadata.a": 1}}]
    with pytest.raises(ValueError, match="distance_metric"):
        ElasticsearchVectorStore(distance_metric="hamming", _client=object())


def _fake_elasticsearch_module(monkeypatch: pytest.MonkeyPatch) -> type:
    """Stand in for the ``elasticsearch`` package; the client records its kwargs."""

    class AsyncElasticsearch:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    module = types.ModuleType("elasticsearch")
    module.AsyncElasticsearch = AsyncElasticsearch  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "elasticsearch", module)
    return AsyncElasticsearch


def test_client_built_from_hosts_with_auth_and_ca_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    client_cls = _fake_elasticsearch_module(monkeypatch)
    store = ElasticsearchVectorStore(
        hosts=["https://es.internal:9200"],
        api_key=("id", "secret"),
        basic_auth=("user", "pass"),
        verify_certs=False,
        ca_certs="/etc/ssl/es-ca.pem",
    )
    client = store._get_client()
    assert isinstance(client, client_cls)
    assert client.kwargs == {
        "verify_certs": False,
        "hosts": ["https://es.internal:9200"],
        "api_key": ("id", "secret"),
        "basic_auth": ("user", "pass"),
        "ca_certs": "/etc/ssl/es-ca.pem",
    }
    assert store._get_client() is client  # cached


def test_client_defaults_to_localhost_and_cloud_id_replaces_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_elasticsearch_module(monkeypatch)
    local = ElasticsearchVectorStore()._get_client()
    assert local.kwargs == {"verify_certs": True, "hosts": ["http://localhost:9200"]}
    cloud = ElasticsearchVectorStore(
        cloud_id="deployment:abc", hosts=["http://ignored:9200"], api_key="key"
    )._get_client()
    assert cloud.kwargs == {"verify_certs": True, "cloud_id": "deployment:abc", "api_key": "key"}


def test_missing_elasticsearch_package_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "elasticsearch", None)
    with pytest.raises(ImportError, match=r"locus-sdk\[elasticsearch\]"):
        ElasticsearchVectorStore()._get_client()


async def test_close_is_a_no_op_without_a_client_and_closes_a_built_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ElasticsearchVectorStore()
    await store.close()  # nothing built yet
    _fake_elasticsearch_module(monkeypatch)
    client = store._get_client()
    await store.close()
    assert client.closed is True
    assert store._client is None


def test_to_document_tolerates_malformed_created_at() -> None:
    doc = ElasticsearchVectorStore._to_document({"doc_id": "a", "created_at": "not-a-date"})
    assert doc.id == "a"
    assert doc.created_at.tzinfo is not None
    assert doc.embedding is None


async def test_add_batch_empty_is_a_no_op() -> None:
    store, fake = _store()
    assert await store.add_batch([]) == []
    assert fake.calls == []


async def test_get_and_delete_reraise_errors_that_are_not_404() -> None:
    store, fake = _store()
    fake.indices.exists_result = True

    class _UnavailableError(Exception):
        status_code = 503

    async def failing(**kwargs: Any) -> dict[str, Any]:
        raise _UnavailableError("cluster unavailable")

    fake.get = failing  # type: ignore[method-assign]
    fake.delete = failing  # type: ignore[method-assign]
    with pytest.raises(_UnavailableError):
        await store.get("a")
    with pytest.raises(_UnavailableError):
        await store.delete("a")


def test_l2_and_dot_score_mapping() -> None:
    l2, _ = _store(distance_metric="l2")
    score, distance = l2._score(0.5)  # 1 / (1 + d²) = 0.5 → d = 1
    assert score == pytest.approx(0.5)
    assert distance == pytest.approx(1.0)
    dot, _ = _store(distance_metric="dot_product")
    assert dot._score(0.75) == (pytest.approx(0.5), None)
    assert dot.config.distance_metric == "dot_product"
