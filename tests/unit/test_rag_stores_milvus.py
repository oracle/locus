# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the Milvus store.

A fake ``pymilvus`` module is injected into ``sys.modules`` so the adapter's
lazy imports resolve to an in-memory client and the tests need no library;
a few tests at the end run the same scenarios on Milvus Lite when it is
installed.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from locus.rag import RAGRetriever
from locus.rag.stores.base import Document, supports_hybrid_search
from locus.rag.stores.milvus import MilvusVectorStore, build_filter_expr


def _importable(*modules: str) -> bool:
    try:
        for name in modules:
            importlib.import_module(name)
    except ImportError:
        return False
    return True


needs_milvus_lite = pytest.mark.skipif(
    not _importable("pymilvus", "milvus_lite"), reason="pymilvus with Milvus Lite not installed"
)


class _FakeEmbedder:
    async def embed_query(self, text: str) -> Any:
        return SimpleNamespace(embedding=[1.0, 0.0, 0.0])


def _doc(doc_id: str, content: str, vector: list[float], **metadata: Any) -> Document:
    return Document(id=doc_id, content=content, embedding=vector, metadata=metadata)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


# ------------------------------------------------------------ fake pymilvus


class _Record:
    """Records constructor arguments (``Function``, ``AnnSearchRequest``, ``WeightedRanker``)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


class _Schema:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.fields: list[dict[str, Any]] = []
        self.functions: list[_Record] = []

    def add_field(self, name: str, dtype: Any, **kwargs: Any) -> None:
        self.fields.append({"name": name, "type": dtype, **kwargs})

    def add_function(self, function: _Record) -> None:
        self.functions.append(function)


class _IndexParams:
    def __init__(self) -> None:
        self.indexes: list[dict[str, Any]] = []

    def add_index(self, **kwargs: Any) -> None:
        self.indexes.append(kwargs)


class _FakeMilvusClient:
    """In-memory ``MilvusClient``: collections live in a dict shared by every client."""

    def __init__(self, collections: dict[str, dict[str, Any]], **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.collections = collections
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.hybrid_scores: list[float] = []
        self.closed = False

    def has_collection(self, name: str) -> bool:
        return name in self.collections

    def create_schema(self, **kwargs: Any) -> _Schema:
        return _Schema(**kwargs)

    def prepare_index_params(self) -> _IndexParams:
        return _IndexParams()

    def create_collection(self, name: str, *, schema: _Schema, index_params: _IndexParams) -> None:
        self.calls.append(("create_collection", {"schema": schema, "index_params": index_params}))
        self.collections[name] = {"schema": schema, "index_params": index_params, "rows": {}}

    def describe_collection(self, name: str) -> dict[str, Any]:
        self.calls.append(("describe_collection", {"name": name}))
        schema = self.collections[name]["schema"]
        return {
            "fields": [{"name": field["name"]} for field in schema.fields],
            "functions": list(schema.functions),
        }

    def _rows(self, name: str) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = self.collections[name]["rows"]
        return rows

    @staticmethod
    def _entity(row: dict[str, Any], output_fields: list[str]) -> dict[str, Any]:
        return {key: value for key, value in row.items() if key in output_fields}

    def upsert(self, name: str, *, data: list[dict[str, Any]]) -> None:
        self.calls.append(("upsert", {"data": data}))
        for row in data:
            self._rows(name)[row["id"]] = dict(row)

    def get(self, name: str, *, ids: list[str], output_fields: list[str]) -> list[dict[str, Any]]:
        self.calls.append(("get", {"ids": ids, "output_fields": output_fields}))
        rows = self._rows(name)
        return [self._entity(rows[i], output_fields) for i in ids if i in rows]

    def delete(self, name: str, *, ids: list[str]) -> None:
        self.calls.append(("delete", {"ids": ids}))
        for i in ids:
            self._rows(name).pop(i, None)

    def search(
        self,
        name: str,
        *,
        data: list[list[float]],
        anns_field: str,
        limit: int,
        filter: str,  # noqa: A002 — the pymilvus keyword
        output_fields: list[str],
        search_params: dict[str, Any],
    ) -> list[list[dict[str, Any]]]:
        self.calls.append(
            (
                "search",
                {"filter": filter, "limit": limit, "anns_field": anns_field, **search_params},
            )
        )
        metric = search_params["metric_type"]
        hits: list[dict[str, Any]] = []
        for row in self._rows(name).values():
            if metric == "L2":
                distance = sum((x - y) ** 2 for x, y in zip(row["embedding"], data[0], strict=True))
            else:
                distance = _dot(row["embedding"], data[0])
            hits.append(
                {"id": row["id"], "distance": distance, "entity": self._entity(row, output_fields)}
            )
        hits.sort(key=lambda hit: hit["distance"], reverse=metric != "L2")
        return [hits[:limit]]

    def hybrid_search(
        self,
        name: str,
        *,
        reqs: list[_Record],
        ranker: _Record,
        limit: int,
        output_fields: list[str],
    ) -> list[list[dict[str, Any]]]:
        self.calls.append(("hybrid_search", {"reqs": reqs, "ranker": ranker, "limit": limit}))
        rows = list(self._rows(name).values())
        scores = self.hybrid_scores or [1.0 / (i + 1) for i in range(len(rows))]
        hits = [
            {"id": row["id"], "distance": score, "entity": self._entity(row, output_fields)}
            for row, score in zip(rows, scores, strict=False)
        ]
        return [hits[:limit]]

    def query(
        self,
        name: str,
        *,
        filter: str,  # noqa: A002 — the pymilvus keyword
        output_fields: list[str],
    ) -> list[dict[str, Any]]:
        self.calls.append(("query", {"filter": filter, "output_fields": output_fields}))
        return [{"count(*)": len(self._rows(name))}]

    def drop_collection(self, name: str) -> None:
        self.calls.append(("drop_collection", {"name": name}))
        self.collections.pop(name, None)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_pymilvus(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fake ``pymilvus``; collections persist across clients within the test."""
    collections: dict[str, dict[str, Any]] = {}
    module = types.ModuleType("pymilvus")

    class MilvusClient(_FakeMilvusClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(collections, **kwargs)

    module.MilvusClient = MilvusClient  # type: ignore[attr-defined]
    module.DataType = SimpleNamespace(  # type: ignore[attr-defined]
        VARCHAR="VARCHAR",
        FLOAT_VECTOR="FLOAT_VECTOR",
        SPARSE_FLOAT_VECTOR="SPARSE_FLOAT_VECTOR",
        JSON="JSON",
    )
    module.FunctionType = SimpleNamespace(BM25="BM25")  # type: ignore[attr-defined]
    module.Function = _Record  # type: ignore[attr-defined]
    module.AnnSearchRequest = _Record  # type: ignore[attr-defined]
    module.WeightedRanker = _Record  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pymilvus", module)
    return module


def _store(**kwargs: Any) -> MilvusVectorStore:
    kwargs.setdefault("uri", "http://milvus.test:19530")
    kwargs.setdefault("collection_name", "t")
    return MilvusVectorStore(dimension=3, **kwargs)


# ------------------------------------------------------------- expressions


def test_filter_expr_keeps_non_ascii_literal() -> None:
    expr = build_filter_expr({"ville": "café", "naïve": True, "n": 2})
    assert expr == (
        'metadata["ville"] == "café" and metadata["naïve"] == true and metadata["n"] == 2'
    )
    assert "\\u" not in expr


def test_filter_expr_quotes_breakouts_and_rejects_non_scalars() -> None:
    assert build_filter_expr(None) == ""
    assert build_filter_expr({'a"b': 'x" or 1 == 1'}) == 'metadata["a\\"b"] == "x\\" or 1 == 1"'
    assert build_filter_expr({"f": 1.5}) == 'metadata["f"] == 1.5'
    with pytest.raises(TypeError, match="metadata_filter"):
        build_filter_expr({"a": [1]})


# ------------------------------------------------------------ fake-backed


def test_client_kwargs_carry_token_and_db_name(fake_pymilvus: types.ModuleType) -> None:
    plain = _store()
    assert plain._get_client().kwargs == {"uri": "http://milvus.test:19530"}
    assert plain._get_client() is plain._get_client()  # cached
    secured = _store(token="user:pw", db_name="kb")  # noqa: S106 — fake credential
    assert secured._get_client().kwargs == {
        "uri": "http://milvus.test:19530",
        "token": "user:pw",
        "db_name": "kb",
    }
    assert "milvus.test" in repr(secured)


async def test_injected_client_is_used_and_not_closed(fake_pymilvus: types.ModuleType) -> None:
    client = _FakeMilvusClient({})
    store = MilvusVectorStore(dimension=3, collection_name="t", _client=client)
    assert await store.count() == 0
    assert client.calls[0][0] == "create_collection"
    await store.close()
    assert client.closed is False  # injected clients belong to the caller


def test_constructor_validation() -> None:
    with pytest.raises(ValueError, match="distance_metric"):
        MilvusVectorStore(distance_metric="hamming", _client=object())
    store = MilvusVectorStore(dimension=8, distance_metric="l2", _client=object())
    assert store.config.dimension == 8
    assert store.config.distance_metric == "l2"
    assert store.config.index_type == "hnsw"


def test_missing_pymilvus_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pymilvus", None)
    with pytest.raises(ImportError, match=r"locus-sdk\[milvus\]"):
        _store()


async def test_plain_collection_schema_and_index(fake_pymilvus: types.ModuleType) -> None:
    store = _store(distance_metric="l2")
    await store.count()
    client = store._get_client()
    (create,) = [c for c in client.calls if c[0] == "create_collection"]
    schema, index_params = create[1]["schema"], create[1]["index_params"]
    assert schema.kwargs == {"auto_id": False, "enable_dynamic_field": False}
    assert [f["name"] for f in schema.fields] == [
        "id",
        "embedding",
        "content",
        "content_type",
        "metadata",
        "created_at",
    ]
    assert schema.fields[0]["is_primary"] is True
    assert schema.fields[1] == {"name": "embedding", "type": "FLOAT_VECTOR", "dim": 3}
    assert "enable_analyzer" not in schema.fields[2]
    assert schema.functions == []
    assert index_params.indexes == [
        {"field_name": "embedding", "index_type": "AUTOINDEX", "metric_type": "L2"}
    ]
    assert store.hybrid_search_available is False
    # A second operation finds the collection and does not create it again.
    await store.count()
    assert len([c for c in client.calls if c[0] == "create_collection"]) == 1


async def test_hybrid_collection_adds_bm25_function_and_sparse_index(
    fake_pymilvus: types.ModuleType,
) -> None:
    store = _store(enable_hybrid=True)
    await store.count()
    (create,) = [c for c in store._get_client().calls if c[0] == "create_collection"]
    schema, index_params = create[1]["schema"], create[1]["index_params"]
    assert [f["name"] for f in schema.fields] == [
        "id",
        "embedding",
        "content",
        "sparse",
        "content_type",
        "metadata",
        "created_at",
    ]
    assert schema.fields[2]["enable_analyzer"] is True
    assert schema.fields[3]["type"] == "SPARSE_FLOAT_VECTOR"
    (function,) = schema.functions
    assert function.kwargs == {
        "name": "locus_bm25",
        "input_field_names": ["content"],
        "output_field_names": ["sparse"],
        "function_type": "BM25",
    }
    assert index_params.indexes[1] == {
        "field_name": "sparse",
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "BM25",
    }
    assert store.hybrid_search_available is True
    assert supports_hybrid_search(store) is True


async def test_add_get_delete_count_clear_close(fake_pymilvus: types.ModuleType) -> None:
    store = _store()
    assert await store.add_batch([]) == []
    assert await store.add(_doc("a", "alpha", [1.0, 0.0, 0.0], topic="x", year=2026)) == "a"
    client = store._get_client()
    (upsert,) = [c for c in client.calls if c[0] == "upsert"]
    row = upsert[1]["data"][0]
    assert row["id"] == "a"
    assert row["embedding"] == [1.0, 0.0, 0.0]
    assert row["metadata"] == {"topic": "x", "year": 2026}
    assert row["content_type"] == "text"
    assert row["created_at"].endswith("+00:00")

    got = await store.get("a")
    assert got is not None
    assert got.content == "alpha"
    assert got.metadata == {"topic": "x", "year": 2026}
    assert got.embedding == [1.0, 0.0, 0.0]
    assert got.created_at.tzinfo is not None
    assert await store.get("missing") is None

    assert await store.count() == 1
    assert await store.delete("a") is True
    assert await store.delete("a") is False
    assert await store.add_batch([_doc("b", "b", [0.0, 1.0, 0.0])]) == ["b"]
    assert await store.clear() == 1
    assert ("drop_collection", {"name": "t"}) in client.calls
    assert await store.clear() == 0  # collection gone: nothing to count or drop
    assert await store.count() == 0  # recreated empty

    await store.close()
    assert client.closed is True
    assert store._client is None
    await store.close()  # no client: no-op


async def test_add_requires_embedding(fake_pymilvus: types.ModuleType) -> None:
    with pytest.raises(ValueError, match="embedding"):
        await _store().add(Document(id="a", content="x"))


async def test_search_filter_threshold_and_cosine_scores(fake_pymilvus: types.ModuleType) -> None:
    store = _store()
    await store.add_batch(
        [
            _doc("a", "a", [1.0, 0.0, 0.0], topic="x"),
            _doc("b", "b", [0.6, 0.8, 0.0], topic="x"),
            _doc("c", "c", [0.0, 0.0, 1.0], topic="y"),
        ]
    )
    results = await store.search([1.0, 0.0, 0.0], limit=3, metadata_filter={"topic": "x"})
    search = store._get_client().calls[-1]
    assert search[0] == "search"
    assert search[1]["filter"] == 'metadata["topic"] == "x"'
    assert search[1]["anns_field"] == "embedding"
    assert search[1]["metric_type"] == "COSINE"
    assert [r.document.id for r in results] == ["a", "b", "c"]
    assert results[0].score == pytest.approx(1.0)
    assert results[0].distance == pytest.approx(0.0)
    assert results[1].score == pytest.approx(0.6)
    assert results[1].distance == pytest.approx(0.4)
    assert results[0].document.metadata == {"topic": "x"}

    kept = await store.search([1.0, 0.0, 0.0], limit=3, threshold=0.5)
    assert [r.document.id for r in kept] == ["a", "b"]
    assert store._get_client().calls[-1][1]["filter"] == ""

    calls_before = len(store._get_client().calls)
    assert await store.search([1.0, 0.0, 0.0], limit=0) == []
    assert await store.search([1.0, 0.0, 0.0], limit=-1) == []
    assert len(store._get_client().calls) == calls_before


async def test_score_mapping_for_l2_and_inner_product(fake_pymilvus: types.ModuleType) -> None:
    l2 = _store(distance_metric="l2", collection_name="l2")
    await l2.add(_doc("a", "a", [0.0, 0.0, 0.0]))
    (hit,) = await l2.search([1.0, 0.0, 0.0], limit=1)
    assert hit.distance == pytest.approx(1.0)
    assert hit.score == pytest.approx(0.5)

    ip = _store(distance_metric="dot_product", collection_name="ip")
    await ip.add(_doc("a", "a", [0.5, 0.0, 0.0]))
    (hit,) = await ip.search([1.0, 0.0, 0.0], limit=1)
    assert hit.score == pytest.approx(0.5)
    assert hit.distance is None


async def test_hybrid_search_requests_ranker_and_fused_scores(
    fake_pymilvus: types.ModuleType,
) -> None:
    store = _store(enable_hybrid=True)
    await store.add_batch(
        [_doc("a", "card refunds", [1.0, 0.0, 0.0]), _doc("b", "b", [0.0, 1.0, 0.0])]
    )
    client = store._get_client()
    client.hybrid_scores = [0.9, 0.3]

    results = await store.hybrid_search(
        "card refunds", [1.0, 0.0, 0.0], limit=2, alpha=0.3, metadata_filter={"topic": "x"}
    )

    call = client.calls[-1]
    assert call[0] == "hybrid_search"
    dense, sparse = call[1]["reqs"]
    assert dense.kwargs["anns_field"] == "embedding"
    assert dense.kwargs["data"] == [[1.0, 0.0, 0.0]]
    assert dense.kwargs["param"] == {"metric_type": "COSINE"}
    assert dense.kwargs["expr"] == 'metadata["topic"] == "x"'
    assert sparse.kwargs["anns_field"] == "sparse"
    assert sparse.kwargs["data"] == ["card refunds"]
    assert sparse.kwargs["param"] == {"metric_type": "BM25"}
    assert call[1]["ranker"].args == pytest.approx((0.3, 0.7))
    assert call[1]["limit"] == 2
    assert [(r.document.id, r.score) for r in results] == [("a", 0.9), ("b", 0.3)]
    assert results[0].distance is None

    kept = await store.hybrid_search("card refunds", [1.0, 0.0, 0.0], limit=2, threshold=0.5)
    assert [r.document.id for r in kept] == ["a"]
    assert client.calls[-1][1]["reqs"][0].kwargs["expr"] is None

    assert await store.hybrid_search("card refunds", [1.0, 0.0, 0.0], limit=0) == []


async def test_hybrid_search_empty_query_and_validation(fake_pymilvus: types.ModuleType) -> None:
    store = _store(enable_hybrid=True)
    await store.add(_doc("a", "a", [1.0, 0.0, 0.0]))
    results = await store.hybrid_search("   ", [1.0, 0.0, 0.0], limit=2, alpha=0.3)
    assert [r.document.id for r in results] == ["a"]
    assert store._get_client().calls[-1][0] == "search"
    with pytest.raises(ValueError, match="alpha"):
        await store.hybrid_search("q", [1.0, 0.0, 0.0], alpha=2.0)


async def test_hybrid_search_disabled_at_construction_raises() -> None:
    store = MilvusVectorStore(dimension=3, _client=object())
    assert store.hybrid_search_available is False
    with pytest.raises(ValueError, match="enable_hybrid=True"):
        await store.hybrid_search("q", [1.0, 0.0, 0.0])


async def test_existing_plain_collection_turns_hybrid_off_and_retriever_falls_back(
    fake_pymilvus: types.ModuleType, caplog: pytest.LogCaptureFixture
) -> None:
    plain = _store()
    await plain.add(_doc("a", "card refunds", [1.0, 0.0, 0.0]))
    await plain.close()

    reopened = _store(enable_hybrid=True)
    assert reopened.hybrid_search_available is True  # not opened yet: mirrors the flag
    assert await reopened.count() == 1
    assert reopened.hybrid_search_available is False  # the schema has no sparse field
    assert supports_hybrid_search(reopened) is False
    assert ("describe_collection", {"name": "t"}) in reopened._get_client().calls
    # Reopened without enable_hybrid: no schema lookup, hybrid simply off.
    plain_again = _store()
    assert await plain_again.count() == 1
    assert plain_again.hybrid_search_available is False
    assert not [c for c in plain_again._get_client().calls if c[0] == "describe_collection"]

    retriever = RAGRetriever(embedder=_FakeEmbedder(), store=reopened, hybrid=True)
    with caplog.at_level(logging.WARNING, logger="locus.rag.retriever"):
        result = await retriever.retrieve("card refunds", limit=3)
    assert [r.document.id for r in result.documents] == ["a"]
    assert "falling back to vector-only" in caplog.text

    # hybrid_search as the very first operation: answered by vector, with a warning.
    direct = _store(enable_hybrid=True)
    with caplog.at_level(logging.WARNING, logger="locus.rag.stores.milvus"):
        hits = await direct.hybrid_search("card refunds", [1.0, 0.0, 0.0], limit=3)
    assert [h.document.id for h in hits] == ["a"]
    assert "has no BM25 sparse field" in caplog.text
    assert direct.hybrid_search_available is False
    assert direct._get_client().calls[-1][0] == "search"


def test_to_document_tolerates_bad_rows() -> None:
    doc = MilvusVectorStore._to_document({"id": "a", "created_at": "garbage", "content": None})
    assert doc.id == "a"
    assert doc.content == ""
    assert doc.embedding is None
    assert doc.metadata == {}
    assert doc.content_type == "text"
    assert doc.created_at.tzinfo is not None
    assert MilvusVectorStore._to_document({}).id == ""


# ------------------------------------------------------------- Milvus Lite


@needs_milvus_lite
async def test_non_ascii_filter_matches_on_milvus_lite(tmp_path: Path) -> None:
    store = MilvusVectorStore(uri=str(tmp_path / "x.db"), dimension=3, collection_name="t")
    try:
        await store.add_batch(
            [
                _doc("a", "café au lait", [1.0, 0.0, 0.0], ville="café"),
                _doc("b", "tea", [0.0, 1.0, 0.0], ville="Zürich"),
            ]
        )
        hits = await store.search([1.0, 0.0, 0.0], limit=5, metadata_filter={"ville": "café"})
        assert [h.document.id for h in hits] == ["a"]
        hits = await store.search([1.0, 0.0, 0.0], limit=5, metadata_filter={"ville": "Zürich"})
        assert [h.document.id for h in hits] == ["b"]
    finally:
        await store.close()


@needs_milvus_lite
async def test_non_positive_limit_returns_empty_on_milvus_lite(tmp_path: Path) -> None:
    store = MilvusVectorStore(
        uri=str(tmp_path / "x.db"), dimension=3, collection_name="h", enable_hybrid=True
    )
    try:
        await store.add(_doc("a", "card refunds", [1.0, 0.0, 0.0]))
        assert await store.search([1.0, 0.0, 0.0], limit=0) == []
        assert await store.hybrid_search("card refunds", [1.0, 0.0, 0.0], limit=0) == []
        assert len(await store.hybrid_search("card refunds", [1.0, 0.0, 0.0], limit=1)) == 1
    finally:
        await store.close()


@needs_milvus_lite
async def test_existing_plain_collection_falls_back_on_milvus_lite(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    uri = str(tmp_path / "reopen.db")
    plain = MilvusVectorStore(uri=uri, dimension=3, collection_name="t")
    await plain.add(_doc("a", "card refunds", [1.0, 0.0, 0.0]))
    await plain.close()

    reopened = MilvusVectorStore(uri=uri, dimension=3, collection_name="t", enable_hybrid=True)
    try:
        with caplog.at_level(logging.WARNING, logger="locus.rag.stores.milvus"):
            hits = await reopened.hybrid_search("card refunds", [1.0, 0.0, 0.0], limit=3)
        assert [h.document.id for h in hits] == ["a"]
        assert "has no BM25 sparse field" in caplog.text
        assert reopened.hybrid_search_available is False
        retriever = RAGRetriever(embedder=_FakeEmbedder(), store=reopened, hybrid=True)
        result = await retriever.retrieve("card refunds", limit=3)
        assert [r.document.id for r in result.documents] == ["a"]
    finally:
        await reopened.close()


@needs_milvus_lite
async def test_hybrid_collection_keeps_hybrid_after_reopen_and_clear(tmp_path: Path) -> None:
    uri = str(tmp_path / "hybrid.db")
    first = MilvusVectorStore(uri=uri, dimension=3, collection_name="h", enable_hybrid=True)
    await first.add(_doc("a", "card refunds", [1.0, 0.0, 0.0]))
    await first.close()

    second = MilvusVectorStore(uri=uri, dimension=3, collection_name="h", enable_hybrid=True)
    try:
        assert await second.count() == 1
        assert second.hybrid_search_available is True
        hits = await second.hybrid_search("card refunds", [1.0, 0.0, 0.0], limit=3, alpha=0.0)
        assert [h.document.id for h in hits] == ["a"]
        assert await second.clear() == 1
        assert await second.count() == 0
        assert second.hybrid_search_available is True
    finally:
        await second.close()
