# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the Milvus store — filter expressions and Milvus Lite behaviours."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from locus.rag import RAGRetriever
from locus.rag.stores.base import Document, supports_hybrid_search
from locus.rag.stores.milvus import MilvusVectorStore, build_filter_expr


needs_milvus_lite = pytest.mark.skipif(
    not (importlib.util.find_spec("pymilvus") and importlib.util.find_spec("milvus_lite")),
    reason="pymilvus with Milvus Lite not installed",
)


class _FakeEmbedder:
    async def embed_query(self, text: str) -> Any:
        return SimpleNamespace(embedding=[1.0, 0.0, 0.0])


def _doc(doc_id: str, content: str, vector: list[float], **metadata: Any) -> Document:
    return Document(id=doc_id, content=content, embedding=vector, metadata=metadata)


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
    with pytest.raises(TypeError, match="metadata_filter"):
        build_filter_expr({"a": [1]})


async def test_hybrid_search_disabled_at_construction_raises() -> None:
    store = MilvusVectorStore(dimension=3, _client=object())
    assert store.hybrid_search_available is False
    with pytest.raises(ValueError, match="enable_hybrid=True"):
        await store.hybrid_search("q", [1.0, 0.0, 0.0])


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
async def test_non_positive_limit_returns_empty(tmp_path: Path) -> None:
    store = MilvusVectorStore(
        uri=str(tmp_path / "x.db"), dimension=3, collection_name="h", enable_hybrid=True
    )
    try:
        await store.add(_doc("a", "card refunds", [1.0, 0.0, 0.0]))
        assert await store.search([1.0, 0.0, 0.0], limit=0) == []
        assert await store.search([1.0, 0.0, 0.0], limit=-1) == []
        assert await store.hybrid_search("card refunds", [1.0, 0.0, 0.0], limit=0) == []
        assert len(await store.hybrid_search("card refunds", [1.0, 0.0, 0.0], limit=1)) == 1
    finally:
        await store.close()


@needs_milvus_lite
async def test_existing_plain_collection_reports_no_hybrid_and_retriever_falls_back(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    uri = str(tmp_path / "reopen.db")
    plain = MilvusVectorStore(uri=uri, dimension=3, collection_name="t")
    await plain.add(_doc("a", "card refunds", [1.0, 0.0, 0.0]))
    await plain.close()

    reopened = MilvusVectorStore(uri=uri, dimension=3, collection_name="t", enable_hybrid=True)
    try:
        # Not opened yet: the flag mirrors the constructor.
        assert reopened.hybrid_search_available is True
        assert await reopened.count() == 1
        # Opened: the schema has no sparse field, so hybrid is off.
        assert reopened.hybrid_search_available is False
        assert supports_hybrid_search(reopened) is False

        retriever = RAGRetriever(embedder=_FakeEmbedder(), store=reopened, hybrid=True)
        with caplog.at_level(logging.WARNING, logger="locus.rag.retriever"):
            result = await retriever.retrieve("card refunds", limit=3)
        assert [r.document.id for r in result.documents] == ["a"]
        assert "falling back to vector-only" in caplog.text
    finally:
        await reopened.close()


@needs_milvus_lite
async def test_hybrid_search_first_on_plain_collection_answers_by_vector(
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
