# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Unit tests for the Qdrant vector store.

Runs against an in-memory Qdrant instance (``location=":memory:"``) — no
server, no network, free. Skips if ``qdrant-client`` is not installed.
"""

import pytest


pytest.importorskip("qdrant_client")

from tulip.rag.stores.base import Document  # noqa: E402
from tulip.rag.stores.qdrant import QdrantVectorStore  # noqa: E402


def _store() -> QdrantVectorStore:
    return QdrantVectorStore(location=":memory:", dimension=3, collection_name="t")


def _doc(doc_id: str, vec, **meta) -> Document:
    return Document(id=doc_id, content=f"content {doc_id}", embedding=vec, metadata=meta)


async def test_add_get_count_search():
    store = _store()
    await store.add(_doc("a", [1.0, 0.0, 0.0], topic="x"))
    await store.add(_doc("b", [0.0, 1.0, 0.0], topic="y"))

    assert await store.count() == 2

    got = await store.get("a")
    assert got is not None
    assert got.id == "a"
    assert got.metadata["topic"] == "x"

    results = await store.search([1.0, 0.0, 0.0], limit=2)
    assert results[0].document.id == "a"
    assert results[0].score >= results[1].score


async def test_metadata_filter():
    store = _store()
    await store.add(_doc("a", [1.0, 0.0, 0.0], topic="x"))
    await store.add(_doc("b", [0.9, 0.1, 0.0], topic="y"))

    results = await store.search([1.0, 0.0, 0.0], limit=5, metadata_filter={"topic": "y"})
    assert [r.document.id for r in results] == ["b"]


async def test_delete_and_missing():
    store = _store()
    await store.add(_doc("a", [1.0, 0.0, 0.0]))

    assert await store.delete("a") is True
    assert await store.delete("a") is False
    assert await store.get("a") is None
    assert await store.count() == 0


async def test_add_requires_embedding():
    store = _store()
    with pytest.raises(ValueError, match="embedding"):
        await store.add(Document(id="x", content="no vector"))


async def test_clear():
    store = _store()
    await store.add(_doc("a", [1.0, 0.0, 0.0]))
    await store.add(_doc("b", [0.0, 1.0, 0.0]))
    removed = await store.clear()
    assert removed == 2
    assert await store.count() == 0
