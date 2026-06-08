# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Unit tests for the Chroma vector store.

Runs against an ephemeral in-memory Chroma client — no server, no disk,
free. Skips if ``chromadb`` is not installed.
"""

import uuid

import pytest


# chromadb can fail to import for reasons beyond "not installed" (e.g. a
# protobuf/opentelemetry version clash in the environment). Skip the whole
# module on ANY import failure rather than only ImportError.
try:
    import chromadb  # noqa: F401
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"chromadb unavailable: {exc}", allow_module_level=True)

from tulip.rag.stores.base import Document  # noqa: E402
from tulip.rag.stores.chroma import ChromaVectorStore  # noqa: E402


def _store() -> ChromaVectorStore:
    # Unique collection name per test to isolate the shared ephemeral client.
    return ChromaVectorStore(dimension=3, collection_name=f"t_{uuid.uuid4().hex}")


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
    assert got.metadata.get("topic") == "x"
    # Internal bookkeeping key is not leaked back to the caller.
    assert "_content_type" not in got.metadata

    results = await store.search([1.0, 0.0, 0.0], limit=2)
    assert results[0].document.id == "a"


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


async def test_add_requires_embedding():
    store = _store()
    with pytest.raises(ValueError, match="embedding"):
        await store.add(Document(id="x", content="no vector"))
