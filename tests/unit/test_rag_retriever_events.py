# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``RAGRetriever`` emits ``rag.embedding.generated``, ``rag.store.search`` and
``rag.store.upsert`` on every query and ingest path."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from locus.observability import get_event_bus, run_context
from locus.rag import RAGRetriever
from locus.rag.embeddings.base import EmbeddingResult
from locus.rag.stores.base import Document, SearchResult
from locus.rag.stores.memory import InMemoryVectorStore


class _FakeEmbedder:
    async def embed(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(embedding=[1.0, 0.0, 0.5], text=text, model="fake")

    async def embed_query(self, text: str) -> EmbeddingResult:
        return await self.embed(text)

    async def embed_documents(self, texts: list[str]) -> list[EmbeddingResult]:
        return [await self.embed(t) for t in texts]


class _HybridStore(InMemoryVectorStore):
    async def hybrid_search(self, query_text: str, query_embedding: Any, **kwargs: Any) -> Any:
        return [SearchResult(document=Document(id="h", content="h"), score=0.5)]


def _events(rid: str) -> list[Any]:
    return list(get_event_bus()._history.get(rid, ()))  # type: ignore[attr-defined]


def _kinds(rid: str) -> list[str]:
    return [e.event_type for e in _events(rid)]


def _one(rid: str, kind: str) -> dict[str, Any]:
    matches = [e.data for e in _events(rid) if e.event_type == kind]
    assert len(matches) == 1, _kinds(rid)
    return matches[0]


def _retriever(store: Any = None, **kw: Any) -> RAGRetriever:
    return RAGRetriever(
        embedder=_FakeEmbedder(),
        store=store or InMemoryVectorStore(),
        chunk_size=50,
        chunk_overlap=0,
        **kw,
    )


@pytest.mark.asyncio
async def test_retrieve_emits_in_order() -> None:
    retriever = _retriever()
    async with run_context("rag-ev-query") as rid:
        await retriever.retrieve("hello", limit=3)
    assert _kinds(rid) == [
        "rag.query.started",
        "rag.embedding.generated",
        "rag.store.search",
        "rag.query.completed",
    ]
    emb = _one(rid, "rag.embedding.generated")
    assert emb == {
        "embedder_type": "_FakeEmbedder",
        "count": 1,
        "dimension": 3,
        "purpose": "query",
    }
    search = _one(rid, "rag.store.search")
    assert search["store_type"] == "InMemoryVectorStore"
    assert search["limit"] == 3
    assert search["hit_count"] == 0
    assert search["hybrid"] is False


@pytest.mark.asyncio
async def test_hybrid_flag_true_only_on_hybrid_path() -> None:
    async with run_context("rag-ev-hybrid") as rid:
        await _retriever(_HybridStore(), hybrid=True).retrieve("q")
    assert _one(rid, "rag.store.search")["hybrid"] is True

    # hybrid requested but the store cannot do it: vector fallback, flag False.
    async with run_context("rag-ev-hybrid-fallback") as rid2:
        await _retriever(hybrid=True).retrieve("q")
    assert _one(rid2, "rag.store.search")["hybrid"] is False


@pytest.mark.asyncio
async def test_add_document_emits_embedding_and_upsert_per_chunks() -> None:
    retriever = _retriever()
    content = "\n\n".join(f"paragraph number {i} with some filler text" for i in range(6))
    async with run_context("rag-ev-doc") as rid:
        ids = await retriever.add_document(content, doc_id="doc1")
    assert len(ids) > 1
    emb = _one(rid, "rag.embedding.generated")
    up = _one(rid, "rag.store.upsert")
    assert emb["purpose"] == "documents"
    assert emb["count"] == len(ids)
    assert up == {"store_type": "InMemoryVectorStore", "doc_id": "doc1", "chunk_count": len(ids)}


@pytest.mark.asyncio
async def test_add_file_emits(tmp_path: Any) -> None:
    f = tmp_path / "note.txt"
    f.write_text("a small note")
    async with run_context("rag-ev-file") as rid:
        await _retriever().add_file(f, doc_id="f1")
    assert _kinds(rid).count("rag.embedding.generated") == 1
    assert _one(rid, "rag.store.upsert")["chunk_count"] == 1


@pytest.mark.asyncio
async def test_add_image_emits(monkeypatch: pytest.MonkeyPatch) -> None:
    from locus.rag import multimodal

    async def fake_process(self: Any, image: Any) -> Any:
        return SimpleNamespace(text="ocr text", metadata={}, raw_content=None)

    monkeypatch.setattr(multimodal.ImageProcessor, "process", fake_process)
    async with run_context("rag-ev-image") as rid:
        await _retriever().add_image(b"not-really-an-image", doc_id="img1")
    assert _kinds(rid).count("rag.embedding.generated") == 1
    assert _one(rid, "rag.store.upsert")["doc_id"] == "img1"
