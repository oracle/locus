# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Shared conformance suite for ``locus.rag.stores`` backends.

Every vector store — in-memory, embedded or served — has to honour the same
contract behind the ``VectorStore`` protocol: ids round-trip, ``search``
orders by similarity and respects ``limit``/``threshold``/``metadata_filter``,
``delete`` reports what it removed, ``clear`` empties the store and stores
that advertise hybrid search blend the two legs the way ``alpha`` says.

Subclass :class:`VectorStoreConformance`, implement ``make_store`` and flip
the capability flags the backend does not support. The unit suite runs it
against the free in-process backends (in-memory, Qdrant, Chroma, Milvus
Lite); ``tests/integration/rag/test_store_conformance_live.py`` runs the
same class against served backends behind environment gates.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from locus.rag.stores.base import Document, VectorStore, supports_hybrid_search


DIMENSION = 4

# Unit vectors: ``A`` and ``B`` are orthogonal, ``C`` sits between them.
VEC_A = [1.0, 0.0, 0.0, 0.0]
VEC_B = [0.0, 1.0, 0.0, 0.0]
VEC_C = [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]
VEC_D = [0.0, 0.0, 1.0, 0.0]


def doc(doc_id: str, vector: list[float], content: str | None = None, **metadata: Any) -> Document:
    return Document(
        id=doc_id,
        content=content if content is not None else f"content for {doc_id}",
        embedding=list(vector),
        metadata=metadata,
    )


class VectorStoreConformance:
    """Mixin with the contract tests. Subclasses provide ``make_store``."""

    #: The store blends vector and lexical relevance (``hybrid_search``).
    supports_hybrid: bool = False
    #: ``metadata_filter`` with an ``int`` value is matched by value.
    supports_numeric_filter: bool = True
    #: ``search`` hits carry ``document.embedding``.
    returns_embedding_on_search: bool = True
    #: Scores for an exact match are close to 1.0 and far above an orthogonal vector.
    exact_match_min_score: float = 0.9

    async def make_store(self) -> VectorStore:
        raise NotImplementedError

    @pytest.fixture
    async def store(self) -> Any:
        store = await self.make_store()
        try:
            await store.clear()
            yield store
        finally:
            await store.clear()
            await store.close()

    # ------------------------------------------------------------ add / get

    async def test_add_and_get_round_trip(self, store: VectorStore) -> None:
        original = doc("a", VEC_A, "alpha text", topic="x", year=2026, flag=True)
        assert await store.add(original) == "a"

        got = await store.get("a")
        assert got is not None
        assert got.id == "a"
        assert got.content == "alpha text"
        assert got.embedding is not None
        assert [round(x, 5) for x in got.embedding] == VEC_A
        assert got.metadata["topic"] == "x"
        assert got.metadata["year"] == 2026
        assert got.metadata["flag"] is True
        assert got.content_type == "text"

    async def test_get_missing_returns_none(self, store: VectorStore) -> None:
        assert await store.get("missing") is None

    async def test_add_requires_embedding(self, store: VectorStore) -> None:
        with pytest.raises(ValueError, match="embedding"):
            await store.add(Document(id="x", content="no vector"))

    async def test_add_batch_and_count(self, store: VectorStore) -> None:
        ids = await store.add_batch([doc("a", VEC_A), doc("b", VEC_B), doc("c", VEC_C)])
        assert ids == ["a", "b", "c"]
        assert await store.count() == 3
        assert await store.add_batch([]) == []

    async def test_upsert_same_id_replaces(self, store: VectorStore) -> None:
        await store.add(doc("a", VEC_A, "first"))
        await store.add(doc("a", VEC_A, "second", topic="new"))
        assert await store.count() == 1
        got = await store.get("a")
        assert got is not None
        assert got.content == "second"
        assert got.metadata.get("topic") == "new"

    # --------------------------------------------------------------- search

    async def test_search_orders_by_similarity(self, store: VectorStore) -> None:
        await store.add_batch([doc("a", VEC_A), doc("b", VEC_B), doc("c", VEC_C)])
        results = await store.search(VEC_A, limit=3)
        assert [r.document.id for r in results] == ["a", "c", "b"]
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] >= self.exact_match_min_score
        assert scores[0] > scores[2]
        if self.returns_embedding_on_search:
            assert results[0].document.embedding is not None
            assert len(results[0].document.embedding) == DIMENSION

    async def test_search_respects_limit(self, store: VectorStore) -> None:
        await store.add_batch([doc("a", VEC_A), doc("b", VEC_B), doc("c", VEC_C)])
        results = await store.search(VEC_A, limit=2)
        assert len(results) == 2
        assert results[0].document.id == "a"

    async def test_search_threshold_drops_weak_matches(self, store: VectorStore) -> None:
        await store.add_batch([doc("a", VEC_A), doc("b", VEC_B), doc("c", VEC_C)])
        results = await store.search(VEC_A, limit=3, threshold=self.exact_match_min_score)
        assert [r.document.id for r in results] == ["a"]

    async def test_search_metadata_filter_string(self, store: VectorStore) -> None:
        await store.add_batch(
            [doc("a", VEC_A, topic="x"), doc("b", VEC_B, topic="y"), doc("c", VEC_C, topic="x")]
        )
        results = await store.search(VEC_A, limit=5, metadata_filter={"topic": "y"})
        assert [r.document.id for r in results] == ["b"]

        results = await store.search(VEC_A, limit=5, metadata_filter={"topic": "x"})
        assert [r.document.id for r in results] == ["a", "c"]

    async def test_search_metadata_filter_numeric(self, store: VectorStore) -> None:
        if not self.supports_numeric_filter:
            pytest.skip("store matches metadata filters as strings only")
        await store.add_batch([doc("a", VEC_A, year=2025), doc("b", VEC_B, year=2026)])
        results = await store.search(VEC_A, limit=5, metadata_filter={"year": 2026})
        assert [r.document.id for r in results] == ["b"]

    async def test_search_filter_without_match_is_empty(self, store: VectorStore) -> None:
        await store.add(doc("a", VEC_A, topic="x"))
        assert await store.search(VEC_A, limit=5, metadata_filter={"topic": "nope"}) == []

    async def test_search_empty_store(self, store: VectorStore) -> None:
        assert await store.search(VEC_A, limit=3) == []

    # ------------------------------------------------------- delete / clear

    async def test_delete_reports_removal(self, store: VectorStore) -> None:
        await store.add(doc("a", VEC_A))
        assert await store.delete("a") is True
        assert await store.delete("a") is False
        assert await store.get("a") is None
        assert await store.count() == 0

    async def test_clear_returns_count_and_empties(self, store: VectorStore) -> None:
        await store.add_batch([doc("a", VEC_A), doc("b", VEC_B)])
        assert await store.clear() == 2
        assert await store.count() == 0
        assert await store.search(VEC_A, limit=3) == []
        # The store is usable again after ``clear``.
        await store.add(doc("d", VEC_D))
        assert await store.count() == 1

    # --------------------------------------------------------------- hybrid

    async def test_hybrid_capability_is_advertised(self, store: VectorStore) -> None:
        assert supports_hybrid_search(store) is self.supports_hybrid

    async def test_hybrid_search_blends_vector_and_lexical(self, store: VectorStore) -> None:
        if not self.supports_hybrid:
            pytest.skip("store has no hybrid search")
        await store.add_batch(
            [
                doc("a", VEC_A, "card refunds settle in five business days"),
                doc("b", VEC_B, "chargeback reason code for merchandise not received"),
                doc("c", VEC_D, "bank account debit returned for insufficient funds"),
            ]
        )
        hybrid = getattr(store, "hybrid_search")  # noqa: B009 — protocol method looked up by name
        # Pure vector: the query vector is document ``a``.
        vector_only = await hybrid("chargeback merchandise", VEC_A, limit=3, alpha=1.0)
        assert vector_only[0].document.id == "a"
        # Pure lexical: the words belong to document ``b``.
        lexical_only = await hybrid("chargeback merchandise", VEC_A, limit=3, alpha=0.0)
        assert lexical_only[0].document.id == "b"
        # A blend ranks both above the document that matches neither leg.
        blended = await hybrid("chargeback merchandise", VEC_A, limit=3, alpha=0.5)
        assert len(blended) >= 2
        top_two = {blended[0].document.id, blended[1].document.id}
        assert top_two == {"a", "b"}
        scores = [r.score for r in blended]
        assert scores == sorted(scores, reverse=True)

    async def test_hybrid_search_honours_metadata_filter(self, store: VectorStore) -> None:
        if not self.supports_hybrid:
            pytest.skip("store has no hybrid search")
        await store.add_batch(
            [
                doc("a", VEC_A, "refund policy for digital goods", topic="x"),
                doc("b", VEC_B, "refund policy for physical goods", topic="y"),
            ]
        )
        hybrid = getattr(store, "hybrid_search")  # noqa: B009
        results = await hybrid(
            "refund policy", VEC_A, limit=5, alpha=0.5, metadata_filter={"topic": "y"}
        )
        assert [r.document.id for r in results] == ["b"]

    async def test_hybrid_search_rejects_bad_alpha(self, store: VectorStore) -> None:
        if not self.supports_hybrid:
            pytest.skip("store has no hybrid search")
        hybrid = getattr(store, "hybrid_search")  # noqa: B009
        with pytest.raises(ValueError, match="alpha"):
            await hybrid("q", VEC_A, limit=3, alpha=1.5)
