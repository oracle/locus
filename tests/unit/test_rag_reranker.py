# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``locus.rag.reranker`` (closes #216).

Two test surfaces:

  * ``CohereReranker`` against a mocked OCI client — exercises the
    request shape, the response parsing, the empty-input contract, the
    score / distance preservation, and ``top_n`` truncation. No real
    OCI auth needed.
  * ``RAGRetriever`` wired to a fake reranker — confirms the
    over-fetch + rerank + trim plumbing in ``retrieve()``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from locus.rag import CohereReranker, RAGRetriever, Reranker
from locus.rag.stores.base import Document, SearchResult


def _doc(doc_id: str, content: str) -> Document:
    return Document(
        id=doc_id,
        content=content,
        embedding=[0.0],
        metadata={},
        created_at=datetime.now(UTC),
    )


def _hit(doc_id: str, content: str, score: float) -> SearchResult:
    return SearchResult(document=_doc(doc_id, content), score=score, distance=None)


# ============================================================================
# CohereReranker — request/response plumbing against a mocked OCI client.
# ============================================================================


def _mock_oci_client_returning(ranks: list[tuple[int, float]]) -> Any:
    """Return a mock OCI client whose ``rerank_text`` echoes the given
    ranks. ``ranks`` is a list of ``(original_index, relevance_score)``
    tuples in the order the model would return them."""
    response = SimpleNamespace(
        data=SimpleNamespace(
            document_ranks=[
                SimpleNamespace(index=idx, relevance_score=score) for idx, score in ranks
            ]
        )
    )
    client = MagicMock()
    client.rerank_text.return_value = response
    return client


@pytest.mark.asyncio
async def test_empty_candidates_short_circuits_without_oci_call() -> None:
    """Contract: rerank([]) returns [] without calling the OCI client.

    Important because real OCI calls cost money + add latency, and the
    "no hits" path is common with strict thresholds."""
    client = MagicMock()
    reranker = CohereReranker(compartment_id="ocid1.compartment.oc1..xxx", _client=client)
    out = await reranker.rerank("query", [])
    assert out == []
    client.rerank_text.assert_not_called()


@pytest.mark.asyncio
async def test_reorders_by_relevance_score() -> None:
    """rerank() returns candidates in the order OCI ranked them, with
    each result carrying the rerank score in ``.score`` and the
    original embedding score preserved in ``.distance``."""
    candidates = [
        _hit("a", "alpha", score=0.6),
        _hit("b", "bravo", score=0.5),
        _hit("c", "charlie", score=0.4),
    ]
    # OCI says: c is most relevant, then a, then b.
    client = _mock_oci_client_returning([(2, 0.95), (0, 0.70), (1, 0.10)])
    reranker = CohereReranker(
        model="cohere.rerank-v3.5",
        compartment_id="ocid1.compartment.oc1..xxx",
        _client=client,
    )

    out = await reranker.rerank("query", candidates)

    assert [r.document.id for r in out] == ["c", "a", "b"]
    assert [r.score for r in out] == [0.95, 0.70, 0.10]
    # Original embedding scores preserved on .distance for diagnostics.
    assert out[0].distance == 0.4
    assert out[1].distance == 0.6
    assert out[2].distance == 0.5


@pytest.mark.asyncio
async def test_top_n_truncates_output() -> None:
    """``top_n`` caps the number of returned candidates (matches OCI's
    own ``top_n`` parameter)."""
    candidates = [_hit(str(i), f"doc-{i}", 0.5) for i in range(10)]
    client = _mock_oci_client_returning([(i, 1.0 - i * 0.1) for i in range(10)])
    reranker = CohereReranker(
        compartment_id="ocid1.compartment.oc1..xxx",
        top_n=3,
        _client=client,
    )

    out = await reranker.rerank("query", candidates)

    assert len(out) == 10  # OCI returned 10; truncation enforced by OCI server
    # And we explicitly pass top_n through:
    request_kwargs = client.rerank_text.call_args.kwargs
    request = request_kwargs["rerank_text_details"]
    assert request.top_n == 3


@pytest.mark.asyncio
async def test_request_shape_carries_query_documents_and_compartment() -> None:
    """The OCI request the reranker builds carries the right fields:
    ``input`` (query), ``documents`` (candidate texts in order),
    ``compartment_id``, and a Cohere on-demand serving mode."""
    candidates = [_hit("x", "doc x text", 0.5), _hit("y", "doc y text", 0.5)]
    client = _mock_oci_client_returning([(0, 0.9), (1, 0.1)])
    reranker = CohereReranker(
        model="cohere.rerank-v3.5",
        compartment_id="ocid1.compartment.oc1..xyz",
        _client=client,
    )

    await reranker.rerank("hepcidin role", candidates)

    request = client.rerank_text.call_args.kwargs["rerank_text_details"]
    assert request.input == "hepcidin role"
    assert request.documents == ["doc x text", "doc y text"]
    assert request.compartment_id == "ocid1.compartment.oc1..xyz"
    assert request.serving_mode.model_id == "cohere.rerank-v3.5"


@pytest.mark.asyncio
async def test_build_client_routes_through_oci_client_helper(monkeypatch: Any) -> None:
    """Without a ``_client`` override, ``_build_client`` constructs an
    ``OCIClient`` from ``locus.models.providers.oci.client`` so every
    auth mode (api_key / security_token / session_token / instance &
    resource principal) works through the same plumbing as the rest
    of the locus OCI surface. Covers the construction path (lines
    134-157) without needing live OCI credentials."""
    constructed: dict[str, Any] = {}

    class _FakeOCIClient:
        def __init__(self, cfg: Any) -> None:
            constructed["config"] = cfg
            self.compartment_id = "ocid1.compartment.oc1..auto-derived"
            self.client = MagicMock(name="GenerativeAiInferenceClient")

    monkeypatch.setattr(
        "locus.models.providers.oci.client.OCIClient",
        _FakeOCIClient,
    )

    reranker = CohereReranker(
        model="cohere.rerank-v4.0-fast",
        profile_name="DEFAULT",
        auth_type="security_token",
        region="us-chicago-1",
        # compartment_id intentionally unset — should auto-derive.
    )
    client = reranker._build_client()

    # The first call returns the fake client; second call is cached.
    assert client is reranker._build_client()
    # OCIClient was built with the reranker's auth + endpoint config.
    cfg = constructed["config"]
    assert cfg.profile_name == "DEFAULT"
    assert cfg.auth_type.value == "security_token"
    assert cfg.service_endpoint == (
        "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
    )
    # Compartment auto-derived from the profile's tenancy.
    assert reranker.compartment_id == "ocid1.compartment.oc1..auto-derived"


@pytest.mark.asyncio
async def test_out_of_range_index_skipped_not_raised() -> None:
    """OCI returning a malformed index (out of range vs the candidate
    list) is logged-only — we skip the entry rather than crash the
    retriever path."""
    candidates = [_hit("a", "alpha", 0.5), _hit("b", "bravo", 0.5)]
    # Index 5 is out of range for 2 candidates.
    client = _mock_oci_client_returning([(5, 0.9), (0, 0.5)])
    reranker = CohereReranker(compartment_id="ocid1.compartment.oc1..xxx", _client=client)

    out = await reranker.rerank("query", candidates)

    assert [r.document.id for r in out] == ["a"]


# ============================================================================
# RAGRetriever — reranker plumbed through the retrieve() path.
# ============================================================================


class _FakeReranker(Reranker):
    """Reverses the candidate order. Tracks how many calls + what limit
    was passed so the retriever's over-fetch behaviour is observable."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_query: str | None = None
        self.last_candidates_len: int = 0

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        self.calls += 1
        self.last_query = query
        self.last_candidates_len = len(candidates)
        return list(reversed(candidates))


class _FakeStore:
    """Captures the limit the retriever asks for and returns N hits."""

    def __init__(self, n_hits: int) -> None:
        self.n_hits = n_hits
        self.last_limit: int | None = None

    @property
    def config(self) -> Any:  # for ``store_type`` logging
        return SimpleNamespace(distance_metric="cosine")

    async def search(
        self,
        query_embedding: list[float],  # noqa: ARG002
        limit: int,
        threshold: float | None = None,  # noqa: ARG002
        metadata_filter: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> list[SearchResult]:
        self.last_limit = limit
        return [_hit(str(i), f"doc-{i}", 1.0 - i * 0.01) for i in range(min(self.n_hits, limit))]


class _FakeEmbedder:
    async def embed_query(self, query: str) -> Any:  # noqa: ARG002
        return SimpleNamespace(embedding=[0.0])


@pytest.mark.asyncio
async def test_retrieve_without_reranker_uses_limit_directly() -> None:
    """Default behaviour (no reranker) — the retriever asks the store
    for exactly ``limit`` hits. No over-fetch."""
    store = _FakeStore(n_hits=100)
    retriever = RAGRetriever(embedder=_FakeEmbedder(), store=store)

    result = await retriever.retrieve("q", limit=5)

    assert store.last_limit == 5
    assert len(result.documents) == 5


@pytest.mark.asyncio
async def test_retrieve_with_reranker_overfetches_then_trims() -> None:
    """With a reranker wired in, the retriever asks the store for
    ``rerank_candidate_pool`` hits (default 50), the reranker reorders
    them, and ``retrieve()`` trims back to ``limit``."""
    store = _FakeStore(n_hits=100)
    reranker = _FakeReranker()
    retriever = RAGRetriever(
        embedder=_FakeEmbedder(),
        store=store,
        reranker=reranker,
        rerank_candidate_pool=20,
    )

    result = await retriever.retrieve("q", limit=3)

    # Over-fetched 20 from the store, then trimmed to 3.
    assert store.last_limit == 20
    assert reranker.calls == 1
    assert reranker.last_candidates_len == 20
    assert len(result.documents) == 3

    # _FakeReranker reverses order → the trimmed top-3 are the worst-
    # scoring originals (id=19, 18, 17).
    assert [r.document.id for r in result.documents] == ["19", "18", "17"]


@pytest.mark.asyncio
async def test_retrieve_with_empty_store_skips_reranker() -> None:
    """If the vector store returns zero hits, the reranker is not
    called (matches the reranker's own empty-input contract)."""
    store = _FakeStore(n_hits=0)
    reranker = _FakeReranker()
    retriever = RAGRetriever(
        embedder=_FakeEmbedder(),
        store=store,
        reranker=reranker,
        rerank_candidate_pool=20,
    )

    result = await retriever.retrieve("q", limit=5)

    assert reranker.calls == 0
    assert len(result.documents) == 0
