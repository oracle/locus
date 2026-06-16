# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the MMR diversity re-ranker.

The reranker is pure Python (no numpy) and operates on a
:class:`SearchResult` candidate pool. These tests pin its three
canonical behaviours:

* ``lambda=1.0`` collapses to plain top-N relevance (no diversity).
* ``lambda=0.0`` maximises diversity — never picks two near-duplicates
  back-to-back when a more distinct candidate is available.
* ``lambda=0.5`` blends — first pick is the most relevant, later picks
  trade some relevance for distance from earlier picks.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from locus.rag.stores._mmr import _cosine, mmr_rerank
from locus.rag.stores.base import Document, SearchResult


def _result(id_: str, embedding: list[float], score: float = 1.0) -> SearchResult:
    return SearchResult(
        document=Document(
            id=id_,
            content=f"doc-{id_}",
            embedding=embedding,
            metadata={},
            created_at=datetime.now(UTC),
        ),
        score=score,
        distance=1.0 - score,
    )


class TestCosine:
    def test_identical_vectors_score_one(self) -> None:
        assert _cosine([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_score_minus_one(self) -> None:
        assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_norm_falls_back_to_zero(self) -> None:
        # Avoid ZeroDivisionError when either side is the zero vector.
        assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestMMRRerank:
    def _two_clusters(self) -> list[SearchResult]:
        # Three near-duplicates clustered around (1, 0, 0) plus one
        # outlier near (0, 0, 1). Query is exactly (1, 0, 0).
        return [
            _result("a1", [1.00, 0.00, 0.00], score=1.00),
            _result("a2", [0.99, 0.05, 0.00], score=0.98),
            _result("a3", [0.98, 0.10, 0.00], score=0.96),
            _result("b1", [0.00, 0.00, 1.00], score=0.30),
        ]

    def test_lambda_one_is_pure_relevance(self) -> None:
        # lambda=1 → no diversity penalty → top-3 by relevance.
        out = mmr_rerank(
            self._two_clusters(),
            query_embedding=[1.0, 0.0, 0.0],
            limit=3,
            lambda_=1.0,
        )
        assert [r.document.id for r in out] == ["a1", "a2", "a3"]

    def test_lambda_zero_maximises_diversity(self) -> None:
        out = mmr_rerank(
            self._two_clusters(),
            query_embedding=[1.0, 0.0, 0.0],
            limit=2,
            lambda_=0.0,
        )
        # First pick = most relevant; second = whichever maximises
        # distance from the first. b1 is orthogonal to a1, so it wins.
        ids = [r.document.id for r in out]
        assert ids[0] == "a1"
        assert ids[1] == "b1"

    def test_lambda_third_favours_diversity(self) -> None:
        out = mmr_rerank(
            self._two_clusters(),
            query_embedding=[1.0, 0.0, 0.0],
            limit=2,
            lambda_=0.3,
        )
        # First pick is always the highest relevance; with a diversity
        # bias the second pick should land in the other cluster (b1),
        # not on another near-duplicate of a1.
        assert out[0].document.id == "a1"
        assert out[1].document.id == "b1"

    def test_limit_honoured(self) -> None:
        out = mmr_rerank(
            self._two_clusters(),
            query_embedding=[1.0, 0.0, 0.0],
            limit=2,
            lambda_=0.5,
        )
        assert len(out) == 2

    def test_empty_candidates(self) -> None:
        assert mmr_rerank([], query_embedding=[1.0], limit=5) == []

    def test_zero_limit(self) -> None:
        assert mmr_rerank(self._two_clusters(), query_embedding=[1.0, 0.0, 0.0], limit=0) == []

    def test_lambda_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="lambda_"):
            mmr_rerank([], query_embedding=[1.0], limit=1, lambda_=1.5)
        with pytest.raises(ValueError, match="lambda_"):
            mmr_rerank([], query_embedding=[1.0], limit=1, lambda_=-0.1)

    def test_handles_missing_embedding(self) -> None:
        # Candidates without embeddings still surface — they just
        # don't contribute to the diversity penalty.
        bare = SearchResult(
            document=Document(
                id="bare",
                content="no embedding",
                embedding=None,
                metadata={},
                created_at=datetime.now(UTC),
            ),
            score=0.5,
            distance=0.5,
        )
        out = mmr_rerank(
            self._two_clusters() + [bare],
            query_embedding=[1.0, 0.0, 0.0],
            limit=5,
            lambda_=0.5,
        )
        assert any(r.document.id == "bare" for r in out)
