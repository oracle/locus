# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Unit tests for the local cross-encoder reranker.

Uses an injected fake model (``_model=``) so the tests run fully offline
with no ``sentence-transformers`` / torch dependency.
"""

from tulip.rag.reranker.cross_encoder import CrossEncoderReranker
from tulip.rag.stores.base import Document, SearchResult


class _FakeCrossEncoder:
    """Stand-in for sentence_transformers.CrossEncoder.

    Scores each [query, doc] pair by a lookup keyed on the document text.
    """

    def __init__(self, scores_by_text):
        self.scores_by_text = scores_by_text
        self.calls = []

    def predict(self, pairs):
        self.calls.append(pairs)
        return [self.scores_by_text[doc] for _query, doc in pairs]


def _candidate(doc_id: str, content: str, score: float) -> SearchResult:
    return SearchResult(document=Document(id=doc_id, content=content), score=score, distance=None)


async def test_empty_candidates_short_circuits():
    model = _FakeCrossEncoder({})
    reranker = CrossEncoderReranker(_model=model)
    assert await reranker.rerank("q", []) == []
    assert model.calls == []  # no model call for empty input


async def test_reorders_by_cross_encoder_score():
    candidates = [
        _candidate("a", "low relevance", 0.9),  # high embedding score...
        _candidate("b", "high relevance", 0.1),  # ...but reranker disagrees
    ]
    model = _FakeCrossEncoder({"low relevance": 0.2, "high relevance": 0.95})
    reranker = CrossEncoderReranker(_model=model)

    out = await reranker.rerank("query", candidates)

    assert [r.document.id for r in out] == ["b", "a"]
    assert out[0].score == 0.95
    # Original embedding score preserved on .distance
    assert out[0].distance == 0.1


async def test_top_n_truncates():
    candidates = [_candidate(str(i), f"doc {i}", 0.5) for i in range(5)]
    scores = {f"doc {i}": float(i) for i in range(5)}
    reranker = CrossEncoderReranker(top_n=2, _model=_FakeCrossEncoder(scores))

    out = await reranker.rerank("query", candidates)

    assert len(out) == 2
    assert [r.document.id for r in out] == ["4", "3"]


async def test_does_not_mutate_input():
    candidates = [_candidate("a", "x", 0.3), _candidate("b", "y", 0.4)]
    original_order = [r.document.id for r in candidates]
    reranker = CrossEncoderReranker(_model=_FakeCrossEncoder({"x": 0.1, "y": 0.9}))

    await reranker.rerank("query", candidates)

    assert [r.document.id for r in candidates] == original_order
