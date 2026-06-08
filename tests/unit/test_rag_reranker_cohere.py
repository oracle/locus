# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Unit tests for the Cohere reranker.

Uses an injected fake async client (``_client=``) so the tests run with
no ``cohere`` dependency and no network access.
"""

from dataclasses import dataclass

from tulip.rag.reranker.cohere import CohereReranker
from tulip.rag.stores.base import Document, SearchResult


@dataclass
class _Result:
    index: int
    relevance_score: float


@dataclass
class _Response:
    results: list


class _FakeCohereClient:
    """Returns a fixed ranking; records the kwargs of the last call."""

    def __init__(self, ranking):
        self._ranking = ranking
        self.last_kwargs = None

    async def rerank(self, **kwargs):
        self.last_kwargs = kwargs
        return _Response(results=[_Result(index=i, relevance_score=s) for i, s in self._ranking])


def _candidate(doc_id: str, content: str, score: float) -> SearchResult:
    return SearchResult(document=Document(id=doc_id, content=content), score=score)


async def test_empty_candidates_short_circuits():
    client = _FakeCohereClient([])
    reranker = CohereReranker(_client=client)
    assert await reranker.rerank("q", []) == []
    assert client.last_kwargs is None


async def test_reorders_and_maps_indices():
    candidates = [
        _candidate("a", "doc a", 0.5),
        _candidate("b", "doc b", 0.6),
        _candidate("c", "doc c", 0.7),
    ]
    # Cohere returns index 2 first, then 0.
    client = _FakeCohereClient([(2, 0.99), (0, 0.40)])
    reranker = CohereReranker(_client=client, top_n=2)

    out = await reranker.rerank("query", candidates)

    assert [r.document.id for r in out] == ["c", "a"]
    assert out[0].score == 0.99
    assert out[0].distance == 0.7  # original score preserved
    assert client.last_kwargs["documents"] == ["doc a", "doc b", "doc c"]
    assert client.last_kwargs["top_n"] == 2
    assert client.last_kwargs["query"] == "query"


async def test_missing_api_key_does_not_crash_construction(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    reranker = CohereReranker()
    assert reranker.api_key is None
