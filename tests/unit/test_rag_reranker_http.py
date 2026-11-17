# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the Jina and Voyage rerankers — request shaping on ``httpx.MockTransport``."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from locus.rag.reranker.jina import DEFAULT_JINA_BASE_URL, JinaReranker
from locus.rag.reranker.voyage import DEFAULT_VOYAGE_BASE_URL, VoyageReranker
from locus.rag.stores.base import Document, SearchResult


def _candidate(doc_id: str, content: str, score: float) -> SearchResult:
    return SearchResult(document=Document(id=doc_id, content=content), score=score)


CANDIDATES = [
    _candidate("a", "doc a", 0.5),
    _candidate("b", "doc b", 0.6),
    _candidate("c", "doc c", 0.7),
]


class _Recorder:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self.payload = payload
        self.status = status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, json=self.payload)


def _client(recorder: _Recorder) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(recorder))


# ------------------------------------------------------------------- Jina


async def test_jina_request_shape_and_ordering() -> None:
    recorder = _Recorder(
        {"results": [{"index": 2, "relevance_score": 0.95}, {"index": 0, "relevance_score": 0.4}]}
    )
    async with _client(recorder) as client:
        reranker = JinaReranker(api_key="k", top_n=2, _client=client)
        out = await reranker.rerank("query", CANDIDATES)

    request = recorder.requests[0]
    assert str(request.url) == DEFAULT_JINA_BASE_URL
    assert request.headers["authorization"] == "Bearer k"
    body = json.loads(request.content)
    assert body == {
        "model": "jina-reranker-v2-base-multilingual",
        "query": "query",
        "documents": ["doc a", "doc b", "doc c"],
        "return_documents": False,
        "top_n": 2,
    }
    assert [r.document.id for r in out] == ["c", "a"]
    assert out[0].score == 0.95
    assert out[0].distance == 0.7


async def test_jina_empty_candidates_and_missing_key() -> None:
    recorder = _Recorder({"results": []})
    async with _client(recorder) as client:
        assert await JinaReranker(api_key="k", _client=client).rerank("q", []) == []
        assert recorder.requests == []
        with pytest.raises(ValueError, match="JINA_API_KEY"):
            await JinaReranker(api_key=None, _client=client).rerank("q", CANDIDATES)


async def test_jina_http_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    recorder = _Recorder({"detail": "nope"}, status=401)
    async with _client(recorder) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await JinaReranker(api_key="bad", _client=client).rerank("q", CANDIDATES)


def test_jina_reads_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JINA_API_KEY", "from-env")
    assert JinaReranker().api_key == "from-env"


# ----------------------------------------------------------------- Voyage


async def test_voyage_request_shape_and_ordering() -> None:
    recorder = _Recorder(
        {"data": [{"index": 1, "relevance_score": 0.8}, {"index": 2, "relevance_score": 0.9}]}
    )
    async with _client(recorder) as client:
        reranker = VoyageReranker(api_key="k", top_n=2, model="rerank-2.5-lite", _client=client)
        out = await reranker.rerank("query", CANDIDATES)

    request = recorder.requests[0]
    assert str(request.url) == DEFAULT_VOYAGE_BASE_URL
    assert request.headers["authorization"] == "Bearer k"
    body = json.loads(request.content)
    assert body == {
        "model": "rerank-2.5-lite",
        "query": "query",
        "documents": ["doc a", "doc b", "doc c"],
        "truncation": True,
        "top_k": 2,
    }
    # Sorted by score even if the service returns another order.
    assert [r.document.id for r in out] == ["c", "b"]
    assert out[0].distance == 0.7


async def test_voyage_empty_candidates_and_missing_key() -> None:
    recorder = _Recorder({"data": []})
    async with _client(recorder) as client:
        assert await VoyageReranker(api_key="k", _client=client).rerank("q", []) == []
        assert recorder.requests == []
        with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
            await VoyageReranker(api_key=None, _client=client).rerank("q", CANDIDATES)


def test_voyage_reads_env_key_and_omits_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "from-env")
    reranker = VoyageReranker(truncation=False)
    assert reranker.api_key == "from-env"
    assert reranker.build_request("q", ["d"]) == {
        "model": "rerank-2.5",
        "query": "q",
        "documents": ["d"],
        "truncation": False,
    }
