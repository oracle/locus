# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Cohere embeddings.

Uses an injected fake async client (``_client=``) so the tests run with
no ``cohere`` dependency and no network access.
"""

from dataclasses import dataclass

from tulip.rag.embeddings.cohere import CohereEmbeddings


@dataclass
class _Embeddings:
    float: list


@dataclass
class _Response:
    embeddings: _Embeddings


class _FakeCohereClient:
    def __init__(self):
        self.calls = []

    async def embed(self, **kwargs):
        self.calls.append(kwargs)
        # One deterministic 4-d vector per input text.
        vectors = [[float(len(t)), 0.0, 1.0, 2.0] for t in kwargs["texts"]]
        return _Response(embeddings=_Embeddings(float=vectors))


def test_capabilities_and_dimension():
    emb = CohereEmbeddings(_client=_FakeCohereClient())
    assert emb.capabilities.supports_query_vs_doc is True
    assert emb.dimension == 1024  # embed-english-v3.0 default


async def test_embed_uses_document_input_type():
    client = _FakeCohereClient()
    emb = CohereEmbeddings(_client=client)

    result = await emb.embed("hello")

    assert result.text == "hello"
    assert result.embedding == [5.0, 0.0, 1.0, 2.0]
    assert client.calls[-1]["input_type"] == "search_document"


async def test_embed_query_uses_query_input_type():
    client = _FakeCohereClient()
    emb = CohereEmbeddings(_client=client)

    await emb.embed_query("a query")

    assert client.calls[-1]["input_type"] == "search_query"


async def test_embed_batch():
    client = _FakeCohereClient()
    emb = CohereEmbeddings(_client=client)

    results = await emb.embed_batch(["aa", "bbbb"])

    assert [r.text for r in results] == ["aa", "bbbb"]
    assert results[0].embedding[0] == 2.0
    assert results[1].embedding[0] == 4.0


async def test_embed_batch_empty():
    emb = CohereEmbeddings(_client=_FakeCohereClient())
    assert await emb.embed_batch([]) == []
