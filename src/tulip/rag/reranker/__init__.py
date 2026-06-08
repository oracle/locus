"""Rerankers — reorder retriever candidates by relevance to a query.

The retrieve-then-rerank pattern materially improves answer grounding in
RAG: pull a wide candidate set (top-K) from the vector store cheaply,
then have a *cross-encoder* reranker score each candidate against the
query before the top-N hits the LLM. The reranker sees both query and
candidate together, so it catches relevance signals an embedding-only
score misses.

This subpackage adds:

  * :class:`Reranker` — abstract base every reranker implements.
  * :class:`CrossEncoderReranker` — local ``sentence-transformers``
    cross-encoder; runs fully offline, no API key.
  * :class:`CohereReranker` — Cohere's hosted rerank API (direct).

Wire one into a retriever with ``RAGRetriever(reranker=...)`` to swap the
default semantic-only ordering for a reranked one.
"""

from typing import Any

from tulip.rag.reranker.base import Reranker


__all__ = ["CohereReranker", "CrossEncoderReranker", "Reranker"]


def __getattr__(name: str) -> Any:
    """Lazy import rerankers to avoid requiring optional dependencies."""
    if name == "CrossEncoderReranker":
        from tulip.rag.reranker.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker

    if name == "CohereReranker":
        from tulip.rag.reranker.cohere import CohereReranker

        return CohereReranker

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
