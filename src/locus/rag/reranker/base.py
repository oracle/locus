# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Reranker abstract base — reorder ``SearchResult``s by query relevance."""

from __future__ import annotations

from abc import ABC, abstractmethod

from locus.rag.stores.base import SearchResult


class Reranker(ABC):
    """Cross-encoder reranker over retriever candidates.

    Implementations score each ``SearchResult.document.content`` against
    the query string and return the candidates reordered (and optionally
    truncated). The ``SearchResult.score`` field is replaced with the
    reranker's relevance score (typically 0.0-1.0, higher = more relevant).

    Two contracts every implementation honours:

    1. **Empty input → empty output.** ``rerank("...", [])`` returns
       ``[]`` without making an API call.
    2. **No mutation of input.** The original list is left untouched; a
       new list of new ``SearchResult`` instances is returned. Safe to
       call from concurrent contexts.

    Example::

        from locus.rag.reranker import CohereReranker

        reranker = CohereReranker(model="cohere.rerank-v3.5", top_n=5)
        top = await reranker.rerank("hepcidin role in iron", candidates)
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        """Return ``candidates`` reordered (and optionally truncated) by
        relevance to ``query``.

        Args:
            query: The query string to score candidates against.
            candidates: The retriever's hits, typically wide (top-K=50).

        Returns:
            A *new* list of ``SearchResult`` in descending relevance,
            length ≤ ``top_n`` if the implementation has that bound set.
            Each returned ``SearchResult`` carries the reranker's
            relevance score in ``.score``; the original embedding score
            is preserved on ``.distance`` so callers can compare.
        """
