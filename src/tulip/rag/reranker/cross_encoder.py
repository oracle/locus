# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Local cross-encoder reranker backed by ``sentence-transformers``.

A cross-encoder scores each ``(query, document)`` pair jointly, so it
catches relevance signals an embedding-only similarity misses. This
implementation runs fully **offline** — the model is downloaded once and
then executes locally on CPU (or GPU if available), with no API key and
no per-call cost. That makes it the default reranker for tests and
air-gapped deployments.

Usage::

    from tulip.rag.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker(top_n=5)
    top = await reranker.rerank("hepcidin in iron metabolism", candidates)

Plugged into a retriever::

    retriever = RAGRetriever(embedder=embedder, store=store, reranker=reranker)
    results = await retriever.retrieve("query", limit=5)
"""

from __future__ import annotations

import asyncio
from typing import Any

from tulip.rag.reranker.base import Reranker
from tulip.rag.stores.base import SearchResult


#: Small, fast, widely-used MS-MARCO cross-encoder. ~80 MB, CPU-friendly.
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker(Reranker):
    """Reranker backed by a local ``sentence-transformers`` CrossEncoder.

    Args:
        model: HuggingFace cross-encoder model id. Defaults to
            ``cross-encoder/ms-marco-MiniLM-L-6-v2``.
        top_n: Trim the reranked output to the top N candidates. ``None``
            returns every candidate, reordered.
        device: Torch device string (``"cpu"``, ``"cuda"``). ``None``
            lets sentence-transformers auto-select.
        max_length: Max sequence length for the cross-encoder. ``None``
            uses the model default.
        _model: Injection seam for tests — a pre-built object exposing a
            ``predict(pairs) -> list[float]`` method bypasses the lazy
            ``sentence-transformers`` import.

    Notes:
        The model's ``predict`` call is synchronous; ``rerank`` dispatches
        it to a threadpool via :func:`asyncio.to_thread` so it composes
        with the async embedding + vector-store calls in a retriever.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_CROSS_ENCODER_MODEL,
        top_n: int | None = None,
        device: str | None = None,
        max_length: int | None = None,
        _model: Any = None,
    ) -> None:
        self.model = model
        self.top_n = top_n
        self.device = device
        self.max_length = max_length
        self._model_override = _model
        self._cached_model: Any = None

    def _get_model(self) -> Any:
        if self._model_override is not None:
            return self._model_override
        if self._cached_model is not None:
            return self._cached_model
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. "
                'Install with: pip install "tulip-agents[rerank-local]"'
            ) from e
        kwargs: dict[str, Any] = {}
        if self.device is not None:
            kwargs["device"] = self.device
        if self.max_length is not None:
            kwargs["max_length"] = self.max_length
        self._cached_model = CrossEncoder(self.model, **kwargs)
        return self._cached_model

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        """Reorder ``candidates`` by local cross-encoder relevance."""
        if not candidates:
            return []

        model = self._get_model()
        pairs = [[query, c.document.content] for c in candidates]
        scores = await asyncio.to_thread(model.predict, pairs)

        ranked = [
            SearchResult(
                document=candidate.document,
                score=float(score),
                # Preserve the original embedding score so callers can
                # compare semantic vs reranked ordering.
                distance=candidate.score,
            )
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        ranked.sort(key=lambda r: r.score, reverse=True)

        if self.top_n is not None:
            return ranked[: self.top_n]
        return ranked
