# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Maximal Marginal Relevance (MMR) diversity re-ranker.

Pure Python, no numpy / scipy dependency — keeps locus's optional
extras list small. Operates on a candidate pool of
:class:`SearchResult` objects already scored against a query, picking
``limit`` of them that balance relevance to the query against
dissimilarity from each other.

Reference: Carbonell & Goldstein, 1998
(<https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf>).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from locus.rag.stores.base import SearchResult


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]. Returns 0 on zero-norm inputs."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        ai = a[i]
        bi = b[i]
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def mmr_rerank(
    candidates: list[SearchResult],
    *,
    query_embedding: list[float],
    limit: int,
    lambda_: float = 0.5,
) -> list[SearchResult]:
    """Greedy MMR over ``candidates``.

    Each pick maximises::

        lambda * sim(doc, query) - (1 - lambda) * max(sim(doc, picked))

    Candidates without an ``embedding`` are skipped from the diversity
    computation but kept as fallbacks when the embedded pool is
    exhausted.

    Args:
        candidates: Result pool, ideally over-fetched (e.g. ``limit*4``).
        query_embedding: Query vector.
        limit: Final result count.
        lambda_: Trade-off in ``[0.0, 1.0]``. ``1.0`` reduces to plain
            top-N relevance; ``0.0`` is pure diversity.

    Returns:
        The selected results, in MMR pick order.
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError(f"lambda_ must be in [0.0, 1.0], got {lambda_}")
    if limit <= 0 or not candidates:
        return []

    # Precompute relevance against the query for every candidate that
    # carries an embedding. Candidates without embeddings still use
    # their reported ``score`` as a relevance proxy.
    pool: list[tuple[SearchResult, float]] = []
    for cand in candidates:
        emb = cand.document.embedding
        if emb:
            rel = _cosine(query_embedding, emb)
            # Normalise [-1, 1] → [0, 1] so it composes with diversity.
            rel = (rel + 1.0) / 2.0
        else:
            rel = float(cand.score)
        pool.append((cand, rel))

    selected: list[SearchResult] = []
    remaining = list(range(len(pool)))

    while remaining and len(selected) < limit:
        best_idx = remaining[0]
        best_score = -float("inf")
        for idx in remaining:
            cand, rel = pool[idx]
            if not selected:
                diversity_penalty = 0.0
            else:
                emb = cand.document.embedding or []
                if not emb:
                    diversity_penalty = 0.0
                else:
                    max_sim = 0.0
                    for picked in selected:
                        pemb = picked.document.embedding or []
                        if not pemb:
                            continue
                        sim = (_cosine(emb, pemb) + 1.0) / 2.0
                        max_sim = max(max_sim, sim)
                    diversity_penalty = max_sim
            mmr_score = lambda_ * rel - (1.0 - lambda_) * diversity_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        selected.append(pool[best_idx][0])
        remaining.remove(best_idx)

    return selected
