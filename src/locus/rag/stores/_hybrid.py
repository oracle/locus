# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Client-side blending for stores without a native hybrid query.

OpenSearch and Elasticsearch both answer a k-NN query and a BM25 ``match``
query, but a single blended query needs server-side plumbing (a search
pipeline with a normalisation processor on OpenSearch, a licensed retriever
on Elasticsearch). Running the two legs separately and blending the
normalised scores here gives every store the same ``alpha`` semantics as
:meth:`locus.rag.stores.oracle.OracleVectorStore.hybrid_search` with no
cluster configuration.
"""

from __future__ import annotations

from locus.rag.stores.base import Document, SearchResult


def blend_hybrid(
    vector_hits: list[SearchResult],
    lexical_hits: list[tuple[Document, float]],
    *,
    alpha: float,
    limit: int,
    threshold: float | None = None,
) -> list[SearchResult]:
    """Blend a vector leg and a lexical leg into one ranked list.

    Args:
        vector_hits: The store's own ``search()`` results. ``score`` is read
            as the vector similarity in ``[0, 1]`` (clamped).
        lexical_hits: ``(document, raw_score)`` pairs from a BM25 query. Raw
            scores are divided by the leg's maximum so the best lexical match
            scores ``1.0``.
        alpha: Weight of the vector leg; ``1 - alpha`` goes to the lexical leg.
        limit: Maximum number of results.
        threshold: Minimum *blended* score to keep.

    Returns:
        ``SearchResult`` list sorted by blended score, ``distance`` carrying
        the vector leg's distance when the document appeared there.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")

    pool: dict[str, tuple[Document, float, float, float | None]] = {}
    for hit in vector_hits:
        vec = min(1.0, max(0.0, float(hit.score)))
        pool[hit.document.id] = (hit.document, vec, 0.0, hit.distance)

    max_lex = max((s for _, s in lexical_hits), default=0.0)
    for doc, raw in lexical_hits:
        lex = float(raw) / max_lex if max_lex > 0 else 0.0
        if doc.id in pool:
            existing = pool[doc.id]
            pool[doc.id] = (existing[0], existing[1], lex, existing[3])
        else:
            pool[doc.id] = (doc, 0.0, lex, None)

    blended: list[SearchResult] = []
    for doc, vec, lex, distance in pool.values():
        score = alpha * vec + (1.0 - alpha) * lex
        if threshold is not None and score < threshold:
            continue
        blended.append(SearchResult(document=doc, score=score, distance=distance))

    blended.sort(key=lambda r: r.score, reverse=True)
    return blended[:limit]
