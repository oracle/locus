# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""In-memory vector store for testing and development."""

from __future__ import annotations

import math
from typing import Any

from locus.rag.stores.base import (
    BaseVectorStore,
    Document,
    SearchResult,
    VectorStoreConfig,
)


class InMemoryVectorStore(BaseVectorStore):
    """
    In-memory vector store for testing and development.

    Fast but not persistent - data is lost when process exits.

    Example:
        >>> store = InMemoryVectorStore(dimension=1024)
        >>> await store.add(document)
        >>> results = await store.search(query_embedding, limit=5)
    """

    def __init__(
        self,
        dimension: int = 1024,
        distance_metric: str = "cosine",
    ):
        self._dimension = dimension
        self._distance_metric = distance_metric
        self._documents: dict[str, Document] = {}

    @property
    def config(self) -> VectorStoreConfig:
        return VectorStoreConfig(
            dimension=self._dimension,
            distance_metric=self._distance_metric,
            index_type="flat",
        )

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def _euclidean_distance(self, a: list[float], b: list[float]) -> float:
        """Compute Euclidean distance between two vectors."""
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=False)))

    def _dot_product(self, a: list[float], b: list[float]) -> float:
        """Compute dot product between two vectors."""
        return sum(x * y for x, y in zip(a, b, strict=False))

    async def add(self, document: Document) -> str:
        """Add a document."""
        if document.embedding is None:
            raise ValueError("Document must have an embedding")
        self._documents[document.id] = document
        return document.id

    async def add_batch(self, documents: list[Document]) -> list[str]:
        """Add multiple documents."""
        ids = []
        for doc in documents:
            doc_id = await self.add(doc)
            ids.append(doc_id)
        return ids

    async def get(self, doc_id: str) -> Document | None:
        """Get a document by ID."""
        return self._documents.get(doc_id)

    async def delete(self, doc_id: str) -> bool:
        """Delete a document."""
        if doc_id in self._documents:
            del self._documents[doc_id]
            return True
        return False

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""
        results = []

        for doc in self._documents.values():
            if doc.embedding is None:
                continue

            # Apply metadata filter
            if metadata_filter:
                match = True
                for key, value in metadata_filter.items():
                    if doc.metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            # Compute similarity/distance
            if self._distance_metric == "cosine":
                score = self._cosine_similarity(query_embedding, doc.embedding)
                distance = 1.0 - score
            elif self._distance_metric == "euclidean":
                distance = self._euclidean_distance(query_embedding, doc.embedding)
                score = 1.0 / (1.0 + distance)
            else:  # dot_product
                score = self._dot_product(query_embedding, doc.embedding)
                distance = -score  # Higher is better for dot product

            # Apply threshold
            if threshold is not None and score < threshold:
                continue

            results.append(
                SearchResult(
                    document=doc,
                    score=score,
                    distance=distance,
                )
            )

        # Sort by score (descending)
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:limit]

    async def count(self) -> int:
        """Count documents."""
        return len(self._documents)

    async def clear(self) -> int:
        """Delete all documents."""
        count = len(self._documents)
        self._documents.clear()
        return count

    def __repr__(self) -> str:
        return f"InMemoryVectorStore(dimension={self._dimension}, count={len(self._documents)})"
