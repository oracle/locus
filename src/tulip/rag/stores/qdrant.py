# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Qdrant vector store.

Backed by the official ``qdrant-client``. Works against a Qdrant server
(``url=``) or, for tests and small local corpora, an **in-memory**
instance (``location=":memory:"``) that needs no server and no network —
making it a free, offline test target.

Usage::

    from tulip.rag.stores import QdrantVectorStore

    # Local, in-memory (tests / demos)
    store = QdrantVectorStore(location=":memory:", dimension=1024)

    # Server
    store = QdrantVectorStore(url="http://localhost:6333", dimension=1024)
"""

from __future__ import annotations

import uuid
from typing import Any

from tulip.rag.stores.base import (
    BaseVectorStore,
    Document,
    SearchResult,
    VectorStoreConfig,
)


# Deterministic namespace so a given ``Document.id`` always maps to the
# same Qdrant point id (Qdrant ids must be unsigned ints or UUIDs).
_QDRANT_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")

_DISTANCE_MAP = {
    "cosine": "Cosine",
    "l2": "Euclid",
    "euclidean": "Euclid",
    "dot_product": "Dot",
    "dot": "Dot",
}


def _point_id(doc_id: str) -> str:
    """Map an arbitrary string ``Document.id`` to a deterministic UUID."""
    return str(uuid.uuid5(_QDRANT_NAMESPACE, doc_id))


class QdrantVectorStore(BaseVectorStore):
    """Vector store backed by Qdrant.

    Args:
        dimension: Embedding dimension. Used when the collection is
            created.
        collection_name: Qdrant collection. Defaults to ``"tulip"``.
        distance_metric: ``cosine`` (default), ``l2``/``euclidean``, or
            ``dot_product``.
        location: Qdrant location string — ``":memory:"`` for an
            in-process instance (tests / local). Mutually exclusive with
            ``url``.
        url: Qdrant server URL (e.g. ``http://localhost:6333``).
        api_key: Qdrant Cloud API key.
        _client: Injection seam for tests — a pre-built
            ``AsyncQdrantClient`` bypasses the lazy import.
    """

    def __init__(
        self,
        dimension: int = 1024,
        collection_name: str = "tulip",
        distance_metric: str = "cosine",
        *,
        location: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
        _client: Any = None,
    ) -> None:
        self._dimension = dimension
        self._collection = collection_name
        self._distance_metric = distance_metric
        self._location = location
        self._url = url
        self._api_key = api_key
        self._client_override = _client
        self._client: Any = None
        self._ensured = False

    @property
    def config(self) -> VectorStoreConfig:
        return VectorStoreConfig(
            dimension=self._dimension,
            distance_metric=self._distance_metric,
            index_type="hnsw",
        )

    def _get_client(self) -> Any:
        if self._client_override is not None:
            return self._client_override
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import AsyncQdrantClient  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                'qdrant-client is not installed. Install with: pip install "tulip-agents[qdrant]"'
            ) from e
        if self._url is not None:
            self._client = AsyncQdrantClient(url=self._url, api_key=self._api_key)
        else:
            self._client = AsyncQdrantClient(location=self._location or ":memory:")
        return self._client

    async def _ensure_collection(self) -> None:
        if self._ensured:
            return
        from qdrant_client import models  # noqa: PLC0415

        client = self._get_client()
        if not await client.collection_exists(self._collection):
            distance = _DISTANCE_MAP.get(self._distance_metric, "Cosine")
            await client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=self._dimension,
                    distance=models.Distance(distance),
                ),
            )
        self._ensured = True

    async def add(self, document: Document) -> str:
        ids = await self.add_batch([document])
        return ids[0]

    async def add_batch(self, documents: list[Document]) -> list[str]:
        if not documents:
            return []
        from qdrant_client import models  # noqa: PLC0415

        await self._ensure_collection()
        client = self._get_client()
        points = []
        ids: list[str] = []
        for doc in documents:
            if doc.embedding is None:
                raise ValueError("Document must have an embedding")
            payload = {
                "_doc_id": doc.id,
                "content": doc.content,
                "content_type": doc.content_type,
                "metadata": doc.metadata,
            }
            points.append(
                models.PointStruct(
                    id=_point_id(doc.id),
                    vector=doc.embedding,
                    payload=payload,
                )
            )
            ids.append(doc.id)
        await client.upsert(collection_name=self._collection, points=points)
        return ids

    def _to_document(self, payload: dict[str, Any], vector: Any = None) -> Document:
        return Document(
            id=payload.get("_doc_id", ""),
            content=payload.get("content", ""),
            embedding=list(vector) if vector is not None else None,
            metadata=payload.get("metadata", {}) or {},
            content_type=payload.get("content_type", "text"),
        )

    async def get(self, doc_id: str) -> Document | None:
        await self._ensure_collection()
        client = self._get_client()
        points = await client.retrieve(
            collection_name=self._collection,
            ids=[_point_id(doc_id)],
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            return None
        return self._to_document(points[0].payload or {}, points[0].vector)

    async def delete(self, doc_id: str) -> bool:
        existing = await self.get(doc_id)
        if existing is None:
            return False
        from qdrant_client import models  # noqa: PLC0415

        client = self._get_client()
        await client.delete(
            collection_name=self._collection,
            points_selector=models.PointIdsList(points=[_point_id(doc_id)]),
        )
        return True

    def _build_filter(self, metadata_filter: dict[str, Any] | None) -> Any:
        if not metadata_filter:
            return None
        from qdrant_client import models  # noqa: PLC0415

        return models.Filter(
            must=[
                models.FieldCondition(
                    key=f"metadata.{key}",
                    match=models.MatchValue(value=value),
                )
                for key, value in metadata_filter.items()
            ]
        )

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        await self._ensure_collection()
        client = self._get_client()
        response = await client.query_points(
            collection_name=self._collection,
            query=query_embedding,
            limit=limit,
            query_filter=self._build_filter(metadata_filter),
            score_threshold=threshold,
            with_payload=True,
            with_vectors=True,
        )
        results: list[SearchResult] = []
        for point in response.points:
            score = float(point.score)
            results.append(
                SearchResult(
                    document=self._to_document(point.payload or {}, point.vector),
                    score=score,
                    distance=1.0 - score if self._distance_metric == "cosine" else None,
                )
            )
        return results

    async def count(self) -> int:
        await self._ensure_collection()
        client = self._get_client()
        result = await client.count(collection_name=self._collection)
        return int(result.count)

    async def clear(self) -> int:
        client = self._get_client()
        if await client.collection_exists(self._collection):
            n = await self.count()
            await client.delete_collection(collection_name=self._collection)
            self._ensured = False
            return n
        return 0

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
