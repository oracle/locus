# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""MongoDB Atlas Vector Search store.

Backed by ``pymongo``'s asynchronous client. Documents are plain BSON
documents in one collection (``_id`` is the Locus document id) and
similarity search is the ``$vectorSearch`` aggregation stage against an
Atlas Vector Search index on the ``embedding`` field.

Atlas requires every filterable path to be declared in that index, so
``filter_fields`` names the metadata keys the index should cover.
:meth:`MongoDBAtlasVectorStore.ensure_index` creates the index when it is
missing (index builds are asynchronous on Atlas and take about a minute
to become queryable); ``$vectorSearch`` silently ignores a filter on an
undeclared path, which is why this store raises instead when
``metadata_filter`` names a key outside ``filter_fields``.

Usage::

    from locus.rag.stores import MongoDBAtlasVectorStore

    store = MongoDBAtlasVectorStore(
        connection_string="mongodb+srv://...",
        database="locus",
        collection="runbooks",
        dimension=1024,
        filter_fields=["tenant", "channel"],
    )
    await store.ensure_index(wait=True)
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from typing import Any

from locus.rag.stores.base import (
    BaseVectorStore,
    Document,
    SearchResult,
    VectorStoreConfig,
)


_MISSING_PYMONGO = 'pymongo is not installed. Install with: pip install "locus-sdk[mongodb]"'

_SIMILARITY_MAP = {
    "cosine": "cosine",
    "l2": "euclidean",
    "euclidean": "euclidean",
    "dot_product": "dotProduct",
    "dot": "dotProduct",
}


class MongoDBAtlasVectorStore(BaseVectorStore):
    """Vector store backed by MongoDB Atlas Vector Search.

    Args:
        dimension: Embedding dimension; written into the index definition.
        connection_string: MongoDB connection string. Defaults to the
            ``MONGODB_URI`` environment variable.
        database: Database name. Defaults to ``"locus"``.
        collection: Collection name. Defaults to ``"documents"``.
        index_name: Atlas Vector Search index name. Defaults to
            ``"vector_index"``.
        distance_metric: ``cosine`` (default), ``l2``/``euclidean``, or
            ``dot_product`` — mapped to the index ``similarity``.
        filter_fields: Metadata keys declared as ``filter`` paths in the
            index. Only these keys are accepted in ``metadata_filter``.
        num_candidates: ``$vectorSearch`` candidate pool; ``None`` uses
            ``max(limit * 10, 100)``.
        _collection: Injection seam for tests — an object with the async
            collection method surface.
    """

    def __init__(
        self,
        dimension: int = 1024,
        *,
        connection_string: str | None = None,
        database: str = "locus",
        collection: str = "documents",
        index_name: str = "vector_index",
        distance_metric: str = "cosine",
        filter_fields: list[str] | None = None,
        num_candidates: int | None = None,
        _collection: Any = None,
    ) -> None:
        if distance_metric not in _SIMILARITY_MAP:
            raise ValueError(
                f"Unsupported distance_metric {distance_metric!r}; "
                f"choose one of {sorted(_SIMILARITY_MAP)}"
            )
        self._dimension = dimension
        self._connection_string = connection_string or os.environ.get("MONGODB_URI")
        self._database = database
        self._collection_name = collection
        self._index_name = index_name
        self._distance_metric = distance_metric
        self._similarity = _SIMILARITY_MAP[distance_metric]
        self._filter_fields = list(filter_fields or [])
        self._num_candidates = num_candidates
        self._collection_override = _collection
        self._client: Any = None
        self._collection: Any = None

    @property
    def config(self) -> VectorStoreConfig:
        return VectorStoreConfig(
            dimension=self._dimension,
            distance_metric=self._distance_metric,
            index_type="hnsw",
        )

    # ----------------------------------------------------------------- client

    def _get_collection(self) -> Any:
        if self._collection_override is not None:
            return self._collection_override
        if self._collection is not None:
            return self._collection
        try:
            from pymongo import AsyncMongoClient  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(_MISSING_PYMONGO) from e
        if not self._connection_string:
            raise ValueError(
                "MongoDBAtlasVectorStore needs a connection string: pass "
                "connection_string= or set MONGODB_URI"
            )
        self._client = AsyncMongoClient(self._connection_string)
        self._collection = self._client[self._database][self._collection_name]
        return self._collection

    def index_definition(self) -> dict[str, Any]:
        """Atlas Vector Search index definition (public so operators can create it)."""
        fields: list[dict[str, Any]] = [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": self._dimension,
                "similarity": self._similarity,
            }
        ]
        fields.extend({"type": "filter", "path": f"metadata.{f}"} for f in self._filter_fields)
        return {"fields": fields}

    async def ensure_index(self, *, wait: bool = False, wait_timeout: float = 300.0) -> None:
        """Create the vector search index if it is missing.

        Args:
            wait: Poll until Atlas reports the index ``queryable``.
            wait_timeout: Seconds to wait before giving up when ``wait`` is set.
        """
        coll = self._get_collection()
        existing = [idx async for idx in await coll.list_search_indexes()]
        if not any(idx.get("name") == self._index_name for idx in existing):
            from pymongo.operations import SearchIndexModel  # noqa: PLC0415

            await coll.create_search_index(
                SearchIndexModel(
                    definition=self.index_definition(),
                    name=self._index_name,
                    type="vectorSearch",
                )
            )
        if not wait:
            return
        deadline = asyncio.get_running_loop().time() + wait_timeout
        while True:
            current = [idx async for idx in await coll.list_search_indexes(self._index_name)]
            if any(idx.get("queryable") for idx in current):
                return
            if asyncio.get_running_loop().time() > deadline:
                raise TimeoutError(
                    f"Atlas index {self._index_name!r} was not queryable after {wait_timeout:.0f}s"
                )
            await asyncio.sleep(5.0)

    # ------------------------------------------------------------------ rows

    @staticmethod
    def _body(doc: Document) -> dict[str, Any]:
        if doc.embedding is None:
            raise ValueError("Document must have an embedding")
        return {
            "_id": doc.id,
            "content": doc.content,
            "content_type": doc.content_type,
            "embedding": [float(x) for x in doc.embedding],
            "metadata": dict(doc.metadata),
            "created_at": doc.created_at,
        }

    @staticmethod
    def _to_document(row: dict[str, Any]) -> Document:
        created = row.get("created_at")
        if isinstance(created, str):
            try:
                created = datetime.fromisoformat(created)
            except ValueError:
                created = None
        if isinstance(created, datetime) and created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        embedding = row.get("embedding")
        return Document(
            id=str(row.get("_id", "")),
            content=row.get("content", "") or "",
            embedding=[float(x) for x in embedding] if embedding is not None else None,
            metadata=dict(row.get("metadata") or {}),
            created_at=created if isinstance(created, datetime) else datetime.now(UTC),
            content_type=row.get("content_type", "text") or "text",
        )

    def _score(self, raw: float) -> tuple[float, float | None]:
        """Map the ``vectorSearchScore`` back to ``(similarity, distance)``.

        Atlas reports ``(1 + cos) / 2`` for cosine, ``1 / (1 + d)`` for
        euclidean and ``(1 + dot) / 2`` for dotProduct.
        """
        if self._similarity == "cosine":
            sim = 2.0 * raw - 1.0
            return sim, 1.0 - sim
        if self._similarity == "dotProduct":
            return 2.0 * raw - 1.0, None
        return raw, (1.0 / raw - 1.0) if raw > 0 else float("inf")

    def build_filter(self, metadata_filter: dict[str, Any] | None) -> dict[str, Any] | None:
        """``$vectorSearch`` filter document; rejects keys outside ``filter_fields``."""
        if not metadata_filter:
            return None
        unknown = [key for key in metadata_filter if key not in self._filter_fields]
        if unknown:
            raise ValueError(
                f"metadata_filter keys {unknown!r} are not in filter_fields={self._filter_fields!r}; "
                "Atlas ignores filters on paths the index does not declare"
            )
        return {f"metadata.{key}": {"$eq": value} for key, value in metadata_filter.items()}

    # ------------------------------------------------------------------- API

    async def add(self, document: Document) -> str:
        coll = self._get_collection()
        await coll.replace_one({"_id": document.id}, self._body(document), upsert=True)
        return document.id

    async def add_batch(self, documents: list[Document]) -> list[str]:
        if not documents:
            return []
        from pymongo import ReplaceOne  # noqa: PLC0415

        coll = self._get_collection()
        await coll.bulk_write(
            [ReplaceOne({"_id": doc.id}, self._body(doc), upsert=True) for doc in documents]
        )
        return [doc.id for doc in documents]

    async def get(self, doc_id: str) -> Document | None:
        coll = self._get_collection()
        row = await coll.find_one({"_id": doc_id})
        return self._to_document(dict(row)) if row else None

    async def delete(self, doc_id: str) -> bool:
        coll = self._get_collection()
        result = await coll.delete_one({"_id": doc_id})
        return int(result.deleted_count) > 0

    def pipeline(
        self,
        query_embedding: list[float],
        limit: int,
        metadata_filter: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """The ``$vectorSearch`` aggregation pipeline for a query."""
        stage: dict[str, Any] = {
            "index": self._index_name,
            "path": "embedding",
            "queryVector": [float(x) for x in query_embedding],
            "numCandidates": self._num_candidates or max(limit * 10, 100),
            "limit": limit,
        }
        filter_doc = self.build_filter(metadata_filter)
        if filter_doc:
            stage["filter"] = filter_doc
        return [
            {"$vectorSearch": stage},
            {
                "$project": {
                    "content": 1,
                    "content_type": 1,
                    "embedding": 1,
                    "metadata": 1,
                    "created_at": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        coll = self._get_collection()
        cursor = await coll.aggregate(self.pipeline(query_embedding, limit, metadata_filter))
        results: list[SearchResult] = []
        async for row in cursor:
            score, distance = self._score(float(row.get("score", 0.0)))
            if threshold is not None and score < threshold:
                continue
            results.append(
                SearchResult(document=self._to_document(dict(row)), score=score, distance=distance)
            )
        return results

    async def count(self) -> int:
        coll = self._get_collection()
        return int(await coll.count_documents({}))

    async def clear(self) -> int:
        coll = self._get_collection()
        result = await coll.delete_many({})
        return int(result.deleted_count)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._collection = None

    def __repr__(self) -> str:
        return (
            f"MongoDBAtlasVectorStore(database={self._database!r}, "
            f"collection={self._collection_name!r}, index={self._index_name!r})"
        )
