# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Pinecone vector store.

Backed by the official ``pinecone`` client against a serverless index.
Pinecone stores vectors with a flat metadata map, so the document content,
content type, timestamp and the full metadata dict (as JSON) ride along
under reserved ``_``-prefixed keys, and scalar metadata values are
flattened next to them for server-side ``metadata_filter``.

The client is synchronous, so each call is dispatched to a worker thread
with :func:`asyncio.to_thread`. Pinecone's metadata limit is 40 KB per
vector, so keep chunks well under that.

Usage::

    from locus.rag.stores import PineconeVectorStore

    store = PineconeVectorStore(
        index_name="runbooks", dimension=1024
    )  # PINECONE_API_KEY
    store = PineconeVectorStore(
        index_name="runbooks", dimension=1024, namespace="tenant-a"
    )
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import UTC, datetime
from typing import Any

from locus.rag.stores.base import (
    BaseVectorStore,
    Document,
    SearchResult,
    VectorStoreConfig,
)


_MISSING_PINECONE = 'pinecone is not installed. Install with: pip install "locus-sdk[pinecone]"'

_METRIC_MAP = {
    "cosine": "cosine",
    "l2": "euclidean",
    "euclidean": "euclidean",
    "dot_product": "dotproduct",
    "dot": "dotproduct",
}

_SCALAR = (str, int, float, bool)
_RESERVED = ("_content", "_content_type", "_created_at", "_metadata_json")


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Attribute-or-key access: the client returns objects, fakes may return dicts."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class PineconeVectorStore(BaseVectorStore):
    """Vector store backed by a Pinecone serverless index.

    Args:
        dimension: Embedding dimension; fixed when the index is created.
        index_name: Pinecone index. Defaults to ``"locus"``.
        distance_metric: ``cosine`` (default), ``l2``/``euclidean``, or
            ``dot_product``.
        api_key: Pinecone API key. Defaults to the ``PINECONE_API_KEY``
            environment variable.
        namespace: Namespace inside the index. Defaults to ``""``.
        cloud: Serverless cloud used when the index has to be created.
        region: Serverless region used when the index has to be created.
        create_index: Create the index when it does not exist (default
            ``True``). Set ``False`` to require a pre-provisioned index.
        _index: Injection seam for tests — an object with the ``Index``
            method surface (``upsert``, ``fetch``, ``query``, ``delete``,
            ``describe_index_stats``).
    """

    def __init__(
        self,
        dimension: int = 1024,
        index_name: str = "locus",
        distance_metric: str = "cosine",
        *,
        api_key: str | None = None,
        namespace: str = "",
        cloud: str = "aws",
        region: str = "us-east-1",
        create_index: bool = True,
        _index: Any = None,
    ) -> None:
        if distance_metric not in _METRIC_MAP:
            raise ValueError(
                f"Unsupported distance_metric {distance_metric!r}; "
                f"choose one of {sorted(_METRIC_MAP)}"
            )
        self._dimension = dimension
        self._index_name = index_name
        self._distance_metric = distance_metric
        self._metric = _METRIC_MAP[distance_metric]
        self._api_key = api_key or os.environ.get("PINECONE_API_KEY")
        self._namespace = namespace
        self._cloud = cloud
        self._region = region
        self._create_index = create_index
        self._index_override = _index
        self._index: Any = None

    @property
    def config(self) -> VectorStoreConfig:
        return VectorStoreConfig(
            dimension=self._dimension,
            distance_metric=self._distance_metric,
            index_type="hnsw",
        )

    # ----------------------------------------------------------------- client

    def _get_index_sync(self) -> Any:
        if self._index_override is not None:
            return self._index_override
        if self._index is not None:
            return self._index
        try:
            from pinecone import Pinecone, ServerlessSpec  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(_MISSING_PINECONE) from e
        if not self._api_key:
            raise ValueError(
                "PineconeVectorStore needs an API key: pass api_key= or set PINECONE_API_KEY"
            )
        pc = Pinecone(api_key=self._api_key)
        if not pc.has_index(self._index_name):
            if not self._create_index:
                raise ValueError(
                    f"Pinecone index {self._index_name!r} does not exist and create_index=False"
                )
            pc.create_index(
                name=self._index_name,
                dimension=self._dimension,
                metric=self._metric,
                spec=ServerlessSpec(cloud=self._cloud, region=self._region),
            )
            deadline = time.monotonic() + 300
            while not _get(
                _get(pc.describe_index(self._index_name), "status", {}), "ready", default=False
            ):
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Pinecone index {self._index_name!r} not ready after 300s")
                time.sleep(2.0)
        self._index = pc.Index(self._index_name)
        return self._index

    # ------------------------------------------------------------------ rows

    @staticmethod
    def _metadata(doc: Document) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "_content": doc.content,
            "_content_type": doc.content_type,
            "_created_at": doc.created_at.isoformat(),
            "_metadata_json": json.dumps(doc.metadata, default=str),
        }
        for key, value in doc.metadata.items():
            if key.startswith("_"):
                continue
            if isinstance(value, _SCALAR) or (
                isinstance(value, list) and all(isinstance(v, str) for v in value)
            ):
                meta[key] = value
        return meta

    def _vector(self, doc: Document) -> dict[str, Any]:
        if doc.embedding is None:
            raise ValueError("Document must have an embedding")
        return {
            "id": doc.id,
            "values": [float(x) for x in doc.embedding],
            "metadata": self._metadata(doc),
        }

    @staticmethod
    def _to_document(doc_id: str, values: Any, meta: dict[str, Any] | None) -> Document:
        meta = dict(meta or {})
        try:
            metadata = json.loads(meta.get("_metadata_json") or "{}")
        except (TypeError, ValueError):
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {k: v for k, v in meta.items() if k not in _RESERVED}
        created = meta.get("_created_at")
        try:
            created_at = datetime.fromisoformat(created) if created else datetime.now(UTC)
        except ValueError:
            created_at = datetime.now(UTC)
        return Document(
            id=doc_id,
            content=meta.get("_content", "") or "",
            embedding=[float(x) for x in values] if values else None,
            metadata=metadata,
            created_at=created_at,
            content_type=meta.get("_content_type", "text") or "text",
        )

    def _score(self, raw: float) -> tuple[float, float | None]:
        """Map Pinecone's ``score`` to ``(score, distance)``.

        Cosine and dotproduct scores are the similarity itself; euclidean
        is the squared distance.
        """
        if self._metric == "cosine":
            return raw, 1.0 - raw
        if self._metric == "dotproduct":
            return raw, None
        return 1.0 / (1.0 + raw), raw

    @staticmethod
    def build_filter(metadata_filter: dict[str, Any] | None) -> dict[str, Any] | None:
        """Pinecone ``$eq`` filter on the flattened metadata keys."""
        if not metadata_filter:
            return None
        return {key: {"$eq": value} for key, value in metadata_filter.items()}

    # ------------------------------------------------------------------- API

    def _add_sync(self, documents: list[Document]) -> list[str]:
        index = self._get_index_sync()
        vectors = [self._vector(doc) for doc in documents]
        for start in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[start : start + 100], namespace=self._namespace)
        return [doc.id for doc in documents]

    async def add(self, document: Document) -> str:
        ids = await self.add_batch([document])
        return ids[0]

    async def add_batch(self, documents: list[Document]) -> list[str]:
        if not documents:
            return []
        return await asyncio.to_thread(self._add_sync, documents)

    def _fetch_sync(self, doc_id: str) -> Document | None:
        index = self._get_index_sync()
        response = index.fetch(ids=[doc_id], namespace=self._namespace)
        vectors = _get(response, "vectors", {}) or {}
        record = vectors.get(doc_id) if isinstance(vectors, dict) else None
        if record is None:
            return None
        return self._to_document(doc_id, _get(record, "values"), _get(record, "metadata"))

    async def get(self, doc_id: str) -> Document | None:
        return await asyncio.to_thread(self._fetch_sync, doc_id)

    def _delete_sync(self, doc_id: str) -> bool:
        if self._fetch_sync(doc_id) is None:
            return False
        self._get_index_sync().delete(ids=[doc_id], namespace=self._namespace)
        return True

    async def delete(self, doc_id: str) -> bool:
        return await asyncio.to_thread(self._delete_sync, doc_id)

    def _search_sync(
        self,
        query_embedding: list[float],
        limit: int,
        threshold: float | None,
        metadata_filter: dict[str, Any] | None,
    ) -> list[SearchResult]:
        index = self._get_index_sync()
        response = index.query(
            vector=[float(x) for x in query_embedding],
            top_k=limit,
            namespace=self._namespace,
            filter=self.build_filter(metadata_filter),
            include_metadata=True,
            include_values=True,
        )
        results: list[SearchResult] = []
        for match in _get(response, "matches", []) or []:
            score, distance = self._score(float(_get(match, "score", 0.0)))
            if threshold is not None and score < threshold:
                continue
            results.append(
                SearchResult(
                    document=self._to_document(
                        str(_get(match, "id")), _get(match, "values"), _get(match, "metadata")
                    ),
                    score=score,
                    distance=distance,
                )
            )
        return results

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        return await asyncio.to_thread(
            self._search_sync, query_embedding, limit, threshold, metadata_filter
        )

    def _count_sync(self) -> int:
        stats = self._get_index_sync().describe_index_stats()
        namespaces = _get(stats, "namespaces", {}) or {}
        entry = namespaces.get(self._namespace) if isinstance(namespaces, dict) else None
        if entry is None:
            return 0
        return int(_get(entry, "vector_count", 0) or 0)

    async def count(self) -> int:
        """Vector count for the namespace. Pinecone statistics are eventually consistent."""
        return await asyncio.to_thread(self._count_sync)

    def _clear_sync(self) -> int:
        n = self._count_sync()
        if n:
            self._get_index_sync().delete(delete_all=True, namespace=self._namespace)
        return n

    async def clear(self) -> int:
        return await asyncio.to_thread(self._clear_sync)

    async def close(self) -> None:
        self._index = None

    def __repr__(self) -> str:
        return f"PineconeVectorStore(index={self._index_name!r}, namespace={self._namespace!r})"
