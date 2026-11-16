# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Elasticsearch vector store.

Backed by the official ``elasticsearch`` client (``AsyncElasticsearch``,
8.x and 9.x). Documents live in one index with a ``dense_vector`` field
for the embedding, a ``text`` field for the content (BM25) and a
``flattened`` field for metadata, so ``metadata_filter`` is an exact
``term`` match on ``metadata.<key>`` without per-key mappings.

Hybrid search runs the k-NN leg and a ``match`` leg as two requests and
blends the normalised scores client-side (see
:mod:`locus.rag.stores._hybrid`); no retriever licence or ingest pipeline
is needed on the cluster.

Usage::

    from locus.rag.stores import ElasticsearchVectorStore

    store = ElasticsearchVectorStore(
        hosts=["http://localhost:9200"], dimension=1024
    )
    store = ElasticsearchVectorStore(cloud_id="...", api_key="...", dimension=1024)
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from locus.core.loop_bound import loop_bound
from locus.rag.stores._hybrid import blend_hybrid
from locus.rag.stores.base import (
    BaseVectorStore,
    Document,
    SearchResult,
    VectorStoreConfig,
)


_MISSING_ES = 'elasticsearch is not installed. Install with: pip install "locus-sdk[elasticsearch]"'

_SIMILARITY_MAP = {
    "cosine": "cosine",
    "l2": "l2_norm",
    "euclidean": "l2_norm",
    "dot_product": "dot_product",
    "dot": "dot_product",
}

_SOURCE_FIELDS = ["doc_id", "content", "content_type", "embedding", "metadata", "created_at"]


def _is_not_found(exc: BaseException) -> bool:
    return type(exc).__name__ == "NotFoundError" or getattr(exc, "status_code", None) == 404


def build_filter_clauses(metadata_filter: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Equality filters as ``term`` queries on the flattened ``metadata`` field."""
    if not metadata_filter:
        return []
    return [{"term": {f"metadata.{key}": value}} for key, value in metadata_filter.items()]


class ElasticsearchVectorStore(BaseVectorStore):
    """Vector store backed by Elasticsearch ``dense_vector`` k-NN search.

    Args:
        dimension: Embedding dimension; fixed in the index mapping.
        index_name: Index to use. Defaults to ``"locus_vectors"``.
        distance_metric: ``cosine`` (default), ``l2``/``euclidean``, or
            ``dot_product`` — mapped to the ``dense_vector`` ``similarity``.
        hosts: Node URLs, e.g. ``["http://localhost:9200"]``. Mutually
            exclusive with ``cloud_id``.
        cloud_id: Elastic Cloud deployment id.
        api_key: API key (string or ``(id, key)`` tuple).
        basic_auth: ``(username, password)`` tuple.
        verify_certs: Verify TLS certificates. Defaults to ``True``.
        ca_certs: Path to a CA bundle.
        num_candidates: k-NN ``num_candidates`` per shard; ``None`` uses
            ``max(limit * 10, 100)``.
        _client: Injection seam for tests — an object with the
            ``AsyncElasticsearch`` method surface.
    """

    def __init__(
        self,
        dimension: int = 1024,
        index_name: str = "locus_vectors",
        distance_metric: str = "cosine",
        *,
        hosts: list[str] | None = None,
        cloud_id: str | None = None,
        api_key: str | tuple[str, str] | None = None,
        basic_auth: tuple[str, str] | None = None,
        verify_certs: bool = True,
        ca_certs: str | None = None,
        num_candidates: int | None = None,
        _client: Any = None,
    ) -> None:
        if distance_metric not in _SIMILARITY_MAP:
            raise ValueError(
                f"Unsupported distance_metric {distance_metric!r}; "
                f"choose one of {sorted(_SIMILARITY_MAP)}"
            )
        self._dimension = dimension
        self._index = index_name
        self._distance_metric = distance_metric
        self._similarity = _SIMILARITY_MAP[distance_metric]
        self._hosts = hosts
        self._cloud_id = cloud_id
        self._api_key = api_key
        self._basic_auth = basic_auth
        self._verify_certs = verify_certs
        self._ca_certs = ca_certs
        self._num_candidates = num_candidates
        self._client: Any = _client
        self._ensured = False

    @property
    def config(self) -> VectorStoreConfig:
        return VectorStoreConfig(
            dimension=self._dimension,
            distance_metric=self._distance_metric,
            index_type="hnsw",
        )

    # ----------------------------------------------------------------- client

    def _get_client(self) -> Any:
        def build() -> Any:
            try:
                from elasticsearch import AsyncElasticsearch  # noqa: PLC0415
            except ImportError as e:
                raise ImportError(_MISSING_ES) from e
            kwargs: dict[str, Any] = {"verify_certs": self._verify_certs}
            if self._cloud_id is not None:
                kwargs["cloud_id"] = self._cloud_id
            else:
                kwargs["hosts"] = self._hosts or ["http://localhost:9200"]
            if self._api_key is not None:
                kwargs["api_key"] = self._api_key
            if self._basic_auth is not None:
                kwargs["basic_auth"] = self._basic_auth
            if self._ca_certs is not None:
                kwargs["ca_certs"] = self._ca_certs
            return AsyncElasticsearch(**kwargs)

        return loop_bound(self, "_client", build)

    def mapping(self) -> dict[str, Any]:
        """Index mapping for the documents (public so operators can pre-create it)."""
        return {
            "properties": {
                "doc_id": {"type": "keyword"},
                "content": {"type": "text"},
                "content_type": {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": self._dimension,
                    "index": True,
                    "similarity": self._similarity,
                },
                "metadata": {"type": "flattened"},
                "created_at": {"type": "date"},
            }
        }

    async def _ensure_index(self) -> Any:
        client = self._get_client()
        if self._ensured:
            return client
        if not await client.indices.exists(index=self._index):
            await client.indices.create(index=self._index, mappings=self.mapping())
        self._ensured = True
        return client

    # ------------------------------------------------------------------ rows

    @staticmethod
    def _body(doc: Document) -> dict[str, Any]:
        if doc.embedding is None:
            raise ValueError("Document must have an embedding")
        return {
            "doc_id": doc.id,
            "content": doc.content,
            "content_type": doc.content_type,
            "embedding": [float(x) for x in doc.embedding],
            "metadata": dict(doc.metadata),
            "created_at": doc.created_at.isoformat(),
        }

    @staticmethod
    def _to_document(source: dict[str, Any]) -> Document:
        created = source.get("created_at")
        try:
            created_at = datetime.fromisoformat(created) if created else datetime.now(UTC)
        except ValueError:
            created_at = datetime.now(UTC)
        embedding = source.get("embedding")
        return Document(
            id=str(source.get("doc_id", "")),
            content=source.get("content", "") or "",
            embedding=[float(x) for x in embedding] if embedding is not None else None,
            metadata=dict(source.get("metadata") or {}),
            created_at=created_at,
            content_type=source.get("content_type", "text") or "text",
        )

    def _score(self, raw: float) -> tuple[float, float | None]:
        """Map the k-NN ``_score`` back to ``(similarity, distance)``.

        Elasticsearch reports ``(1 + cos) / 2`` for cosine, ``1 / (1 + d²)``
        for ``l2_norm`` and ``(1 + dot) / 2`` for ``dot_product``.
        """
        if self._similarity == "cosine":
            sim = 2.0 * raw - 1.0
            return sim, 1.0 - sim
        if self._similarity == "dot_product":
            return 2.0 * raw - 1.0, None
        distance = math.sqrt(max(0.0, 1.0 / raw - 1.0)) if raw > 0 else float("inf")
        return raw, distance

    # ------------------------------------------------------------------- API

    async def add(self, document: Document) -> str:
        client = await self._ensure_index()
        await client.index(
            index=self._index, id=document.id, document=self._body(document), refresh="wait_for"
        )
        return document.id

    async def add_batch(self, documents: list[Document]) -> list[str]:
        if not documents:
            return []
        client = await self._ensure_index()
        operations: list[dict[str, Any]] = []
        for doc in documents:
            operations.append({"index": {"_index": self._index, "_id": doc.id}})
            operations.append(self._body(doc))
        response = await client.bulk(operations=operations, refresh="wait_for")
        if response.get("errors"):
            failed = [
                item["index"].get("error")
                for item in response.get("items", [])
                if item.get("index", {}).get("error")
            ]
            raise RuntimeError(f"Elasticsearch rejected {len(failed)} document(s): {failed[:1]}")
        return [doc.id for doc in documents]

    async def get(self, doc_id: str) -> Document | None:
        client = await self._ensure_index()
        try:
            response = await client.get(index=self._index, id=doc_id)
        except Exception as exc:
            if _is_not_found(exc):
                return None
            raise
        return self._to_document(dict(response["_source"]))

    async def delete(self, doc_id: str) -> bool:
        client = await self._ensure_index()
        try:
            response = await client.delete(index=self._index, id=doc_id, refresh="wait_for")
        except Exception as exc:
            if _is_not_found(exc):
                return False
            raise
        return bool(response.get("result") == "deleted")

    def _knn(
        self, query_embedding: list[float], k: int, filters: list[dict[str, Any]]
    ) -> dict[str, Any]:
        knn: dict[str, Any] = {
            "field": "embedding",
            "query_vector": [float(x) for x in query_embedding],
            "k": k,
            "num_candidates": self._num_candidates or max(k * 10, 100),
        }
        if filters:
            knn["filter"] = filters
        return knn

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        client = await self._ensure_index()
        response = await client.search(
            index=self._index,
            knn=self._knn(query_embedding, limit, build_filter_clauses(metadata_filter)),
            size=limit,
            source=_SOURCE_FIELDS,
        )
        results: list[SearchResult] = []
        for hit in response["hits"]["hits"]:
            score, distance = self._score(float(hit["_score"]))
            if threshold is not None and score < threshold:
                continue
            results.append(
                SearchResult(
                    document=self._to_document(dict(hit["_source"])), score=score, distance=distance
                )
            )
        return results

    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        *,
        limit: int = 10,
        alpha: float = 0.5,
        threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Blend k-NN similarity with BM25 ``match`` relevance on ``content``.

        Both legs fetch ``max(limit * 3, 20)`` candidates with the same
        metadata filter; scores are normalised to ``[0, 1]`` and combined as
        ``alpha * vector + (1 - alpha) * lexical``. An empty ``query_text``
        is answered by :meth:`search`.

        ``threshold`` applies to the blended score. When only one leg returns
        hits, that leg's scores are returned unblended, so a threshold chosen
        for vector search still keeps an exact match.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
        if not query_text.strip():
            return await self.search(
                query_embedding, limit=limit, threshold=threshold, metadata_filter=metadata_filter
            )
        pool = max(limit * 3, 20)
        filters = build_filter_clauses(metadata_filter)
        vector_hits = await self.search(
            query_embedding, limit=pool, metadata_filter=metadata_filter
        )

        client = await self._ensure_index()
        query: dict[str, Any] = {"bool": {"must": [{"match": {"content": query_text}}]}}
        if filters:
            query["bool"]["filter"] = filters
        response = await client.search(
            index=self._index, query=query, size=pool, source=_SOURCE_FIELDS
        )
        lexical_hits = [
            (self._to_document(dict(hit["_source"])), float(hit["_score"] or 0.0))
            for hit in response["hits"]["hits"]
        ]
        return blend_hybrid(
            vector_hits, lexical_hits, alpha=alpha, limit=limit, threshold=threshold
        )

    async def count(self) -> int:
        client = await self._ensure_index()
        response = await client.count(index=self._index)
        return int(response["count"])

    async def clear(self) -> int:
        client = await self._ensure_index()
        n = await self.count()
        await client.delete_by_query(index=self._index, query={"match_all": {}}, refresh=True)
        return n

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._ensured = False

    def __repr__(self) -> str:
        return f"ElasticsearchVectorStore(index={self._index!r})"
