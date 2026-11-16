# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Milvus vector store.

Backed by the official ``pymilvus`` ``MilvusClient``. Points at a Milvus
server or Zilliz Cloud through ``uri``/``token``; with a local file path as
the ``uri`` it runs **Milvus Lite** in-process, which needs no server, no
container and no account — the free, offline test target for this store.

The client is synchronous, so each call is dispatched to a worker thread
with :func:`asyncio.to_thread`, like the Chroma store.

Hybrid search: with ``enable_hybrid=True`` the collection carries a sparse
BM25 vector derived from ``content`` by a Milvus ``Function``, and
:meth:`MilvusVectorStore.hybrid_search` fuses the dense and BM25 legs with
a ``WeightedRanker`` whose weights are ``alpha`` and ``1 - alpha``.

Usage::

    from locus.rag.stores import MilvusVectorStore

    # Milvus Lite — a file, no server
    store = MilvusVectorStore(uri="./runbooks.db", dimension=1024)

    # Server / Zilliz Cloud
    store = MilvusVectorStore(uri="http://localhost:19530", dimension=1024)
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
from datetime import UTC, datetime
from typing import Any

from locus.rag.stores.base import (
    BaseVectorStore,
    Document,
    SearchResult,
    VectorStoreConfig,
)


logger = logging.getLogger(__name__)

_MISSING_PYMILVUS = (
    'pymilvus is not installed. Install with: pip install "locus-sdk[milvus]" '
    "(Milvus Lite ships in that extra; a server needs only pymilvus)"
)

_METRIC_MAP = {
    "cosine": "COSINE",
    "l2": "L2",
    "euclidean": "L2",
    "dot_product": "IP",
    "dot": "IP",
}

_SCALAR = (str, int, float, bool)
_OUTPUT_FIELDS = ["id", "embedding", "content", "content_type", "metadata", "created_at"]


def _require_pymilvus() -> Any:
    """Return the ``pymilvus`` module, or explain how to install it."""
    try:
        import pymilvus  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - depends on install shape
        raise ImportError(_MISSING_PYMILVUS) from e
    return pymilvus


def _literal(value: Any) -> str:
    """Render a Python scalar as a Milvus boolean-expression literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    # ``ensure_ascii=False``: Milvus has no ``\uXXXX`` escape, so non-ASCII
    # text must stay literal inside the quoted string.
    return json.dumps(str(value), ensure_ascii=False)


def build_filter_expr(metadata_filter: dict[str, Any] | None) -> str:
    """Compile an equality metadata filter into a Milvus filter expression.

    ``{"topic": "x", "year": 2026}`` becomes
    ``metadata["topic"] == "x" and metadata["year"] == 2026``. Keys and
    string values are JSON-quoted, so neither can break out of the
    expression. Non-scalar values are rejected.
    """
    if not metadata_filter:
        return ""
    clauses: list[str] = []
    for key, value in metadata_filter.items():
        if not isinstance(value, _SCALAR):
            raise TypeError(
                f"metadata_filter[{key!r}] must be a str, int, float or bool, "
                f"got {type(value).__name__}"
            )
        clauses.append(f"metadata[{json.dumps(str(key), ensure_ascii=False)}] == {_literal(value)}")
    return " and ".join(clauses)


def _is_lite_uri(uri: str) -> bool:
    """A local ``.db`` path, which pymilvus serves with embedded Milvus Lite."""
    return uri.endswith(".db") and not uri.startswith(("http://", "https://", "unix:", "tcp://"))


def _release_lite_server(uri: str) -> None:
    """Stop the embedded Milvus Lite server behind ``uri``.

    ``MilvusClient.close()`` drops the connection and leaves the in-process
    server running until interpreter exit, when it flushes and recreates the
    database directory, so a caller that removed the file after ``close()``
    found it back. Stopping the server here makes ``close()`` the last write.
    """
    try:
        manager = importlib.import_module("milvus_lite.server_manager")
    except ImportError:
        return
    release = getattr(getattr(manager, "server_manager_instance", None), "release_server", None)
    if callable(release):
        release(uri)


class MilvusVectorStore(BaseVectorStore):
    """Vector store backed by Milvus (server, Zilliz Cloud or Milvus Lite).

    Args:
        dimension: Embedding dimension; fixed when the collection is created.
        collection_name: Milvus collection. Defaults to ``"locus"``.
        distance_metric: ``cosine`` (default), ``l2``/``euclidean``, or
            ``dot_product``.
        uri: Server URI (``http://host:19530``), Zilliz Cloud endpoint, or a
            local file path for Milvus Lite. Defaults to ``"./locus_milvus.db"``.
        token: Server/cloud credentials (``user:password`` or an API key).
        db_name: Milvus database to use. ``None`` keeps the client default.
        enable_hybrid: Add a BM25 sparse vector derived from ``content`` so
            :meth:`hybrid_search` is available. Set at collection creation; an
            existing collection created without it stays vector-only (see
            :attr:`hybrid_search_available`).
        _client: Injection seam for tests — a pre-built ``MilvusClient``
            (or any object with the same method surface).
    """

    def __init__(
        self,
        dimension: int = 1024,
        collection_name: str = "locus",
        distance_metric: str = "cosine",
        *,
        uri: str = "./locus_milvus.db",
        token: str | None = None,
        db_name: str | None = None,
        enable_hybrid: bool = False,
        _client: Any = None,
    ) -> None:
        if _client is None:
            _require_pymilvus()
        if distance_metric not in _METRIC_MAP:
            raise ValueError(
                f"Unsupported distance_metric {distance_metric!r}; "
                f"choose one of {sorted(_METRIC_MAP)}"
            )
        self._dimension = dimension
        self._collection = collection_name
        self._distance_metric = distance_metric
        self._metric = _METRIC_MAP[distance_metric]
        self._uri = uri
        self._token = token
        self._db_name = db_name
        self._enable_hybrid = enable_hybrid
        self._client_override = _client
        self._client: Any = None
        self._ensured = False
        # Derived from the collection schema once it has been opened.
        self._hybrid_ready: bool | None = None

    @property
    def config(self) -> VectorStoreConfig:
        return VectorStoreConfig(
            dimension=self._dimension,
            distance_metric=self._distance_metric,
            index_type="hnsw",
        )

    @property
    def hybrid_search_available(self) -> bool:
        """True when the collection carries the BM25 sparse field.

        Until the collection has been opened this mirrors ``enable_hybrid``;
        after the first operation it is read from the collection schema, so
        an existing collection created without the BM25 function reports
        ``False`` and ``RAGRetriever(hybrid=True)`` searches it by vector
        alone.
        """
        if self._hybrid_ready is None:
            return self._enable_hybrid
        return self._hybrid_ready

    # ----------------------------------------------------------------- client

    def _get_client(self) -> Any:
        if self._client_override is not None:
            return self._client_override
        if self._client is not None:
            return self._client
        pymilvus = _require_pymilvus()
        kwargs: dict[str, Any] = {"uri": self._uri}
        if self._token is not None:
            kwargs["token"] = self._token
        if self._db_name is not None:
            kwargs["db_name"] = self._db_name
        self._client = pymilvus.MilvusClient(**kwargs)
        return self._client

    def _ensure_collection_sync(self) -> Any:
        client = self._get_client()
        if self._ensured:
            return client
        if not client.has_collection(self._collection):
            pymilvus = _require_pymilvus()
            dt = pymilvus.DataType
            schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("id", dt.VARCHAR, is_primary=True, max_length=512)
            schema.add_field("embedding", dt.FLOAT_VECTOR, dim=self._dimension)
            if self._enable_hybrid:
                schema.add_field("content", dt.VARCHAR, max_length=65535, enable_analyzer=True)
                schema.add_field("sparse", dt.SPARSE_FLOAT_VECTOR)
                schema.add_function(
                    pymilvus.Function(
                        name="locus_bm25",
                        input_field_names=["content"],
                        output_field_names=["sparse"],
                        function_type=pymilvus.FunctionType.BM25,
                    )
                )
            else:
                schema.add_field("content", dt.VARCHAR, max_length=65535)
            schema.add_field("content_type", dt.VARCHAR, max_length=64)
            schema.add_field("metadata", dt.JSON)
            schema.add_field("created_at", dt.VARCHAR, max_length=64)

            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="embedding", index_type="AUTOINDEX", metric_type=self._metric
            )
            if self._enable_hybrid:
                index_params.add_index(
                    field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25"
                )
            client.create_collection(self._collection, schema=schema, index_params=index_params)
            has_sparse = self._enable_hybrid
        else:
            # A collection that already exists may not be loaded: a Milvus
            # Lite file reopened after ``close()`` stopped its server or in a
            # new process, a server after a release or restart. Loading is
            # idempotent, and a search on an unloaded collection fails.
            client.load_collection(self._collection)
            has_sparse = False
            if self._enable_hybrid:
                description = client.describe_collection(self._collection)
                has_sparse = any(
                    field.get("name") == "sparse" for field in description.get("fields") or []
                )
        self._hybrid_ready = has_sparse
        self._ensured = True
        return client

    # ------------------------------------------------------------------ rows

    @staticmethod
    def _row(doc: Document) -> dict[str, Any]:
        if doc.embedding is None:
            raise ValueError("Document must have an embedding")
        return {
            "id": doc.id,
            "embedding": [float(x) for x in doc.embedding],
            "content": doc.content,
            "content_type": doc.content_type,
            "metadata": dict(doc.metadata),
            "created_at": doc.created_at.isoformat(),
        }

    @staticmethod
    def _to_document(entity: dict[str, Any]) -> Document:
        created = entity.get("created_at")
        try:
            created_at = datetime.fromisoformat(created) if created else datetime.now(UTC)
        except ValueError:
            created_at = datetime.now(UTC)
        embedding = entity.get("embedding")
        return Document(
            id=str(entity.get("id", "")),
            content=entity.get("content", "") or "",
            embedding=[float(x) for x in embedding] if embedding is not None else None,
            metadata=dict(entity.get("metadata") or {}),
            created_at=created_at,
            content_type=entity.get("content_type", "text") or "text",
        )

    def _score(self, raw: float) -> tuple[float, float | None]:
        """Map Milvus's ``distance`` field to ``(score, distance)``.

        For COSINE and IP Milvus reports the *similarity* (higher is
        closer); for L2 it reports the squared distance.
        """
        if self._metric == "COSINE":
            return raw, 1.0 - raw
        if self._metric == "IP":
            return raw, None
        return 1.0 / (1.0 + raw), raw

    # ------------------------------------------------------------------- API

    def _add_sync(self, documents: list[Document]) -> list[str]:
        client = self._ensure_collection_sync()
        rows = [self._row(doc) for doc in documents]
        client.upsert(self._collection, data=rows)
        return [doc.id for doc in documents]

    async def add(self, document: Document) -> str:
        ids = await self.add_batch([document])
        return ids[0]

    async def add_batch(self, documents: list[Document]) -> list[str]:
        if not documents:
            return []
        return await asyncio.to_thread(self._add_sync, documents)

    def _get_sync(self, doc_id: str) -> Document | None:
        client = self._ensure_collection_sync()
        rows = client.get(self._collection, ids=[doc_id], output_fields=_OUTPUT_FIELDS)
        if not rows:
            return None
        return self._to_document(dict(rows[0]))

    async def get(self, doc_id: str) -> Document | None:
        return await asyncio.to_thread(self._get_sync, doc_id)

    def _delete_sync(self, doc_id: str) -> bool:
        client = self._ensure_collection_sync()
        if not client.get(self._collection, ids=[doc_id], output_fields=["id"]):
            return False
        client.delete(self._collection, ids=[doc_id])
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
        client = self._ensure_collection_sync()
        response = client.search(
            self._collection,
            data=[[float(x) for x in query_embedding]],
            anns_field="embedding",
            limit=limit,
            filter=build_filter_expr(metadata_filter),
            output_fields=_OUTPUT_FIELDS,
            search_params={"metric_type": self._metric},
        )
        results: list[SearchResult] = []
        for hit in response[0] if response else []:
            score, distance = self._score(float(hit["distance"]))
            if threshold is not None and score < threshold:
                continue
            entity = dict(hit.get("entity") or {})
            entity.setdefault("id", hit.get("id"))
            results.append(
                SearchResult(document=self._to_document(entity), score=score, distance=distance)
            )
        return results

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if limit <= 0:
            return []
        return await asyncio.to_thread(
            self._search_sync, query_embedding, limit, threshold, metadata_filter
        )

    def _hybrid_sync(
        self,
        query_text: str,
        query_embedding: list[float],
        limit: int,
        alpha: float,
        threshold: float | None,
        metadata_filter: dict[str, Any] | None,
    ) -> list[SearchResult]:
        pymilvus = _require_pymilvus()
        client = self._ensure_collection_sync()
        expr = build_filter_expr(metadata_filter)
        dense = pymilvus.AnnSearchRequest(
            data=[[float(x) for x in query_embedding]],
            anns_field="embedding",
            param={"metric_type": self._metric},
            limit=limit,
            expr=expr or None,
        )
        sparse = pymilvus.AnnSearchRequest(
            data=[query_text],
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=limit,
            expr=expr or None,
        )
        response = client.hybrid_search(
            self._collection,
            reqs=[dense, sparse],
            ranker=pymilvus.WeightedRanker(alpha, 1.0 - alpha),
            limit=limit,
            output_fields=_OUTPUT_FIELDS,
        )
        results: list[SearchResult] = []
        for hit in response[0] if response else []:
            # The ranker returns the fused score in [0, 1] under ``distance``.
            score = float(hit["distance"])
            if threshold is not None and score < threshold:
                continue
            entity = dict(hit.get("entity") or {})
            entity.setdefault("id", hit.get("id"))
            results.append(SearchResult(document=self._to_document(entity), score=score))
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
        """Blend dense similarity with BM25 relevance on ``content``.

        Requires ``enable_hybrid=True`` at construction (the collection
        needs its sparse BM25 field). ``alpha=1.0`` is pure vector search,
        ``alpha=0.0`` pure BM25. An empty ``query_text`` has no lexical
        signal and is answered by :meth:`search`.

        The fused score comes from Milvus's ``WeightedRanker``: it keeps the
        ranking of the weighted legs but is not the raw vector similarity,
        even at ``alpha=1.0``, so ``threshold`` applies to the fused score.
        An existing collection created without the BM25 function cannot
        answer the lexical leg; the call is answered by :meth:`search` with
        a warning and :attr:`hybrid_search_available` turns ``False``.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
        if not self._enable_hybrid:
            raise ValueError(
                "hybrid_search needs MilvusVectorStore(enable_hybrid=True); the collection "
                "was created without the BM25 sparse field."
            )
        if limit <= 0:
            return []
        await asyncio.to_thread(self._ensure_collection_sync)
        if not self.hybrid_search_available:
            logger.warning(
                "MilvusVectorStore(enable_hybrid=True) but collection %r has no BM25 sparse "
                "field; answering hybrid_search by vector alone",
                self._collection,
            )
            return await self.search(
                query_embedding, limit=limit, threshold=threshold, metadata_filter=metadata_filter
            )
        if not query_text.strip():
            return await self.search(
                query_embedding, limit=limit, threshold=threshold, metadata_filter=metadata_filter
            )
        return await asyncio.to_thread(
            self._hybrid_sync, query_text, query_embedding, limit, alpha, threshold, metadata_filter
        )

    def _count_sync(self) -> int:
        client = self._ensure_collection_sync()
        rows = client.query(self._collection, filter="", output_fields=["count(*)"])
        if not rows:
            return 0
        return int(rows[0]["count(*)"])

    async def count(self) -> int:
        return await asyncio.to_thread(self._count_sync)

    def _clear_sync(self) -> int:
        client = self._get_client()
        if not client.has_collection(self._collection):
            return 0
        n = self._count_sync()
        client.drop_collection(self._collection)
        self._ensured = False
        self._hybrid_ready = None
        return n

    async def clear(self) -> int:
        return await asyncio.to_thread(self._clear_sync)

    async def close(self) -> None:
        if self._client is not None:
            await asyncio.to_thread(self._client.close)
            self._client = None
            self._ensured = False
            self._hybrid_ready = None
            if _is_lite_uri(self._uri):
                await asyncio.to_thread(_release_lite_server, self._uri)

    def __repr__(self) -> str:
        return f"MilvusVectorStore(uri={self._uri!r}, collection={self._collection!r})"
