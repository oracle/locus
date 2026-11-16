# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Weaviate vector store.

Backed by the official ``weaviate-client`` (v4). Connects to a local or
self-hosted instance (``host``/``port``), to Weaviate Cloud
(``cluster_url`` + ``api_key``) or, with ``embedded=True``, starts
**embedded Weaviate** — the database binary runs as a child process and
persists under ``persistence_data_path``, so tests and small corpora need
no container and no account. The first embedded start downloads the binary.

Vectors are self-provided (the Locus embedder produces them). Scalar
metadata is written to ``meta_<key>`` properties so ``metadata_filter``
runs server-side with exact ``field`` tokenization; the complete metadata
dict also round-trips through a JSON property.

Hybrid search is native: :meth:`WeaviateVectorStore.hybrid_search` is
Weaviate's ``hybrid`` query (BM25 on ``content`` fused with the vector
leg) and ``alpha`` keeps Weaviate's meaning — ``1.0`` pure vector, ``0.0``
pure keyword.

Usage::

    from locus.rag.stores import WeaviateVectorStore

    store = WeaviateVectorStore(
        embedded=True, persistence_data_path="./weaviate", dimension=1024
    )
    store = WeaviateVectorStore(host="localhost", port=8080, dimension=1024)
    store = WeaviateVectorStore(
        cluster_url="https://...weaviate.cloud", api_key="...", dimension=1024
    )
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime
from typing import Any

from locus.rag.stores.base import (
    BaseVectorStore,
    Document,
    SearchResult,
    VectorStoreConfig,
)


_MISSING_WEAVIATE = (
    'weaviate-client is not installed. Install with: pip install "locus-sdk[weaviate]"'
)

_DISTANCE_MAP = {
    "cosine": "cosine",
    "l2": "l2-squared",
    "euclidean": "l2-squared",
    "dot_product": "dot",
    "dot": "dot",
}

_SCALAR = (str, int, float, bool)
_PROPERTY_RE = re.compile(r"^[_A-Za-z][_0-9A-Za-z]*$")
_RESERVED = {"doc_id", "content", "content_type", "metadata_json", "created_at"}


def _require_weaviate() -> Any:
    try:
        import weaviate  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - depends on install shape
        raise ImportError(_MISSING_WEAVIATE) from e
    return weaviate


def meta_property(key: str) -> str | None:
    """Property name for a metadata key, or ``None`` if it cannot be a property."""
    if not _PROPERTY_RE.match(key):
        return None
    return f"meta_{key}"


class WeaviateVectorStore(BaseVectorStore):
    """Vector store backed by Weaviate.

    Args:
        dimension: Embedding dimension (advertised in ``config``; Weaviate
            infers it from the first vector).
        collection_name: Weaviate collection. Weaviate capitalises the first
            letter, so ``"locus"`` and ``"Locus"`` name the same collection.
        distance_metric: ``cosine`` (default), ``l2``/``euclidean``, or
            ``dot_product``.
        host: Host of a local / self-hosted instance (``connect_to_local``).
        port: HTTP port of that instance. Defaults to ``8080``.
        grpc_port: gRPC port of that instance. Defaults to ``50051``.
        cluster_url: Weaviate Cloud cluster URL (``connect_to_weaviate_cloud``).
        api_key: API key for Weaviate Cloud or an authenticated instance.
        embedded: Start embedded Weaviate instead of connecting.
        persistence_data_path: Data directory for embedded Weaviate.
        embedded_port: HTTP port for embedded Weaviate. Defaults to ``8079``.
        embedded_grpc_port: gRPC port for embedded Weaviate. Defaults to ``50050``.
        _client: Injection seam for tests — a connected ``WeaviateClient``.
    """

    def __init__(
        self,
        dimension: int = 1024,
        collection_name: str = "Locus",
        distance_metric: str = "cosine",
        *,
        host: str | None = None,
        port: int = 8080,
        grpc_port: int = 50051,
        cluster_url: str | None = None,
        api_key: str | None = None,
        embedded: bool = False,
        persistence_data_path: str | None = None,
        embedded_port: int = 8079,
        embedded_grpc_port: int = 50050,
        _client: Any = None,
    ) -> None:
        if _client is None:
            _require_weaviate()
        if distance_metric not in _DISTANCE_MAP:
            raise ValueError(
                f"Unsupported distance_metric {distance_metric!r}; "
                f"choose one of {sorted(_DISTANCE_MAP)}"
            )
        self._dimension = dimension
        self._collection_name = collection_name[:1].upper() + collection_name[1:]
        self._distance_metric = distance_metric
        self._host = host
        self._port = port
        self._grpc_port = grpc_port
        self._cluster_url = cluster_url
        self._api_key = api_key
        self._embedded = embedded
        self._persistence_data_path = persistence_data_path
        self._embedded_port = embedded_port
        self._embedded_grpc_port = embedded_grpc_port
        self._client_override = _client
        self._client: Any = None
        self._collection: Any = None
        self._known_props: dict[str, str] = {}

    @property
    def config(self) -> VectorStoreConfig:
        return VectorStoreConfig(
            dimension=self._dimension,
            distance_metric=self._distance_metric,
            index_type="hnsw",
        )

    # ----------------------------------------------------------------- client

    def _get_client(self) -> Any:
        if self._client_override is not None:
            return self._client_override
        if self._client is not None:
            return self._client
        weaviate = _require_weaviate()
        auth = weaviate.classes.init.Auth.api_key(self._api_key) if self._api_key else None
        if self._embedded:
            kwargs: dict[str, Any] = {
                "port": self._embedded_port,
                "grpc_port": self._embedded_grpc_port,
            }
            if self._persistence_data_path is not None:
                kwargs["persistence_data_path"] = self._persistence_data_path
            self._client = weaviate.connect_to_embedded(**kwargs)
        elif self._cluster_url is not None:
            self._client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self._cluster_url, auth_credentials=auth
            )
        else:
            self._client = weaviate.connect_to_local(
                host=self._host or "localhost",
                port=self._port,
                grpc_port=self._grpc_port,
                auth_credentials=auth,
            )
        return self._client

    def _get_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        weaviate = _require_weaviate()
        wc = weaviate.classes.config
        client = self._get_client()
        if not client.collections.exists(self._collection_name):
            distance = getattr(
                wc.VectorDistances, _DISTANCE_MAP[self._distance_metric].upper().replace("-", "_")
            )
            client.collections.create(
                self._collection_name,
                vector_config=wc.Configure.Vectors.self_provided(
                    vector_index_config=wc.Configure.VectorIndex.hnsw(distance_metric=distance)
                ),
                properties=[
                    wc.Property(
                        name="doc_id",
                        data_type=wc.DataType.TEXT,
                        tokenization=wc.Tokenization.FIELD,
                        index_searchable=False,
                    ),
                    wc.Property(name="content", data_type=wc.DataType.TEXT),
                    wc.Property(
                        name="content_type",
                        data_type=wc.DataType.TEXT,
                        tokenization=wc.Tokenization.FIELD,
                        index_searchable=False,
                    ),
                    wc.Property(
                        name="metadata_json",
                        data_type=wc.DataType.TEXT,
                        index_filterable=False,
                        index_searchable=False,
                    ),
                    wc.Property(
                        name="created_at",
                        data_type=wc.DataType.TEXT,
                        index_filterable=False,
                        index_searchable=False,
                    ),
                ],
            )
        self._collection = client.collections.get(self._collection_name)
        self._known_props = {
            p.name: str(p.data_type) for p in self._collection.config.get().properties
        }
        return self._collection

    def _ensure_meta_property(self, collection: Any, key: str, value: Any) -> str | None:
        """Make sure ``meta_<key>`` exists with the type of ``value``."""
        prop = meta_property(key)
        if prop is None or prop in _RESERVED:
            return None
        if prop in self._known_props:
            return prop
        wc = _require_weaviate().classes.config
        if isinstance(value, bool):
            data_type = wc.DataType.BOOL
        elif isinstance(value, int | float):
            data_type = wc.DataType.NUMBER
        else:
            data_type = wc.DataType.TEXT
        kwargs: dict[str, Any] = {"name": prop, "data_type": data_type, "index_searchable": False}
        if data_type is wc.DataType.TEXT:
            kwargs["tokenization"] = wc.Tokenization.FIELD
        collection.config.add_property(wc.Property(**kwargs))
        self._known_props[prop] = str(data_type)
        return prop

    # ------------------------------------------------------------------ rows

    def _properties(self, collection: Any, doc: Document) -> dict[str, Any]:
        props: dict[str, Any] = {
            "doc_id": doc.id,
            "content": doc.content,
            "content_type": doc.content_type,
            "metadata_json": json.dumps(doc.metadata, default=str),
            "created_at": doc.created_at.isoformat(),
        }
        for key, value in doc.metadata.items():
            if not isinstance(value, _SCALAR):
                continue
            prop = self._ensure_meta_property(collection, key, value)
            if prop is not None:
                props[prop] = (
                    float(value)
                    if isinstance(value, int) and not isinstance(value, bool)
                    else value
                )
        return props

    @staticmethod
    def _to_document(obj: Any) -> Document:
        props = obj.properties or {}
        try:
            metadata = json.loads(props.get("metadata_json") or "{}")
        except (TypeError, ValueError):
            metadata = {}
        created = props.get("created_at")
        try:
            created_at = datetime.fromisoformat(created) if created else datetime.now(UTC)
        except ValueError:
            created_at = datetime.now(UTC)
        vector = obj.vector.get("default") if isinstance(obj.vector, dict) else obj.vector
        return Document(
            id=str(props.get("doc_id", "")),
            content=props.get("content", "") or "",
            embedding=[float(x) for x in vector] if vector else None,
            metadata=metadata if isinstance(metadata, dict) else {},
            created_at=created_at,
            content_type=props.get("content_type", "text") or "text",
        )

    @staticmethod
    def _uuid(doc_id: str) -> str:
        from weaviate.util import generate_uuid5  # noqa: PLC0415

        return str(generate_uuid5(doc_id))

    def _build_filter(self, metadata_filter: dict[str, Any] | None) -> Any:
        if not metadata_filter:
            return None
        wq = _require_weaviate().classes.query
        combined: Any = None
        for key, value in metadata_filter.items():
            prop = meta_property(key)
            if prop is None:
                raise ValueError(
                    f"metadata_filter key {key!r} is not a valid Weaviate property name"
                )
            literal = (
                float(value) if isinstance(value, int) and not isinstance(value, bool) else value
            )
            clause = wq.Filter.by_property(prop).equal(literal)
            combined = clause if combined is None else combined & clause
        return combined

    def _score(self, distance: float) -> float:
        if self._distance_metric == "cosine":
            return 1.0 - distance
        if self._distance_metric in ("dot", "dot_product"):
            return -distance
        return 1.0 / (1.0 + distance)

    # ------------------------------------------------------------------- API

    def _add_sync(self, documents: list[Document]) -> list[str]:
        collection = self._get_collection()
        ids: list[str] = []
        with collection.batch.fixed_size(batch_size=100) as batch:
            for doc in documents:
                if doc.embedding is None:
                    raise ValueError("Document must have an embedding")
                batch.add_object(
                    properties=self._properties(collection, doc),
                    vector=[float(x) for x in doc.embedding],
                    uuid=self._uuid(doc.id),
                )
                ids.append(doc.id)
        failed = collection.batch.failed_objects
        if failed:
            raise RuntimeError(f"Weaviate rejected {len(failed)} object(s): {failed[0].message}")
        return ids

    async def add(self, document: Document) -> str:
        ids = await self.add_batch([document])
        return ids[0]

    async def add_batch(self, documents: list[Document]) -> list[str]:
        if not documents:
            return []
        return await asyncio.to_thread(self._add_sync, documents)

    def _get_sync(self, doc_id: str) -> Document | None:
        collection = self._get_collection()
        obj = collection.query.fetch_object_by_id(self._uuid(doc_id), include_vector=True)
        if obj is None:
            return None
        return self._to_document(obj)

    async def get(self, doc_id: str) -> Document | None:
        return await asyncio.to_thread(self._get_sync, doc_id)

    def _delete_sync(self, doc_id: str) -> bool:
        collection = self._get_collection()
        uuid = self._uuid(doc_id)
        if collection.query.fetch_object_by_id(uuid) is None:
            return False
        collection.data.delete_by_id(uuid)
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
        wq = _require_weaviate().classes.query
        collection = self._get_collection()
        response = collection.query.near_vector(
            near_vector=[float(x) for x in query_embedding],
            limit=limit,
            filters=self._build_filter(metadata_filter),
            return_metadata=wq.MetadataQuery(distance=True),
            include_vector=True,
        )
        results: list[SearchResult] = []
        for obj in response.objects:
            distance = float(obj.metadata.distance or 0.0)
            score = self._score(distance)
            if threshold is not None and score < threshold:
                continue
            results.append(
                SearchResult(document=self._to_document(obj), score=score, distance=distance)
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

    def _hybrid_sync(
        self,
        query_text: str,
        query_embedding: list[float],
        limit: int,
        alpha: float,
        threshold: float | None,
        metadata_filter: dict[str, Any] | None,
    ) -> list[SearchResult]:
        wq = _require_weaviate().classes.query
        collection = self._get_collection()
        response = collection.query.hybrid(
            query=query_text,
            vector=[float(x) for x in query_embedding],
            alpha=alpha,
            limit=limit,
            filters=self._build_filter(metadata_filter),
            query_properties=["content"],
            return_metadata=wq.MetadataQuery(score=True),
            include_vector=True,
        )
        results: list[SearchResult] = []
        for obj in response.objects:
            score = float(obj.metadata.score or 0.0)
            if threshold is not None and score < threshold:
                continue
            results.append(SearchResult(document=self._to_document(obj), score=score))
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
        """Weaviate's native hybrid query: BM25 on ``content`` fused with the vector leg.

        ``alpha=1.0`` is pure vector search, ``alpha=0.0`` pure keyword; the
        fused score is in ``[0, 1]``. An empty ``query_text`` is answered by
        :meth:`search`.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
        if not query_text.strip():
            return await self.search(
                query_embedding, limit=limit, threshold=threshold, metadata_filter=metadata_filter
            )
        return await asyncio.to_thread(
            self._hybrid_sync, query_text, query_embedding, limit, alpha, threshold, metadata_filter
        )

    def _count_sync(self) -> int:
        collection = self._get_collection()
        result = collection.aggregate.over_all(total_count=True)
        return int(result.total_count or 0)

    async def count(self) -> int:
        return await asyncio.to_thread(self._count_sync)

    def _clear_sync(self) -> int:
        client = self._get_client()
        if not client.collections.exists(self._collection_name):
            return 0
        n = self._count_sync()
        client.collections.delete(self._collection_name)
        self._collection = None
        self._known_props = {}
        return n

    async def clear(self) -> int:
        return await asyncio.to_thread(self._clear_sync)

    async def close(self) -> None:
        if self._client is not None:
            await asyncio.to_thread(self._client.close)
            self._client = None
            self._collection = None
            self._known_props = {}

    def __repr__(self) -> str:
        target = "embedded" if self._embedded else (self._cluster_url or self._host or "localhost")
        return f"WeaviateVectorStore(target={target!r}, collection={self._collection_name!r})"
