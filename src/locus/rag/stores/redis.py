# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Redis vector store.

Backed by ``redis.asyncio`` and the Redis Query Engine (the ``search``
module shipped in Redis Stack and in the Redis 8 bundle images). Each
document is one hash under ``key_prefix``; an ``FT.CREATE`` index over
that prefix holds an HNSW ``VECTOR`` field for the embedding and ``TAG``
fields for the metadata keys named in ``metadata_fields``.

Metadata filtering: keys listed in ``metadata_fields`` are filtered
server-side in the ``FT.SEARCH`` query. Any other key is filtered
client-side after an over-fetch, so the store answers every filter and
operators can promote hot keys into the index.

Usage::

    from locus.rag.stores import RedisVectorStore

    store = RedisVectorStore(url="redis://localhost:6379", dimension=1024)
    store = RedisVectorStore(
        url="redis://localhost:6379",
        dimension=1024,
        metadata_fields=["tenant", "channel"],
    )
"""

from __future__ import annotations

import json
import struct
from datetime import UTC, datetime
from typing import Any

from locus.core.loop_bound import loop_bound
from locus.rag.stores.base import (
    BaseVectorStore,
    Document,
    SearchResult,
    VectorStoreConfig,
)


_MISSING_REDIS = 'redis is not installed. Install with: pip install "locus-sdk[redis-vector]"'

_METRIC_MAP = {
    "cosine": "COSINE",
    "l2": "L2",
    "euclidean": "L2",
    "dot_product": "IP",
    "dot": "IP",
}

_TAG_ESCAPE = set(",.<>{}[]\"':;!@#$%^&*()-+=~ /\\|")
_RETURN_FIELDS = ["doc_id", "content", "content_type", "metadata_json", "created_at", "embedding"]


def escape_tag(value: Any) -> str:
    """Escape a TAG value for an ``FT.SEARCH`` query (``@field:{value}``)."""
    text = json.dumps(value) if isinstance(value, bool) else str(value)
    return "".join(f"\\{ch}" if ch in _TAG_ESCAPE else ch for ch in text)


def pack_vector(values: list[float]) -> bytes:
    """Little-endian FLOAT32 bytes, the layout the ``VECTOR`` field expects."""
    return struct.pack(f"<{len(values)}f", *[float(x) for x in values])


def unpack_vector(blob: bytes) -> list[float]:
    """Inverse of :func:`pack_vector`."""
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob[: n * 4]))


class RedisVectorStore(BaseVectorStore):
    """Vector store backed by the Redis Query Engine (``FT.*``).

    The ``TAG`` fields for ``metadata_fields`` are declared ``CASESENSITIVE``
    so a server-side filter is the same exact match as every other store's
    ``metadata_filter``; an empty filter value on a declared key is rejected
    with ``ValueError`` because the query syntax has no form for it. The
    schema is fixed when the index is created: a key added to
    ``metadata_fields`` afterwards is not added to an existing index, so its
    server-side filter fails until the index is altered
    (``FT.ALTER <index> SCHEMA ADD meta_<key> TAG CASESENSITIVE``) or dropped
    and recreated.

    Args:
        dimension: Embedding dimension; fixed in the index schema.
        index_name: ``FT.CREATE`` index name. Defaults to ``"locus"``.
        distance_metric: ``cosine`` (default), ``l2``/``euclidean``, or
            ``dot_product``.
        url: Redis URL. Defaults to ``redis://localhost:6379``.
        key_prefix: Hash key prefix. Defaults to ``"<index_name>:doc:"``.
        metadata_fields: Metadata keys indexed as ``TAG`` fields for
            server-side filtering. Other keys still filter, client-side.
            Fixed when the index is created.
        algorithm: ``"HNSW"`` (default) or ``"FLAT"``.
        _client: Injection seam for tests — a ``redis.asyncio.Redis``-shaped
            object returning raw bytes.
    """

    def __init__(
        self,
        dimension: int = 1024,
        index_name: str = "locus",
        distance_metric: str = "cosine",
        *,
        url: str = "redis://localhost:6379",
        key_prefix: str | None = None,
        metadata_fields: list[str] | None = None,
        algorithm: str = "HNSW",
        _client: Any = None,
    ) -> None:
        if distance_metric not in _METRIC_MAP:
            raise ValueError(
                f"Unsupported distance_metric {distance_metric!r}; "
                f"choose one of {sorted(_METRIC_MAP)}"
            )
        if algorithm.upper() not in ("HNSW", "FLAT"):
            raise ValueError(f"algorithm must be 'HNSW' or 'FLAT', got {algorithm!r}")
        self._dimension = dimension
        self._index = index_name
        self._distance_metric = distance_metric
        self._metric = _METRIC_MAP[distance_metric]
        self._url = url
        self._prefix = key_prefix if key_prefix is not None else f"{index_name}:doc:"
        self._metadata_fields = list(metadata_fields or [])
        self._algorithm = algorithm.upper()
        self._client: Any = _client
        self._ensured = False

    @property
    def config(self) -> VectorStoreConfig:
        return VectorStoreConfig(
            dimension=self._dimension,
            distance_metric=self._distance_metric,
            index_type=self._algorithm.lower(),
        )

    # ----------------------------------------------------------------- client

    def _get_client(self) -> Any:
        def build() -> Any:
            try:
                import redis.asyncio as redis_asyncio  # noqa: PLC0415
            except ImportError as e:
                raise ImportError(_MISSING_REDIS) from e
            return redis_asyncio.from_url(self._url, decode_responses=False)

        return loop_bound(self, "_client", build)

    def _key(self, doc_id: str) -> str:
        return f"{self._prefix}{doc_id}"

    def schema_args(self) -> list[str]:
        """``FT.CREATE`` arguments for the index (public so operators can pre-create it)."""
        args = [
            "FT.CREATE",
            self._index,
            "ON",
            "HASH",
            "PREFIX",
            "1",
            self._prefix,
            "SCHEMA",
            "doc_id",
            "TAG",
            "content",
            "TEXT",
            "content_type",
            "TAG",
        ]
        for field in self._metadata_fields:
            args += [f"meta_{field}", "TAG", "CASESENSITIVE"]
        args += [
            "embedding",
            "VECTOR",
            self._algorithm,
            "6",
            "TYPE",
            "FLOAT32",
            "DIM",
            str(self._dimension),
            "DISTANCE_METRIC",
            self._metric,
        ]
        return args

    async def _ensure_index(self) -> Any:
        client = self._get_client()
        if self._ensured:
            return client
        try:
            await client.execute_command("FT.INFO", self._index)
        except Exception as exc:
            if "unknown index" not in str(exc).lower() and "no such index" not in str(exc).lower():
                raise
            await client.execute_command(*self.schema_args())
        self._ensured = True
        return client

    # ------------------------------------------------------------------ rows

    def _mapping(self, doc: Document) -> dict[str, Any]:
        if doc.embedding is None:
            raise ValueError("Document must have an embedding")
        mapping: dict[str, Any] = {
            "doc_id": doc.id,
            "content": doc.content,
            "content_type": doc.content_type,
            "metadata_json": json.dumps(doc.metadata, default=str),
            "created_at": doc.created_at.isoformat(),
            "embedding": pack_vector(doc.embedding),
        }
        for field in self._metadata_fields:
            if field in doc.metadata:
                value = doc.metadata[field]
                mapping[f"meta_{field}"] = (
                    json.dumps(value) if isinstance(value, bool) else str(value)
                )
        return mapping

    @staticmethod
    def _text(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value) if value is not None else ""

    def _to_document(self, fields: dict[Any, Any]) -> Document:
        data = {self._text(k): v for k, v in fields.items()}
        try:
            metadata = json.loads(self._text(data.get("metadata_json")) or "{}")
        except ValueError:
            metadata = {}
        created = self._text(data.get("created_at"))
        try:
            created_at = datetime.fromisoformat(created) if created else datetime.now(UTC)
        except ValueError:
            created_at = datetime.now(UTC)
        blob = data.get("embedding")
        return Document(
            id=self._text(data.get("doc_id")),
            content=self._text(data.get("content")),
            embedding=unpack_vector(blob) if isinstance(blob, bytes) and blob else None,
            metadata=metadata if isinstance(metadata, dict) else {},
            created_at=created_at,
            content_type=self._text(data.get("content_type")) or "text",
        )

    def _score(self, raw: float) -> tuple[float, float]:
        """Map the ``VECTOR`` distance to ``(score, distance)``.

        COSINE returns ``1 - cos``, IP returns ``1 - dot``, L2 the squared
        Euclidean distance.
        """
        if self._metric in ("COSINE", "IP"):
            return 1.0 - raw, raw
        return 1.0 / (1.0 + raw), raw

    def _split_filter(self, metadata_filter: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
        """Server-side TAG clauses for indexed keys; the rest for client-side filtering."""
        if not metadata_filter:
            return "*", {}
        clauses: list[str] = []
        remaining: dict[str, Any] = {}
        for key, value in metadata_filter.items():
            if key in self._metadata_fields:
                tag = escape_tag(value)
                if not tag:
                    raise ValueError(
                        f"metadata_filter[{key!r}] is empty; a TAG filter needs a value"
                    )
                clauses.append(f"@meta_{key}:{{{tag}}}")
            else:
                remaining[key] = value
        return (" ".join(clauses) if clauses else "*"), remaining

    @staticmethod
    def _lookup(mapping: dict[Any, Any], name: str) -> Any:
        return mapping.get(name, mapping.get(name.encode()))

    @classmethod
    def _parse_search(cls, reply: Any) -> list[dict[Any, Any]]:
        """Per-hit field dicts from either ``FT.SEARCH`` reply shape.

        Older redis-py hands back the raw ``[total, key, [f, v, ...], ...]``
        array; redis-py 5+ parses the command into
        ``{"total_results": n, "results": [{"id": ..., "extra_attributes": {...}}]}``.
        """
        if isinstance(reply, dict):
            return [
                dict(cls._lookup(hit, "extra_attributes") or {})
                for hit in cls._lookup(reply, "results") or []
            ]
        rows: list[dict[Any, Any]] = []
        for i in range(1, len(reply), 2):
            fields = reply[i + 1] if i + 1 < len(reply) else []
            rows.append({fields[j]: fields[j + 1] for j in range(0, len(fields) - 1, 2)})
        return rows

    @classmethod
    def _total(cls, reply: Any) -> int:
        if isinstance(reply, dict):
            return int(cls._lookup(reply, "total_results") or 0)
        return int(reply[0]) if reply else 0

    # ------------------------------------------------------------------- API

    async def add(self, document: Document) -> str:
        ids = await self.add_batch([document])
        return ids[0]

    async def add_batch(self, documents: list[Document]) -> list[str]:
        if not documents:
            return []
        client = await self._ensure_index()
        pipe = client.pipeline(transaction=False)
        for doc in documents:
            mapping = self._mapping(doc)
            pipe.delete(self._key(doc.id))
            pipe.hset(self._key(doc.id), mapping=mapping)
        await pipe.execute()
        return [doc.id for doc in documents]

    async def get(self, doc_id: str) -> Document | None:
        client = await self._ensure_index()
        fields = await client.hgetall(self._key(doc_id))
        if not fields:
            return None
        return self._to_document(fields)

    async def delete(self, doc_id: str) -> bool:
        client = await self._ensure_index()
        removed = await client.delete(self._key(doc_id))
        return int(removed) > 0

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        client = await self._ensure_index()
        prefilter, post = self._split_filter(metadata_filter)
        k = limit * 4 if post else limit
        query = f"({prefilter})=>[KNN {k} @embedding $vec AS vector_distance]"
        reply = await client.execute_command(
            "FT.SEARCH",
            self._index,
            query,
            "PARAMS",
            "2",
            "vec",
            pack_vector(query_embedding),
            "SORTBY",
            "vector_distance",
            "ASC",
            "LIMIT",
            "0",
            str(k),
            "RETURN",
            str(len(_RETURN_FIELDS) + 1),
            *_RETURN_FIELDS,
            "vector_distance",
            "DIALECT",
            "2",
        )
        results: list[SearchResult] = []
        for fields in self._parse_search(reply):
            doc = self._to_document(fields)
            if post and any(doc.metadata.get(key) != value for key, value in post.items()):
                continue
            raw = float(self._text(self._lookup(fields, "vector_distance")) or 0.0)
            score, distance = self._score(raw)
            if threshold is not None and score < threshold:
                continue
            results.append(SearchResult(document=doc, score=score, distance=distance))
            if len(results) >= limit:
                break
        return results

    async def count(self) -> int:
        client = await self._ensure_index()
        reply = await client.execute_command(
            "FT.SEARCH", self._index, "*", "LIMIT", "0", "0", "DIALECT", "2"
        )
        return self._total(reply)

    async def clear(self) -> int:
        client = await self._ensure_index()
        removed = 0
        cursor: int | bytes = 0
        while True:
            cursor, keys = await client.scan(cursor=cursor, match=f"{self._prefix}*", count=500)
            if keys:
                removed += int(await client.delete(*keys))
            if int(cursor) == 0:
                break
        return removed

    async def close(self) -> None:
        if self._client is not None:
            closer = getattr(self._client, "aclose", None) or self._client.close
            await closer()
            self._client = None
            self._ensured = False

    def __repr__(self) -> str:
        return f"RedisVectorStore(index={self._index!r}, prefix={self._prefix!r})"
