# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Chroma vector store.

Backed by ``chromadb``. Defaults to an **ephemeral in-memory** client
(no server, no disk) which makes it a free, offline test target; pass
``persist_directory`` for an on-disk embedded store, or ``host``/``port``
to talk to a Chroma server.

Chroma's embedded client is synchronous, so each call is dispatched to a
threadpool via :func:`asyncio.to_thread` to keep the store awaitable
alongside the async embedder / retriever.

Usage::

    from tulip.rag.stores import ChromaVectorStore

    store = ChromaVectorStore()  # in-memory
    store = ChromaVectorStore(persist_directory="./chroma")  # on-disk
"""

from __future__ import annotations

import asyncio
from typing import Any

from tulip.rag.stores.base import (
    BaseVectorStore,
    Document,
    SearchResult,
    VectorStoreConfig,
)


# Chroma metadata values must be scalars.
_SCALAR = (str, int, float, bool)


class ChromaVectorStore(BaseVectorStore):
    """Vector store backed by Chroma.

    Args:
        dimension: Embedding dimension (advertised in ``config``; Chroma
            itself infers it from the first vector added).
        collection_name: Chroma collection. Defaults to ``"tulip"``.
        distance_metric: ``cosine`` (default), ``l2``, or ``dot_product``
            — mapped to Chroma's ``hnsw:space``.
        persist_directory: On-disk path for a ``PersistentClient``. When
            ``None`` (default) an in-memory ``EphemeralClient`` is used.
        host: Chroma server host (uses ``HttpClient`` when set).
        port: Chroma server port.
        _client: Injection seam for tests — a pre-built Chroma client
            bypasses the lazy import.
    """

    _SPACE_MAP = {
        "cosine": "cosine",
        "l2": "l2",
        "euclidean": "l2",
        "dot_product": "ip",
        "dot": "ip",
    }

    def __init__(
        self,
        dimension: int = 1024,
        collection_name: str = "tulip",
        distance_metric: str = "cosine",
        *,
        persist_directory: str | None = None,
        host: str | None = None,
        port: int = 8000,
        _client: Any = None,
    ) -> None:
        self._dimension = dimension
        self._collection_name = collection_name
        self._distance_metric = distance_metric
        self._persist_directory = persist_directory
        self._host = host
        self._port = port
        self._client_override = _client
        self._client: Any = None
        self._collection: Any = None

    @property
    def config(self) -> VectorStoreConfig:
        return VectorStoreConfig(
            dimension=self._dimension,
            distance_metric=self._distance_metric,
            index_type="hnsw",
        )

    def _get_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        if self._client_override is not None:
            client = self._client_override
        elif self._client is not None:
            client = self._client
        else:
            try:
                import chromadb  # noqa: PLC0415
            except ImportError as e:
                raise ImportError(
                    'chromadb is not installed. Install with: pip install "tulip-agents[chroma]"'
                ) from e
            if self._host is not None:
                client = chromadb.HttpClient(host=self._host, port=self._port)
            elif self._persist_directory is not None:
                client = chromadb.PersistentClient(path=self._persist_directory)
            else:
                client = chromadb.EphemeralClient()
            self._client = client
        space = self._SPACE_MAP.get(self._distance_metric, "cosine")
        self._collection = client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": space},
        )
        return self._collection

    @staticmethod
    def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in metadata.items() if isinstance(v, _SCALAR)}

    def _add_sync(self, documents: list[Document]) -> list[str]:
        collection = self._get_collection()
        ids, embeddings, contents, metadatas = [], [], [], []
        for doc in documents:
            if doc.embedding is None:
                raise ValueError("Document must have an embedding")
            ids.append(doc.id)
            embeddings.append(doc.embedding)
            contents.append(doc.content)
            meta = self._clean_metadata(doc.metadata)
            meta["_content_type"] = doc.content_type
            metadatas.append(meta)
        collection.upsert(ids=ids, embeddings=embeddings, documents=contents, metadatas=metadatas)
        return ids

    async def add(self, document: Document) -> str:
        ids = await self.add_batch([document])
        return ids[0]

    async def add_batch(self, documents: list[Document]) -> list[str]:
        if not documents:
            return []
        return await asyncio.to_thread(self._add_sync, documents)

    def _to_document(self, doc_id: str, content: str, embedding: Any, meta: dict) -> Document:
        meta = dict(meta or {})
        content_type = meta.pop("_content_type", "text")
        return Document(
            id=doc_id,
            content=content,
            embedding=list(embedding) if embedding is not None else None,
            metadata=meta,
            content_type=content_type,
        )

    def _get_sync(self, doc_id: str) -> Document | None:
        collection = self._get_collection()
        result = collection.get(ids=[doc_id], include=["embeddings", "documents", "metadatas"])
        if not result["ids"]:
            return None
        embeddings = result.get("embeddings")
        embedding = embeddings[0] if embeddings is not None and len(embeddings) else None
        metadatas = result.get("metadatas") or [{}]
        return self._to_document(
            result["ids"][0], result["documents"][0], embedding, metadatas[0] or {}
        )

    async def get(self, doc_id: str) -> Document | None:
        return await asyncio.to_thread(self._get_sync, doc_id)

    def _delete_sync(self, doc_id: str) -> bool:
        collection = self._get_collection()
        existing = collection.get(ids=[doc_id])
        if not existing["ids"]:
            return False
        collection.delete(ids=[doc_id])
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
        collection = self._get_collection()
        where = dict(metadata_filter) if metadata_filter else None
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        ids = result["ids"][0]
        documents = result["documents"][0]
        distances = result["distances"][0]
        metadatas = result.get("metadatas", [[{}] * len(ids)])[0]
        embeddings = result.get("embeddings")
        emb_row = embeddings[0] if embeddings is not None and len(embeddings) else None

        results: list[SearchResult] = []
        for i, doc_id in enumerate(ids):
            distance = float(distances[i])
            # Chroma cosine/l2 are distances (lower = closer). Map to a
            # similarity-ish score in [0,1] for cosine.
            score = 1.0 - distance if self._distance_metric == "cosine" else 1.0 / (1.0 + distance)
            if threshold is not None and score < threshold:
                continue
            emb = emb_row[i] if emb_row is not None else None
            results.append(
                SearchResult(
                    document=self._to_document(
                        doc_id, documents[i], emb, (metadatas[i] if metadatas else {}) or {}
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

    async def count(self) -> int:
        return await asyncio.to_thread(lambda: self._get_collection().count())

    def _clear_sync(self) -> int:
        collection = self._get_collection()
        n = int(collection.count())
        client = self._client_override or self._client
        client.delete_collection(name=self._collection_name)
        self._collection = None
        return n

    async def clear(self) -> int:
        return await asyncio.to_thread(self._clear_sync)
