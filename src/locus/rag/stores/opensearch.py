# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""OpenSearch vector store with k-NN plugin.

Uses OpenSearch's k-NN plugin for efficient vector similarity search.
Supports hybrid search combining vector similarity with BM25 text search.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field

from locus.rag.stores.base import (
    BaseVectorStore,
    Document,
    SearchResult,
    VectorStoreConfig,
)


if TYPE_CHECKING:
    from opensearchpy._async.client import AsyncOpenSearch


class OpenSearchVectorConfig(BaseModel):
    """Configuration for OpenSearch Vector Store."""

    hosts: list[str] = Field(
        default=["localhost:9200"],
        description="OpenSearch hosts",
    )
    http_auth: tuple[str, str] | None = Field(
        default=None,
        description="HTTP auth credentials (username, password)",
    )
    # Secure by default: assume the deployment terminates TLS. Flip to False
    # only for explicit local-dev / docker-compose setups (see
    # examples/docker-compose.yaml).
    use_ssl: bool = Field(default=True, description="Use SSL/TLS")
    verify_certs: bool = Field(default=True, description="Verify SSL certificates")

    index_name: str = Field(default="locus_vectors", description="Index name")
    dimension: int = Field(default=1024, description="Vector dimension")
    distance_metric: str = Field(
        default="cosinesimil",
        description="Distance metric: cosinesimil, l2, innerproduct",
    )

    # k-NN settings
    ef_construction: int = Field(default=256, description="HNSW ef_construction")
    m: int = Field(default=16, description="HNSW M parameter")


class OpenSearchVectorStore(BaseModel, BaseVectorStore):
    """
    OpenSearch vector store with k-NN plugin.

    Uses OpenSearch's k-NN plugin for efficient approximate nearest
    neighbor search. Supports hybrid search combining vectors with
    full-text search.

    Example:
        >>> store = OpenSearchVectorStore(
        ...     hosts=["localhost:9200"],
        ...     index_name="my_vectors",
        ...     dimension=1024,
        ... )
        >>> await store.add(document)
        >>> results = await store.search(query_embedding, limit=5)

    Example with authentication:
        >>> store = OpenSearchVectorStore(
        ...     hosts=["search.example.com:443"],
        ...     http_auth=("admin", "password"),
        ...     use_ssl=True,
        ... )
    """

    os_config: OpenSearchVectorConfig = Field(default_factory=OpenSearchVectorConfig)
    _client: AsyncOpenSearch | None = None
    _initialized: bool = False

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        hosts: list[str] | None = None,
        http_auth: tuple[str, str] | None = None,
        use_ssl: bool = False,
        index_name: str = "locus_vectors",
        dimension: int = 1024,
        distance_metric: str = "cosinesimil",
        **kwargs: Any,
    ) -> None:
        os_config = OpenSearchVectorConfig(
            hosts=hosts or ["localhost:9200"],
            http_auth=http_auth,
            use_ssl=use_ssl,
            index_name=index_name,
            dimension=dimension,
            distance_metric=distance_metric,
            **kwargs,
        )
        super().__init__(os_config=os_config)

    @property
    def config(self) -> VectorStoreConfig:
        """Get store configuration."""
        return VectorStoreConfig(
            dimension=self.os_config.dimension,
            distance_metric=self.os_config.distance_metric,
            index_type="hnsw",
        )

    async def _get_client(self) -> AsyncOpenSearch:
        """Get or create OpenSearch client."""
        if self._client is None:
            try:
                from opensearchpy._async.client import AsyncOpenSearch
            except ImportError as e:
                raise ImportError(
                    "OpenSearchVectorStore requires 'opensearch-py[async]'. "
                    "Install with: pip install opensearch-py aiohttp"
                ) from e

            self._client = AsyncOpenSearch(
                hosts=self.os_config.hosts,
                http_auth=self.os_config.http_auth,
                use_ssl=self.os_config.use_ssl,
                verify_certs=self.os_config.verify_certs,
            )

        return self._client

    async def _ensure_index(self) -> None:
        """Create index if not exists."""
        if self._initialized:
            return

        client = await self._get_client()

        exists = await client.indices.exists(index=self.os_config.index_name)
        if not exists:
            # Create index with k-NN settings
            mappings = {
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 100,
                    }
                },
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "content": {"type": "text"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": self.os_config.dimension,
                            "method": {
                                "name": "hnsw",
                                "space_type": self.os_config.distance_metric,
                                "engine": "lucene",
                                "parameters": {
                                    "ef_construction": self.os_config.ef_construction,
                                    "m": self.os_config.m,
                                },
                            },
                        },
                        "metadata": {"type": "object", "enabled": True},
                        "created_at": {"type": "date"},
                    }
                },
            }

            await client.indices.create(
                index=self.os_config.index_name,
                body=mappings,
            )

        self._initialized = True

    async def add(self, document: Document) -> str:
        """Add a document."""
        await self._ensure_index()
        client = await self._get_client()

        doc_id = document.id or uuid4().hex

        if document.embedding is None:
            raise ValueError("Document must have an embedding")

        body = {
            "id": doc_id,
            "content": document.content,
            "embedding": document.embedding,
            "metadata": document.metadata,
            "created_at": document.created_at.isoformat(),
        }

        await client.index(
            index=self.os_config.index_name,
            id=doc_id,
            body=body,
            refresh=True,
        )

        return doc_id

    async def add_batch(self, documents: list[Document]) -> list[str]:
        """Add multiple documents using bulk API."""
        await self._ensure_index()
        client = await self._get_client()

        # The OpenSearch bulk API alternates control headers and source bodies;
        # both shapes are dicts with disparate value types, so widen to ``Any``.
        actions: list[dict[str, Any]] = []
        ids = []

        for doc in documents:
            doc_id = doc.id or uuid4().hex
            ids.append(doc_id)

            if doc.embedding is None:
                raise ValueError(f"Document {doc_id} must have an embedding")

            actions.append({"index": {"_index": self.os_config.index_name, "_id": doc_id}})
            actions.append(
                {
                    "id": doc_id,
                    "content": doc.content,
                    "embedding": doc.embedding,
                    "metadata": doc.metadata,
                    "created_at": doc.created_at.isoformat(),
                }
            )

        if actions:
            await client.bulk(body=actions, refresh=True)

        return ids

    async def get(self, doc_id: str) -> Document | None:
        """Get a document by ID."""
        await self._ensure_index()
        client = await self._get_client()

        try:
            result = await client.get(
                index=self.os_config.index_name,
                id=doc_id,
            )
        except Exception:  # noqa: BLE001 — vector store lookup/delete; return falsy on any failure
            return None

        source = result["_source"]
        return Document(
            id=source["id"],
            content=source["content"],
            embedding=source.get("embedding"),
            metadata=source.get("metadata", {}),
            created_at=datetime.fromisoformat(source["created_at"])
            if source.get("created_at")
            else datetime.now(UTC),
        )

    async def delete(self, doc_id: str) -> bool:
        """Delete a document."""
        await self._ensure_index()
        client = await self._get_client()

        try:
            result: dict[str, Any] = await client.delete(
                index=self.os_config.index_name,
                id=doc_id,
                refresh=True,
            )
            return result.get("result") == "deleted"
        except Exception:  # noqa: BLE001 — vector store lookup/delete; return falsy on any failure
            return False

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents using k-NN."""
        await self._ensure_index()
        client = await self._get_client()

        # Build k-NN query
        knn_query = {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": limit,
                }
            }
        }

        # Add metadata filter if provided
        query: dict[str, Any]
        if metadata_filter:
            must_clauses: list[dict[str, Any]] = [knn_query]
            for key, value in metadata_filter.items():
                must_clauses.append({"term": {f"metadata.{key}": value}})
            query = {"bool": {"must": must_clauses}}
        else:
            query = knn_query

        result = await client.search(
            index=self.os_config.index_name,
            body={
                "size": limit,
                "query": query,
                "_source": ["id", "content", "embedding", "metadata", "created_at"],
            },
        )

        results = []
        for hit in result["hits"]["hits"]:
            source = hit["_source"]
            score = hit["_score"]

            # Normalize score to 0-1 range
            # OpenSearch k-NN scores depend on distance metric
            if self.os_config.distance_metric == "cosinesimil":
                # Cosine similarity already in 0-1 range (approximately)
                normalized_score = min(1.0, max(0.0, score))
            else:
                # For L2/innerproduct, scores can vary
                normalized_score = 1.0 / (1.0 + (1.0 / max(score, 0.001)))

            if threshold is not None and normalized_score < threshold:
                continue

            doc = Document(
                id=source["id"],
                content=source["content"],
                embedding=source.get("embedding"),
                metadata=source.get("metadata", {}),
                created_at=datetime.fromisoformat(source["created_at"])
                if source.get("created_at")
                else datetime.now(UTC),
            )

            results.append(
                SearchResult(
                    document=doc,
                    score=normalized_score,
                    distance=1.0 / max(score, 0.001) if score > 0 else float("inf"),
                )
            )

        return results

    async def count(self) -> int:
        """Count documents."""
        await self._ensure_index()
        client = await self._get_client()

        result = await client.count(index=self.os_config.index_name)
        n: int = result["count"]
        return n

    async def clear(self) -> int:
        """Delete all documents."""
        await self._ensure_index()
        client = await self._get_client()

        count = await self.count()

        # Delete by query (all documents)
        await client.delete_by_query(
            index=self.os_config.index_name,
            body={"query": {"match_all": {}}},
            refresh=True,
        )

        return count

    async def close(self) -> None:
        """Close the client."""
        if self._client:
            await self._client.close()
            self._client = None

    def __repr__(self) -> str:
        return f"OpenSearchVectorStore(index={self.os_config.index_name!r})"
