# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""OpenSearch checkpoint backend - 100% Pydantic."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from opensearchpy._async.client import AsyncOpenSearch


class OpenSearchConfig(BaseModel):
    """Configuration for OpenSearch backend."""

    hosts: list[str] = Field(default_factory=lambda: ["localhost:9200"])
    index_name: str = "locus-checkpoints"
    username: str | None = None
    password: str | None = None
    # Secure by default: assume the deployment terminates TLS. Flip to False
    # only for explicit local-dev / docker-compose setups (see
    # examples/docker-compose.yaml).
    use_ssl: bool = True
    verify_certs: bool = True
    ca_certs: str | None = None


class OpenSearchBackend(BaseModel):
    """
    OpenSearch checkpoint backend.

    Scalable document storage with full-text search capabilities.

    Example:
        >>> backend = OpenSearchBackend(hosts=["localhost:9200"])
        >>> await backend.save("thread_1", state.model_dump())
        >>> data = await backend.load("thread_1")
        >>> results = await backend.search("user query")
    """

    config: OpenSearchConfig = Field(default_factory=OpenSearchConfig)
    _client: AsyncOpenSearch | None = None
    _initialized: bool = False

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        hosts: list[str] | None = None,
        index_name: str = "locus-checkpoints",
        username: str | None = None,
        password: str | None = None,
        **kwargs: Any,
    ) -> None:
        config = OpenSearchConfig(
            hosts=hosts or ["localhost:9200"],
            index_name=index_name,
            username=username,
            password=password,
            **kwargs,
        )
        super().__init__(config=config)

    async def _get_client(self) -> AsyncOpenSearch:
        """Get or create OpenSearch client."""
        if self._client is None:
            try:
                from opensearchpy._async.client import AsyncOpenSearch
            except ImportError as e:
                raise ImportError(
                    "OpenSearchBackend requires the 'opensearch-py' package. "
                    "Install with: pip install locus[opensearch]"
                ) from e

            auth = None
            if self.config.username and self.config.password:
                auth = (self.config.username, self.config.password)

            self._client = AsyncOpenSearch(
                hosts=self.config.hosts,
                http_auth=auth,
                use_ssl=self.config.use_ssl,
                verify_certs=self.config.verify_certs,
                ca_certs=self.config.ca_certs,
            )

        return self._client

    async def _ensure_index(self) -> None:
        """Create index if not exists."""
        if self._initialized:
            return

        client = await self._get_client()

        # Check if index exists
        exists = await client.indices.exists(index=self.config.index_name)
        if not exists:
            # Create index with mapping
            await client.indices.create(
                index=self.config.index_name,
                body={
                    "mappings": {
                        "properties": {
                            "thread_id": {"type": "keyword"},
                            "data": {"type": "object", "enabled": False},
                            "data_json": {"type": "text"},
                            "created_at": {"type": "date"},
                            "updated_at": {"type": "date"},
                            "metadata": {"type": "object"},
                        }
                    },
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                    },
                },
            )

        self._initialized = True

    async def save(
        self,
        thread_id: str,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save checkpoint to OpenSearch."""
        await self._ensure_index()
        client = await self._get_client()

        now = datetime.now(UTC).isoformat()

        doc = {
            "thread_id": thread_id,
            "data": data,
            "data_json": json.dumps(data),  # For text search
            "updated_at": now,
            "metadata": metadata or {},
        }

        # Check if exists for created_at
        try:
            existing = await client.get(
                index=self.config.index_name,
                id=thread_id,
            )
            doc["created_at"] = existing["_source"].get("created_at", now)
        except Exception:  # noqa: BLE001 — first-write path; any lookup failure == "no prior"
            doc["created_at"] = now

        await client.index(
            index=self.config.index_name,
            id=thread_id,
            body=doc,
            refresh=True,
        )

    async def load(self, thread_id: str) -> dict[str, Any] | None:
        """Load checkpoint from OpenSearch."""
        await self._ensure_index()
        client = await self._get_client()

        try:
            result = await client.get(
                index=self.config.index_name,
                id=thread_id,
            )
            data: dict[str, Any] = result["_source"]["data"]
            return data
        except Exception:  # noqa: BLE001 — missing document == None by design
            return None

    async def delete(self, thread_id: str) -> bool:
        """Delete checkpoint from OpenSearch."""
        await self._ensure_index()
        client = await self._get_client()

        try:
            await client.delete(
                index=self.config.index_name,
                id=thread_id,
                refresh=True,
            )
            return True
        except Exception:  # noqa: BLE001 — delete is idempotent; report boolean result
            return False

    async def exists(self, thread_id: str) -> bool:
        """Check if checkpoint exists."""
        await self._ensure_index()
        client = await self._get_client()

        present: bool = await client.exists(
            index=self.config.index_name,
            id=thread_id,
        )
        return present

    async def list_threads(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[str]:
        """List all thread IDs."""
        await self._ensure_index()
        client = await self._get_client()

        result = await client.search(
            index=self.config.index_name,
            body={
                "query": {"match_all": {}},
                "size": limit,
                "from": offset,
                "_source": ["thread_id"],
                "sort": [{"updated_at": "desc"}],
            },
        )

        return [hit["_source"]["thread_id"] for hit in result["hits"]["hits"]]

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search checkpoints by content.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching checkpoints with scores
        """
        await self._ensure_index()
        client = await self._get_client()

        result = await client.search(
            index=self.config.index_name,
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["data_json", "thread_id"],
                    }
                },
                "size": limit,
                "_source": ["thread_id", "data", "updated_at"],
            },
        )

        return [
            {
                "thread_id": hit["_source"]["thread_id"],
                "data": hit["_source"]["data"],
                "score": hit["_score"],
                "updated_at": hit["_source"]["updated_at"],
            }
            for hit in result["hits"]["hits"]
        ]

    async def get_by_metadata(
        self,
        key: str,
        value: Any,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get checkpoints by metadata field."""
        await self._ensure_index()
        client = await self._get_client()

        result = await client.search(
            index=self.config.index_name,
            body={
                "query": {"term": {f"metadata.{key}": value}},
                "size": limit,
            },
        )

        return [
            {
                "thread_id": hit["_source"]["thread_id"],
                "data": hit["_source"]["data"],
            }
            for hit in result["hits"]["hits"]
        ]

    async def close(self) -> None:
        """Close OpenSearch connection."""
        if self._client:
            await self._client.close()
            self._client = None
