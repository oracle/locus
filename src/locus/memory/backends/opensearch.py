# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""OpenSearch checkpoint backend - 100% Pydantic."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from locus.memory.backends.checkpoint_index import index_key


if TYPE_CHECKING:
    from opensearchpy._async.client import AsyncOpenSearch


#: How many times the service retries a scripted index update that lost a
#: version race with another writer before it fails (``retry_on_conflict``).
_INDEX_UPDATE_RETRIES = 25

#: Painless: replace the entry with the same id, append the new one, newest first.
_INDEX_ADD_SCRIPT = """
if (ctx._source.data == null) { ctx._source.data = [:]; }
if (!(ctx._source.data.checkpoints instanceof List)) { ctx._source.data.checkpoints = []; }
List entries = ctx._source.data.checkpoints;
entries.removeIf(e -> e.checkpoint_id == params.entry.checkpoint_id);
entries.add(params.entry);
entries.sort((a, b) -> String.valueOf(b.timestamp).compareTo(String.valueOf(a.timestamp)));
ctx._source.updated_at = params.now;
"""

#: Painless: drop the entry with the id; a no-op (``noop`` result) when there is none.
_INDEX_REMOVE_SCRIPT = """
if (ctx._source.data == null || !(ctx._source.data.checkpoints instanceof List)) {
  ctx.op = 'none';
} else if (ctx._source.data.checkpoints.removeIf(e -> e.checkpoint_id == params.checkpoint_id)) {
  ctx._source.updated_at = params.now;
} else {
  ctx.op = 'none';
}
"""


def _is_not_found(exc: BaseException) -> bool:
    """Whether ``exc`` is the client's 404 (``opensearchpy.exceptions.NotFoundError``).

    Told by status code rather than by class so the backend does not import
    the optional client package just to separate "no such document" from a
    version conflict, an authorisation failure or a script error, which the
    caller must see.
    """
    return getattr(exc, "status_code", None) == 404


class OpenSearchConfig(BaseModel):
    """Configuration for OpenSearch backend."""

    # Reject unknown keys rather than absorbing them. The backends take
    # **kwargs and splat them in here, so a misspelled or wrong-named
    # parameter used to vanish silently — notebook 65 passed namespace=
    # instead of prefix= and every run quietly shared one keyspace.
    model_config = {"extra": "forbid"}

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

    Checkpoint index:
        Through ``opensearch_checkpointer()`` the thread's checkpoint index
        is the ``<thread_id>:_checkpoints`` document's ``data``, as it
        always was, but each change is one scripted ``_update``: the
        service applies the script to the document's current version and
        retries it on a version conflict, so two processes saving to one
        thread never lose an entry.
    """

    config: OpenSearchConfig = Field(default_factory=OpenSearchConfig)
    _client: AsyncOpenSearch | None = None
    #: The loop the cached client was built on, or ``None`` when the client
    #: did not come from here — a caller (or a test) that assigns ``_client``
    #: owns its lifecycle, and second-guessing that would discard it.
    _client_loop: asyncio.AbstractEventLoop | None = None
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
        """Get or create the OpenSearch client for the running event loop.

        The underlying aiohttp session binds to the loop that created it, so a
        cached client is only usable inside that one loop; a second loop fails
        with ``Event loop is closed``. Keying the cache on the loop keeps the
        usual single-loop case free and rebuilds only when the loop changed.
        """
        loop = asyncio.get_running_loop()
        stale = self._client_loop is not None and self._client_loop is not loop
        if self._client is None or stale:
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

            self._client_loop = loop
            self._initialized = False
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

    # -- checkpoint index (CheckpointIndexStore) -------------------------------

    async def index_add(self, thread_id: str, entry: dict[str, Any]) -> None:
        """Add the entry to the index document in one scripted update (upserting it)."""
        await self._ensure_index()
        client = await self._get_client()
        now = datetime.now(UTC).isoformat()
        key = index_key(thread_id)

        # ``scripted_upsert``: the script also runs on the upsert body when
        # the document does not exist yet, so the first entry goes through
        # the same path as every later one.
        await client.update(
            index=self.config.index_name,
            id=key,
            body={
                "scripted_upsert": True,
                "script": {
                    "lang": "painless",
                    "source": _INDEX_ADD_SCRIPT,
                    "params": {"entry": entry, "now": now},
                },
                "upsert": {
                    "thread_id": key,
                    "data": {"checkpoints": []},
                    "created_at": now,
                    "updated_at": now,
                    "metadata": {},
                },
            },
            refresh=True,
            retry_on_conflict=_INDEX_UPDATE_RETRIES,
        )

    async def index_remove(self, thread_id: str, checkpoint_id: str) -> bool:
        """Drop the entry from the index document in one scripted update.

        ``False`` when the entry was not there or the thread has no index
        document; any other failure of the update propagates.
        """
        await self._ensure_index()
        client = await self._get_client()

        try:
            result = await client.update(
                index=self.config.index_name,
                id=index_key(thread_id),
                body={
                    "script": {
                        "lang": "painless",
                        "source": _INDEX_REMOVE_SCRIPT,
                        "params": {
                            "checkpoint_id": checkpoint_id,
                            "now": datetime.now(UTC).isoformat(),
                        },
                    },
                },
                refresh=True,
                retry_on_conflict=_INDEX_UPDATE_RETRIES,
            )
        except Exception as exc:
            if _is_not_found(exc):
                return False  # no index document: nothing to remove
            raise
        return bool(result.get("result") == "updated")

    async def index_list(self, thread_id: str) -> list[dict[str, Any]]:
        """The index document's entries; ``[]`` without the document."""
        existing = await self.load(index_key(thread_id))
        return list((existing or {}).get("checkpoints", []))

    async def index_clear(self, thread_id: str) -> bool:
        """Delete the index document."""
        return await self.delete(index_key(thread_id))

    async def close(self) -> None:
        """Close OpenSearch connection."""
        if self._client:
            await self._client.close()
            self._client = None
