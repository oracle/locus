# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Redis checkpoint backend - 100% Pydantic."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field

from locus.memory.backends.checkpoint_index import index_key


if TYPE_CHECKING:
    from redis.asyncio import Redis


class RedisConfig(BaseModel):
    """Configuration for Redis backend."""

    # Reject unknown keys rather than absorbing them. The backends take
    # **kwargs and splat them in here, so a misspelled or wrong-named
    # parameter used to vanish silently — notebook 65 passed namespace=
    # instead of prefix= and every run quietly shared one keyspace.
    model_config = {"extra": "forbid"}

    url: str = "redis://localhost:6379"
    prefix: str = "locus:checkpoint:"
    ttl_seconds: int | None = None  # None = no expiry
    db: int = 0


class RedisBackend(BaseModel):
    """
    Redis checkpoint backend.

    Fast key-value storage with optional TTL for checkpoints.

    Example:
        >>> backend = RedisBackend(url="redis://localhost:6379")
        >>> await backend.save("thread_1", state.model_dump())
        >>> data = await backend.load("thread_1")

    Checkpoint index:
        Through ``redis_checkpointer()`` the thread's checkpoint index is a
        hash at ``<prefix><thread_id>:_checkpoints:hash``, one field per
        checkpoint (``HSET`` / ``HDEL`` are atomic), so two processes saving
        to one thread never lose an entry. An index written by an earlier
        release as one JSON string at ``<prefix><thread_id>:_checkpoints``
        is still read, and moved into the hash on the thread's next save
        or removal.
    """

    config: RedisConfig = Field(default_factory=RedisConfig)
    _client: Redis | None = None
    #: The loop the cached client was built on, or ``None`` when the client
    #: did not come from here — a caller (or a test) that assigns ``_client``
    #: owns its lifecycle, and second-guessing that would discard it.
    _client_loop: asyncio.AbstractEventLoop | None = None

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, url: str = "redis://localhost:6379", **kwargs: Any) -> None:
        config = RedisConfig(url=url, **kwargs)
        super().__init__(config=config)

    async def _get_client(self) -> Redis:
        """Get or create the Redis client for the running event loop.

        ``redis.asyncio`` binds a client's connection pool to the loop that
        created it, so a cached client is only usable inside that one loop.
        Caching it on the instance alone means a second loop inherits a dead
        pool and fails with ``Event loop is closed`` — which is not exotic:
        FastAPI's ``TestClient`` runs each request through its own portal, and
        anything calling ``asyncio.run()`` more than once hits it too. The
        error surfaces from deep inside ``redis.asyncio``, so it reads as a
        Redis outage rather than a lifecycle bug here.

        Keying the cache on the loop leaves the single-loop case — every real
        deployment — exactly as cheap as before, and rebuilds only when the
        loop actually changed.
        """
        loop = asyncio.get_running_loop()
        stale = self._client_loop is not None and self._client_loop is not loop
        if self._client is None or stale:
            try:
                from redis.asyncio import Redis
            except ImportError as e:
                raise ImportError(
                    "RedisBackend requires the 'redis' package. "
                    "Install with: pip install locus[redis]"
                ) from e

            # Any previous client belongs to a loop that is gone. Dropping the
            # reference is all that can safely be done — closing it would need
            # its own loop, which is exactly what is no longer available.
            self._client = Redis.from_url(
                self.config.url,
                db=self.config.db,
                decode_responses=True,
            )
            self._client_loop = loop
        return self._client

    def _key(self, thread_id: str) -> str:
        """Generate Redis key for thread."""
        return f"{self.config.prefix}{thread_id}"

    async def save(self, thread_id: str, data: dict[str, Any]) -> None:
        """Save checkpoint to Redis."""
        client = await self._get_client()
        key = self._key(thread_id)
        value = json.dumps(data)

        if self.config.ttl_seconds:
            await client.setex(key, self.config.ttl_seconds, value)
        else:
            await client.set(key, value)

    async def load(self, thread_id: str) -> dict[str, Any] | None:
        """Load checkpoint from Redis."""
        client = await self._get_client()
        key = self._key(thread_id)
        value = await client.get(key)

        if value is None:
            return None

        data: dict[str, Any] = json.loads(value)
        return data

    async def delete(self, thread_id: str) -> bool:
        """Delete checkpoint from Redis."""
        client = await self._get_client()
        key = self._key(thread_id)
        result: int = await client.delete(key)
        return result > 0

    async def exists(self, thread_id: str) -> bool:
        """Check if checkpoint exists."""
        client = await self._get_client()
        key = self._key(thread_id)
        existing: int = await client.exists(key)
        return existing > 0

    async def list_threads(
        self,
        limit: int = 100,
        offset: int = 0,
        pattern: str = "*",
    ) -> list[str]:
        """List all thread IDs matching pattern."""
        client = await self._get_client()
        full_pattern = f"{self.config.prefix}{pattern}"
        # ``decode_responses=True`` (see _get_client) guarantees str keys at
        # runtime; the redis stub still types ``keys()`` as list[bytes | str].
        keys = cast("list[str]", await client.keys(full_pattern))
        prefix_len = len(self.config.prefix)
        threads = [k[prefix_len:] for k in keys]
        # Apply offset and limit
        return threads[offset : offset + limit]

    # -- checkpoint index (CheckpointIndexStore) -------------------------------

    def _index_hash_key(self, thread_id: str) -> str:
        """The hash holding a thread's checkpoint index: one field per checkpoint."""
        return f"{self._key(index_key(thread_id))}:hash"

    def _legacy_index_key(self, thread_id: str) -> str:
        """The key an earlier release kept the index at, as one JSON string."""
        return self._key(index_key(thread_id))

    @staticmethod
    def _legacy_entries(raw: str | bytes | None) -> list[dict[str, Any]]:
        """The entries of a legacy index string; ``[]`` when there is none."""
        if raw is None:
            return []
        entries = json.loads(raw).get("checkpoints", [])
        return [entry for entry in entries if entry.get("checkpoint_id")]

    async def _migrate_legacy_index(self, client: Redis, thread_id: str) -> None:
        """Move a legacy index string into the hash, once.

        Each field is set only where the hash has none (``HSETNX``), so an
        entry the hash already carries is never replaced by its older copy,
        and the string is deleted in the same transaction. Two processes
        migrating at once do the same idempotent work.
        """
        raw = await client.get(self._legacy_index_key(thread_id))
        if raw is None:
            return
        pipe = client.pipeline(transaction=True)
        for entry in self._legacy_entries(raw):
            pipe.hsetnx(
                self._index_hash_key(thread_id), str(entry["checkpoint_id"]), json.dumps(entry)
            )
        pipe.delete(self._legacy_index_key(thread_id))
        await pipe.execute()

    async def index_add(self, thread_id: str, entry: dict[str, Any]) -> None:
        """Set the entry's field in the index hash (``HSET``: atomic, replaces a duplicate).

        The hash is shared by every checkpoint of the thread, so its expiry
        is the longest any writer asked for: a ``ttl`` only ever lengthens
        it (a hash this call creates gets the ttl outright), and a writer
        without a ttl makes it persist. Otherwise a short-lived save after
        a long-lived one would take the index — and the listing of the
        checkpoints still live — down with it.
        """
        client = await self._get_client()
        await self._migrate_legacy_index(client, thread_id)
        key = self._index_hash_key(thread_id)
        pipe = client.pipeline(transaction=True)
        pipe.hset(key, str(entry["checkpoint_id"]), json.dumps(entry))
        if not self.config.ttl_seconds:
            pipe.persist(key)
            await pipe.execute()
            return
        pipe.hlen(key)
        pipe.pttl(key)
        added, length, remaining = await pipe.execute()
        wanted = int(self.config.ttl_seconds) * 1000
        created = int(added) == 1 and int(length) == 1
        if created or 0 <= int(remaining) < wanted:
            await client.pexpire(key, wanted)

    async def index_remove(self, thread_id: str, checkpoint_id: str) -> bool:
        """Delete the entry's field from the index hash (``HDEL``)."""
        client = await self._get_client()
        await self._migrate_legacy_index(client, thread_id)
        removed: int = await client.hdel(self._index_hash_key(thread_id), checkpoint_id)
        return removed > 0

    async def index_list(self, thread_id: str) -> list[dict[str, Any]]:
        """Every entry: the hash's fields, plus a legacy string's not yet migrated."""
        client = await self._get_client()
        pipe = client.pipeline(transaction=False)
        pipe.hgetall(self._index_hash_key(thread_id))
        pipe.get(self._legacy_index_key(thread_id))
        fields, raw = await pipe.execute()
        entries = {str(e["checkpoint_id"]): e for e in self._legacy_entries(raw)}
        for checkpoint_id, value in cast("dict[str, str]", fields).items():
            entries[checkpoint_id] = json.loads(value)
        return list(entries.values())

    async def index_clear(self, thread_id: str) -> bool:
        """Delete the index hash and any legacy string."""
        client = await self._get_client()
        deleted: int = await client.delete(
            self._index_hash_key(thread_id), self._legacy_index_key(thread_id)
        )
        return deleted > 0

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
            self._client_loop = None
