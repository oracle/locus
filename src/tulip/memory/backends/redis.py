# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Redis checkpoint backend - 100% Pydantic."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from redis.asyncio import Redis


class RedisConfig(BaseModel):
    """Configuration for Redis backend."""

    url: str = "redis://localhost:6379"
    prefix: str = "tulip:checkpoint:"
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
                    "Install with: pip install tulip[redis]"
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

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
            self._client_loop = None
