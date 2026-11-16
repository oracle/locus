# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Approval store in Redis: one key per held call, changed under a lock.

Same connection conventions as
:class:`~locus.memory.backends.redis.RedisBackend`: a ``redis://`` URL, a key
prefix, a database number::

    from locus.control import RedisApprovals

    store = RedisApprovals(
        "redis://cache.internal:6379", prefix="billing:approvals:"
    )

Under ``prefix`` the store keeps ``record:<approval_id>`` (the JSON record),
``digest:<digest>`` (the ids of every record of that call), ``pending`` (the ids
still waiting), and ``lock``. Every submit, decision and consumption holds the
lock for its read-modify-write, so replicas sharing the prefix never both
redeem one approval. Install with ``pip install locus-sdk[redis]``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

from locus.control.approvals_db import (
    DatabaseApprovals,
    record_from_json,
    record_to_json,
    without_password,
)
from locus.core.loop_bound import loop_bound


if TYPE_CHECKING:
    from redis.asyncio import Redis

    from locus.control.approvals import ApprovalAuthority, ApprovalRecord


__all__ = ["RedisApprovals"]


class _RedisTransaction:
    def __init__(self, client: Redis, prefix: str) -> None:
        self._client = client
        self._prefix = prefix

    def _record_key(self, approval_id: str) -> str:
        return f"{self._prefix}record:{approval_id}"

    async def _records(self, ids: list[str]) -> list[ApprovalRecord]:
        if not ids:
            return []
        documents = await self._client.mget([self._record_key(i) for i in ids])
        return [record_from_json(d) for d in documents if d is not None]

    async def get(self, approval_id: str, *, lock: bool = False) -> ApprovalRecord | None:
        del lock  # the unit of work holds the store's lock already
        document = await self._client.get(self._record_key(approval_id))
        return record_from_json(document) if document is not None else None

    async def same_call(self, digest: str, *, lock: bool = False) -> list[ApprovalRecord]:
        del lock
        ids = cast("set[str]", await self._client.smembers(f"{self._prefix}digest:{digest}"))
        return await self._records(sorted(ids))

    async def insert(self, record: ApprovalRecord) -> bool:
        added = await self._client.set(
            self._record_key(record.approval_id), record_to_json(record), nx=True
        )
        if not added:
            return False
        await self._client.sadd(f"{self._prefix}digest:{record.digest}", record.approval_id)
        await self._client.sadd(f"{self._prefix}pending", record.approval_id)
        return True

    async def update(self, record: ApprovalRecord) -> None:
        await self._client.set(self._record_key(record.approval_id), record_to_json(record))
        if record.status != "pending":
            await self._client.srem(f"{self._prefix}pending", record.approval_id)

    async def waiting(self) -> list[ApprovalRecord]:
        ids = cast("set[str]", await self._client.smembers(f"{self._prefix}pending"))
        return await self._records(sorted(ids))


class RedisApprovals(DatabaseApprovals):
    """An approval store in Redis.

    Args:
        url: The ``redis://`` or ``rediss://`` URL.
        prefix: What every key of this store starts with.
        db: The database number.
        lock_timeout: Seconds a unit of work may hold the store's lock before it
            is released anyway, so a process that dies mid-write cannot block
            the store for good.
        authority: Who may decide, as for
            :class:`~locus.control.approvals.InMemoryApprovals`.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        *,
        prefix: str = "locus:approvals:",
        db: int = 0,
        lock_timeout: float = 10.0,
        authority: ApprovalAuthority
        | Callable[[ApprovalRecord], ApprovalAuthority | None]
        | None = None,
    ) -> None:
        super().__init__(authority)
        self.url = url
        self.prefix = prefix
        self.db = db
        self.lock_timeout = lock_timeout
        self._client: Redis | None = None
        self._client_loop: Any = None

    @property
    def spec(self) -> str:
        where = without_password(self.url)
        joiner = "&" if "?" in where else "?"
        return f"{where}{joiner}prefix={self.prefix}"

    def _build_client(self) -> Redis:
        try:
            from redis.asyncio import Redis  # noqa: PLC0415
        except ImportError as error:
            raise ImportError(
                "RedisApprovals needs the 'redis' package: pip install locus-sdk[redis]"
            ) from error
        return Redis.from_url(self.url, db=self.db, decode_responses=True)

    async def _get_client(self) -> Redis:
        # The client lives on the store's own loop; the shared cache keeps a
        # call from another loop correct rather than mysterious.
        return loop_bound(self, "_client", self._build_client)

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[_RedisTransaction]:
        client = await self._get_client()
        async with client.lock(
            f"{self.prefix}lock",
            timeout=self.lock_timeout,
            blocking_timeout=self.lock_timeout,
        ):
            yield _RedisTransaction(client, self.prefix)

    async def _disconnect(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
