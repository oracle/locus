# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Adapters to make storage backends compatible with BaseCheckpointer."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from tulip.core.protocols import CheckpointerCapabilities
from tulip.memory.checkpointer import BaseCheckpointer


if TYPE_CHECKING:
    from tulip.core.state import AgentState


def _callable_has_parameter(fn: Any, parameter_name: str) -> bool:
    """Return whether a callable signature includes a parameter, if introspectable."""
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return parameter_name in sig.parameters


class StorageBackendAdapter(BaseCheckpointer):
    """
    Adapter that wraps simple storage backends to implement BaseCheckpointer.

    Storage backends have a simple interface:
    - save(thread_id: str, data: dict) -> None
    - load(thread_id: str) -> dict | None
    - delete(thread_id: str) -> bool
    - exists(thread_id: str) -> bool
    - list_threads() -> list[str]

    This adapter converts between AgentState and dict representations.

    Key improvement: Checkpoint IDs are now stored IN the backend,
    not in memory. This ensures persistence across restarts.

    Storage schema:
    - `{thread_id}:{checkpoint_id}` -> checkpoint data
    - `{thread_id}:latest` -> latest checkpoint (for quick access)
    - `{thread_id}:_checkpoints` -> list of checkpoint metadata

    Example:
        >>> from tulip.memory.backends import RedisBackend
        >>> from tulip.memory.backends.adapters import StorageBackendAdapter
        >>>
        >>> # Create storage backend
        >>> storage = RedisBackend(url="redis://localhost:6379")
        >>>
        >>> # Wrap with adapter for use with Agent
        >>> checkpointer = StorageBackendAdapter(storage)
        >>>
        >>> # Use with Agent
        >>> agent = Agent(model=model, checkpointer=checkpointer)

    Concurrency:
        The ``{thread}:_checkpoints`` index is updated under a per-thread
        ``asyncio.Lock``, so concurrent saves/removals to the same thread are
        safe *within a single process*. The lock is per adapter instance and
        does NOT serialize separate processes sharing one store, so
        cross-process writes to the same thread can still drop index entries.
        Making that case safe requires backend-native atomic index updates;
        tracked in issue #301.
    """

    def __init__(self, backend: Any) -> None:
        """
        Initialize adapter with a storage backend.

        Args:
            backend: Storage backend with save/load/delete/exists methods
        """
        self._backend = backend
        self._capabilities_cache: CheckpointerCapabilities | None = None
        # Per-thread locks serializing the read-modify-write of the checkpoint
        # index (`{thread}:_checkpoints`). Without this, concurrent saves to the
        # same thread interleave load/save and silently drop index entries.
        self._index_locks: dict[str, asyncio.Lock] = {}

    def _index_lock(self, thread_id: str) -> asyncio.Lock:
        """Get (or lazily create) the index lock for a thread."""
        lock = self._index_locks.get(thread_id)
        if lock is None:
            lock = asyncio.Lock()
            self._index_locks[thread_id] = lock
        return lock

    @property
    def capabilities(self) -> CheckpointerCapabilities:
        """Derive capabilities from backend methods."""
        if self._capabilities_cache is not None:
            return self._capabilities_cache

        self._capabilities_cache = CheckpointerCapabilities(
            search=hasattr(self._backend, "search"),
            metadata_query=(
                hasattr(self._backend, "query_by_metadata")
                or hasattr(self._backend, "get_by_metadata")
                or hasattr(self._backend, "get_metadata")
            ),
            vacuum=hasattr(self._backend, "vacuum"),
            branching=hasattr(self._backend, "copy_thread"),
            ttl=(
                hasattr(self._backend, "config")
                and hasattr(getattr(self._backend, "config", None), "ttl_seconds")
            ),
            list_threads=hasattr(self._backend, "list_threads"),
            list_with_metadata=hasattr(self._backend, "list_with_metadata"),
            persistent_checkpoint_ids=True,  # Now stored in backend!
        )
        return self._capabilities_cache

    async def save(
        self,
        state: AgentState,
        thread_id: str,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save agent state with persistent checkpoint ID tracking."""
        checkpoint_id = checkpoint_id or uuid4().hex
        now = datetime.now(UTC)

        # Create storage key
        storage_key = f"{thread_id}:{checkpoint_id}"

        # Convert state to dict and save
        data = state.to_checkpoint()
        data["_checkpoint_id"] = checkpoint_id
        data["_checkpoint_timestamp"] = now.isoformat()
        data["_metadata"] = metadata or {}

        # Save the checkpoint (some backends support metadata parameter)
        save_method = self._backend.save
        if _callable_has_parameter(save_method, "metadata"):
            await save_method(storage_key, data, metadata=metadata)
        else:
            await save_method(storage_key, data)

        # Also save as "latest" for easy retrieval
        await self._backend.save(f"{thread_id}:latest", data)

        # Update checkpoint index (persistent checkpoint ID list)
        await self._update_checkpoint_index(thread_id, checkpoint_id, now, metadata)

        return checkpoint_id

    async def _update_checkpoint_index(
        self,
        thread_id: str,
        checkpoint_id: str,
        timestamp: datetime,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update the persistent checkpoint index."""
        index_key = f"{thread_id}:_checkpoints"

        # Serialize the read-modify-write so concurrent saves to the same
        # thread cannot clobber each other's index entries.
        async with self._index_lock(thread_id):
            # Load existing index
            existing = await self._backend.load(index_key)
            if existing is None:
                existing = {"checkpoints": []}

            # Remove duplicate if exists (update case)
            existing["checkpoints"] = [
                cp
                for cp in existing.get("checkpoints", [])
                if cp.get("checkpoint_id") != checkpoint_id
            ]

            # Add new/updated checkpoint
            existing["checkpoints"].append(
                {
                    "checkpoint_id": checkpoint_id,
                    "timestamp": timestamp.isoformat(),
                    "metadata": metadata or {},
                }
            )

            # Sort by timestamp (newest first)
            existing["checkpoints"].sort(
                key=lambda x: x.get("timestamp", ""),
                reverse=True,
            )

            # Save updated index
            await self._backend.save(index_key, existing)

    async def load(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> AgentState | None:
        """Load agent state from the storage backend."""
        from tulip.core.state import AgentState

        # Determine storage key
        if checkpoint_id:
            storage_key = f"{thread_id}:{checkpoint_id}"
        else:
            storage_key = f"{thread_id}:latest"

        # Load data
        data = await self._backend.load(storage_key)
        if data is None:
            return None

        # Remove adapter metadata before restoring
        data.pop("_checkpoint_id", None)
        data.pop("_checkpoint_timestamp", None)
        data.pop("_metadata", None)

        return AgentState.from_checkpoint(data)

    async def list_checkpoints(
        self,
        thread_id: str,
        limit: int = 10,
    ) -> list[str]:
        """List available checkpoints from persistent index."""
        index_key = f"{thread_id}:_checkpoints"

        existing = await self._backend.load(index_key)
        if existing is None:
            return []

        checkpoints = existing.get("checkpoints", [])
        return [cp.get("checkpoint_id") for cp in checkpoints[:limit] if cp.get("checkpoint_id")]

    async def delete(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> bool:
        """Delete checkpoint(s) with index update."""
        if checkpoint_id:
            # Delete specific checkpoint
            storage_key = f"{thread_id}:{checkpoint_id}"
            result: bool = await self._backend.delete(storage_key)

            # Update index
            await self._remove_from_index(thread_id, checkpoint_id)

            return result
        else:
            # Delete all checkpoints for thread
            deleted = False

            # Get all checkpoint IDs from index
            checkpoints = await self.list_checkpoints(thread_id, limit=1000)

            # Delete each checkpoint
            for cp_id in checkpoints:
                if await self._backend.delete(f"{thread_id}:{cp_id}"):
                    deleted = True

            # Delete latest pointer
            if await self._backend.exists(f"{thread_id}:latest"):
                await self._backend.delete(f"{thread_id}:latest")
                deleted = True

            # Delete index
            if await self._backend.exists(f"{thread_id}:_checkpoints"):
                await self._backend.delete(f"{thread_id}:_checkpoints")
                deleted = True

            return deleted

    async def _remove_from_index(
        self,
        thread_id: str,
        checkpoint_id: str,
    ) -> None:
        """Remove checkpoint from index."""
        index_key = f"{thread_id}:_checkpoints"

        async with self._index_lock(thread_id):
            existing = await self._backend.load(index_key)
            if existing is None:
                return

            existing["checkpoints"] = [
                cp
                for cp in existing.get("checkpoints", [])
                if cp.get("checkpoint_id") != checkpoint_id
            ]

            await self._backend.save(index_key, existing)

    async def exists(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> bool:
        """Check if checkpoint exists."""
        if checkpoint_id:
            storage_key = f"{thread_id}:{checkpoint_id}"
        else:
            storage_key = f"{thread_id}:latest"

        present: bool = await self._backend.exists(storage_key)
        return present

    # =========================================================================
    # Extended Methods - Delegate to Backend
    # =========================================================================

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Delegate to backend search."""
        self._require_capability("search")
        results: list[dict[str, Any]] = await self._backend.search(query, limit=limit)
        return results

    async def query_by_metadata(
        self,
        key: str,
        value: Any,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Delegate to backend metadata query."""
        self._require_capability("metadata_query")
        if hasattr(self._backend, "query_by_metadata"):
            results: list[dict[str, Any]] = await self._backend.query_by_metadata(
                key, value, limit=limit
            )
            return results
        if hasattr(self._backend, "get_by_metadata"):
            via_get: list[dict[str, Any]] = await self._backend.get_by_metadata(
                key, value, limit=limit
            )
            return via_get
        raise NotImplementedError("Backend has no metadata query method")

    async def get_metadata(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Get checkpoint metadata from index or backend."""
        # First try the backend's native method
        if hasattr(self._backend, "get_metadata"):
            storage_key = f"{thread_id}:{checkpoint_id}" if checkpoint_id else f"{thread_id}:latest"
            meta: dict[str, Any] | None = await self._backend.get_metadata(storage_key)
            return meta

        # Fallback to checkpoint index
        index_key = f"{thread_id}:_checkpoints"
        existing = await self._backend.load(index_key)
        if existing is None:
            return None

        checkpoints = existing.get("checkpoints", [])

        if checkpoint_id:
            for cp in checkpoints:
                if cp.get("checkpoint_id") == checkpoint_id:
                    matched: dict[str, Any] = cp
                    return matched
            return None
        # Return latest
        latest: dict[str, Any] | None = checkpoints[0] if checkpoints else None
        return latest

    async def vacuum(
        self,
        older_than_days: int = 30,
    ) -> int:
        """Delegate to backend vacuum."""
        self._require_capability("vacuum")
        deleted: int = await self._backend.vacuum(older_than_days)
        return deleted

    async def copy_thread(
        self,
        source_thread_id: str,
        dest_thread_id: str,
    ) -> bool:
        """Copy all checkpoints from one thread to another (branching)."""
        self._require_capability("branching")

        # Always use manual implementation since adapter uses different key structure
        # ({thread_id}:{checkpoint_id}) than backends expect
        checkpoints = await self.list_checkpoints(source_thread_id, limit=1000)
        if not checkpoints:
            return False

        for cp_id in checkpoints:
            state = await self.load(source_thread_id, cp_id)
            if state:
                meta = await self.get_metadata(source_thread_id, cp_id)
                await self.save(
                    state, dest_thread_id, cp_id, metadata=meta.get("metadata") if meta else None
                )
        return True

    async def list_threads(
        self,
        limit: int = 100,
        pattern: str = "*",
    ) -> list[str]:
        """Delegate to backend list_threads."""
        self._require_capability("list_threads")

        # Backend might have different signature
        if hasattr(self._backend, "list_threads"):
            if _callable_has_parameter(self._backend.list_threads, "pattern"):
                with_pattern: list[str] = await self._backend.list_threads(
                    pattern=pattern, limit=limit
                )
                return with_pattern
            if _callable_has_parameter(self._backend.list_threads, "limit"):
                threads: list[str] = await self._backend.list_threads(limit=limit)
            else:
                threads = await self._backend.list_threads()

            # Apply pattern filter if backend doesn't support it
            if pattern != "*":
                import fnmatch

                threads = [t for t in threads if fnmatch.fnmatch(t, pattern)]

            return threads[:limit]

        raise NotImplementedError("Backend has no list_threads method")

    async def list_with_metadata(
        self,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Delegate to backend list_with_metadata."""
        self._require_capability("list_with_metadata")
        items: list[dict[str, Any]] = await self._backend.list_with_metadata(limit=limit)
        return items

    async def close(self) -> None:
        """Close the underlying backend if it supports it."""
        if hasattr(self._backend, "close"):
            await self._backend.close()

    def __repr__(self) -> str:
        return f"StorageBackendAdapter({self._backend!r})"


# =============================================================================
# Convenience Factory Functions
# =============================================================================


def redis_checkpointer(
    url: str = "redis://localhost:6379",
    prefix: str = "tulip:state:",
    **kwargs: Any,
) -> StorageBackendAdapter:
    """
    Create a Redis-backed checkpointer.

    Args:
        url: Redis URL
        prefix: Key prefix for all checkpoints
        **kwargs: Additional RedisBackend options (ttl_seconds, db)

    Returns:
        StorageBackendAdapter wrapping RedisBackend

    Capabilities:
        - ttl: Yes (via ttl_seconds)
        - list_threads: Yes
        - persistent_checkpoint_ids: Yes

    Example:
        >>> checkpointer = redis_checkpointer("redis://localhost:6379")
        >>> agent = Agent(model=model, checkpointer=checkpointer)
    """
    from tulip.memory.backends.redis import RedisBackend

    backend = RedisBackend(url=url, prefix=prefix, **kwargs)
    return StorageBackendAdapter(backend)


def postgresql_checkpointer(
    host: str = "localhost",
    port: int = 5432,
    database: str = "tulip",
    user: str = "postgres",
    password: str = "",
    dsn: str | None = None,
    **kwargs: Any,
) -> StorageBackendAdapter:
    """
    Create a PostgreSQL-backed checkpointer.

    Args:
        host: PostgreSQL host
        port: PostgreSQL port
        database: Database name
        user: Database user
        password: Database password
        dsn: Connection string (overrides other params)
        **kwargs: Additional PostgreSQLBackend options

    Returns:
        StorageBackendAdapter wrapping PostgreSQLBackend

    Capabilities:
        - search: Yes (via search_data)
        - metadata_query: Yes (via query_by_metadata)
        - vacuum: Yes
        - list_threads: Yes
        - persistent_checkpoint_ids: Yes

    Example:
        >>> checkpointer = postgresql_checkpointer(database="myapp")
        >>> agent = Agent(model=model, checkpointer=checkpointer)
    """
    from tulip.memory.backends.postgresql import PostgreSQLBackend

    backend = PostgreSQLBackend(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        dsn=dsn,
        **kwargs,
    )
    return StorageBackendAdapter(backend)


def mysql_checkpointer(
    host: str = "localhost",
    port: int = 3306,
    database: str = "tulip",
    user: str = "root",
    password: str = "",
    dsn: str | None = None,
    **kwargs: Any,
) -> StorageBackendAdapter:
    """
    Create a MySQL-backed checkpointer.

    Args:
        host: MySQL host
        port: MySQL port
        database: Database name
        user: Database user
        password: Database password
        dsn: Connection string (overrides other params)
        **kwargs: Additional MySQLBackend options

    Returns:
        StorageBackendAdapter wrapping MySQLBackend

    Capabilities:
        - search: Yes (via search_data)
        - metadata_query: Yes (via query_by_metadata)
        - vacuum: Yes
        - list_threads: Yes
        - persistent_checkpoint_ids: Yes

    Example:
        >>> checkpointer = mysql_checkpointer(database="myapp")
        >>> agent = Agent(model=model, checkpointer=checkpointer)
    """
    from tulip.memory.backends.mysql import MySQLBackend

    backend = MySQLBackend(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        dsn=dsn,
        **kwargs,
    )
    return StorageBackendAdapter(backend)


def opensearch_checkpointer(
    hosts: list[str] | None = None,
    index_name: str = "tulip-checkpoints",
    **kwargs: Any,
) -> StorageBackendAdapter:
    """
    Create an OpenSearch-backed checkpointer.

    Args:
        hosts: OpenSearch hosts
        index_name: Index name for checkpoints
        **kwargs: Additional OpenSearchBackend options (username, password, use_ssl)

    Returns:
        StorageBackendAdapter wrapping OpenSearchBackend

    Capabilities:
        - search: Yes (full-text search)
        - metadata_query: Yes (via get_by_metadata)
        - list_threads: Yes
        - persistent_checkpoint_ids: Yes

    Example:
        >>> checkpointer = opensearch_checkpointer(hosts=["localhost:9200"])
        >>> agent = Agent(model=model, checkpointer=checkpointer)
    """
    from tulip.memory.backends.opensearch import OpenSearchBackend

    backend = OpenSearchBackend(hosts=hosts, index_name=index_name, **kwargs)
    return StorageBackendAdapter(backend)


def s3_checkpointer(
    bucket: str,
    prefix: str = "tulip/checkpoints/",
    **kwargs: Any,
) -> BaseCheckpointer:
    """
    Create an S3-compatible object-storage-backed checkpointer.

    Works against AWS S3, MinIO, Cloudflare R2, and any other
    S3-compatible endpoint. ``S3Backend`` is a native ``BaseCheckpointer``;
    this factory is a thin convenience alias for parity with the other
    backend factories.

    Args:
        bucket: Bucket name
        prefix: Object key prefix
        **kwargs: Additional S3Backend options (endpoint_url, region_name,
            aws_access_key_id, aws_secret_access_key)

    Example:
        >>> checkpointer = s3_checkpointer(
        ...     bucket="my-checkpoints",
        ...     endpoint_url="http://localhost:9000",  # MinIO
        ... )
        >>> agent = Agent(config=cfg, checkpointer=checkpointer)
    """
    from tulip.memory.backends.s3 import S3Backend

    return S3Backend(bucket=bucket, prefix=prefix, **kwargs)
