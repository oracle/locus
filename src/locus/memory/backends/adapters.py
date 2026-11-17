# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Adapters to make storage backends compatible with BaseCheckpointer."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from locus.core.protocols import CheckpointerCapabilities
from locus.memory.backends.checkpoint_index import (
    CheckpointIndexStore,
    index_key,
    keeps_index_natively,
    newest_first,
)
from locus.memory.checkpointer import BaseCheckpointer
from locus.memory.expiry import EXPIRES_AT_KEY, is_expired, ttl_seconds
from locus.memory.time_travel import (
    CheckpointInfo,
    checkpoint_iteration,
    checkpoint_size,
    info_from,
    link_parents,
)


if TYPE_CHECKING:
    from locus.core.state import AgentState
    from locus.memory.encryption import EncryptedSerializer
    from locus.memory.scope import CheckpointScope

#: The document field the adapter records a checkpoint's expiry under.
_EXPIRES_FIELD = "_expires_at"
#: The key suffix of a thread's ``latest`` copy.
_LATEST_SUFFIX = ":latest"


def _callable_has_parameter(fn: Any, parameter_name: str) -> bool:
    """Return whether a callable signature includes a parameter, if introspectable."""
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return parameter_name in sig.parameters


def _parameter_default(fn: Any, parameter_name: str) -> Any:
    """The default of a callable's parameter, or ``None`` when it has none (or is not introspectable)."""
    import inspect

    try:
        parameter = inspect.signature(fn).parameters[parameter_name]
    except (TypeError, ValueError, KeyError):
        return None
    return None if parameter.default is inspect.Parameter.empty else parameter.default


def _glob_to_like(pattern: str) -> str:
    """A shell-style pattern as a SQL ``LIKE`` pattern (``\\`` escapes)."""
    return (
        pattern.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("*", "%")
        .replace("?", "_")
    )


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
    - `{thread_id}:_checkpoints` -> list of checkpoint metadata: for each
      checkpoint its id, timestamp, metadata, iteration and size, so
      ``list`` reads one key and never a checkpoint's state. A backend
      that keeps the index itself (``CheckpointIndexStore``) may keep it
      elsewhere — Redis in a hash beside that key.

    Every checkpoint is its own key, so the thread's history is complete
    and grows with the square of the conversation's length (each
    checkpoint holds the whole conversation). ``max_checkpoints_per_thread``
    bounds it: the save that makes a checkpoint surplus deletes it. The
    tool-effect ledger's sidecar thread is never trimmed.

    Example:
        >>> from locus.memory.backends import RedisBackend
        >>> from locus.memory.backends.adapters import StorageBackendAdapter
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
        A backend that implements
        :class:`~locus.memory.backends.checkpoint_index.CheckpointIndexStore`
        (Redis, PostgreSQL, MySQL and OpenSearch do) keeps the index in an
        operation its store makes atomic, so concurrent saves and removals
        on one thread are safe across processes sharing the store. For any
        other backend the index is a read-modify-write of the
        ``{thread}:_checkpoints`` key under a per-thread ``asyncio.Lock``:
        safe within one process, not between processes (the lock is per
        adapter instance).

    Production settings:
        ``ttl`` records each checkpoint's expiry in its document
        (``_expires_at``) and its index entry, hands it to a backend whose
        ``save`` takes ``expires_at`` (the SQL backends keep it in a column)
        and, on Redis, sets the key's own ``EXPIRE``; reads skip what has
        expired and :meth:`expire` removes it. ``encryption`` seals the
        state inside each document; the ``_checkpoint_id``,
        ``_checkpoint_timestamp``, ``_metadata`` and ``_expires_at`` fields
        and the index stay plaintext. ``scope`` prefixes every key with the
        scope's prefix.
    """

    def __init__(
        self,
        backend: Any,
        *,
        max_checkpoints_per_thread: int | None = None,
        ttl: float | timedelta | None = None,
        encryption: EncryptedSerializer | None = None,
        scope: CheckpointScope | None = None,
    ) -> None:
        """
        Initialize adapter with a storage backend.

        Args:
            backend: Storage backend with save/load/delete/exists methods
            max_checkpoints_per_thread: Keep only this many of a thread's
                newest checkpoints (``None``, the default, keeps them all).
            ttl: How long a checkpoint stays loadable after its save (seconds
                or a ``timedelta``). Defaults to the backend's own
                ``ttl_seconds`` when it has one (Redis), so the two agree;
                given here, it is also set on such a backend.
            encryption: Seals the state before it is stored
                (:class:`~locus.memory.encryption.EncryptedSerializer`).
            scope: The application and user the threads belong to
                (:class:`~locus.memory.scope.CheckpointScope`).
        """
        if max_checkpoints_per_thread is not None and max_checkpoints_per_thread < 1:
            raise ValueError("max_checkpoints_per_thread must be at least 1")
        self._backend = backend
        self.max_checkpoints_per_thread = max_checkpoints_per_thread
        config = getattr(backend, "config", None)
        native_ttl = getattr(config, "ttl_seconds", None) if config is not None else None
        if not isinstance(native_ttl, int | float) or isinstance(native_ttl, bool):
            native_ttl = None
        if ttl is None and native_ttl:
            ttl = native_ttl
        self._configure(ttl=ttl, encryption=encryption, scope=scope)
        if self.ttl is not None and config is not None and hasattr(config, "ttl_seconds"):
            if not native_ttl:
                # The store expires the keys itself (Redis): the same ttl, in whole seconds.
                config.ttl_seconds = ttl_seconds(self.ttl)
        self._saves_expiry = _callable_has_parameter(getattr(backend, "save", None), "expires_at")
        self._capabilities_cache: CheckpointerCapabilities | None = None
        # Per-thread locks serializing the read-modify-write of the checkpoint
        # index (`{thread}:_checkpoints`). Without this, concurrent saves to the
        # same thread interleave load/save and silently drop index entries.
        self._index_locks: dict[str, asyncio.Lock] = {}
        # The backend itself, when it keeps the index atomically (see
        # ``CheckpointIndexStore``); ``None`` means the locked read-modify-write.
        self._native_index: CheckpointIndexStore | None = (
            backend if keeps_index_natively(backend) else None
        )

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
        key = self._scoped(thread_id)
        expires_at = self._expires_at(now)

        # Create storage key
        storage_key = f"{key}:{checkpoint_id}"

        # Convert state to dict and save
        data = self._encode_state(state, key, checkpoint_id)
        data["_checkpoint_id"] = checkpoint_id
        data["_checkpoint_timestamp"] = now.isoformat()
        data["_metadata"] = metadata or {}
        if expires_at is not None:
            data[_EXPIRES_FIELD] = expires_at.isoformat()

        # Save the checkpoint (some backends support metadata parameter)
        save_method = self._backend.save
        extra: dict[str, Any] = {}
        if self._saves_expiry:
            extra["expires_at"] = expires_at
        if _callable_has_parameter(save_method, "metadata"):
            await save_method(storage_key, data, metadata=metadata, **extra)
        else:
            await save_method(storage_key, data, **extra)

        # Also save as "latest" for easy retrieval
        await self._backend.save(f"{key}:latest", data, **extra)

        # Update checkpoint index (persistent checkpoint ID list)
        await self._update_checkpoint_index(
            key,
            checkpoint_id,
            now,
            metadata,
            iteration=checkpoint_iteration(data),
            size=checkpoint_size(data),
            expires_at=expires_at,
        )
        if self.max_checkpoints_per_thread is not None:
            await self._trim_history(thread_id, self.max_checkpoints_per_thread)

        return checkpoint_id

    async def _update_checkpoint_index(
        self,
        thread_id: str,
        checkpoint_id: str,
        timestamp: datetime,
        metadata: dict[str, Any] | None = None,
        *,
        iteration: int | None = None,
        size: int | None = None,
        expires_at: datetime | None = None,
    ) -> None:
        """Update the persistent checkpoint index (``thread_id`` is the stored, scoped id)."""
        entry = {
            "checkpoint_id": checkpoint_id,
            "timestamp": timestamp.isoformat(),
            "metadata": metadata or {},
            "iteration": iteration,
            "size": size,
        }
        if expires_at is not None:
            entry[EXPIRES_AT_KEY] = expires_at.isoformat()
        if self._native_index is not None:
            # One atomic operation on the store: safe against another
            # process saving to the same thread at the same time.
            await self._native_index.index_add(thread_id, entry)
            return

        # Serialize the read-modify-write so concurrent saves to the same
        # thread cannot clobber each other's index entries.
        async with self._index_lock(thread_id):
            # Load existing index
            existing = await self._backend.load(index_key(thread_id))
            if existing is None:
                existing = {"checkpoints": []}

            # Remove duplicate if exists (update case)
            existing["checkpoints"] = [
                cp
                for cp in existing.get("checkpoints", [])
                if cp.get("checkpoint_id") != checkpoint_id
            ]

            # Add new/updated checkpoint, newest first
            existing["checkpoints"] = newest_first([*existing["checkpoints"], entry])

            # Save updated index
            await self._backend.save(index_key(thread_id), existing)

    async def _index_entries(self, key: str, *, live: bool = True) -> list[dict[str, Any]]:
        """The thread's index entries, newest first, from wherever the backend keeps them.

        ``key`` is the stored (scoped) thread id. With ``live`` (the default)
        the entries whose expiry has passed are left out.
        """
        if self._native_index is not None:
            entries = newest_first(await self._native_index.index_list(key))
        else:
            existing = await self._backend.load(index_key(key))
            entries = newest_first((existing or {}).get("checkpoints", []))
        if not live:
            return entries
        now = datetime.now(UTC)
        return [e for e in entries if not is_expired(e.get(EXPIRES_AT_KEY), now)]

    async def load(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> AgentState | None:
        """Load agent state from the storage backend.

        Without a ``checkpoint_id`` the thread's ``latest`` copy is read; when
        that copy is gone or has expired — it was written with a shorter
        ``ttl`` than an earlier checkpoint, or the store dropped it — the
        newest live checkpoint in the index is loaded instead.
        """
        key = self._scoped(thread_id)
        if checkpoint_id:
            return await self._load_document(key, f"{key}:{checkpoint_id}", checkpoint_id)
        latest = await self._load_document(key, f"{key}:latest", None)
        if latest is not None:
            return latest
        for entry in await self._index_entries(key):
            cid = str(entry.get("checkpoint_id") or "")
            if cid:
                state = await self._load_document(key, f"{key}:{cid}", cid)
                if state is not None:
                    return state
        return None

    async def _load_document(
        self, key: str, storage_key: str, checkpoint_id: str | None
    ) -> AgentState | None:
        """The state a stored document holds, or ``None`` when it is missing or expired."""
        data = await self._backend.load(storage_key)
        if data is None or is_expired(data.get(_EXPIRES_FIELD)):
            return None
        # The ``latest`` copy names the checkpoint it is; the envelope is bound to it.
        stored_id = data.pop("_checkpoint_id", None)
        data.pop("_checkpoint_timestamp", None)
        data.pop("_metadata", None)
        data.pop(_EXPIRES_FIELD, None)
        return self._decode_state(data, key, str(checkpoint_id or stored_id or ""))

    async def list_checkpoints(
        self,
        thread_id: str,
        limit: int = 10,
    ) -> list[str]:
        """List available checkpoints from persistent index."""
        checkpoints = await self._index_entries(self._scoped(thread_id))
        return [str(cp["checkpoint_id"]) for cp in checkpoints[:limit] if cp.get("checkpoint_id")]

    async def delete(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> bool:
        """Delete checkpoint(s) with index update."""
        key = self._scoped(thread_id)
        if checkpoint_id:
            # Delete specific checkpoint
            storage_key = f"{key}:{checkpoint_id}"
            deleted: bool = await self._backend.delete(storage_key)

            # Dropping the index entry is a deletion in its own right: an
            # entry whose checkpoint is already gone leaves list_checkpoints().
            unlisted = await self._remove_from_index(key, checkpoint_id)

            return deleted or unlisted
        else:
            # Delete all checkpoints for thread
            deleted = False

            # Get all checkpoint IDs from index — expired ones included.
            entries = await self._index_entries(key, live=False)

            # Delete each checkpoint
            for entry in entries[:1000]:
                cp_id = entry.get("checkpoint_id")
                if cp_id and await self._backend.delete(f"{key}:{cp_id}"):
                    deleted = True

            # Delete latest pointer
            if await self._backend.exists(f"{key}:latest"):
                await self._backend.delete(f"{key}:latest")
                deleted = True

            # Delete index
            if self._native_index is not None:
                if await self._native_index.index_clear(key):
                    deleted = True
            elif await self._backend.exists(index_key(key)):
                await self._backend.delete(index_key(key))
                deleted = True

            return deleted

    async def _indexed_threads(self) -> list[str]:
        """The stored ids of every thread, from the backend's listing of ``<thread>:latest`` keys.

        Every thread has its ``latest`` copy whatever store keeps the index,
        so that key names the threads; a backend whose ``list_threads`` takes
        no pattern cannot be swept this way and yields none.
        """
        list_threads = getattr(self._backend, "list_threads", None)
        if list_threads is None or not _callable_has_parameter(list_threads, "pattern"):
            return []
        pattern = f"*{_LATEST_SUFFIX}"
        if _parameter_default(list_threads, "pattern") == "%":
            pattern = _glob_to_like(pattern)
        keys: list[str] = await list_threads(pattern=pattern, limit=1_000_000)
        return sorted(k[: -len(_LATEST_SUFFIX)] for k in keys if k.endswith(_LATEST_SUFFIX))

    async def expire(self) -> int:
        """Remove the checkpoints whose ``ttl`` has run out, from every thread and scope.

        Each expired index entry's document is deleted and the entry dropped;
        a ``latest`` copy that has expired goes with them. A backend with an
        ``expire`` of its own (the SQL backends' ``expires_at`` column) sweeps
        its rows as well. Returns the checkpoints removed.
        """
        now = datetime.now(UTC)
        removed = 0
        for key in await self._indexed_threads():
            for entry in await self._index_entries(key, live=False):
                if not is_expired(entry.get(EXPIRES_AT_KEY), now):
                    continue
                checkpoint_id = str(entry.get("checkpoint_id") or "")
                if checkpoint_id:
                    await self._backend.delete(f"{key}:{checkpoint_id}")
                    await self._remove_from_index(key, checkpoint_id)
                removed += 1
            latest = await self._backend.load(f"{key}:latest")
            if latest is not None and is_expired(latest.get(_EXPIRES_FIELD), now):
                await self._backend.delete(f"{key}:latest")
        native_expire = getattr(self._backend, "expire", None)
        if native_expire is not None:
            await native_expire()
        return removed

    async def _remove_from_index(
        self,
        thread_id: str,
        checkpoint_id: str,
    ) -> bool:
        """Remove the checkpoint from the index; ``True`` when the index listed it."""
        if self._native_index is not None:
            return await self._native_index.index_remove(thread_id, checkpoint_id)

        async with self._index_lock(thread_id):
            existing = await self._backend.load(index_key(thread_id))
            if existing is None:
                return False

            entries = existing.get("checkpoints", [])
            kept = [cp for cp in entries if cp.get("checkpoint_id") != checkpoint_id]
            if len(kept) == len(entries):
                return False

            existing["checkpoints"] = kept
            await self._backend.save(index_key(thread_id), existing)
            return True

    async def exists(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> bool:
        """Check if checkpoint exists and has not expired.

        A document that carries an expiry is filtered whether or not this
        instance has a ``ttl``: the writer's expiry is the checkpoint's.
        Without a ``checkpoint_id`` the thread exists when any live
        checkpoint does, the ``latest`` copy's own expiry notwithstanding.
        """
        key = self._scoped(thread_id)
        if checkpoint_id:
            storage_key = f"{key}:{checkpoint_id}"
            if not await self._backend.exists(storage_key):
                return False
            data = await self._backend.load(storage_key)
            return data is not None and not is_expired(data.get(_EXPIRES_FIELD))
        if await self._backend.exists(f"{key}:latest"):
            data = await self._backend.load(f"{key}:latest")
            if data is not None and not is_expired(data.get(_EXPIRES_FIELD)):
                return True
        return bool(await self._index_entries(key))

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
        key = self._scoped(thread_id)
        # First try the backend's native method
        if hasattr(self._backend, "get_metadata"):
            storage_key = f"{key}:{checkpoint_id}" if checkpoint_id else f"{key}:latest"
            meta: dict[str, Any] | None = await self._backend.get_metadata(storage_key)
            return meta

        # Fallback to checkpoint index
        checkpoints = await self._index_entries(key)
        if not checkpoints:
            return None

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
        *,
        scope: CheckpointScope | None = None,
    ) -> list[str]:
        """The backend's keys of one scope, their scope prefix removed.

        A shell-style ``pattern`` is matched by the backend when its
        ``list_threads`` takes one (translated to ``LIKE`` for a backend whose
        pattern defaults to ``%``), else here.
        """
        self._require_capability("list_threads")

        # Backend might have different signature
        if hasattr(self._backend, "list_threads"):
            listing = self._listing_scope(scope)
            list_threads = self._backend.list_threads
            if _callable_has_parameter(list_threads, "pattern"):
                stored_pattern = f"{listing.prefix}{pattern}"
                if _parameter_default(list_threads, "pattern") == "%":
                    stored_pattern = _glob_to_like(stored_pattern)
                with_pattern: list[str] = await list_threads(pattern=stored_pattern, limit=limit)
                return listing.strip(with_pattern)
            if _callable_has_parameter(list_threads, "limit"):
                threads: list[str] = await list_threads(limit=limit)
            else:
                threads = await list_threads()
            threads = listing.strip(threads)

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

    # Last on purpose: a method named ``list`` shadows the builtin for the
    # annotations of anything defined after it.
    async def list(self, thread_id: str, limit: int = 10) -> list[CheckpointInfo]:
        """The thread's checkpoints, newest first, from the index key alone.

        An index entry written before the step fields existed has no
        ``iteration`` or ``size``; they are read from the checkpoint then.
        """
        key = self._scoped(thread_id)
        entries = await self._index_entries(key)
        # The entry just past the window is the parent of the oldest in it.
        older = entries[limit].get("checkpoint_id") if len(entries) > limit else None
        infos: list[CheckpointInfo] = []
        for entry in entries[:limit]:
            checkpoint_id = entry.get("checkpoint_id")
            if not checkpoint_id:
                continue
            iteration, size = entry.get("iteration"), entry.get("size")
            if iteration is None or size is None:
                # The same figures a save records: the document as stored.
                data = await self._backend.load(f"{key}:{checkpoint_id}")
                if data is not None:
                    iteration, size = data.get("iteration"), checkpoint_size(data)
            infos.append(
                info_from(
                    str(checkpoint_id),
                    thread_id,
                    created_at=entry.get("timestamp"),
                    metadata=entry.get("metadata") or {},
                    iteration=iteration,
                    size=size,
                )
            )
        return link_parents(infos, older)


# =============================================================================
# Convenience Factory Functions
# =============================================================================


def redis_checkpointer(
    url: str = "redis://localhost:6379",
    prefix: str = "locus:state:",
    max_checkpoints_per_thread: int | None = None,
    *,
    ttl: float | timedelta | None = None,
    encryption: EncryptedSerializer | None = None,
    scope: CheckpointScope | None = None,
    **kwargs: Any,
) -> StorageBackendAdapter:
    """
    Create a Redis-backed checkpointer.

    Args:
        url: Redis URL
        prefix: Key prefix for all checkpoints
        max_checkpoints_per_thread: Keep only this many of a thread's newest
            checkpoints (``None`` keeps all); see ``StorageBackendAdapter``
        ttl: How long a checkpoint stays loadable (seconds or a ``timedelta``);
            every key the thread writes gets Redis's own ``EXPIRE`` for it
            (whole seconds), the index hash and the tool-effect ledger's
            sidecar included. ``ttl_seconds`` in ``kwargs`` is the same
            setting by its older name.
        encryption: Seals the state (:class:`~locus.memory.encryption.EncryptedSerializer`)
        scope: The application and user the threads belong to
            (:class:`~locus.memory.scope.CheckpointScope`)
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
    from locus.memory.backends.redis import RedisBackend

    backend = RedisBackend(url=url, prefix=prefix, **kwargs)
    return StorageBackendAdapter(
        backend,
        max_checkpoints_per_thread=max_checkpoints_per_thread,
        ttl=ttl,
        encryption=encryption,
        scope=scope,
    )


def postgresql_checkpointer(
    host: str = "localhost",
    port: int = 5432,
    database: str = "locus",
    user: str = "postgres",
    password: str = "",
    dsn: str | None = None,
    max_checkpoints_per_thread: int | None = None,
    *,
    ttl: float | timedelta | None = None,
    encryption: EncryptedSerializer | None = None,
    scope: CheckpointScope | None = None,
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
        max_checkpoints_per_thread: Keep only this many of a thread's newest
            checkpoints (``None`` keeps all); see ``StorageBackendAdapter``
        ttl: How long a checkpoint stays loadable (seconds or a ``timedelta``);
            kept in the table's ``expires_at`` column, swept by ``expire()``
        encryption: Seals the state (:class:`~locus.memory.encryption.EncryptedSerializer`)
        scope: The application and user the threads belong to
            (:class:`~locus.memory.scope.CheckpointScope`)
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
    from locus.memory.backends.postgresql import PostgreSQLBackend

    backend = PostgreSQLBackend(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        dsn=dsn,
        **kwargs,
    )
    return StorageBackendAdapter(
        backend,
        max_checkpoints_per_thread=max_checkpoints_per_thread,
        ttl=ttl,
        encryption=encryption,
        scope=scope,
    )


def mysql_checkpointer(
    host: str = "localhost",
    port: int = 3306,
    database: str = "locus",
    user: str = "root",
    password: str = "",
    dsn: str | None = None,
    max_checkpoints_per_thread: int | None = None,
    *,
    ttl: float | timedelta | None = None,
    encryption: EncryptedSerializer | None = None,
    scope: CheckpointScope | None = None,
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
        max_checkpoints_per_thread: Keep only this many of a thread's newest
            checkpoints (``None`` keeps all); see ``StorageBackendAdapter``
        ttl: How long a checkpoint stays loadable (seconds or a ``timedelta``);
            kept in the table's ``expires_at`` column, swept by ``expire()``
        encryption: Seals the state (:class:`~locus.memory.encryption.EncryptedSerializer`)
        scope: The application and user the threads belong to
            (:class:`~locus.memory.scope.CheckpointScope`)
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
    from locus.memory.backends.mysql import MySQLBackend

    backend = MySQLBackend(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        dsn=dsn,
        **kwargs,
    )
    return StorageBackendAdapter(
        backend,
        max_checkpoints_per_thread=max_checkpoints_per_thread,
        ttl=ttl,
        encryption=encryption,
        scope=scope,
    )


def opensearch_checkpointer(
    hosts: list[str] | None = None,
    index_name: str = "locus-checkpoints",
    max_checkpoints_per_thread: int | None = None,
    *,
    ttl: float | timedelta | None = None,
    encryption: EncryptedSerializer | None = None,
    scope: CheckpointScope | None = None,
    **kwargs: Any,
) -> StorageBackendAdapter:
    """
    Create an OpenSearch-backed checkpointer.

    Args:
        hosts: OpenSearch hosts
        index_name: Index name for checkpoints
        max_checkpoints_per_thread: Keep only this many of a thread's newest
            checkpoints (``None`` keeps all); see ``StorageBackendAdapter``
        ttl: How long a checkpoint stays loadable (seconds or a ``timedelta``);
            recorded in each document and swept by ``expire()``
        encryption: Seals the state (:class:`~locus.memory.encryption.EncryptedSerializer`)
        scope: The application and user the threads belong to
            (:class:`~locus.memory.scope.CheckpointScope`)
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
    from locus.memory.backends.opensearch import OpenSearchBackend

    backend = OpenSearchBackend(hosts=hosts, index_name=index_name, **kwargs)
    return StorageBackendAdapter(
        backend,
        max_checkpoints_per_thread=max_checkpoints_per_thread,
        ttl=ttl,
        encryption=encryption,
        scope=scope,
    )


def s3_checkpointer(
    bucket: str,
    prefix: str = "locus/checkpoints/",
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
    from locus.memory.backends.s3 import S3Backend

    return S3Backend(bucket=bucket, prefix=prefix, **kwargs)


def oci_bucket_checkpointer(
    bucket_name: str,
    namespace: str,
    prefix: str = "locus/checkpoints/",
    **kwargs: Any,
) -> BaseCheckpointer:
    """
    Create an OCI Object Storage-backed checkpointer.

    ``OCIBucketBackend`` is a native ``BaseCheckpointer`` — this factory is
    kept as a thin convenience alias for parity with the other backend
    factories. You can just as well instantiate the class directly.

    Args:
        bucket_name: OCI bucket name
        namespace: OCI namespace
        prefix: Object prefix
        **kwargs: Additional OCIBucketBackend options (profile_name, auth_type, region)

    Example:
        >>> checkpointer = oci_bucket_checkpointer(
        ...     bucket_name="my-checkpoints",
        ...     namespace="my-namespace",
        ... )
        >>> agent = Agent(config=cfg, checkpointer=checkpointer)
    """
    from locus.memory.backends.oci_bucket import OCIBucketBackend

    return OCIBucketBackend(
        bucket_name=bucket_name,
        namespace=namespace,
        prefix=prefix,
        **kwargs,
    )


def oracle_checkpointer(
    dsn: str | None = None,
    user: str = "admin",
    password: str = "",
    max_checkpoints_per_thread: int | None = None,
    *,
    ttl: float | timedelta | None = None,
    encryption: EncryptedSerializer | None = None,
    scope: CheckpointScope | None = None,
    **kwargs: Any,
) -> StorageBackendAdapter:
    """
    Create an Oracle Database-backed checkpointer.

    Args:
        dsn: Oracle connection string (e.g., "host:port/service_name") or a
            tnsnames alias
        user: Database user
        password: Database password
        max_checkpoints_per_thread: Keep only this many of a thread's newest
            checkpoints (``None`` keeps all); see ``StorageBackendAdapter``
        ttl: How long a checkpoint stays loadable (seconds or a ``timedelta``);
            recorded in each row's document and swept by ``expire()``
        encryption: Seals the state (:class:`~locus.memory.encryption.EncryptedSerializer`)
        scope: The application and user the threads belong to
            (:class:`~locus.memory.scope.CheckpointScope`)
        **kwargs: Additional OracleBackend options: ``table_name``;
            ``config_dir`` (the directory holding ``tnsnames.ora``, for an
            alias on a TLS-only database); ``wallet_location`` and
            ``wallet_password`` for Autonomous Database with mTLS

    Returns:
        StorageBackendAdapter wrapping OracleBackend

    Capabilities:
        - search: Yes (via JSON search)
        - metadata_query: Yes
        - vacuum: Yes
        - list_threads: Yes
        - persistent_checkpoint_ids: Yes

    Example:
        >>> checkpointer = oracle_checkpointer(
        ...     dsn="mydb_high",
        ...     user="admin",
        ...     password="secret",
        ... )
        >>> agent = Agent(model=model, checkpointer=checkpointer)
    """
    from locus.memory.backends.oracle import OracleBackend

    backend = OracleBackend(
        dsn=dsn,
        user=user,
        password=password,
        **kwargs,
    )
    return StorageBackendAdapter(
        backend,
        max_checkpoints_per_thread=max_checkpoints_per_thread,
        ttl=ttl,
        encryption=encryption,
        scope=scope,
    )
