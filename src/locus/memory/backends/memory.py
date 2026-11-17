# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""In-memory checkpoint backend for Locus.

This backend stores checkpoints in a dictionary, making it ideal for:
- Unit testing
- Development
- Short-lived sessions
- Caching layer

Note: All data is lost when the process exits.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, NamedTuple
from uuid import uuid4

from locus.core.protocols import CheckpointerCapabilities
from locus.memory.checkpointer import BaseCheckpointer
from locus.memory.expiry import is_expired
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


class _Stored(NamedTuple):
    """One checkpoint as the dictionary holds it: the state and what ``list`` reports about it."""

    data: dict[str, Any]
    created_at: datetime
    metadata: dict[str, Any]
    iteration: int | None
    size: int
    expires_at: datetime | None = None


class MemoryCheckpointer(BaseCheckpointer):
    """
    In-memory checkpointer for testing and development.

    Stores all checkpoints in a dictionary. Data is not persistent
    and will be lost when the process terminates.

    Useful for:
    - Unit and integration testing
    - Development and prototyping
    - Short-lived agent sessions
    - As a fast caching layer

    Capabilities:
    - list_threads: Yes
    - persistent_checkpoint_ids: Yes (within process lifetime)

    Args:
        max_checkpoints_per_thread: Keep only this many of a thread's newest
            checkpoints, dropping older ones after each save. ``None`` (the
            default) keeps them all. Every checkpoint holds the whole
            conversation, so a long thread's memory grows with the square of
            its length; a cap bounds it, at the cost of ``fork`` and ``list``
            reaching back only that far. The tool-effect ledger's sidecar
            thread is never trimmed.
        ttl: How long a checkpoint stays loadable after its save (seconds or
            a ``timedelta``); ``None`` keeps it until deleted. See
            :mod:`locus.memory.expiry`.
        encryption: Seals the state before it is stored
            (:class:`~locus.memory.encryption.EncryptedSerializer`).
        scope: The application and user the threads belong to
            (:class:`~locus.memory.scope.CheckpointScope`).

    Example:
        ```python
        checkpointer = MemoryCheckpointer()

        # Save state
        checkpoint_id = await checkpointer.save(state, "thread-1")

        # Load state
        restored = await checkpointer.load("thread-1")
        ```
    """

    def __init__(
        self,
        max_checkpoints_per_thread: int | None = None,
        *,
        ttl: float | timedelta | None = None,
        encryption: EncryptedSerializer | None = None,
        scope: CheckpointScope | None = None,
    ) -> None:
        if max_checkpoints_per_thread is not None and max_checkpoints_per_thread < 1:
            raise ValueError("max_checkpoints_per_thread must be at least 1")
        self.max_checkpoints_per_thread = max_checkpoints_per_thread
        self._configure(ttl=ttl, encryption=encryption, scope=scope)
        # Storage: {stored thread id: {checkpoint_id: _Stored}}; the tuple's
        # first three fields are (state_data, timestamp, metadata), as they
        # always were. A scoped thread is stored under its scope's prefix.
        self._storage: dict[str, dict[str, _Stored]] = {}

    def _live(self, thread_key: str) -> dict[str, _Stored]:
        """A thread's checkpoints that have not expired."""
        now = datetime.now(UTC)
        return {
            checkpoint_id: stored
            for checkpoint_id, stored in self._storage.get(thread_key, {}).items()
            if not is_expired(stored.expires_at, now)
        }

    @property
    def capabilities(self) -> CheckpointerCapabilities:
        """Memory checkpointer capabilities."""
        return CheckpointerCapabilities(
            list_threads=True,
            persistent_checkpoint_ids=True,
        )

    async def save(
        self,
        state: AgentState,
        thread_id: str,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Save agent state to memory.

        Args:
            state: Current agent state
            thread_id: Thread identifier
            checkpoint_id: Optional specific checkpoint ID
            metadata: Optional metadata for the checkpoint

        Returns:
            Checkpoint ID for the saved state
        """
        checkpoint_id = checkpoint_id or uuid4().hex
        key = self._scoped(thread_id)

        if key not in self._storage:
            self._storage[key] = {}

        data = self._encode_state(state, key, checkpoint_id)
        now = datetime.now(UTC)
        self._storage[key][checkpoint_id] = _Stored(
            data,
            now,
            metadata or {},
            checkpoint_iteration(data),
            checkpoint_size(data),
            self._expires_at(now),
        )
        if self.max_checkpoints_per_thread is not None:
            await self._trim_history(thread_id, self.max_checkpoints_per_thread)

        return checkpoint_id

    async def load(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> AgentState | None:
        """
        Load agent state from memory.

        Args:
            thread_id: Thread identifier
            checkpoint_id: Specific checkpoint ID (latest if None)

        Returns:
            Restored AgentState or None if not found
        """
        key = self._scoped(thread_id)
        thread_data = self._live(key)

        if not thread_data:
            return None

        if checkpoint_id is None:
            # Get latest checkpoint by timestamp
            latest_id = max(
                thread_data.keys(),
                key=lambda k: thread_data[k][1],
            )
            checkpoint_id = latest_id

        if checkpoint_id not in thread_data:
            return None

        return self._decode_state(thread_data[checkpoint_id].data, key, checkpoint_id)

    async def list_checkpoints(
        self,
        thread_id: str,
        limit: int = 10,
    ) -> list[str]:
        """
        List available checkpoints for a thread.

        Args:
            thread_id: Thread identifier
            limit: Maximum number to return

        Returns:
            List of checkpoint IDs, newest first
        """
        thread_data = self._live(self._scoped(thread_id))

        # Sort by timestamp descending
        sorted_ids = sorted(
            thread_data.keys(),
            key=lambda k: thread_data[k][1],
            reverse=True,
        )

        return sorted_ids[:limit]

    async def delete(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> bool:
        """
        Delete checkpoint(s) from memory.

        Args:
            thread_id: Thread identifier
            checkpoint_id: Specific checkpoint to delete (all if None)

        Returns:
            True if deletion was successful
        """
        key = self._scoped(thread_id)
        if key not in self._storage:
            return False

        if checkpoint_id is None:
            # Delete all checkpoints for thread
            del self._storage[key]
            return True
        if checkpoint_id in self._storage[key]:
            del self._storage[key][checkpoint_id]
            return True
        return False

    async def expire(self) -> int:
        """Drop every expired checkpoint, and the threads left empty; returns the count."""
        now = datetime.now(UTC)
        removed = 0
        for key in list(self._storage):
            thread_data = self._storage[key]
            expired = [cid for cid, s in thread_data.items() if is_expired(s.expires_at, now)]
            for checkpoint_id in expired:
                del thread_data[checkpoint_id]
            removed += len(expired)
            if not thread_data:
                del self._storage[key]
        return removed

    def clear(self) -> None:
        """Clear all stored checkpoints."""
        self._storage.clear()

    def get_thread_ids(self) -> list[str]:
        """Get list of all thread IDs with checkpoints, as stored (scoped ids carry their prefix)."""
        return list(self._storage.keys())

    async def list_threads(
        self,
        limit: int = 100,
        pattern: str = "*",
        *,
        scope: CheckpointScope | None = None,
    ) -> list[str]:
        """
        List the thread IDs of one scope.

        Args:
            limit: Maximum threads to return
            pattern: Pattern to filter (supports * as wildcard)
            scope: The scope to list; default the call's, else the unscoped threads

        Returns:
            List of thread IDs
        """
        import fnmatch

        threads = self._listing_scope(scope).strip(key for key in self._storage if self._live(key))

        if pattern != "*":
            threads = [t for t in threads if fnmatch.fnmatch(t, pattern)]

        return threads[:limit]

    def get_checkpoint_count(self, thread_id: str | None = None) -> int:
        """
        Get count of stored checkpoints.

        Args:
            thread_id: Specific thread (all threads if None)

        Returns:
            Number of checkpoints
        """
        if thread_id is not None:
            return len(self._storage.get(self._scoped(thread_id), {}))
        return sum(len(t) for t in self._storage.values())

    def __repr__(self) -> str:
        total = self.get_checkpoint_count()
        threads = len(self._storage)
        return f"MemoryCheckpointer(threads={threads}, checkpoints={total})"

    # Last on purpose: a method named ``list`` shadows the builtin for the
    # annotations of anything defined after it.
    async def list(self, thread_id: str, limit: int = 10) -> list[CheckpointInfo]:
        """The thread's checkpoints, newest first, read off the stored records."""
        thread_data = self._live(self._scoped(thread_id))
        newest_first = sorted(
            thread_data.items(), key=lambda item: item[1].created_at, reverse=True
        )
        # The checkpoint just past the window is the parent of the oldest in it.
        older = newest_first[limit][0] if len(newest_first) > limit else None
        return link_parents(
            [
                info_from(
                    checkpoint_id,
                    thread_id,
                    created_at=stored.created_at,
                    metadata=stored.metadata,
                    iteration=stored.iteration,
                    size=stored.size,
                )
                for checkpoint_id, stored in newest_first[:limit]
            ],
            older,
        )
