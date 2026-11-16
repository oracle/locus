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

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, NamedTuple
from uuid import uuid4

from locus.core.protocols import CheckpointerCapabilities
from locus.memory.checkpointer import BaseCheckpointer
from locus.memory.time_travel import (
    CheckpointInfo,
    checkpoint_iteration,
    checkpoint_size,
    info_from,
    link_parents,
)


if TYPE_CHECKING:
    from locus.core.state import AgentState


class _Stored(NamedTuple):
    """One checkpoint as the dictionary holds it: the state and what ``list`` reports about it."""

    data: dict[str, Any]
    created_at: datetime
    metadata: dict[str, Any]
    iteration: int | None
    size: int


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

    Example:
        ```python
        checkpointer = MemoryCheckpointer()

        # Save state
        checkpoint_id = await checkpointer.save(state, "thread-1")

        # Load state
        restored = await checkpointer.load("thread-1")
        ```
    """

    def __init__(self, max_checkpoints_per_thread: int | None = None) -> None:
        if max_checkpoints_per_thread is not None and max_checkpoints_per_thread < 1:
            raise ValueError("max_checkpoints_per_thread must be at least 1")
        self.max_checkpoints_per_thread = max_checkpoints_per_thread
        # Storage: {thread_id: {checkpoint_id: _Stored}}; the tuple's first
        # three fields are (state_data, timestamp, metadata), as they always were.
        self._storage: dict[str, dict[str, _Stored]] = {}

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

        if thread_id not in self._storage:
            self._storage[thread_id] = {}

        data = state.to_checkpoint()
        self._storage[thread_id][checkpoint_id] = _Stored(
            data,
            datetime.now(UTC),
            metadata or {},
            checkpoint_iteration(data),
            checkpoint_size(data),
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
        from locus.core.state import AgentState

        if thread_id not in self._storage:
            return None

        thread_data = self._storage[thread_id]

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

        return AgentState.from_checkpoint(thread_data[checkpoint_id].data)

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
        if thread_id not in self._storage:
            return []

        thread_data = self._storage[thread_id]

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
        if thread_id not in self._storage:
            return False

        if checkpoint_id is None:
            # Delete all checkpoints for thread
            del self._storage[thread_id]
            return True
        if checkpoint_id in self._storage[thread_id]:
            del self._storage[thread_id][checkpoint_id]
            return True
        return False

    def clear(self) -> None:
        """Clear all stored checkpoints."""
        self._storage.clear()

    def get_thread_ids(self) -> list[str]:
        """Get list of all thread IDs with checkpoints."""
        return list(self._storage.keys())

    async def list_threads(
        self,
        limit: int = 100,
        pattern: str = "*",
    ) -> list[str]:
        """
        List all thread IDs.

        Args:
            limit: Maximum threads to return
            pattern: Pattern to filter (supports * as wildcard)

        Returns:
            List of thread IDs
        """
        import fnmatch

        threads = list(self._storage.keys())

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
            return len(self._storage.get(thread_id, {}))
        return sum(len(t) for t in self._storage.values())

    def __repr__(self) -> str:
        total = self.get_checkpoint_count()
        threads = len(self._storage)
        return f"MemoryCheckpointer(threads={threads}, checkpoints={total})"

    # Last on purpose: a method named ``list`` shadows the builtin for the
    # annotations of anything defined after it.
    async def list(self, thread_id: str, limit: int = 10) -> list[CheckpointInfo]:
        """The thread's checkpoints, newest first, read off the stored records."""
        thread_data = self._storage.get(thread_id, {})
        newest_first = sorted(
            thread_data.items(), key=lambda item: item[1].created_at, reverse=True
        )
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
            ]
        )
