# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Base checkpointer for Locus - state persistence abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from locus.core.protocols import CheckpointerCapabilities


if TYPE_CHECKING:
    from locus.core.state import AgentState


class BaseCheckpointer(ABC):
    """
    Abstract base class for checkpointer implementations.

    Checkpointers handle saving and loading agent state, enabling
    features like:
    - Conversation persistence
    - Session recovery
    - Branching conversations
    - State inspection and debugging
    - Full-text search (backend-dependent)
    - Metadata queries (backend-dependent)

    All methods are async to support various backends (file, database,
    network storage, etc.).

    Use the `capabilities` property to check which features are available
    before calling extended methods.

    Example:
        >>> if checkpointer.capabilities.search:
        ...     results = await checkpointer.search("error handling")
        >>> if checkpointer.capabilities.branching:
        ...     await checkpointer.copy_thread("main", "experiment")
    """

    @property
    def capabilities(self) -> CheckpointerCapabilities:
        """
        Return the capabilities of this checkpointer.

        Override in subclasses to advertise supported features.
        """
        return CheckpointerCapabilities()

    # =========================================================================
    # Core Methods (Required)
    # =========================================================================

    @abstractmethod
    async def save(
        self,
        state: AgentState,
        thread_id: str,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Save agent state.

        Args:
            state: Current agent state to persist
            thread_id: Unique identifier for the conversation thread
            checkpoint_id: Optional specific checkpoint ID. If not provided,
                          a new ID will be generated.
            metadata: Optional metadata for querying/filtering checkpoints

        Returns:
            Checkpoint ID that can be used to restore this state
        """
        ...

    @abstractmethod
    async def load(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> AgentState | None:
        """
        Load agent state.

        Args:
            thread_id: Thread identifier to load from
            checkpoint_id: Optional specific checkpoint ID. If not provided,
                          loads the latest checkpoint.

        Returns:
            Restored AgentState or None if not found
        """
        ...

    @abstractmethod
    async def list_checkpoints(
        self,
        thread_id: str,
        limit: int = 10,
    ) -> list[str]:
        """
        List available checkpoints for a thread.

        Args:
            thread_id: Thread identifier
            limit: Maximum number of checkpoint IDs to return

        Returns:
            List of checkpoint IDs, newest first
        """
        ...

    # =========================================================================
    # Optional Core Methods (Default Implementations)
    # =========================================================================

    async def delete(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> bool:
        """
        Delete a checkpoint or all checkpoints for a thread.

        Args:
            thread_id: Thread identifier
            checkpoint_id: Specific checkpoint to delete. If None,
                          deletes all checkpoints for the thread.

        Returns:
            True if deletion was successful
        """
        raise NotImplementedError("delete not implemented for this backend")

    async def exists(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> bool:
        """
        Check if a checkpoint exists.

        Args:
            thread_id: Thread identifier
            checkpoint_id: Specific checkpoint to check. If None,
                          checks if any checkpoint exists for the thread.

        Returns:
            True if the checkpoint exists
        """
        if checkpoint_id is None:
            checkpoints = await self.list_checkpoints(thread_id, limit=1)
            return len(checkpoints) > 0
        state = await self.load(thread_id, checkpoint_id)
        return state is not None

    async def close(self) -> None:
        """
        Close any resources (connections, files, etc.).

        Override in subclasses if cleanup is needed.
        """

    # =========================================================================
    # Extended Methods (Capability-Dependent)
    # =========================================================================

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Full-text search across checkpoints.

        Requires: capabilities.search = True

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching checkpoints with scores

        Raises:
            NotImplementedError: If backend doesn't support search
        """
        self._require_capability("search")
        raise NotImplementedError("search not implemented")

    async def query_by_metadata(
        self,
        key: str,
        value: Any,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Query checkpoints by metadata field.

        Requires: capabilities.metadata_query = True

        Args:
            key: Metadata field name
            value: Value to match
            limit: Maximum results

        Returns:
            List of matching checkpoints
        """
        self._require_capability("metadata_query")
        raise NotImplementedError("query_by_metadata not implemented")

    async def get_metadata(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Get checkpoint metadata.

        Requires: capabilities.metadata_query = True

        Args:
            thread_id: Thread identifier
            checkpoint_id: Specific checkpoint (latest if None)

        Returns:
            Metadata dict or None if not found
        """
        self._require_capability("metadata_query")
        raise NotImplementedError("get_metadata not implemented")

    async def vacuum(
        self,
        older_than_days: int = 30,
    ) -> int:
        """
        Delete old checkpoints.

        Requires: capabilities.vacuum = True

        Args:
            older_than_days: Delete checkpoints older than this

        Returns:
            Number of deleted checkpoints
        """
        self._require_capability("vacuum")
        raise NotImplementedError("vacuum not implemented")

    async def copy_thread(
        self,
        source_thread_id: str,
        dest_thread_id: str,
    ) -> bool:
        """
        Copy a thread to create a branch.

        Requires: capabilities.branching = True

        Args:
            source_thread_id: Source thread to copy from
            dest_thread_id: Destination thread ID

        Returns:
            True if successful
        """
        self._require_capability("branching")
        raise NotImplementedError("copy_thread not implemented")

    async def list_threads(
        self,
        limit: int = 100,
        pattern: str = "*",
    ) -> list[str]:
        """
        List all thread IDs.

        Requires: capabilities.list_threads = True

        Args:
            limit: Maximum threads to return
            pattern: Pattern to filter threads (backend-specific)

        Returns:
            List of thread IDs
        """
        self._require_capability("list_threads")
        raise NotImplementedError("list_threads not implemented")

    async def list_with_metadata(
        self,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        List checkpoints with their metadata.

        Requires: capabilities.list_with_metadata = True

        Args:
            limit: Maximum results

        Returns:
            List of {thread_id, checkpoint_id, metadata, ...} dicts
        """
        self._require_capability("list_with_metadata")
        raise NotImplementedError("list_with_metadata not implemented")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _require_capability(self, capability: str) -> None:
        """
        Raise NotImplementedError if capability is not supported.

        Args:
            capability: Name of the capability to check

        Raises:
            NotImplementedError: If capability is not available
        """
        if not getattr(self.capabilities, capability, False):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support '{capability}'. "
                f"Check capabilities before calling: checkpointer.capabilities.{capability}"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
