# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Base checkpointer for Locus - state persistence abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from locus.core.protocols import CheckpointerCapabilities
from locus.memory.encryption import CheckpointEncryptionError, EncryptedSerializer, is_envelope
from locus.memory.expiry import expiry_time, parse_ttl
from locus.memory.scope import CheckpointScope, current_scope


if TYPE_CHECKING:
    from locus.core.state import AgentState
    from locus.memory.time_travel import CheckpointInfo


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

    Every checkpointer has the time-travel surface: :meth:`list` describes
    a thread's checkpoints (iteration, time, size, parent), :meth:`fork`
    starts a new thread from one of them, and ``Agent.run(...,
    from_checkpoint=...)`` continues a thread from a past one. The defaults
    here work on any subclass through ``save`` / ``load`` /
    ``list_checkpoints``; a backend overrides :meth:`list` when it can read
    the fields without loading each state.

    Every checkpointer also takes the three production settings, applied
    here so each backend only stores what they produce: ``ttl`` (a
    checkpoint expires that long after it is written and is then neither
    loaded, listed nor forked; :meth:`expire` removes what has expired),
    ``encryption`` (an :class:`~locus.memory.encryption.EncryptedSerializer`
    that seals the state before it is stored) and ``scope`` (a
    :class:`~locus.memory.scope.CheckpointScope` that keeps one
    application's and user's threads apart from another's). A subclass
    passes them to :meth:`_configure` from its constructor and uses
    :meth:`_scoped`, :meth:`_encode_state`, :meth:`_decode_state` and
    :meth:`_expires_at` where it stores and reads.

    Example:
        >>> if checkpointer.capabilities.search:
        ...     results = await checkpointer.search("error handling")
        >>> for info in await checkpointer.list("main"):
        ...     print(info.checkpoint_id, info.iteration, info.parent_checkpoint_id)
        >>> experiment = await checkpointer.fork("main", info.checkpoint_id)
    """

    #: How long after its save a checkpoint stays loadable; ``None`` keeps it until deleted.
    ttl: timedelta | None = None
    #: Seals the state before a backend stores it; ``None`` stores plaintext JSON.
    encryption: EncryptedSerializer | None = None
    #: The application and user this instance's threads belong to; ``None`` for none.
    scope: CheckpointScope | None = None

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

    @property
    def deletes_single_checkpoints(self) -> bool:
        """Whether :meth:`delete` can remove one checkpoint and leave the rest.

        The agent saves after every iteration only where this holds: those
        saves exist to survive a crash mid-turn, and the turn's final save
        deletes them. A backend that cannot delete would keep every one.

        True when a subclass implements :meth:`delete`. A backend whose
        checkpoints depend on one another (deltas chained to a parent) must
        override this to return False, because deleting one breaks the rest.
        """
        return type(self).delete is not BaseCheckpointer.delete

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
        *,
        scope: CheckpointScope | None = None,
    ) -> list[str]:
        """
        List the thread IDs of one scope.

        Requires: capabilities.list_threads = True

        Args:
            limit: Maximum threads to return
            pattern: Pattern to filter threads (backend-specific)
            scope: The scope whose threads to list. Default: the scope active
                for the call (a scoped run, or the one this instance was
                built with); the unscoped threads when there is none.

        Returns:
            List of thread IDs, as ``load`` takes them (without the scope prefix)
        """
        self._require_capability("list_threads")
        raise NotImplementedError("list_threads not implemented")

    async def expire(self) -> int:
        """Remove the checkpoints whose ``ttl`` has run out; returns how many.

        A read never returns an expired checkpoint whether or not this has
        run; this reclaims the storage. Call it from a scheduler as often as
        the store's size warrants. It sweeps every scope. A backend whose
        store expires keys itself (Redis) has nothing left to remove here.
        """
        raise NotImplementedError("expire not implemented for this backend")

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
    # Time travel
    # =========================================================================

    async def fork(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
        *,
        new_thread_id: str | None = None,
    ) -> str:
        """Start a new thread from one of this thread's checkpoints (the latest by default).

        The new thread's first checkpoint is the state at ``checkpoint_id``,
        saved with ``forked_from`` and ``parent_checkpoint_id`` in its
        metadata; running on it never changes the source thread. The source
        thread's tool-effect ledger is copied up to that checkpoint's
        iteration and no further, so a run on the fork replays nothing the
        source did past the fork point.

        Args:
            thread_id: The thread to fork.
            checkpoint_id: The checkpoint to fork at; the latest when ``None``.
            new_thread_id: The fork's id; ``<thread_id>-fork-<8 hex>`` when ``None``.

        Returns:
            The new thread's id.

        Raises:
            LookupError: No such thread or checkpoint.
            ValueError: ``new_thread_id`` is the source thread.
        """
        from locus.memory.time_travel import default_fork  # noqa: PLC0415

        return await default_fork(self, thread_id, checkpoint_id, new_thread_id)

    async def _trim_history(self, thread_id: str, keep: int | None) -> int:
        """Delete all but the ``keep`` newest checkpoints of ``thread_id``; returns how many.

        For backends with a ``max_checkpoints_per_thread`` bound. The
        tool-effect ledger's sidecar thread (``<thread_id>~ledger``) is never
        trimmed: its checkpoints are one record per executed tool call, and
        dropping one would run that call again on a replay.
        """
        from locus.memory.ledger import LEDGER_SUFFIX  # noqa: PLC0415

        if keep is None or thread_id.endswith(LEDGER_SUFFIX):
            return 0
        surplus = (await self.list_checkpoints(thread_id, limit=1_000_000))[keep:]
        deleted = 0
        for checkpoint_id in surplus:
            if await self.delete(thread_id, checkpoint_id):
                deleted += 1
        return deleted

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

    # =========================================================================
    # Production settings: ttl, encryption, scope (for subclasses)
    # =========================================================================

    def _configure(
        self,
        *,
        ttl: float | timedelta | None = None,
        encryption: EncryptedSerializer | None = None,
        scope: CheckpointScope | None = None,
    ) -> None:
        """Validate and keep the production settings a subclass's constructor took."""
        self.ttl = parse_ttl(ttl)
        if encryption is not None and not isinstance(encryption, EncryptedSerializer):
            raise TypeError("encryption must be an EncryptedSerializer")
        self.encryption = encryption
        if scope is not None and not isinstance(scope, CheckpointScope):
            raise TypeError("scope must be a CheckpointScope")
        self.scope = scope if scope is not None and not scope.is_empty else None

    def _effective_scope(self) -> CheckpointScope | None:
        """The scope of this call: an active one (a scoped run) over the instance's."""
        return current_scope() or self.scope

    def _scoped(self, thread_id: str) -> str:
        """The id ``thread_id`` is stored under: prefixed by the scope of this call."""
        scope = self._effective_scope()
        return scope.key(thread_id) if scope is not None else thread_id

    def _listing_scope(self, scope: CheckpointScope | None) -> CheckpointScope:
        """The scope ``list_threads`` lists: the given one, else the call's, else none."""
        return scope or self._effective_scope() or CheckpointScope()

    def _encode_state(self, state: AgentState) -> dict[str, Any]:
        """``state`` as this checkpointer stores it: its checkpoint dict, sealed when encrypting."""
        data = state.to_checkpoint()
        return self.encryption.encrypt(data) if self.encryption is not None else data

    def _decode_state(self, data: dict[str, Any]) -> AgentState:
        """The state a stored checkpoint dict holds, unsealing an envelope.

        A plaintext checkpoint loads whether or not ``encryption`` is set, so
        turning encryption on leaves earlier checkpoints readable.

        Raises:
            CheckpointEncryptionError: The checkpoint is encrypted and this
                instance has no ``encryption``, or none of its keys fits.
        """
        from locus.core.state import AgentState  # noqa: PLC0415

        if is_envelope(data):
            if self.encryption is None:
                raise CheckpointEncryptionError(
                    "the checkpoint is encrypted; give this checkpointer encryption= "
                    "with the key it was written with"
                )
            data = self.encryption.decrypt(data)
        return AgentState.from_checkpoint(data)

    def _expires_at(self, now: datetime | None = None) -> datetime | None:
        """When a save made ``now`` expires under this instance's ``ttl``; ``None`` without one."""
        return expiry_time(self.ttl, now)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    # ``list`` is the last method of the class on purpose: a class-level name
    # shadows the builtin for the annotations of everything defined after it.
    async def list(self, thread_id: str, limit: int = 10) -> list[CheckpointInfo]:
        """The thread's checkpoints, newest first, with what each one is.

        Each :class:`~locus.memory.time_travel.CheckpointInfo` has the
        checkpoint's id, when it was written, the iteration the state had
        reached, the checkpoint's size in bytes as this backend stores it,
        the checkpoint it descends from and, for a fork's first checkpoint,
        the thread it was forked from. ``parent_checkpoint_id`` is ``None``
        only for a thread's first checkpoint: with a ``limit``, the oldest
        checkpoint described still names the one before it. The default
        loads each state to read these; backends that store the fields
        beside the state override it.

        Args:
            thread_id: Thread identifier.
            limit: Maximum number of checkpoints to describe.
        """
        from locus.memory.time_travel import default_list  # noqa: PLC0415

        return await default_list(self, thread_id, limit)
