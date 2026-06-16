# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Long-term memory manager for Locus agents.

Extracts durable facts from conversation history, persists them via a
:class:`~locus.memory.store.BaseStore` backend, and injects relevant
memories into the system prompt at the start of every new session.

Storage layout
--------------
All memories are namespaced in the configured store under::

    (*namespace_prefix, memory_type)  →  key: memory.key  →  value: {content, metadata}

The default prefix is ``("locus_memory",)``, so memories appear as::

    ("locus_memory", "user")       →  "preferred_language": ...
    ("locus_memory", "feedback")   →  "no_db_mocks": ...
    ("locus_memory", "project")    →  "auth_rewrite": ...
    ("locus_memory", "reference")  →  "linear_pipeline": ...

Scope memories per user or tenant by setting a richer prefix::

    LLMMemoryManager(store=my_store, namespace_prefix=("tenants", tenant_id))

Memory types
------------
``user``
    Who the user is — role, expertise, working style.  Use when
    tailoring explanations or phrasing.
``feedback``
    Behavioural rules — what to do or avoid, and *why*.  Structured as
    ``rule → Why → How to apply`` so the agent can reason about edge
    cases.
``project``
    Ongoing work context — goals, deadlines, key decisions.  Decays
    quickly; include a *Why* so future reads can judge staleness.
``reference``
    Pointers to external systems — Jira projects, dashboards, Slack
    channels, config file locations.

Quick start
-----------
::

    from locus import Agent
    from locus.memory.manager import LLMMemoryManager, Memory, MemoryType
    from locus.memory.store import InMemoryStore

    store = InMemoryStore()


    async def my_extractor(messages):
        # Call an LLM here; return a list of Memory objects.
        return [
            Memory(
                type=MemoryType.USER,
                key="preferred_language",
                content="User writes Python, not Go.",
                metadata={},
            )
        ]


    agent = Agent(
        model="oci:openai.gpt-5",
        memory_manager=LLMMemoryManager(store=store, extract_fn=my_extractor),
    )
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from locus.core.messages import Message
    from locus.core.state import AgentState
    from locus.memory.store import BaseStore


# Callable type for user-supplied extraction functions.
ExtractFn = Callable[
    [list["Message"]],
    Coroutine[Any, Any, list["Memory"]],
]


class MemoryType(StrEnum):
    """Semantic category for a stored memory."""

    USER = "user"
    FEEDBACK = "feedback"
    PROJECT = "project"
    REFERENCE = "reference"


@dataclass
class Memory:
    """A single durable memory entry.

    Attributes:
        type: Semantic category (user / feedback / project / reference).
        key: Stable logical name within the category.  The store uses
            this as the key, so re-extracting the same fact under the
            same key *updates* the record rather than creating a
            duplicate.
        content: The actual fact or rule, as a human-readable string.
        metadata: Arbitrary extra fields — ``confidence``, ``source``,
            ``why``, ``how_to_apply``, ISO timestamps, etc.
    """

    type: MemoryType
    key: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_store_value(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict for the store."""
        return {
            "type": self.type.value,
            "key": self.key,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_store_value(cls, value: dict[str, Any]) -> Memory:
        """Deserialise from a store value dict."""
        return cls(
            type=MemoryType(value["type"]),
            key=value["key"],
            content=value["content"],
            metadata=value.get("metadata", {}),
        )


# =============================================================================
# Abstract base
# =============================================================================


class BaseMemoryManager(ABC):
    """Abstract base for long-term memory managers.

    Subclasses implement :meth:`extract` to decide what is worth
    remembering; the base class handles retrieval, injection, and the
    session-start / session-end lifecycle hooks that the agent calls
    automatically.

    Two concrete implementations are provided:

    * :class:`NoopMemoryManager` — no-op, useful for testing.
    * :class:`LLMMemoryManager` — persists to any :class:`BaseStore`
      backend; accepts an optional LLM-backed extraction function.
    """

    @abstractmethod
    async def extract(self, messages: list[Message]) -> list[Memory]:
        """Extract durable memories from a finished conversation.

        Args:
            messages: The full message history for the completed session.

        Returns:
            List of :class:`Memory` objects to persist.  May be empty.
        """
        ...

    @abstractmethod
    async def retrieve(self, limit: int = 20) -> list[Memory]:
        """Retrieve stored memories for injection at session start.

        Args:
            limit: Maximum number of memories to return.

        Returns:
            List of :class:`Memory` objects, most recently updated first.
        """
        ...

    @abstractmethod
    async def save(self, memories: list[Memory]) -> None:
        """Persist a list of memories to the backing store.

        Implementations should upsert by key — re-extracting the same
        fact updates the record rather than creating a duplicate.

        Args:
            memories: Memories to persist.
        """
        ...

    async def on_session_start(self, state: AgentState) -> AgentState:
        """Retrieve memories and inject them into the agent state.

        Called by the agent runtime at the start of every invocation,
        after ``on_before_invocation`` hooks but before the first model
        call.  The default implementation retrieves all stored memories
        and prepends a formatted system message to ``state.messages``.

        Args:
            state: Current agent state (just-created or loaded from
                checkpointer).

        Returns:
            Possibly-modified state with memory context injected.
        """
        memories = await self.retrieve()
        if not memories:
            return state

        from locus.observability.emit import emit  # noqa: PLC0415

        injected_state = _inject_memories_into_state(state, memories)

        await emit(
            "memory.manager.injected",
            memory_count=len(memories),
            types=[m.type.value for m in memories],
        )

        return injected_state

    async def on_session_end(self, state: AgentState) -> None:
        """Extract memories from the finished session and save them.

        Called by the agent runtime in the ``finally`` block of every
        invocation, after ``on_after_invocation`` hooks but before the
        final checkpoint.

        Args:
            state: Final agent state with the complete message history.
        """
        memories = await self.extract(list(state.messages))
        if not memories:
            return

        await self.save(memories)

        from locus.observability.emit import emit  # noqa: PLC0415

        await emit(
            "memory.manager.extracted",
            memory_count=len(memories),
            types=[m.type.value for m in memories],
            keys=[m.key for m in memories],
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# =============================================================================
# No-op implementation
# =============================================================================


class NoopMemoryManager(BaseMemoryManager):
    """Pass-through memory manager — stores and retrieves nothing.

    Useful as a test double or as a placeholder when you want the
    agent-wiring (``memory_manager=`` kwarg) without actual persistence.
    """

    async def extract(self, messages: list[Message]) -> list[Memory]:
        return []

    async def retrieve(self, limit: int = 20) -> list[Memory]:
        return []

    async def save(self, memories: list[Memory]) -> None:
        pass

    async def on_session_start(self, state: AgentState) -> AgentState:
        return state

    async def on_session_end(self, state: AgentState) -> None:
        pass


# =============================================================================
# LLM-backed implementation
# =============================================================================


class LLMMemoryManager(BaseMemoryManager):
    """Memory manager backed by any :class:`~locus.memory.store.BaseStore`.

    Extraction uses either a caller-supplied async function
    (``extract_fn``) or a built-in heuristic that pattern-matches
    common conversational signals (corrections, confirmations, role
    disclosures).  Supply ``extract_fn`` for production use; the
    heuristic is adequate for demos and tests.

    Args:
        store: Any :class:`~locus.memory.store.BaseStore` implementation.
            ``InMemoryStore`` works for development; use
            ``RedisBackend``, ``PostgreSQLBackend``, ``OracleBackend``, etc.
            for production.
        extract_fn: Optional async callable
            ``(messages: list[Message]) -> list[Memory]``.  When
            provided, the heuristic is bypassed.
        namespace_prefix: Store namespace prefix.  Scope per user or
            tenant by passing ``("tenants", tenant_id)``.  Default
            ``("locus_memory",)``.
        max_memories: Hard cap on the total number of memories kept per
            type.  Oldest entries are pruned when the limit is reached.
        retrieve_limit: Maximum memories returned by :meth:`retrieve`.

    Example::

        from locus.memory.manager import LLMMemoryManager
        from locus.memory.backends import OracleBackend

        manager = LLMMemoryManager(
            store=OracleBackend(dsn="mydb_high", user="admin", password="secret"),
            extract_fn=my_llm_extractor,
            namespace_prefix=("users", user_id),
        )

        agent = Agent(model="oci:openai.gpt-5", memory_manager=manager)
    """

    def __init__(
        self,
        store: BaseStore,
        *,
        extract_fn: ExtractFn | None = None,
        namespace_prefix: tuple[str, ...] = ("locus_memory",),
        max_memories: int = 50,
        retrieve_limit: int = 20,
    ) -> None:
        self.store = store
        self.extract_fn = extract_fn
        self.namespace_prefix = namespace_prefix
        self.max_memories = max_memories
        self.retrieve_limit = retrieve_limit

    def _ns(self, memory_type: MemoryType) -> tuple[str, ...]:
        """Build the store namespace for a memory type."""
        return (*self.namespace_prefix, memory_type.value)

    async def extract(self, messages: list[Message]) -> list[Memory]:
        """Extract memories from a message list.

        Uses ``extract_fn`` when provided; otherwise applies the
        built-in heuristic.
        """
        if self.extract_fn is not None:
            return await self.extract_fn(messages)
        return _heuristic_extract(messages)

    async def retrieve(self, limit: int = 20) -> list[Memory]:
        """Retrieve all stored memories across every type.

        Returns memories sorted by most-recently-updated first, up to
        ``limit`` entries total.
        """
        memories: list[Memory] = []

        for memory_type in MemoryType:
            ns = self._ns(memory_type)
            try:
                items = await self.store.search(ns, query=None, limit=limit)
            except Exception:  # noqa: BLE001
                # Gracefully fall back for backends that don't support search.
                keys = await self.store.list_keys(ns, limit=limit)
                items = []
                for k in keys:
                    raw = await self.store.get(ns, k)
                    if raw is not None:
                        from datetime import UTC, datetime  # noqa: PLC0415

                        from locus.memory.store import StoreItem  # noqa: PLC0415

                        now = datetime.now(UTC)
                        items.append(
                            StoreItem(
                                namespace=ns,
                                key=k,
                                value=raw,
                                metadata={},
                                created_at=now,
                                updated_at=now,
                            )
                        )

            for item in items:
                try:
                    memories.append(Memory.from_store_value(item.value))
                except (KeyError, ValueError):
                    pass

        # Sort newest first by updated_at (best-effort — not all items carry it).
        memories.sort(
            key=lambda m: m.metadata.get("updated_at", ""),
            reverse=True,
        )
        return memories[:limit]

    async def save(self, memories: list[Memory]) -> None:
        """Upsert memories into the backing store.

        Memories with the same ``key`` and ``type`` overwrite the
        previous entry — no duplicates accumulate.
        """
        from datetime import UTC, datetime  # noqa: PLC0415

        now = datetime.now(UTC).isoformat()

        for memory in memories:
            ns = self._ns(memory.type)
            value = memory.to_store_value()
            value["metadata"]["updated_at"] = now

            await self.store.put(
                ns,
                memory.key,
                value,
                metadata={"type": memory.type.value, "updated_at": now},
            )

    def __repr__(self) -> str:
        return (
            f"LLMMemoryManager("
            f"store={type(self.store).__name__}, "
            f"namespace_prefix={self.namespace_prefix!r}, "
            f"retrieve_limit={self.retrieve_limit})"
        )


# =============================================================================
# Helpers
# =============================================================================


def _inject_memories_into_state(
    state: AgentState,
    memories: list[Memory],
) -> AgentState:
    """Prepend a formatted memory block to state.messages.

    Inserts a new system message immediately after the first system
    prompt (position 1), or at position 0 when there is no system
    prompt.  This keeps the primary system prompt intact and first,
    while the memory block follows it.
    """
    from locus.core.messages import Message, Role  # noqa: PLC0415

    block = _format_memory_block(memories)
    memory_msg = Message(role=Role.SYSTEM, content=block)

    msgs = list(state.messages)
    if msgs and msgs[0].role == Role.SYSTEM:
        msgs.insert(1, memory_msg)
    else:
        msgs.insert(0, memory_msg)

    return state.model_copy(update={"messages": tuple(msgs)})


def _format_memory_block(memories: list[Memory]) -> str:
    """Format a list of memories as a compact system-prompt block."""
    lines = ["[Long-term Memory]"]
    for m in memories:
        label = m.type.value.upper()
        lines.append(f"{label} [{m.key}]: {m.content}")
    return "\n".join(lines)


def _heuristic_extract(messages: list[Message]) -> list[Memory]:
    """Cheap pattern-based extractor — no LLM required.

    Recognises common conversational signals:

    * Corrections (``"don't"``, ``"avoid"``, ``"stop"``) → feedback
    * Confirmations (``"exactly"``, ``"perfect"``, ``"yes"``) → feedback
    * Role disclosures (``"I'm a"``, ``"I work on"``) → user
    * Deadline / goal sentences → project
    * URLs and service names → reference

    This is intentionally conservative: it is better to miss a memory
    than to store noise.  Pass a proper ``extract_fn`` for richer recall.
    """
    from locus.core.messages import Role  # noqa: PLC0415

    memories: list[Memory] = []

    feedback_signals = ("don't", "avoid", "stop doing", "never ", "please don't", "do not")
    confirm_signals = ("exactly", "perfect", "yes exactly", "that's right", "keep doing")
    user_signals = ("i'm a ", "i am a ", "i work on", "i've been", "i have been")
    project_signals = (
        "we're working on",
        "we need to",
        "the goal is",
        "deadline",
        "by friday",
        "by monday",
    )
    ref_signals = ("http://", "https://", "jira", "confluence", "slack", "grafana", "dashboard at")

    for msg in messages:
        if not msg.content:
            continue

        text = msg.content.lower()
        role = msg.role

        if role == Role.USER:
            for sig in user_signals:
                if sig in text:
                    key = f"user_context_{uuid.uuid4().hex[:8]}"
                    memories.append(
                        Memory(
                            type=MemoryType.USER,
                            key=key,
                            content=msg.content[:300],
                            metadata={"source": "heuristic"},
                        )
                    )
                    break

            for sig in project_signals:
                if sig in text:
                    key = f"project_context_{uuid.uuid4().hex[:8]}"
                    memories.append(
                        Memory(
                            type=MemoryType.PROJECT,
                            key=key,
                            content=msg.content[:300],
                            metadata={"source": "heuristic"},
                        )
                    )
                    break

            for sig in ref_signals:
                if sig in text:
                    key = f"reference_{uuid.uuid4().hex[:8]}"
                    memories.append(
                        Memory(
                            type=MemoryType.REFERENCE,
                            key=key,
                            content=msg.content[:300],
                            metadata={"source": "heuristic"},
                        )
                    )
                    break

        if role == Role.USER:
            for sig in feedback_signals:
                if sig in text:
                    key = f"feedback_{uuid.uuid4().hex[:8]}"
                    memories.append(
                        Memory(
                            type=MemoryType.FEEDBACK,
                            key=key,
                            content=msg.content[:300],
                            metadata={"source": "heuristic", "signal": "correction"},
                        )
                    )
                    break

            for sig in confirm_signals:
                if sig in text:
                    key = f"feedback_confirmed_{uuid.uuid4().hex[:8]}"
                    memories.append(
                        Memory(
                            type=MemoryType.FEEDBACK,
                            key=key,
                            content=msg.content[:300],
                            metadata={"source": "heuristic", "signal": "confirmation"},
                        )
                    )
                    break

    return memories
