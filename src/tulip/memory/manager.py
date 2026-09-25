# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Long-term memory manager for Tulip agents.

Extracts durable facts from conversation history, persists them via a
:class:`~tulip.memory.store.BaseStore` backend, and injects relevant
memories into the system prompt at the start of every new session.

Storage layout
--------------
All memories are namespaced in the configured store under::

    (*namespace_prefix, memory_type)  →  key: memory.key  →  value: {content, metadata}

The default prefix is ``("tulip_memory",)``, so memories appear as::

    ("tulip_memory", "user")       →  "preferred_language": ...
    ("tulip_memory", "feedback")   →  "no_db_mocks": ...
    ("tulip_memory", "project")    →  "auth_rewrite": ...
    ("tulip_memory", "reference")  →  "linear_pipeline": ...

Scope memories per user or tenant by setting a richer prefix::

    LLMMemoryManager(store=my_store, namespace_prefix=("tenants", tenant_id))

or — for ONE manager shared by every user of a server, with one global bound
on background extractions and one :meth:`~BaseMemoryManager.drain` — resolve
the prefix per run from its metadata::

    LLMMemoryManager(
        store=my_store,
        namespace_resolver=lambda run: ("users", run.metadata["user_id"]),
    )
    agent.run(prompt, metadata={"user_id": user_id})

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

    from tulip import Agent
    from tulip.memory.manager import LLMMemoryManager, Memory, MemoryType
    from tulip.memory.store import InMemoryStore

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
        model="anthropic:claude-sonnet-4-6",
        memory_manager=LLMMemoryManager(store=store, extract_fn=my_extractor),
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Coroutine, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    from tulip.core.events import RunInfo
    from tulip.core.messages import Message
    from tulip.core.state import AgentState
    from tulip.memory.store import BaseStore, StoreItem


logger = logging.getLogger(__name__)

#: ``Message.metadata`` key that marks the system message a memory manager
#: injected. :func:`_inject_memories_into_state` replaces a tagged message
#: rather than adding another one, so a checkpointed thread carries at most one
#: memory block no matter how many turns it has run.
MEMORY_BLOCK_METADATA_KEY = "tulip_memory_block"

#: Header line inside every block :func:`_format_memory_block` renders. Used to
#: recognise blocks injected before the metadata tag existed (still sitting in
#: checkpoints written by older releases) so they are cleaned up too.
_MEMORY_BLOCK_HEADER = "[Long-term Memory]"

#: How :meth:`BaseMemoryManager.on_session_end` runs extraction. ``"inline"``
#: finishes it before the run ends; ``"background"`` schedules it as a tracked
#: task and returns at once.
ExtractMode = Literal["inline", "background"]

# Callable type for user-supplied extraction functions.
ExtractFn = Callable[
    [list["Message"]],
    Coroutine[Any, Any, list["Memory"]],
]


#: ``(run) -> namespace prefix`` for :class:`LLMMemoryManager`. ``run`` is a
#: :class:`~tulip.core.events.RunInfo` built from the run's state: ``run_id``,
#: ``metadata`` (the run's persisted metadata) and ``agent_name`` (the agent
#: id). ``None`` uses the manager's ``namespace_prefix``.
NamespaceResolver = Callable[["RunInfo"], "tuple[str, ...] | None"]

#: ``(manager, prefix)`` a scoped manager call is running under.
_ACTIVE_NAMESPACE: ContextVar[tuple[object, tuple[str, ...]] | None] = ContextVar(
    "tulip_memory_namespace", default=None
)


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

    async def retrieve_relevant(self, query: str | None, limit: int | None = None) -> list[Memory]:
        """Retrieve the memories most relevant to ``query``.

        Called by :meth:`on_session_start` with the text of the current user
        turn. The default ignores ``query`` and delegates to :meth:`retrieve`,
        so subclasses written against the older ``retrieve(limit)`` contract
        keep working unchanged; managers whose backend can rank by relevance
        override this.

        Args:
            query: Text to rank memories against, usually the latest user
                message. ``None`` means "no query" (recency order).
            limit: Maximum number of memories to return. ``None`` uses the
                manager's default.
        """
        if limit is None:
            return await self.retrieve()
        return await self.retrieve(limit)

    async def on_session_start(self, state: AgentState) -> AgentState:
        """Retrieve memories and inject them into the agent state.

        Called by the agent runtime at the start of every invocation,
        after ``on_before_invocation`` hooks but before the first model
        call. The latest user message is used as the retrieval query, and
        the formatted block *replaces* any block injected by an earlier turn
        of the same (checkpointed) thread instead of being added next to it.

        Args:
            state: Current agent state (just-created or loaded from
                checkpointer).

        Returns:
            Possibly-modified state with memory context injected.
        """
        memories = await self.retrieve_relevant(_latest_user_text(state))
        if not memories:
            # Still drop a block left over from an earlier turn: if the store
            # no longer holds those memories (deleted, erased, expired) they
            # must not keep reaching the model from the checkpoint.
            return _strip_memory_blocks(state)

        from tulip.observability.emit import emit  # noqa: PLC0415

        injected_state = _inject_memories_into_state(state, memories)

        await emit(
            "memory.manager.injected",
            memory_count=len(memories),
            types=[m.type.value for m in memories],
        )

        return injected_state

    #: ``"inline"`` (default) or ``"background"`` — see :meth:`on_session_end`.
    #: A class attribute so managers written before the option existed keep
    #: the inline behaviour without calling a base ``__init__``.
    extract_mode: ExtractMode = "inline"
    #: Background extractions allowed to run at once (across namespaces).
    max_concurrent_extractions: int = 4

    async def on_session_end(self, state: AgentState) -> None:
        """Extract memories from the finished session and save them.

        Called by the agent runtime in the ``finally`` block of every
        invocation, after ``on_after_invocation`` hooks but before the
        final checkpoint.

        With ``extract_mode="inline"`` (the default) extraction finishes before
        this returns, so its latency is part of the run. With
        ``extract_mode="background"`` it is scheduled as a tracked task and
        this returns at once: the run ends — and a streaming consumer sees its
        last event — without waiting for the extractor. Background jobs of one
        namespace run in submission order (two turns of one user never race
        their writes), at most :attr:`max_concurrent_extractions` run at once,
        and :meth:`drain` waits for all of them (call it, or
        ``Agent.drain_memory()``, at graceful shutdown and in tests).

        The injected memory block is removed before extraction, so recalled
        memories are not fed back to the extractor as if the user had just
        said them. A failing extractor or store is logged and reported as a
        ``memory.manager.extract_failed`` event rather than raised, in either
        mode: an auxiliary memory write must not cost the conversation its
        turn (inline), and must never surface in a run that already finished
        (background).

        Args:
            state: Final agent state with the complete message history.
        """
        messages = list(_strip_memory_blocks(state).messages)
        if self.extract_mode == "background":
            self._background().submit(
                self._extraction_order_key(), lambda: self._extract_and_save(messages)
            )
            return
        await self._extract_and_save(messages)

    async def _extract_and_save(self, messages: list[Message]) -> None:
        """Run :meth:`extract` + :meth:`save`, reporting instead of raising."""
        from tulip.observability.emit import emit  # noqa: PLC0415

        try:
            memories = await self.extract(messages)
            if not memories:
                return
            await self.save(memories)
        except Exception as exc:  # noqa: BLE001
            logger.warning("long-term memory extraction failed", exc_info=True)
            await emit(
                "memory.manager.extract_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                mode=self.extract_mode,
            )
            return

        await emit(
            "memory.manager.extracted",
            memory_count=len(memories),
            types=[m.type.value for m in memories],
            keys=[m.key for m in memories],
        )

    def _extraction_order_key(self) -> tuple[str, ...]:
        """Background jobs sharing this key run strictly one after another."""
        prefix = getattr(self, "namespace_prefix", None)
        return tuple(prefix) if prefix else ()

    def _background(self) -> _BackgroundExtractions:
        bg: _BackgroundExtractions | None = self.__dict__.get("_bg_extractions")
        if bg is None:
            bg = _BackgroundExtractions(self.max_concurrent_extractions)
            self.__dict__["_bg_extractions"] = bg
        return bg

    @property
    def pending_extractions(self) -> int:
        """Background extractions scheduled or running, not yet finished."""
        bg: _BackgroundExtractions | None = self.__dict__.get("_bg_extractions")
        return 0 if bg is None else len(bg.tasks)

    async def drain(self) -> None:
        """Wait for every scheduled background extraction to finish.

        Jobs submitted while draining are waited for too. Safe to call in
        ``"inline"`` mode (returns at once). Errors inside jobs were already
        logged and emitted; they are not raised here. Bound the wait with
        ``asyncio.timeout(...)``: cancelling ``drain`` leaves the jobs running.
        """
        bg: _BackgroundExtractions | None = self.__dict__.get("_bg_extractions")
        if bg is not None:
            await bg.drain()

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
    """Memory manager backed by any :class:`~tulip.memory.store.BaseStore`.

    Extraction uses either a caller-supplied async function
    (``extract_fn``) or a built-in heuristic that pattern-matches
    common conversational signals (corrections, confirmations, role
    disclosures).  Supply ``extract_fn`` for production use; the
    heuristic is adequate for demos and tests.

    Args:
        store: Any :class:`~tulip.memory.store.BaseStore` implementation.
            ``InMemoryStore`` works for development; use
            ``RedisBackend``, ``PostgreSQLBackend``, etc.
            for production.
        extract_fn: Optional async callable
            ``(messages: list[Message]) -> list[Memory]``.  When
            provided, the heuristic is bypassed.
        namespace_prefix: Store namespace prefix.  Scope per user or
            tenant by passing ``("tenants", tenant_id)``.  Default
            ``("tulip_memory",)``.
        max_memories: Hard cap on the total number of memories kept per
            type.  Oldest entries are pruned when the limit is reached.
        retrieve_limit: Maximum memories returned by :meth:`retrieve`.
        extract_mode: ``"inline"`` (default) runs extraction before the run
            ends; ``"background"`` schedules it after the turn's final event
            so a chat never waits for it — see
            :meth:`BaseMemoryManager.on_session_end`. Background mode needs a
            long-lived event loop (a server); call :meth:`drain` before it
            stops. ``Agent.run_sync`` drains for you.
        max_concurrent_extractions: Background extractions allowed to run at
            once. Jobs of one namespace always run one at a time, in order.
        namespace_resolver: ``(run: RunInfo) -> prefix`` choosing the
            namespace prefix PER RUN, from ``run.metadata`` (what
            ``agent.run(..., metadata=)`` passed). Lets one manager serve
            every user: its ``max_concurrent_extractions`` bound and
            :meth:`drain` then cover all of them. ``None`` from the resolver
            uses ``namespace_prefix``. If the resolver raises, that run
            neither reads nor writes memories (it never falls back to a
            shared namespace).

    Example::

        from tulip.memory.manager import LLMMemoryManager
        from tulip.memory.store import InMemoryStore

        manager = LLMMemoryManager(
            store=InMemoryStore(),
            extract_fn=my_llm_extractor,
            namespace_prefix=("users", user_id),
        )

        agent = Agent(model="anthropic:claude-sonnet-4-6", memory_manager=manager)
    """

    def __init__(
        self,
        store: BaseStore,
        *,
        extract_fn: ExtractFn | None = None,
        namespace_prefix: tuple[str, ...] = ("tulip_memory",),
        max_memories: int = 50,
        retrieve_limit: int = 20,
        extract_mode: ExtractMode = "inline",
        max_concurrent_extractions: int = 4,
        namespace_resolver: NamespaceResolver | None = None,
    ) -> None:
        if extract_mode not in ("inline", "background"):
            raise ValueError(f"extract_mode must be 'inline' or 'background', got {extract_mode!r}")
        if max_concurrent_extractions < 1:
            raise ValueError("max_concurrent_extractions must be at least 1")
        self.extract_mode = extract_mode
        self.max_concurrent_extractions = max_concurrent_extractions
        self.store = store
        self.extract_fn = extract_fn
        self.namespace_prefix = namespace_prefix
        self.max_memories = max_memories
        self.retrieve_limit = retrieve_limit
        self.namespace_resolver = namespace_resolver

    # ------------------------------------------------------------------
    # Namespace scoping
    # ------------------------------------------------------------------

    @property
    def active_namespace(self) -> tuple[str, ...]:
        """The prefix the current call reads and writes under: the one a
        :meth:`scoped` block (or the run's resolver) set, else
        ``namespace_prefix``."""
        active = _ACTIVE_NAMESPACE.get()
        if active is not None and active[0] is self:
            return active[1]
        return tuple(self.namespace_prefix)

    @contextmanager
    def scoped(self, namespace_prefix: tuple[str, ...]) -> Iterator[None]:
        """Run :meth:`retrieve` / :meth:`save` / :meth:`extract` calls — and
        background jobs scheduled — inside the block under
        ``namespace_prefix``.

        Context-local: concurrent tasks each keep their own scope. The agent
        runtime does this for you when ``namespace_resolver`` is set.
        """
        token = _ACTIVE_NAMESPACE.set((self, tuple(namespace_prefix)))
        try:
            yield
        finally:
            _ACTIVE_NAMESPACE.reset(token)

    def _resolve_namespace(self, state: AgentState) -> tuple[str, ...] | None:
        """This run's prefix; ``None`` means the resolver failed (fail closed)."""
        if self.namespace_resolver is None:
            return tuple(self.namespace_prefix)
        from tulip.core.events import RunInfo  # noqa: PLC0415

        run = RunInfo.build(
            run_id=state.run_id,
            thread_id=None,
            metadata=state.metadata,
            agent_name=state.agent_id,
        )
        try:
            resolved = self.namespace_resolver(run)
        except Exception:  # noqa: BLE001 — a bad resolver must not cost the turn
            logger.warning(
                "memory namespace_resolver failed; skipping memory for run %s",
                state.run_id,
                exc_info=True,
            )
            return None
        if resolved is None:
            return tuple(self.namespace_prefix)
        return tuple(resolved)

    async def on_session_start(self, state: AgentState) -> AgentState:
        """Inject this run's memories, read from the run's namespace."""
        namespace = self._resolve_namespace(state)
        if namespace is None:
            return _strip_memory_blocks(state)
        with self.scoped(namespace):
            return await super().on_session_start(state)

    async def on_session_end(self, state: AgentState) -> None:
        """Extract and save this run's memories into the run's namespace.

        A background job is created inside the scope, so it inherits it; it
        shares this manager's one semaphore and one :meth:`drain` with every
        other namespace's jobs.
        """
        namespace = self._resolve_namespace(state)
        if namespace is None:
            return
        with self.scoped(namespace):
            await super().on_session_end(state)

    def _extraction_order_key(self) -> tuple[str, ...]:
        return self.active_namespace

    def _ns(self, memory_type: MemoryType) -> tuple[str, ...]:
        """Build the store namespace for a memory type."""
        return (*self.active_namespace, memory_type.value)

    async def extract(self, messages: list[Message]) -> list[Memory]:
        """Extract memories from a message list.

        Uses ``extract_fn`` when provided; otherwise applies the
        built-in heuristic.
        """
        if self.extract_fn is not None:
            return await self.extract_fn(messages)
        return _heuristic_extract(messages)

    async def retrieve(
        self,
        limit: int | None = None,
        *,
        query: str | None = None,
    ) -> list[Memory]:
        """Retrieve stored memories across every type.

        Without ``query`` this returns the most recently updated memories.
        With ``query`` it first asks the store to rank each type's memories
        against it (``BaseStore.search(namespace, query=...)`` — semantic on
        ``PgMemory`` / ``HolographicStore``), interleaves the best matches of
        every type, then tops the list up with the most recent memories. The
        top-up is what keeps a store whose ``search`` only *filters* (the
        substring match of ``InMemoryStore``) from injecting nothing just
        because no memory contains the whole user message. Stores that cannot
        search at all fall back to recency.

        Args:
            limit: Maximum memories returned. ``None`` uses
                ``retrieve_limit``.
            query: Optional text to rank memories against.
        """
        top = self.retrieve_limit if limit is None else limit
        if top <= 0:
            return []

        ranked: list[Memory] = []
        if query and query.strip():
            per_type: list[list[Memory]] = []
            for memory_type in MemoryType:
                try:
                    items = await self.store.search(self._ns(memory_type), query=query, limit=top)
                except Exception:  # noqa: BLE001
                    # No (or failing) query search: recency below covers it.
                    items = []
                per_type.append(_memories_from_items(items))
            # Scores are not comparable across separate searches, so merge
            # the per-type rankings rank-by-rank instead of by score.
            for rank in range(max((len(r) for r in per_type), default=0)):
                ranked.extend(r[rank] for r in per_type if rank < len(r))

        recent = await self._recent(top)
        return _dedupe(ranked + recent)[:top]

    async def _recent(self, limit: int) -> list[Memory]:
        """Up to ``limit`` memories across every type, newest first."""
        memories: list[Memory] = []
        for memory_type in MemoryType:
            memories.extend(_memories_from_items(await self._list_items(memory_type, limit)))
        # Sort newest first by updated_at (best-effort — not all items carry it).
        memories.sort(
            key=lambda m: m.metadata.get("updated_at", ""),
            reverse=True,
        )
        return memories[:limit]

    async def _list_items(self, memory_type: MemoryType, limit: int) -> list[StoreItem]:
        """List a type's items, via ``search`` or ``list_keys`` + ``get``."""
        ns = self._ns(memory_type)
        try:
            return await self.store.search(ns, query=None, limit=limit)
        except Exception:  # noqa: BLE001
            # Gracefully fall back for backends that don't support search.
            from datetime import UTC, datetime  # noqa: PLC0415

            from tulip.memory.store import StoreItem  # noqa: PLC0415

            items: list[StoreItem] = []
            for k in await self.store.list_keys(ns, limit=limit):
                raw = await self.store.get(ns, k)
                if raw is not None:
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
            return items

    async def retrieve_relevant(self, query: str | None, limit: int | None = None) -> list[Memory]:
        """Rank stored memories against ``query`` — see :meth:`retrieve`."""
        return await self.retrieve(limit, query=query)

    async def save(self, memories: list[Memory]) -> None:
        """Upsert memories into the backing store.

        Memories with the same ``key`` and ``type`` overwrite the
        previous entry — no duplicates accumulate. Afterwards each touched
        type is pruned to ``max_memories`` entries, oldest first.
        """
        from datetime import UTC, datetime  # noqa: PLC0415

        now = datetime.now(UTC).isoformat()

        touched: set[MemoryType] = set()
        for memory in memories:
            ns = self._ns(memory.type)
            value = memory.to_store_value()
            # Copy, so stamping ``updated_at`` doesn't mutate the caller's Memory.
            value["metadata"] = {**value["metadata"], "updated_at": now}

            await self.store.put(
                ns,
                memory.key,
                value,
                metadata={"type": memory.type.value, "updated_at": now},
            )
            touched.add(memory.type)

        for memory_type in touched:
            await self._prune(memory_type)

    async def _prune(self, memory_type: MemoryType) -> None:
        """Delete a type's oldest memories beyond ``max_memories``."""
        if self.max_memories <= 0:
            return
        ns = self._ns(memory_type)
        keys = await self.store.list_keys(ns, limit=_PRUNE_SCAN_LIMIT)
        if len(keys) <= self.max_memories:
            return
        stamped: list[tuple[str, str]] = []
        for key in keys:
            raw = await self.store.get(ns, key)
            updated = ""
            if isinstance(raw, dict):
                meta = raw.get("metadata")
                if isinstance(meta, dict):
                    updated = str(meta.get("updated_at", ""))
            stamped.append((updated, key))
        stamped.sort(reverse=True)
        for _, key in stamped[self.max_memories :]:
            await self.store.delete(ns, key)

    def __repr__(self) -> str:
        return (
            f"LLMMemoryManager("
            f"store={type(self.store).__name__}, "
            f"namespace_prefix={self.namespace_prefix!r}, "
            f"namespace_resolver={'set' if self.namespace_resolver else None}, "
            f"retrieve_limit={self.retrieve_limit}, "
            f"extract_mode={self.extract_mode!r})"
        )


# =============================================================================
# Helpers
# =============================================================================


#: Upper bound on keys scanned when pruning a memory type to ``max_memories``.
_PRUNE_SCAN_LIMIT = 10_000


class _BackgroundExtractions:
    """Tracked background extraction jobs of one memory manager.

    * every task is held until done (never garbage-collected mid-flight);
    * a job waits for the previous job with the same key before it starts, so
      one namespace's writes land in submission order;
    * a semaphore bounds how many jobs run at once — taken only after the
      predecessor finished, so a queued job never holds a slot while waiting;
    * a job never raises: ``_extract_and_save`` reports its own failures, and
      anything else is logged here.
    """

    def __init__(self, max_concurrent: int) -> None:
        self._max_concurrent = max_concurrent
        self._loop: asyncio.AbstractEventLoop | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._tails: dict[tuple[str, ...], asyncio.Task[None]] = {}
        self.tasks: set[asyncio.Task[None]] = set()

    def submit(self, key: tuple[str, ...], job: Callable[[], Awaitable[None]]) -> None:
        loop = asyncio.get_running_loop()
        if loop is not self._loop:
            # A new event loop (``run_sync`` opens one per call): asyncio
            # primitives and tasks of the old loop cannot be used from it.
            self._loop = loop
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
            self._tails = {}
            self.tasks = set()
        previous = self._tails.get(key)
        task = loop.create_task(
            self._run(previous, job), name=f"tulip-memory-extract:{'/'.join(key)}"
        )
        self._tails[key] = task
        self.tasks.add(task)

        def _done(t: asyncio.Task[None]) -> None:
            self.tasks.discard(t)
            if self._tails.get(key) is t:
                del self._tails[key]

        task.add_done_callback(_done)

    async def _run(
        self, previous: asyncio.Task[None] | None, job: Callable[[], Awaitable[None]]
    ) -> None:
        if previous is not None:
            # The predecessor never raises; ``wait`` also shields this job
            # from the predecessor being cancelled.
            await asyncio.wait({previous})
        assert self._semaphore is not None
        async with self._semaphore:
            try:
                await job()
            except Exception:  # noqa: BLE001 — a background job must never raise
                logger.warning("background memory extraction failed", exc_info=True)

    async def drain(self) -> None:
        while self.tasks:
            await asyncio.wait(set(self.tasks))


def _is_memory_block(message: Message) -> bool:
    """Whether ``message`` is a memory block a manager injected."""
    from tulip.core.messages import Role  # noqa: PLC0415

    if message.role != Role.SYSTEM:
        return False
    if message.metadata.get(MEMORY_BLOCK_METADATA_KEY):
        return True
    # Untagged blocks written by releases before the tag existed.
    content = message.content or ""
    return content.startswith("<memory-context>") and _MEMORY_BLOCK_HEADER in content


def without_memory_blocks(state: AgentState) -> AgentState:
    """``state`` minus any injected memory block — the form that is persisted.

    The memory block is ephemeral: a manager injects it for the model calls of
    one turn, and the runtime strips it before every checkpoint save, so a
    checkpoint (and the run's result state) never carries recalled memory.
    """
    return _strip_memory_blocks(state)


def _strip_memory_blocks(state: AgentState) -> AgentState:
    """Return ``state`` without any injected memory block."""
    msgs = tuple(m for m in state.messages if not _is_memory_block(m))
    if len(msgs) == len(state.messages):
        return state
    return state.model_copy(update={"messages": msgs})


def _latest_user_text(state: AgentState) -> str | None:
    """Text of the most recent user message — the current turn's prompt."""
    from tulip.core.messages import Role  # noqa: PLC0415

    for message in reversed(state.messages):
        if message.role == Role.USER and message.content:
            return message.content
    return None


def _memories_from_items(items: list[StoreItem]) -> list[Memory]:
    """Decode store items into memories, skipping malformed values."""
    memories: list[Memory] = []
    for item in items:
        try:
            memories.append(Memory.from_store_value(item.value))
        except (KeyError, ValueError, TypeError):
            pass
    return memories


def _dedupe(memories: list[Memory]) -> list[Memory]:
    """Drop repeated ``(type, key)`` entries, keeping the first."""
    seen: set[tuple[str, str]] = set()
    out: list[Memory] = []
    for m in memories:
        ident = (m.type.value, m.key)
        if ident not in seen:
            seen.add(ident)
            out.append(m)
    return out


def _inject_memories_into_state(
    state: AgentState,
    memories: list[Memory],
) -> AgentState:
    """Place a formatted memory block in state.messages, replacing any old one.

    Inserts a system message immediately after the first system prompt
    (position 1), or at position 0 when there is no system prompt. This
    keeps the primary system prompt intact and first, while the memory
    block follows it. Any memory block already present — typically
    injected by an earlier turn and persisted in the thread's checkpoint —
    is removed first, so the state never carries more than one.
    """
    from tulip.core.messages import Message, Role  # noqa: PLC0415

    block = _format_memory_block(memories)
    memory_msg = Message(
        role=Role.SYSTEM,
        content=block,
        metadata={MEMORY_BLOCK_METADATA_KEY: True},
    )

    msgs = list(_strip_memory_blocks(state).messages)
    if msgs and msgs[0].role == Role.SYSTEM:
        msgs.insert(1, memory_msg)
    else:
        msgs.insert(0, memory_msg)

    return state.model_copy(update={"messages": tuple(msgs)})


def _format_memory_block(memories: list[Memory]) -> str:
    """Format memories as a scrubbed, untrusted-tagged system-prompt block.

    Recalled memory is a prompt-injection surface (a fact written in one run, or
    by a poisoned document, could carry instructions into a later run), so the
    block is passed through :func:`build_memory_context_block`: injected
    system-note/fence markers are stripped and the recall is wrapped as
    *informational background data, not instructions*. The model can use what it
    remembers without obeying it.
    """
    from tulip.memory.scrubber import build_memory_context_block  # noqa: PLC0415

    lines = [_MEMORY_BLOCK_HEADER]
    for m in memories:
        label = m.type.value.upper()
        lines.append(f"{label} [{m.key}]: {m.content}")
    return build_memory_context_block("\n".join(lines))


def _content_digest(content: str) -> str:
    """Stable short key suffix for a message's text.

    Heuristic memories are keyed by what was said rather than by a random id:
    on a checkpointed thread every turn re-extracts the whole history, and a
    random key turned each re-extraction into another copy of the same fact.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]


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
    from tulip.core.messages import Role  # noqa: PLC0415

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
                    key = f"user_context_{_content_digest(msg.content)}"
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
                    key = f"project_context_{_content_digest(msg.content)}"
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
                    key = f"reference_{_content_digest(msg.content)}"
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
                    key = f"feedback_{_content_digest(msg.content)}"
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
                    key = f"feedback_confirmed_{_content_digest(msg.content)}"
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
