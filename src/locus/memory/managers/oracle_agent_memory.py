# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Locus ↔ ``oracleagentmemory`` adapter.

`oracleagentmemory <https://pypi.org/project/oracleagentmemory/>`_ is
Oracle's official client for AI agent memory backed by Oracle AI
Database. It ships purpose-built primitives that Locus's home-grown
``LLMMemoryManager + OracleStore`` approximates: durable conversation
threads, scoped retrieval by ``user_id`` / ``agent_id`` / ``thread_id``,
LLM-backed memory extraction, prompt-ready context cards, and
LongMemEval-tuned recall.

This module exposes :class:`OracleAgentMemoryManager` — a thin
adapter that implements Locus's :class:`~locus.memory.manager.BaseMemoryManager`
contract on top of an ``oracleagentmemory.core.OracleAgentMemory``
client. The wins, all without changing the Agent runtime:

- ``on_session_start(state)`` calls the underlying
  ``Thread.get_context_card_async(...)`` for the resolved user/thread
  and injects the rendered card as a system-prompt block — the same
  injection point :class:`LLMMemoryManager` uses, just with a richer
  source.
- ``on_session_end(state)`` ships the conversation transcript to
  ``Thread.add_messages_async(...)`` so oracleagentmemory's LLM-backed
  extractor runs in the background.
- ``extract`` / ``retrieve`` / ``save`` delegate to the same client so
  callers that drive memory directly (outside the agent lifecycle) get
  the same behaviour.

The ``oracleagentmemory`` package is an **optional** dependency
declared in ``pyproject.toml`` as ``[agentmemory]``; this module
imports it lazily so users on the portable ``LLMMemoryManager`` path
never see it.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from locus.core.messages import Message
from locus.memory.manager import (
    BaseMemoryManager,
    Memory,
    MemoryType,
    _inject_memories_into_state,
)


if TYPE_CHECKING:
    from locus.core.state import AgentState


# Resolver callables: given an AgentState, return the scope id.
# Defaults pull from ``state.metadata`` (which the agent runtime
# populates from the ``metadata`` kwarg on ``agent.run_sync(...)``)
# and fall back to ``state.run_id`` so a manager constructed without
# any explicit scoping still works deterministically.
ScopeResolver = Callable[["AgentState"], str]


def _default_thread_id(state: AgentState) -> str:
    value = state.metadata.get("thread_id")
    return str(value) if value else state.run_id


def _default_user_id(state: AgentState) -> str | None:
    value = state.metadata.get("user_id")
    return str(value) if value else None


def _default_agent_id(state: AgentState) -> str | None:
    if state.agent_id:
        return state.agent_id
    value = state.metadata.get("agent_id")
    return str(value) if value else None


class OracleAgentMemoryManager(BaseMemoryManager):
    """``BaseMemoryManager`` backed by Oracle's ``oracleagentmemory`` client.

    Compose with any Locus ``Agent`` via the standard
    ``memory_manager=`` slot::

        from locus.agent import Agent, AgentConfig
        from locus.memory.managers import OracleAgentMemoryManager

        manager = OracleAgentMemoryManager(
            connection=oracledb.connect(
                user="locus_app",
                password=...,
                dsn="mydb_low",
                config_dir="~/.oci/wallets/mydb",
                wallet_location="~/.oci/wallets/mydb",
                wallet_password=...,
            ),
            embedder="oci/cohere.embed-english-v3.0",
            llm="oci/openai.gpt-4.1",
            table_name_prefix="locus_oam_",
        )

        agent = Agent(
            config=AgentConfig(
                model=...,
                memory_manager=manager,
            )
        )

        # Pass user_id (and optional thread_id) via the `metadata` kwarg —
        # the adapter reads them from state.metadata to scope reads + writes.
        agent.run_sync(
            "Hi, I'm Alice — I prefer us-chicago-1.",
            metadata={"user_id": "alice", "thread_id": "t-a"},
        )

    Args:
        connection: A pre-built ``oracledb.Connection`` (sync; the
            ``oracleagentmemory`` runtime drives it through its own
            sync/async helpers). Pass ``store=`` instead if you want
            full control over schema/pool wiring.
        store: An already-constructed
            ``oracleagentmemory.core.OracleMemoryStore``. Mutually
            exclusive with ``connection``.
        embedder: Either a litellm-style model spec (``"oci/..."``,
            ``"openai/..."``, ...) or a custom ``IEmbedder`` instance.
            Defaults to a string that routes to OCI Cohere v3.
        llm: Either a litellm-style model spec or a custom ``ILlm``
            instance. Required when ``extract_memories=True`` (the
            default in ``oracleagentmemory``).
        extract_memories: Whether the underlying client should run
            its LLM-backed extractor when ``add_messages`` is called.
            Defaults to ``True`` — that's the entire point of using
            oracleagentmemory over the portable ``LLMMemoryManager``.
        table_name_prefix: Prefix for the tables the client provisions
            (``<prefix>messages``, ``<prefix>facts``, ...). Choose a
            value that doesn't collide with Locus's other Oracle
            backends (``locus_store``, ``locus_notebook_*``, ...).
        thread_id_resolver: Callable that returns the scoping
            ``thread_id`` from an :class:`AgentState`. Default:
            ``state.metadata["thread_id"] or state.run_id``.
        user_id_resolver: Callable that returns the scoping
            ``user_id``. Default: ``state.metadata.get("user_id")``.
        agent_id_resolver: Callable that returns the scoping
            ``agent_id``. Default: ``state.agent_id`` or
            ``state.metadata.get("agent_id")``.
        max_context_messages: Cap on how many recent messages we ship
            to ``Thread.add_messages_async`` per session-end.
        max_relevant_results: Forwarded to ``get_context_card_async``
            — number of relevant memories to render into the card.
        max_recent_messages: Forwarded to ``get_context_card_async``
            — number of recent messages to render into the card.
    """

    def __init__(
        self,
        *,
        connection: Any = None,
        store: Any = None,
        embedder: Any = "oci/cohere.embed-english-v3.0",
        llm: Any = "oci/openai.gpt-4.1",
        extract_memories: bool = True,
        table_name_prefix: str = "locus_oam_",
        schema_policy: Any = None,
        thread_id_resolver: ScopeResolver = _default_thread_id,
        user_id_resolver: Callable[[AgentState], str | None] = _default_user_id,
        agent_id_resolver: Callable[[AgentState], str | None] = _default_agent_id,
        max_context_messages: int = 50,
        max_relevant_results: int = 5,
        max_recent_messages: int = 5,
        thread_defaults: dict[str, Any] | None = None,
    ) -> None:
        if connection is None and store is None:
            raise ValueError(
                "OracleAgentMemoryManager requires either `connection` (a sync "
                "oracledb.Connection) or `store` (a pre-built "
                "oracleagentmemory OracleMemoryStore)."
            )
        if connection is not None and store is not None:
            raise ValueError(
                "OracleAgentMemoryManager accepts `connection` OR `store`, not both. "
                "Pass `store` when you want full control over schema/pool wiring; "
                "pass `connection` for the default one-shot setup."
            )

        # Lazy import so the optional dep stays optional. Without it
        # this module imports fine — instantiation is what fails.
        try:
            from oracleagentmemory.core import OracleAgentMemory  # noqa: PLC0415
            from oracleagentmemory.core.embedders import Embedder  # noqa: PLC0415
            from oracleagentmemory.core.llms import Llm  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - import-error path
            raise ImportError(
                "OracleAgentMemoryManager requires the `oracleagentmemory` "
                "package. Install it with: pip install 'locus-sdk[agentmemory]' "
                "(or `pip install oracleagentmemory>=26.4.0,<27` directly)."
            ) from exc

        # ``OracleAgentMemory`` auto-wraps a string ``embedder`` but
        # NOT a string ``llm`` — passing a litellm-style model spec for
        # ``llm`` blows up later inside the extractor with
        # "'str' object has no attribute 'generate_async'". Wrap both
        # eagerly so the caller-friendly string shape just works.
        if isinstance(embedder, str):
            embedder = Embedder(model=embedder)
        if isinstance(llm, str):
            llm = Llm(model=llm)

        ctor_kwargs: dict[str, Any] = {
            "embedder": embedder,
            "llm": llm,
            "extract_memories": extract_memories,
            "table_name_prefix": table_name_prefix,
        }
        if connection is not None:
            ctor_kwargs["connection"] = connection
        if store is not None:
            ctor_kwargs["store"] = store
        if schema_policy is not None:
            ctor_kwargs["schema_policy"] = schema_policy

        self._client = OracleAgentMemory(**ctor_kwargs)
        self._thread_id_resolver = thread_id_resolver
        self._user_id_resolver = user_id_resolver
        self._agent_id_resolver = agent_id_resolver
        self._max_context_messages = max_context_messages
        self._max_relevant_results = max_relevant_results
        self._max_recent_messages = max_recent_messages
        # Per-thread defaults forwarded to ``create_thread`` — e.g.
        # ``memory_extraction_frequency=1`` to force extraction on every
        # add_messages call rather than waiting for the upstream's
        # default windowing.
        self._thread_defaults: dict[str, Any] = dict(thread_defaults or {})

        # Per-state cache of (thread_id -> OracleThread). The agent
        # runtime calls on_session_start then on_session_end for the
        # same state, and we want both halves to operate on the same
        # Thread handle.
        self._thread_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # BaseMemoryManager surface
    # ------------------------------------------------------------------

    async def extract(self, messages: list[Message]) -> list[Memory]:
        """No-op for direct callers — the underlying client extracts
        inside :meth:`on_session_end` via ``add_messages_async``. We
        return an empty list so :meth:`save` does nothing, avoiding a
        double-write."""
        return []

    async def retrieve(self, limit: int = 20) -> list[Memory]:
        """Best-effort top-level retrieval — unscoped.

        Direct callers without an :class:`AgentState` get the most
        recent ``limit`` memories across every scope. For per-user
        scoped retrieval, use :meth:`search_for_state` (which the
        lifecycle hooks call) or query the underlying client directly.
        """
        results = await self._client.search_async(query="*", max_results=limit)
        return [_search_result_to_memory(r) for r in results]

    async def save(self, memories: list[Memory]) -> None:
        """Persist memories explicitly. Called by direct users; the
        agent lifecycle path bypasses this in favour of
        ``Thread.add_messages_async`` (which triggers the LLM
        extractor)."""
        for m in memories:
            scope_kwargs: dict[str, Any] = {}
            meta = m.metadata or {}
            for key in ("user_id", "agent_id", "thread_id"):
                if meta.get(key):
                    scope_kwargs[key] = str(meta[key])
            await self._client.add_memory_async(content=m.content, **scope_kwargs)

    # ------------------------------------------------------------------
    # Agent-lifecycle hooks
    # ------------------------------------------------------------------

    async def on_session_start(self, state: AgentState) -> AgentState:
        """Pull oracleagentmemory context for the current scope and
        inject it as a system-prompt block.

        Two complementary signals:

        1. **Per-thread context card** — recent messages + summary of
           this thread (empty on the very first call to a brand-new
           thread).
        2. **Cross-thread user-scoped memories** — what the LLM
           extractor has stored about this ``user_id`` from prior
           threads. Critical for the "new conversation, same user"
           case: without this, a brand-new thread starts blank even
           though prior threads taught durable facts.
        """
        thread = await self._get_or_create_thread(state)
        user_id = self._user_id_resolver(state)
        agent_id = self._agent_id_resolver(state)

        memories: list[Memory] = []

        # 1. Per-thread context card (best-effort — empty for fresh threads).
        with contextlib.suppress(Exception):
            card = await thread.get_context_card_async(
                max_relevant_results=self._max_relevant_results,
                max_recent_messages=self._max_recent_messages,
            )
            rendered = _render_context_card(card)
            if rendered:
                memories.append(
                    Memory(
                        type=MemoryType.PROJECT,
                        key="oam_context_card",
                        content=rendered,
                        metadata={"source": "oracleagentmemory.context_card"},
                    )
                )

        # 2. Cross-thread search scoped to the user, so a brand-new
        #    thread inherits durable facts the LLM extractor stored
        #    on prior threads for the same person.
        if user_id is not None or agent_id is not None:
            with contextlib.suppress(Exception):
                search_kwargs: dict[str, Any] = {
                    "query": "user profile preferences and role",
                    "max_results": self._max_relevant_results,
                }
                if user_id is not None:
                    search_kwargs["user_id"] = user_id
                if agent_id is not None:
                    search_kwargs["agent_id"] = agent_id
                results = await self._client.search_async(**search_kwargs)
                for r in results:
                    memories.append(_search_result_to_memory(r))

        if not memories:
            return state

        injected = _inject_memories_into_state(state, memories)
        with contextlib.suppress(Exception):  # observability is best-effort
            from locus.observability.emit import emit  # noqa: PLC0415

            await emit(
                "memory.manager.injected",
                memory_count=len(memories),
                types=[m.key for m in memories],
                source="oracleagentmemory",
            )
        return injected

    async def on_session_end(self, state: AgentState) -> None:
        """Ship the conversation to the underlying ``Thread`` so the
        client's extractor can run."""
        if not state.messages:
            return

        thread = await self._get_or_create_thread(state)
        tail = list(state.messages)[-self._max_context_messages :]
        payload = [
            {"role": _normalise_role(m.role), "content": m.content or ""}
            for m in tail
            if (m.content or "").strip()
        ]
        if not payload:
            return

        with contextlib.suppress(Exception):  # extractor errors are non-fatal
            await thread.add_messages_async(payload)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _get_or_create_thread(self, state: AgentState) -> Any:
        thread_id = self._thread_id_resolver(state) or uuid4().hex
        cached = self._thread_cache.get(thread_id)
        if cached is not None:
            return cached

        user_id = self._user_id_resolver(state)
        agent_id = self._agent_id_resolver(state)
        kwargs: dict[str, Any] = {"thread_id": thread_id, **self._thread_defaults}
        if user_id is not None:
            kwargs["user_id"] = user_id
        if agent_id is not None:
            kwargs["agent_id"] = agent_id

        # Run the upstream's sync `get_thread` / `create_thread` in a
        # worker thread. Calling them directly from this async hook
        # trips oracleagentmemory's "asynchronous method in a
        # synchronous method from an asynchronous context" guard at
        # `oracleagentmemory/core/oracleagentmemory.py:853` — the
        # upstream doesn't expose async variants of these two yet
        # (as of 26.4.0), so `asyncio.to_thread` is the clean bridge.
        thread = await asyncio.to_thread(self._sync_get_or_create_thread, thread_id, kwargs)
        self._thread_cache[thread_id] = thread
        return thread

    def _sync_get_or_create_thread(self, thread_id: str, kwargs: dict[str, Any]) -> Any:
        try:
            return self._client.get_thread(thread_id)
        except Exception:  # noqa: BLE001 - thread may not exist yet
            return self._client.create_thread(**kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_role(role: Any) -> str:
    """Map Locus Roles to the role strings oracleagentmemory expects."""
    value = role.value if hasattr(role, "value") else str(role)
    if value in {"user", "assistant", "system", "tool"}:
        return value
    return "user"


def _render_context_card(card: Any) -> str:
    """Turn an OracleContextCard into a system-prompt-friendly string.

    The card object stringifies cleanly; we keep this indirection so
    later versions of oracleagentmemory can change the render shape
    without breaking the injection path.
    """
    if card is None:
        return ""
    for attr in ("text", "content", "rendered", "markdown"):
        value = getattr(card, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    rendered = str(card).strip()
    if rendered and rendered != repr(card):
        return rendered
    return ""


def _search_result_to_memory(result: Any) -> Memory:
    """Translate an oracleagentmemory SearchResult into a Locus Memory.

    SearchResult exposes (at minimum) ``content`` / ``record_type`` /
    ``record_id``. We map record_type onto MemoryType conservatively
    and fall back to ``MemoryType.PROJECT`` for the records that don't
    have a clean Locus equivalent.
    """
    content = getattr(result, "content", None) or str(result)
    record_id = getattr(result, "record_id", None) or getattr(result, "id", None)
    record_type = (getattr(result, "record_type", "") or "").lower()
    mtype_map = {
        "preference": MemoryType.USER,
        "userprofile": MemoryType.USER,
        "user_profile": MemoryType.USER,
        "guideline": MemoryType.FEEDBACK,
        "agentprofile": MemoryType.REFERENCE,
        "agent_profile": MemoryType.REFERENCE,
        "thread": MemoryType.PROJECT,
        "fact": MemoryType.PROJECT,
        "memory": MemoryType.PROJECT,
    }
    return Memory(
        type=mtype_map.get(record_type, MemoryType.PROJECT),
        key=str(record_id or uuid4().hex),
        content=str(content),
        metadata={"source": "oracleagentmemory", "record_type": record_type},
    )
