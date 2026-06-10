# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Mem0-backed memory manager.

Delegates long-term memory to the open-source `mem0 <https://github.com/mem0ai/mem0>`_
library, which handles fact extraction, deduplication, and scoped
retrieval over its own (self-hostable) vector store. This is the
vendor-neutral, self-hostable agent-memory backend.

mem0 couples extraction and storage in its ``add`` call, so this manager
overrides :meth:`on_session_end` to hand mem0 the raw conversation and
lets mem0 decide what to remember. Retrieval maps mem0's results back to
Tulip :class:`~tulip.memory.manager.Memory` objects for injection at
session start.

Usage::

    from tulip.memory.managers import Mem0MemoryManager

    manager = Mem0MemoryManager(user_id="alice")
    agent = Agent(model="anthropic:claude-sonnet-4-6", memory_manager=manager)

By default mem0 builds its own OpenAI-backed LLM + embedder + local
vector store; pass ``config=`` to point it at other backends (a fully
a self-hosted OpenAI-compatible stack, for instance).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tulip.memory.manager import BaseMemoryManager, Memory, MemoryType


if TYPE_CHECKING:
    from tulip.core.messages import Message


class Mem0MemoryManager(BaseMemoryManager):
    """Memory manager backed by mem0.

    Args:
        client: A pre-built mem0 async memory client (anything exposing
            async ``add`` / ``search`` / ``get_all``). When ``None``, one
            is created lazily from ``config``.
        config: mem0 configuration dict passed to ``AsyncMemory.from_config``.
            When both ``client`` and ``config`` are ``None``, a default
            ``AsyncMemory()`` is created (OpenAI-backed).
        user_id / agent_id / run_id: mem0 scope identifiers. At least one
            (typically ``user_id``) should be set so memories are scoped
            to the right subject.
        search_query: Optional query used by :meth:`retrieve`. When
            ``None``, retrieval lists all memories in scope via
            ``get_all`` instead of a similarity search.
        retrieve_limit: Maximum memories returned by :meth:`retrieve`.
    """

    def __init__(
        self,
        *,
        client: Any = None,
        config: dict[str, Any] | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        search_query: str | None = None,
        retrieve_limit: int = 20,
    ) -> None:
        self._client_override = client
        self._config = config
        self.user_id = user_id
        self.agent_id = agent_id
        self.run_id = run_id
        self.search_query = search_query
        self.retrieve_limit = retrieve_limit
        self._client: Any = None

    def _scope(self) -> dict[str, str]:
        scope: dict[str, str] = {}
        if self.user_id is not None:
            scope["user_id"] = self.user_id
        if self.agent_id is not None:
            scope["agent_id"] = self.agent_id
        if self.run_id is not None:
            scope["run_id"] = self.run_id
        return scope

    def _get_client(self) -> Any:
        if self._client_override is not None:
            return self._client_override
        if self._client is not None:
            return self._client
        try:
            from mem0 import AsyncMemory  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                'mem0ai is not installed. Install with: pip install "tulip-agents[mem0]"'
            ) from e
        self._client = AsyncMemory.from_config(self._config) if self._config else AsyncMemory()
        return self._client

    @staticmethod
    def _to_mem0_messages(messages: list[Message]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for m in messages:
            role = getattr(m.role, "value", m.role)
            content = m.content
            if isinstance(content, str) and content and role in ("user", "assistant"):
                out.append({"role": role, "content": content})
        return out

    @staticmethod
    def _result_rows(response: Any) -> list[dict[str, Any]]:
        # mem0 returns either {"results": [...]} or a bare list depending
        # on version — normalise both.
        if isinstance(response, dict):
            rows = response.get("results", [])
            return list(rows) if rows else []
        if isinstance(response, list):
            return response
        return []

    # ------------------------------------------------------------------
    # BaseMemoryManager API
    # ------------------------------------------------------------------

    async def extract(self, messages: list[Message]) -> list[Memory]:
        """mem0 extracts internally during ``add`` — see :meth:`on_session_end`."""
        return []

    async def retrieve(self, limit: int = 20) -> list[Memory]:
        """Retrieve in-scope memories from mem0 for injection."""
        client = self._get_client()
        scope = self._scope()
        top = min(limit, self.retrieve_limit)

        if self.search_query:
            response = await client.search(self.search_query, limit=top, **scope)
        else:
            response = await client.get_all(limit=top, **scope)

        memories: list[Memory] = []
        for row in self._result_rows(response)[:top]:
            text = row.get("memory") or row.get("text") or ""
            if not text:
                continue
            metadata = {k: v for k, v in row.items() if k not in ("memory", "text")}
            memories.append(
                Memory(
                    type=MemoryType.REFERENCE,
                    key=str(row.get("id", text[:48])),
                    content=text,
                    metadata=metadata,
                )
            )
        return memories

    async def save(self, memories: list[Memory]) -> None:
        """Persist explicit memories to mem0 (used for direct ``save`` calls)."""
        if not memories:
            return
        client = self._get_client()
        scope = self._scope()
        for memory in memories:
            await client.add(memory.content, metadata=memory.metadata or None, **scope)

    async def on_session_end(self, state: Any) -> None:
        """Hand the finished conversation to mem0 for extraction + storage."""
        mem0_messages = self._to_mem0_messages(list(state.messages))
        if not mem0_messages:
            return

        client = self._get_client()
        await client.add(mem0_messages, **self._scope())

        from tulip.observability.emit import emit  # noqa: PLC0415

        await emit("memory.manager.extracted", backend="mem0", message_count=len(mem0_messages))
