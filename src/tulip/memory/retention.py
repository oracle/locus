# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Message retention: a thread that is in use every day still forgets old turns.

A checkpointer's ``vacuum`` deletes threads nobody has written to for a while.
That does nothing for the thread somebody talks to every day: each new turn
loads the whole conversation, appends to it and saves it again, so an active
thread grows forever and keeps everything ever said in it.

Retention works per message instead. When a message is first checkpointed it
is stamped with the time (``Message.metadata["tulip_at"]``, which no provider
is sent), and a save drops the *exchanges* — a user message and everything up
to the next one — that began before the cut-off. Whole exchanges only, so a
tool result never outlives the call it answers. System messages (the agent's
prompt, a compaction summary) are never dropped, and a message with no stamp is
kept: its age is not known, and retention never guesses.

Two ways to use it:

- :class:`RetainedCheckpointer` wraps any checkpointer — the local default
  ``MemoryCheckpointer`` / ``FileCheckpointer`` as much as a database — and
  trims on every save::

      from datetime import timedelta
      from tulip.memory.backends import FileCheckpointer
      from tulip.memory.retention import RetainedCheckpointer

      checkpointer = RetainedCheckpointer(
          FileCheckpointer("./threads"), max_age=timedelta(days=30)
      )

- :class:`~tulip.memory.backends.PgCheckpointer` takes
  ``message_retention=`` itself and adds ``purge_messages()`` for threads
  nobody is writing to.

A message's age counts from the first save that held it. Within one run the
agent's in-memory state is not stamped, so a turn checkpointed several times
is stamped at its first save; the difference is the length of a turn.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from tulip.core.messages import Message, Role
from tulip.memory.checkpointer import BaseCheckpointer


if TYPE_CHECKING:
    from collections.abc import Sequence

    from tulip.core.protocols import CheckpointerCapabilities
    from tulip.core.state import AgentState


__all__ = [
    "MESSAGE_TIME_KEY",
    "RetainedCheckpointer",
    "message_time",
    "oldest_message_time",
    "stamp_messages",
    "trim_messages",
]

#: ``Message.metadata`` key holding when the message was first checkpointed
#: (ISO 8601, UTC).
MESSAGE_TIME_KEY = "tulip_at"


def message_time(message: Message) -> datetime | None:
    """When ``message`` was first checkpointed, or ``None`` if it was not stamped."""
    raw = (message.metadata or {}).get(MESSAGE_TIME_KEY)
    if not isinstance(raw, str):
        return None
    try:
        when = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return when if when.tzinfo is not None else when.replace(tzinfo=UTC)


def stamp_messages(state: AgentState, now: datetime | None = None) -> AgentState:
    """``state`` with every unstamped message stamped ``now``; stamped ones are kept as they are."""
    if not any(MESSAGE_TIME_KEY not in (m.metadata or {}) for m in state.messages):
        return state
    at = (now or datetime.now(UTC)).isoformat()
    stamped = tuple(
        m
        if MESSAGE_TIME_KEY in (m.metadata or {})
        else m.model_copy(update={"metadata": {**(m.metadata or {}), MESSAGE_TIME_KEY: at}})
        for m in state.messages
    )
    updated: AgentState = state.model_copy(update={"messages": stamped})
    return updated


def _exchanges(messages: Sequence[Message]) -> list[tuple[int, int]]:
    """``(start, end)`` index ranges of each exchange among the non-system messages.

    An exchange opens at a user message and runs to the next one. Anything
    before the first user message belongs to the first exchange.
    """
    starts = [i for i, m in enumerate(messages) if m.role == Role.USER]
    if not starts:
        return [(0, len(messages))] if messages else []
    starts[0] = 0
    ends = [*starts[1:], len(messages)]
    return list(zip(starts, ends, strict=True))


def trim_messages(state: AgentState, older_than: datetime) -> tuple[AgentState, int]:
    """Drop the leading exchanges that began before ``older_than``.

    Returns the trimmed state and how many messages were dropped. Stops at the
    first exchange that is recent or whose opening message has no stamp.
    System messages stay wherever they are.
    """
    talk = [m for m in state.messages if m.role != Role.SYSTEM]
    dropped_upto = 0
    for start, end in _exchanges(talk):
        opened = message_time(talk[start])
        if opened is None or opened >= older_than:
            break
        dropped_upto = end
    if not dropped_upto:
        return state, 0
    gone = {id(m) for m in talk[:dropped_upto]}
    kept = tuple(m for m in state.messages if id(m) not in gone)
    trimmed: AgentState = state.model_copy(update={"messages": kept})
    return trimmed, dropped_upto


def oldest_message_time(state: AgentState) -> datetime | None:
    """The stamp of the oldest non-system message, or ``None`` when none is stamped."""
    times = [message_time(m) for m in state.messages if m.role != Role.SYSTEM]
    known = [t for t in times if t is not None]
    return min(known) if known else None


class RetainedCheckpointer(BaseCheckpointer):
    """Any checkpointer, with per-message retention on every save.

    Every save stamps the messages that are new and, with ``max_age``, drops
    the exchanges older than that before handing the state to ``inner``. All
    else is ``inner``'s: loading, listing, deleting, its capabilities.

    Args:
        inner: The checkpointer that stores the state.
        max_age: How long an exchange is kept. ``None`` only stamps, so a later
            ``max_age`` (or ``trim_messages``) has the times to go by.
    """

    def __init__(self, inner: BaseCheckpointer, max_age: timedelta | None = None) -> None:
        if max_age is not None and max_age <= timedelta(0):
            raise ValueError("max_age must be positive")
        self.inner = inner
        self.max_age = max_age

    @property
    def capabilities(self) -> CheckpointerCapabilities:
        return self.inner.capabilities

    @property
    def deletes_single_checkpoints(self) -> bool:
        return self.inner.deletes_single_checkpoints

    def retain(self, state: AgentState, now: datetime | None = None) -> AgentState:
        """``state`` as it would be saved now: stamped, and trimmed to ``max_age``."""
        now = now or datetime.now(UTC)
        state = stamp_messages(state, now)
        if self.max_age is not None:
            state, _ = trim_messages(state, now - self.max_age)
        return state

    async def save(
        self,
        state: AgentState,
        thread_id: str,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return await self.inner.save(self.retain(state), thread_id, checkpoint_id, metadata)

    async def load(self, thread_id: str, checkpoint_id: str | None = None) -> AgentState | None:
        return await self.inner.load(thread_id, checkpoint_id)

    async def list_checkpoints(self, thread_id: str, limit: int = 10) -> list[str]:
        return await self.inner.list_checkpoints(thread_id, limit)

    async def delete(self, thread_id: str, checkpoint_id: str | None = None) -> bool:
        return await self.inner.delete(thread_id, checkpoint_id)

    async def exists(self, thread_id: str, checkpoint_id: str | None = None) -> bool:
        return await self.inner.exists(thread_id, checkpoint_id)

    async def vacuum(self, older_than_days: int = 30) -> int:
        return await self.inner.vacuum(older_than_days)

    async def list_threads(self, limit: int = 100, pattern: str = "*") -> list[str]:
        return await self.inner.list_threads(limit=limit, pattern=pattern)

    async def close(self) -> None:
        await self.inner.close()

    def __repr__(self) -> str:
        return f"RetainedCheckpointer({self.inner!r}, max_age={self.max_age!r})"
