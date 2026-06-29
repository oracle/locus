# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Coverage tests for ``tulip.memory.manager``.

Targets the branches the end-to-end suite skips: the empty-extraction
early return in ``on_session_end``, the ``retrieve`` fallback path for
stores whose ``search`` raises, and the empty-message / confirmation
arms of the heuristic extractor.
"""

from __future__ import annotations

from typing import Any

from tulip.core.messages import Message, Role
from tulip.core.state import AgentState
from tulip.memory.manager import (
    LLMMemoryManager,
    MemoryType,
    _heuristic_extract,
)
from tulip.memory.store import InMemoryStore


# ---------------------------------------------------------------------------
# on_session_end early return
# ---------------------------------------------------------------------------


async def test_on_session_end_returns_when_nothing_extracted() -> None:
    store = InMemoryStore()
    mgr = LLMMemoryManager(store=store)  # heuristic extractor, no signals below
    state = AgentState(messages=(Message(role=Role.USER, content="hello there friend"),))

    await mgr.on_session_end(state)

    # Nothing was extracted, so nothing was saved.
    assert await mgr.retrieve() == []


# ---------------------------------------------------------------------------
# retrieve() fallback when the store does not support search
# ---------------------------------------------------------------------------


class _SearchlessStore:
    """Duck-typed store whose ``search`` raises, forcing the list+get fallback."""

    async def search(self, ns: tuple[str, ...], query: Any = None, limit: int = 20) -> Any:
        raise RuntimeError("search not supported by this backend")

    async def list_keys(self, ns: tuple[str, ...], limit: int = 100) -> list[str]:
        return ["good", "bad", "missing"]

    async def get(self, ns: tuple[str, ...], key: str) -> Any:
        if key == "good":
            return {
                "type": "user",
                "key": "good",
                "content": "a durable fact",
                "metadata": {"updated_at": "2026-01-01T00:00:00+00:00"},
            }
        if key == "bad":
            return {"type": "user"}  # missing key/content → KeyError on deserialise
        return None  # exercises the ``raw is None`` branch

    async def put(self, *args: Any, **kwargs: Any) -> None:
        return None


async def test_retrieve_falls_back_to_list_and_get() -> None:
    mgr = LLMMemoryManager(store=_SearchlessStore())  # type: ignore[arg-type]
    result = await mgr.retrieve(limit=5)
    assert result
    # Only the well-formed "good" record survives deserialisation.
    assert all(m.key == "good" for m in result)


# ---------------------------------------------------------------------------
# Heuristic extractor edge arms
# ---------------------------------------------------------------------------


def test_heuristic_skips_empty_messages() -> None:
    # A message with no content must be skipped without raising.
    memories = _heuristic_extract([Message(role=Role.USER)])
    assert memories == []


def test_heuristic_detects_confirmation_signal() -> None:
    msgs = [
        Message(role=Role.USER),  # empty → skipped
        Message(role=Role.USER, content="Yes, exactly! That's perfect."),
    ]
    memories = _heuristic_extract(msgs)
    assert any(
        m.type == MemoryType.FEEDBACK and m.metadata.get("signal") == "confirmation"
        for m in memories
    )
