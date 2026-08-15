# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Mem0 memory manager.

Uses an injected fake async client (``client=``) so the tests run with no
``mem0ai`` dependency and no LLM / network access.
"""

import pytest

from tulip.core.messages import Message
from tulip.memory.manager import Memory, MemoryType
from tulip.memory.managers.mem0 import Mem0MemoryManager


class _FakeMem0Client:
    def __init__(self, get_all_rows=None, search_rows=None):
        self._get_all_rows = get_all_rows or []
        self._search_rows = search_rows or []
        self.added = []

    async def add(self, messages, **kwargs):
        self.added.append((messages, kwargs))
        return {"results": []}

    async def get_all(self, **kwargs):
        return {"results": self._get_all_rows}

    async def search(self, query, **kwargs):
        return {"results": self._search_rows}


def test_scope_built_from_ids():
    mgr = Mem0MemoryManager(client=_FakeMem0Client(), user_id="alice", agent_id="bot")
    assert mgr._scope() == {"user_id": "alice", "agent_id": "bot"}


async def test_retrieve_maps_rows_to_memories():
    rows = [
        {"id": "m1", "memory": "User likes dark mode", "score": 0.9},
        {"id": "m2", "memory": "User is based in NYC"},
    ]
    mgr = Mem0MemoryManager(client=_FakeMem0Client(get_all_rows=rows), user_id="alice")

    memories = await mgr.retrieve()

    assert len(memories) == 2
    assert memories[0].type == MemoryType.REFERENCE
    assert memories[0].key == "m1"
    assert memories[0].content == "User likes dark mode"
    assert memories[0].metadata["score"] == 0.9


async def test_retrieve_uses_search_when_query_set():
    client = _FakeMem0Client(search_rows=[{"id": "s1", "memory": "hit"}])
    mgr = Mem0MemoryManager(client=client, user_id="alice", search_query="prefs")

    memories = await mgr.retrieve()

    assert [m.content for m in memories] == ["hit"]


async def test_on_session_end_adds_conversation():
    client = _FakeMem0Client()
    mgr = Mem0MemoryManager(client=client, user_id="alice")

    class _State:
        messages = [
            Message.user("I prefer Python"),
            Message.assistant("Got it, Python it is"),
        ]

    await mgr.on_session_end(_State())

    assert len(client.added) == 1
    sent_messages, kwargs = client.added[0]
    assert kwargs == {"user_id": "alice"}
    assert sent_messages == [
        {"role": "user", "content": "I prefer Python"},
        {"role": "assistant", "content": "Got it, Python it is"},
    ]


async def test_on_session_end_noop_when_no_messages():
    client = _FakeMem0Client()
    mgr = Mem0MemoryManager(client=client, user_id="alice")

    class _State:
        messages = []

    await mgr.on_session_end(_State())
    assert client.added == []


async def test_save_explicit_memories():
    client = _FakeMem0Client()
    mgr = Mem0MemoryManager(client=client, user_id="alice")

    await mgr.save([Memory(type=MemoryType.REFERENCE, key="k", content="a fact")])

    assert client.added[0][0] == "a fact"


# ---------------------------------------------------------------------------
# Scope handling across mem0 versions.
#
# The fake above accepts **kwargs, so it matches whatever the manager sends.
# Real mem0 does not: current releases moved the entity identifiers behind
# ``filters=`` and *raise* when they arrive top-level, which is how the manager
# came to be broken against a real install while these tests stayed green.
# These two fakes model each real contract instead.
# ---------------------------------------------------------------------------


_ENTITY_KEYS = {"user_id", "agent_id", "run_id"}


class _StrictMem0Client:
    """Current mem0: entity ids must arrive via ``filters=``."""

    def __init__(self, rows=None):
        self._rows = rows or [{"id": "m1", "memory": "postgres"}]
        self.seen_filters = None

    @staticmethod
    def _reject_top_level(kwargs):
        bad = _ENTITY_KEYS & set(kwargs)
        if bad:
            raise ValueError(
                f"Top-level entity parameters {frozenset(bad)} are not supported. "
                "Use filters={'user_id': '...'} instead."
            )

    async def get_all(self, *, filters=None, **kwargs):
        self._reject_top_level(kwargs)
        self.seen_filters = filters
        return {"results": self._rows}

    async def search(self, query, *, filters=None, **kwargs):  # noqa: ARG002
        self._reject_top_level(kwargs)
        self.seen_filters = filters
        return {"results": self._rows}


class _LegacyMem0Client:
    """Older mem0: no ``filters`` parameter at all."""

    def __init__(self, rows=None):
        self._rows = rows or [{"id": "m1", "memory": "postgres"}]
        self.seen_scope = None

    async def get_all(self, **kwargs):
        if "filters" in kwargs:
            raise TypeError("get_all() got an unexpected keyword argument 'filters'")
        self.seen_scope = {k: v for k, v in kwargs.items() if k in _ENTITY_KEYS}
        return {"results": self._rows}

    async def search(self, query, **kwargs):  # noqa: ARG002
        if "filters" in kwargs:
            raise TypeError("search() got an unexpected keyword argument 'filters'")
        self.seen_scope = {k: v for k, v in kwargs.items() if k in _ENTITY_KEYS}
        return {"results": self._rows}


@pytest.mark.asyncio
async def test_get_all_scopes_through_filters_on_current_mem0():
    client = _StrictMem0Client()
    mgr = Mem0MemoryManager(client=client, user_id="alice")

    out = await mgr.retrieve(limit=5)

    assert len(out) == 1
    assert client.seen_filters == {"user_id": "alice"}


@pytest.mark.asyncio
async def test_search_scopes_through_filters_on_current_mem0():
    client = _StrictMem0Client()
    mgr = Mem0MemoryManager(client=client, user_id="alice", search_query="prefs")

    out = await mgr.retrieve(limit=5)

    assert len(out) == 1
    assert client.seen_filters == {"user_id": "alice"}


@pytest.mark.asyncio
async def test_falls_back_to_top_level_scope_on_older_mem0():
    """A release without ``filters`` must still be driven correctly."""
    client = _LegacyMem0Client()
    mgr = Mem0MemoryManager(client=client, user_id="alice", agent_id="bot")

    out = await mgr.retrieve(limit=5)

    assert len(out) == 1
    assert client.seen_scope == {"user_id": "alice", "agent_id": "bot"}


@pytest.mark.asyncio
async def test_an_unrelated_error_is_not_swallowed_by_the_fallback():
    """The fallback must only catch the filters-shape mismatch."""

    class _Broken:
        async def get_all(self, **kwargs):
            raise ValueError("vector store unreachable")

    mgr = Mem0MemoryManager(client=_Broken(), user_id="alice")

    with pytest.raises(ValueError, match="vector store unreachable"):
        await mgr.retrieve(limit=5)
