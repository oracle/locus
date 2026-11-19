# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for the Mem0 memory manager.

Uses an injected fake async client (``client=``) so the tests run with no
``mem0ai`` dependency and no LLM / network access.
"""

import pytest

from locus.core.messages import Message
from locus.memory.manager import Memory, MemoryType
from locus.memory.managers.mem0 import Mem0MemoryManager


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


class _TopKMem0Client:
    """mem0 2.x: the page size is ``top_k`` and unknown keywords are ignored."""

    def __init__(self, rows):
        self._rows = rows
        self.seen_top_k = None
        self.seen_kwargs = None

    async def get_all(self, *, filters=None, top_k=20, **kwargs):  # noqa: ARG002
        self.seen_top_k = top_k
        self.seen_kwargs = kwargs
        return {"results": self._rows[:top_k]}

    async def search(self, query, *, top_k=20, filters=None, **kwargs):  # noqa: ARG002
        self.seen_top_k = top_k
        self.seen_kwargs = kwargs
        return {"results": self._rows[:top_k]}


class _LimitMem0Client:
    """mem0 0.1 / 1.x: the page size is ``limit`` and there is no ``**kwargs``."""

    def __init__(self, rows):
        self._rows = rows
        self.seen_limit = None

    async def get_all(self, user_id=None, agent_id=None, run_id=None, limit=100):  # noqa: ARG002
        self.seen_limit = limit
        return self._rows[:limit]

    async def search(  # noqa: PLR0913
        self,
        query,  # noqa: ARG002
        user_id=None,  # noqa: ARG002
        agent_id=None,  # noqa: ARG002
        run_id=None,  # noqa: ARG002
        limit=100,
        filters=None,  # noqa: ARG002
    ):
        self.seen_limit = limit
        return self._rows[:limit]


_ROWS = [{"id": f"m{i}", "memory": f"fact {i}"} for i in range(30)]


@pytest.mark.asyncio
async def test_page_size_is_top_k_on_current_mem0():
    """mem0 2.x dropped ``limit`` for ``top_k`` and ignores unknown keywords, so
    the old name silently fetched the default twenty rows."""
    client = _TopKMem0Client(_ROWS)
    mgr = Mem0MemoryManager(client=client, user_id="alice", retrieve_limit=25)

    out = await mgr.retrieve(limit=25)

    assert client.seen_top_k == 25
    assert "limit" not in client.seen_kwargs
    assert len(out) == 25


@pytest.mark.asyncio
async def test_search_page_size_is_top_k_on_current_mem0():
    client = _TopKMem0Client(_ROWS)
    mgr = Mem0MemoryManager(client=client, user_id="alice", search_query="prefs")

    out = await mgr.retrieve(limit=5)

    assert client.seen_top_k == 5
    assert len(out) == 5


@pytest.mark.asyncio
async def test_page_size_is_limit_on_older_mem0():
    """A release that names the page size ``limit`` and rejects unknown keywords."""
    client = _LimitMem0Client(_ROWS)
    mgr = Mem0MemoryManager(client=client, user_id="alice", search_query="prefs")

    out = await mgr.retrieve(limit=5)

    assert client.seen_limit == 5
    assert len(out) == 5
