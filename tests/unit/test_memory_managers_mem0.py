# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Unit tests for the Mem0 memory manager.

Uses an injected fake async client (``client=``) so the tests run with no
``mem0ai`` dependency and no LLM / network access.
"""

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
