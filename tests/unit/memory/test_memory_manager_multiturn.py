# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Long-term memory on long, checkpointed, multi-user chat threads.

Regression tests for ``LLMMemoryManager`` behaviour that only shows up once a
thread runs more than one turn: memory blocks piling up in the checkpoint,
retrieval that ignores what the user just asked, the heuristic extractor
re-saving the whole history under fresh random keys, and extraction failures
costing the turn its final checkpoint.
"""

from __future__ import annotations

from typing import Any

import pytest

from tulip.agent import Agent
from tulip.core.messages import Message, Role
from tulip.core.state import AgentState
from tulip.memory.backends import MemoryCheckpointer
from tulip.memory.manager import (
    MEMORY_BLOCK_METADATA_KEY,
    LLMMemoryManager,
    Memory,
    MemoryType,
    _format_memory_block,
)
from tulip.memory.store import InMemoryStore, StoreItem
from tulip.models.base import ModelResponse


class _RecordingModel:
    """Answers ``ok`` and records every message list it was sent."""

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls.append(list(messages))
        return ModelResponse(message=Message.assistant("ok"), usage={})

    async def stream(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


class _RankingStore(InMemoryStore):
    """A store whose ``search`` ranks (never filters) by word overlap.

    Stands in for the semantic backends (``PgMemory``, ``HolographicStore``):
    every item comes back, best match first.
    """

    def __init__(self) -> None:
        super().__init__()
        self.queries: list[str | None] = []

    async def search(
        self,
        namespace: tuple[str, ...],
        query: str | None = None,
        limit: int = 10,
    ) -> list[StoreItem]:
        self.queries.append(query)
        items = await super().search(namespace, query=None, limit=10_000)
        if query:
            words = set(query.lower().split())

            def score(item: StoreItem) -> int:
                return len(words & set(str(item.value.get("content", "")).lower().split()))

            items.sort(key=score, reverse=True)
        return items[:limit]


def _memory_blocks(messages: list[Message]) -> list[Message]:
    return [
        m for m in messages if m.role == Role.SYSTEM and "[Long-term Memory]" in (m.content or "")
    ]


async def _turn(agent: Agent, prompt: str, thread_id: str = "thread-1") -> None:
    async for _ in agent.run(prompt, thread_id=thread_id):
        pass


# ---------------------------------------------------------------------------
# 1. Idempotent injection
# ---------------------------------------------------------------------------


async def test_memory_block_does_not_pile_up_on_checkpointed_thread() -> None:
    """Five turns on one thread → the model sees exactly one memory block each time."""
    store = InMemoryStore()
    manager = LLMMemoryManager(store=store)
    await manager.save([Memory(type=MemoryType.USER, key="name", content="User is Ana.")])
    model = _RecordingModel()
    checkpointer = MemoryCheckpointer()
    agent = Agent(
        model=model,
        system_prompt="You are helpful.",
        memory_manager=manager,
        checkpointer=checkpointer,
    )

    for i in range(5):
        await _turn(agent, f"turn {i}")

    assert [len(_memory_blocks(call)) for call in model.calls] == [1, 1, 1, 1, 1]
    saved = await checkpointer.load("thread-1")
    assert saved is not None
    # The block is ephemeral: it never reaches the checkpoint at all.
    assert len(_memory_blocks(list(saved.messages))) == 0
    # The block stays right after the primary system prompt.
    assert model.calls[-1][0].content == "You are helpful."
    assert model.calls[-1][1].metadata.get(MEMORY_BLOCK_METADATA_KEY) is True


async def test_legacy_untagged_block_from_old_checkpoint_is_replaced() -> None:
    """A block written before the metadata tag existed is still recognised."""
    store = InMemoryStore()
    manager = LLMMemoryManager(store=store)
    memory = Memory(type=MemoryType.USER, key="name", content="User is Ana.")
    await manager.save([memory])
    legacy = Message.system(_format_memory_block([memory]))  # no metadata tag
    state = AgentState(
        messages=(Message.system("sys"), legacy, legacy, Message.user("hi")),
    )

    out = await manager.on_session_start(state)

    assert len(_memory_blocks(list(out.messages))) == 1


async def test_stale_block_is_dropped_when_store_is_empty() -> None:
    """Memories erased from the store must stop reaching the model from the checkpoint."""
    manager = LLMMemoryManager(store=InMemoryStore())
    stale = Message(
        role=Role.SYSTEM,
        content=_format_memory_block(
            [Memory(type=MemoryType.USER, key="gone", content="Erased fact.")]
        ),
        metadata={MEMORY_BLOCK_METADATA_KEY: True},
    )
    state = AgentState(messages=(Message.system("sys"), stale, Message.user("hi")))

    out = await manager.on_session_start(state)

    assert _memory_blocks(list(out.messages)) == []
    assert [m.content for m in out.messages] == ["sys", "hi"]


async def test_empty_store_and_no_block_returns_state_unchanged() -> None:
    manager = LLMMemoryManager(store=InMemoryStore())
    state = AgentState(messages=(Message.system("sys"), Message.user("hi")))
    assert await manager.on_session_start(state) is state


# ---------------------------------------------------------------------------
# 2. Query-aware retrieval
# ---------------------------------------------------------------------------


async def test_retrieval_uses_current_user_message_as_query() -> None:
    """The memory relevant to this turn is injected, not merely the newest one."""
    store = _RankingStore()
    manager = LLMMemoryManager(store=store, retrieve_limit=1)
    await manager.save(
        [Memory(type=MemoryType.USER, key="lang", content="User writes Python daily.")]
    )
    # Saved later, so it wins on recency alone.
    await manager.save(
        [Memory(type=MemoryType.PROJECT, key="trip", content="Planning a trip to Lisbon.")]
    )
    state = AgentState(
        messages=(Message.system("sys"), Message.user("Any tips for my Python tests?"))
    )

    out = await manager.on_session_start(state)

    block = _memory_blocks(list(out.messages))[0].content or ""
    assert "Python daily" in block
    assert "Lisbon" not in block
    assert "Any tips for my Python tests?" in store.queries


async def test_query_retrieval_merges_types_rank_by_rank_and_tops_up_with_recency() -> None:
    store = _RankingStore()
    manager = LLMMemoryManager(store=store)
    await manager.save(
        [
            Memory(type=MemoryType.USER, key="u1", content="likes python"),
            Memory(type=MemoryType.USER, key="u2", content="lives in rome"),
            Memory(type=MemoryType.FEEDBACK, key="f1", content="python answers short"),
        ]
    )

    got = await manager.retrieve(query="python")

    # Best of each type first, then the rest; nothing duplicated.
    assert [m.key for m in got[:2]] == ["u1", "f1"]
    assert sorted(m.key for m in got) == ["f1", "u1", "u2"]


async def test_filtering_search_with_no_match_falls_back_to_recency() -> None:
    """``InMemoryStore.search`` filters by substring; a miss must not inject nothing."""
    manager = LLMMemoryManager(store=InMemoryStore())
    await manager.save([Memory(type=MemoryType.USER, key="name", content="User is Ana.")])

    got = await manager.retrieve(query="what's the weather like")

    assert [m.key for m in got] == ["name"]


async def test_store_whose_query_search_raises_falls_back_to_recency() -> None:
    class _NoQueryStore(InMemoryStore):
        async def search(
            self, namespace: tuple[str, ...], query: str | None = None, limit: int = 10
        ) -> list[StoreItem]:
            if query:
                raise RuntimeError("query search unsupported")
            return await super().search(namespace, query=None, limit=limit)

    manager = LLMMemoryManager(store=_NoQueryStore())
    await manager.save([Memory(type=MemoryType.USER, key="name", content="User is Ana.")])

    assert [m.key for m in await manager.retrieve(query="anything")] == ["name"]


async def test_retrieve_limit_is_honoured_by_session_start() -> None:
    """``retrieve_limit`` used to be ignored — session start always asked for 20."""
    store = InMemoryStore()
    manager = LLMMemoryManager(store=store, retrieve_limit=2, max_memories=100)
    await manager.save(
        [Memory(type=MemoryType.USER, key=f"k{i}", content=f"fact {i}") for i in range(5)]
    )
    state = AgentState(messages=(Message.system("sys"), Message.user("hi")))

    out = await manager.on_session_start(state)

    block = _memory_blocks(list(out.messages))[0].content or ""
    assert block.count("USER [") == 2


async def test_retrieve_with_zero_limit_returns_nothing() -> None:
    manager = LLMMemoryManager(store=InMemoryStore())
    await manager.save([Memory(type=MemoryType.USER, key="name", content="User is Ana.")])
    assert await manager.retrieve(0) == []


async def test_base_retrieve_relevant_delegates_to_legacy_retrieve() -> None:
    """Subclasses that only implement ``retrieve(limit)`` keep working."""
    from tulip.memory.manager import BaseMemoryManager

    class _Legacy(BaseMemoryManager):
        def __init__(self) -> None:
            self.limits: list[int] = []

        async def extract(self, messages: list[Message]) -> list[Memory]:
            return []

        async def retrieve(self, limit: int = 20) -> list[Memory]:
            self.limits.append(limit)
            return []

        async def save(self, memories: list[Memory]) -> None:
            return None

    legacy = _Legacy()
    await legacy.retrieve_relevant("q")
    await legacy.retrieve_relevant("q", 3)
    assert legacy.limits == [20, 3]


# ---------------------------------------------------------------------------
# 3. Extraction on multi-turn threads
# ---------------------------------------------------------------------------


async def test_heuristic_extraction_does_not_duplicate_across_turns() -> None:
    """Re-extracting the same history every turn must upsert, not append copies."""
    store = InMemoryStore()
    manager = LLMMemoryManager(store=store)
    agent = Agent(
        model=_RecordingModel(),
        memory_manager=manager,
        checkpointer=MemoryCheckpointer(),
    )

    await _turn(agent, "I'm a data engineer at Acme.")
    for i in range(3):
        await _turn(agent, f"follow-up {i}")

    assert len(await store.list_keys(("tulip_memory", "user"))) == 1


async def test_extractor_does_not_see_injected_memory_block() -> None:
    """Recalled memories must not be fed back to the extractor as new input."""
    seen: list[list[Message]] = []

    async def extractor(messages: list[Message]) -> list[Memory]:
        seen.append(messages)
        return []

    manager = LLMMemoryManager(store=InMemoryStore(), extract_fn=extractor)
    await manager.save([Memory(type=MemoryType.USER, key="name", content="User is Ana.")])
    agent = Agent(model=_RecordingModel(), system_prompt="sys", memory_manager=manager)

    async for _ in agent.run("hello"):
        pass

    assert seen
    assert _memory_blocks(seen[0]) == []
    assert [m.content for m in seen[0] if m.role == Role.USER] == ["hello"]


async def test_failing_extraction_does_not_lose_the_turn_checkpoint() -> None:
    async def broken(messages: list[Message]) -> list[Memory]:
        raise RuntimeError("extraction LLM timed out")

    checkpointer = MemoryCheckpointer()
    agent = Agent(
        model=_RecordingModel(),
        memory_manager=LLMMemoryManager(store=InMemoryStore(), extract_fn=broken),
        checkpointer=checkpointer,
    )

    await _turn(agent, "hello")

    saved = await checkpointer.load("thread-1")
    assert saved is not None
    assert any(m.role == Role.USER and m.content == "hello" for m in saved.messages)


async def test_max_memories_prunes_oldest_per_type() -> None:
    store = InMemoryStore()
    manager = LLMMemoryManager(store=store, max_memories=2)
    for i in range(4):
        await manager.save([Memory(type=MemoryType.USER, key=f"k{i}", content=f"fact {i}")])
    await manager.save([Memory(type=MemoryType.FEEDBACK, key="f", content="rule")])

    assert sorted(await store.list_keys(("tulip_memory", "user"))) == ["k2", "k3"]
    assert await store.list_keys(("tulip_memory", "feedback")) == ["f"]


async def test_max_memories_zero_disables_pruning() -> None:
    store = InMemoryStore()
    manager = LLMMemoryManager(store=store, max_memories=0)
    await manager.save(
        [Memory(type=MemoryType.USER, key=f"k{i}", content=f"fact {i}") for i in range(3)]
    )
    assert len(await store.list_keys(("tulip_memory", "user"))) == 3


async def test_prune_tolerates_values_without_metadata() -> None:
    store = InMemoryStore()
    manager = LLMMemoryManager(store=store, max_memories=1)
    await store.put(("tulip_memory", "user"), "raw", "not-a-dict")
    await manager.save([Memory(type=MemoryType.USER, key="new", content="fresh")])

    assert await store.list_keys(("tulip_memory", "user")) == ["new"]


async def test_save_does_not_mutate_caller_memory_metadata() -> None:
    manager = LLMMemoryManager(store=InMemoryStore())
    memory = Memory(type=MemoryType.USER, key="name", content="User is Ana.", metadata={})

    await manager.save([memory])

    assert memory.metadata == {}


@pytest.mark.parametrize("bad_value", [{"no": "fields"}, {"type": "bogus", "key": "k"}])
async def test_malformed_store_values_are_skipped(bad_value: dict[str, Any]) -> None:
    store = InMemoryStore()
    await store.put(("tulip_memory", "user"), "bad", bad_value)
    manager = LLMMemoryManager(store=store)
    assert await manager.retrieve(query="x") == []
