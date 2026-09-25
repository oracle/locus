# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""ONE ``LLMMemoryManager`` serves every user: the namespace is resolved per run.

A manager per user (the only way to scope memories before) gave every user
their own background-extraction semaphore — no global bound — and
``drain_memory`` drained only one of them. With ``namespace_resolver`` a single
shared manager keeps users apart, bounds all extractions with one semaphore
and drains them all at once.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tulip.agent import Agent
from tulip.core.events import RunInfo
from tulip.core.messages import Message, Role
from tulip.memory.manager import LLMMemoryManager, Memory, MemoryType
from tulip.memory.store import InMemoryStore
from tulip.models.base import ModelResponse


class _Model:
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    async def complete(
        self, messages: list[Message], tools: Any = None, **kwargs: Any
    ) -> ModelResponse:
        self.calls.append(list(messages))
        return ModelResponse(message=Message.assistant("ok"), usage={})

    async def stream(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


def _by_user(run: RunInfo) -> tuple[str, ...]:
    return ("users", run.metadata["user_id"])


def _memory_block(messages: list[Message]) -> str:
    return "".join(
        m.content or ""
        for m in messages
        if m.role == Role.SYSTEM and "[Long-term Memory]" in (m.content or "")
    )


async def test_one_shared_manager_scopes_bounds_and_drains_every_user() -> None:
    running = 0
    peak = 0

    async def extractor(messages: list[Message]) -> list[Memory]:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.05)
        running -= 1
        said = next(m.content for m in messages if m.role == Role.USER)
        return [Memory(type=MemoryType.USER, key="said", content=f"User said {said}")]

    store = InMemoryStore()
    manager = LLMMemoryManager(
        store=store,
        extract_fn=extractor,
        extract_mode="background",
        max_concurrent_extractions=2,
        namespace_resolver=_by_user,
    )
    agent = Agent(model=_Model(), memory_manager=manager, reflexion=False, grounding=False)
    users = [f"u{i}" for i in range(6)]

    await asyncio.gather(*(agent.arun(f"hello from {u}", metadata={"user_id": u}) for u in users))
    assert manager.pending_extractions > 0  # background: the runs did not wait

    await agent.drain_memory()  # ONE drain covers every user
    assert manager.pending_extractions == 0
    assert peak <= 2  # ONE global bound across users
    for u in users:
        item = await store.get(("users", u, "user"), "said")
        assert item is not None
        assert item["content"] == f"User said hello from {u}"
    # Nothing leaked into the constructor's default namespace.
    assert await store.list_keys(("tulip_memory", "user")) == []

    # Retrieval is scoped the same way: each user sees only their own memory.
    model = _Model()
    reader = Agent(model=model, memory_manager=manager, reflexion=False, grounding=False)
    await reader.arun("what did I say?", metadata={"user_id": "u3"})
    block = _memory_block(model.calls[0])
    assert "hello from u3" in block
    assert "hello from u1" not in block


async def test_the_old_fixed_namespace_constructor_still_works() -> None:
    store = InMemoryStore()
    manager = LLMMemoryManager(store=store, namespace_prefix=("tenants", "acme"))
    await manager.save([Memory(type=MemoryType.USER, key="k", content="c")])
    assert await store.get(("tenants", "acme", "user"), "k") is not None
    assert [m.key for m in await manager.retrieve()] == ["k"]


async def test_scoped_block_reads_and_writes_under_its_namespace() -> None:
    store = InMemoryStore()
    manager = LLMMemoryManager(store=store)
    with manager.scoped(("users", "ana")):
        await manager.save([Memory(type=MemoryType.USER, key="k", content="ana's")])
        assert [m.content for m in await manager.retrieve()] == ["ana's"]
    assert await manager.retrieve() == []
    assert await store.get(("users", "ana", "user"), "k") is not None


async def test_a_resolver_returning_none_uses_the_default_prefix() -> None:
    store = InMemoryStore()
    manager = LLMMemoryManager(
        store=store,
        extract_fn=lambda _m: _one("x"),
        namespace_resolver=lambda run: None,
    )
    agent = Agent(model=_Model(), memory_manager=manager, reflexion=False, grounding=False)
    await agent.arun("hi")
    assert await store.get(("tulip_memory", "user"), "x") is not None


async def test_a_failing_resolver_never_falls_back_to_a_shared_namespace(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = InMemoryStore()
    await store.put(
        ("tulip_memory", "user"), "shared", {"content": "someone else's", "metadata": {}}
    )
    extracted: list[int] = []

    async def extractor(messages: list[Message]) -> list[Memory]:
        extracted.append(1)
        return [Memory(type=MemoryType.USER, key="leak", content="x")]

    manager = LLMMemoryManager(store=store, extract_fn=extractor, namespace_resolver=_by_user)
    model = _Model()
    agent = Agent(model=model, memory_manager=manager, reflexion=False, grounding=False)
    await agent.arun("no user_id in metadata")
    assert _memory_block(model.calls[0]) == ""
    assert extracted == []
    assert await store.get(("tulip_memory", "user"), "leak") is None
    assert "namespace_resolver failed" in caplog.text


async def _one(key: str) -> list[Memory]:
    return [Memory(type=MemoryType.USER, key=key, content="c")]
