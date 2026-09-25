# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The injected memory block is ephemeral; extraction can run in the background.

1. A memory manager's block reaches every model call of a turn but is never
   persisted: not in the thread's checkpoint, not in the run's result state,
   and never handed to the extractor.
2. ``LLMMemoryManager(extract_mode="background")`` takes extraction off the
   run's critical path: the run ends without waiting for the extractor, jobs
   are tracked and drainable, failures are reported (never raised), and one
   namespace's jobs run in order.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import time
from typing import Any

import pytest

from tulip.agent import Agent
from tulip.core.events import InterruptEvent, TerminateEvent
from tulip.core.messages import Message, Role, ToolCall
from tulip.memory.backends import MemoryCheckpointer
from tulip.memory.manager import LLMMemoryManager, Memory, MemoryType
from tulip.memory.store import InMemoryStore
from tulip.models.base import ModelResponse
from tulip.tools.decorator import tool


class _RecordingModel:
    """Replays ``responses`` (then answers ``ok``) and records every call."""

    def __init__(self, responses: list[ModelResponse] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls.append(list(messages))
        if self._responses:
            return self._responses.pop(0)
        return ModelResponse(message=Message.assistant("ok"), usage={})

    async def stream(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


def _blocks(messages: Any) -> int:
    return sum(
        1 for m in messages if m.role == Role.SYSTEM and "[Long-term Memory]" in (m.content or "")
    )


async def _seeded_manager(**kwargs: Any) -> LLMMemoryManager:
    manager = LLMMemoryManager(store=InMemoryStore(), **kwargs)
    await manager.save([Memory(type=MemoryType.USER, key="name", content="User is Ana.")])
    return manager


# ---------------------------------------------------------------------------
# 1. Ephemeral memory block
# ---------------------------------------------------------------------------


async def test_checkpoint_never_carries_the_memory_block() -> None:
    """3 turns on a checkpointed thread: every model call saw one block, the checkpoint none."""
    seen_by_extractor: list[list[Message]] = []

    async def extractor(messages: list[Message]) -> list[Memory]:
        seen_by_extractor.append(list(messages))
        return []

    manager = await _seeded_manager(extract_fn=extractor)
    model = _RecordingModel()
    checkpointer = MemoryCheckpointer()
    agent = Agent(
        model=model,
        system_prompt="You are helpful.",
        memory_manager=manager,
        checkpointer=checkpointer,
        checkpoint_every_n_iterations=1,
    )

    results = [await agent.arun(f"turn {i}", thread_id="t") for i in range(3)]

    assert [_blocks(call) for call in model.calls] == [1, 1, 1]
    saved = await checkpointer.load("t")
    assert saved is not None
    assert _blocks(saved.messages) == 0
    # Every checkpoint of the thread, not only the latest.
    for checkpoint_id in await checkpointer.list_checkpoints("t", limit=100):
        old = await checkpointer.load("t", checkpoint_id)
        assert old is not None
        assert _blocks(old.messages) == 0
    # The conversation itself is intact: 3 user turns, 3 replies.
    assert [m.content for m in saved.messages if m.role == Role.USER] == [
        "turn 0",
        "turn 1",
        "turn 2",
    ]
    # Nor is it in the state handed back to the caller, or to the extractor.
    assert all(_blocks(r.state.messages) == 0 for r in results)
    assert len(seen_by_extractor) == 3
    assert all(_blocks(msgs) == 0 for msgs in seen_by_extractor)


@tool
def needs_input() -> str:
    """Return the runtime's interrupt marker (like ask_user does)."""
    return json.dumps({"__interrupt__": True, "question": "Which option?"})


async def test_resume_from_checkpoint_reinjects_the_block() -> None:
    """A cross-process resume loads a block-less checkpoint and injects a fresh block."""
    checkpointer = MemoryCheckpointer()
    pause = ModelResponse(
        message=Message.assistant(
            content="asking", tool_calls=[ToolCall(id="c1", name="needs_input", arguments={})]
        ),
        usage={},
    )
    first = Agent(
        model=_RecordingModel([pause]),
        tools=[needs_input],
        memory_manager=await _seeded_manager(),
        checkpointer=checkpointer,
        reflexion=False,
        grounding=False,
    )
    events = [e async for e in first.run("start", thread_id="t")]
    assert any(isinstance(e, InterruptEvent) for e in events)
    paused = await checkpointer.load("t")
    assert paused is not None
    assert _blocks(paused.messages) == 0

    model = _RecordingModel()
    second = Agent(
        model=model,
        tools=[needs_input],
        memory_manager=await _seeded_manager(),
        checkpointer=checkpointer,
        reflexion=False,
        grounding=False,
    )
    _ = [e async for e in second.resume("a", thread_id="t")]

    assert [_blocks(call) for call in model.calls] == [1]
    final = await checkpointer.load("t")
    assert final is not None
    assert _blocks(final.messages) == 0


# ---------------------------------------------------------------------------
# 2. Background extraction
# ---------------------------------------------------------------------------


def _slow_extractor(delay: float, log: list[str] | None = None, fail: bool = False) -> Any:
    async def extractor(messages: list[Message]) -> list[Memory]:
        last = next(m.content for m in reversed(messages) if m.role == Role.USER)
        if log is not None:
            log.append(f"start {last}")
        await asyncio.sleep(delay)
        if log is not None:
            log.append(f"end {last}")
        if fail:
            raise RuntimeError("extractor down")
        return [Memory(type=MemoryType.PROJECT, key="last_turn", content=str(last))]

    return extractor


async def test_background_extraction_does_not_delay_the_run() -> None:
    """With a 0.5 s extractor the TerminateEvent arrives without waiting for it."""
    manager = LLMMemoryManager(
        store=InMemoryStore(), extract_fn=_slow_extractor(0.5), extract_mode="background"
    )
    agent = Agent(model=_RecordingModel(), memory_manager=manager)

    started = time.perf_counter()
    terminate_at: float | None = None
    async for event in agent.run("hello"):
        if isinstance(event, TerminateEvent):
            terminate_at = time.perf_counter() - started
    finished_at = time.perf_counter() - started

    assert terminate_at is not None
    assert terminate_at < 0.25
    assert finished_at < 0.25  # the generator closed without waiting either
    assert manager.pending_extractions == 1
    assert await manager.store.get(("tulip_memory", "project"), "last_turn") is None

    # drain() flushes the job.
    await agent.drain_memory()
    assert manager.pending_extractions == 0
    saved = await manager.store.get(("tulip_memory", "project"), "last_turn")
    assert saved is not None
    assert saved["content"] == "hello"


async def test_inline_mode_still_waits_for_extraction() -> None:
    """The default stays inline: the memory is saved by the time the run ends."""
    manager = LLMMemoryManager(store=InMemoryStore(), extract_fn=_slow_extractor(0.05))
    agent = Agent(model=_RecordingModel(), memory_manager=manager)

    _ = [e async for e in agent.run("hello")]

    assert manager.extract_mode == "inline"
    assert manager.pending_extractions == 0
    assert await manager.store.get(("tulip_memory", "project"), "last_turn") is not None


async def test_background_extractor_failure_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An extractor exception never reaches the run; it is logged and emitted."""
    emitted: list[tuple[str, dict[str, Any]]] = []

    async def fake_emit(event_type: str, /, **data: Any) -> None:
        emitted.append((event_type, data))

    monkeypatch.setattr(importlib.import_module("tulip.observability.emit"), "emit", fake_emit)
    manager = LLMMemoryManager(
        store=InMemoryStore(),
        extract_fn=_slow_extractor(0.01, fail=True),
        extract_mode="background",
    )
    agent = Agent(model=_RecordingModel(), memory_manager=manager)

    result = await agent.arun("hello")
    assert result.message == "ok"
    await manager.drain()

    failures = [d for t, d in emitted if t == "memory.manager.extract_failed"]
    assert failures == [
        {"error_type": "RuntimeError", "error": "extractor down", "mode": "background"}
    ]
    # The run is still usable afterwards.
    assert (await agent.arun("again")).message == "ok"
    await manager.drain()


async def test_background_jobs_of_one_namespace_run_in_order() -> None:
    """Two quick turns of the same user: the second job starts after the first ends."""
    log: list[str] = []
    manager = LLMMemoryManager(
        store=InMemoryStore(),
        extract_fn=_slow_extractor(0.1, log),
        extract_mode="background",
        max_concurrent_extractions=4,
    )
    agent = Agent(
        model=_RecordingModel(), memory_manager=manager, checkpointer=MemoryCheckpointer()
    )

    await agent.arun("first", thread_id="t")
    await agent.arun("second", thread_id="t")
    assert manager.pending_extractions == 2
    await manager.drain()

    assert log == ["start first", "end first", "start second", "end second"]
    saved = await manager.store.get(("tulip_memory", "project"), "last_turn")
    assert saved is not None
    assert saved["content"] == "second"  # the later turn's write wins


async def test_background_concurrency_is_bounded_across_namespaces() -> None:
    """Different namespaces run in parallel, but never more than the limit at once."""
    running = 0
    peak = 0

    async def job() -> None:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.05)
        running -= 1

    manager = LLMMemoryManager(
        store=InMemoryStore(), extract_mode="background", max_concurrent_extractions=2
    )
    for i in range(5):
        manager._background().submit(("users", f"u{i}"), job)
    assert manager.pending_extractions == 5
    await manager.drain()
    assert peak == 2


async def test_bounded_drain_leaves_jobs_running() -> None:
    manager = LLMMemoryManager(
        store=InMemoryStore(), extract_fn=_slow_extractor(0.3), extract_mode="background"
    )
    agent = Agent(model=_RecordingModel(), memory_manager=manager)
    await agent.arun("hello")

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await manager.drain()
    # Cancelling the wait did not cancel the job.
    assert manager.pending_extractions == 1
    await manager.drain()
    assert manager.pending_extractions == 0
    assert await manager.store.get(("tulip_memory", "project"), "last_turn") is not None


def test_run_sync_drains_background_extraction() -> None:
    """run_sync closes its loop on return, so it must flush background jobs first."""
    manager = LLMMemoryManager(
        store=InMemoryStore(), extract_fn=_slow_extractor(0.05), extract_mode="background"
    )
    agent = Agent(model=_RecordingModel(), memory_manager=manager)

    agent.run_sync("one")
    agent.run_sync("two")  # a second loop: primitives of the first are not reused

    assert manager.pending_extractions == 0
    saved = asyncio.run(manager.store.get(("tulip_memory", "project"), "last_turn"))
    assert saved is not None
    assert saved["content"] == "two"


def test_extract_mode_is_validated() -> None:
    with pytest.raises(ValueError, match="extract_mode"):
        LLMMemoryManager(store=InMemoryStore(), extract_mode="later")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_concurrent_extractions"):
        LLMMemoryManager(store=InMemoryStore(), max_concurrent_extractions=0)
