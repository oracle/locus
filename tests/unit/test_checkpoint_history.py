# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Read, edit and fork a thread from its checkpoints.

A checkpointer already kept every run segment, but the only way back in was
``resume`` from the latest one. These tests pin the history API on top of it:
past states are readable, an edit writes a new checkpoint instead of rewriting
an old one, and a fork diverges without touching its parent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tulip.agent import Agent
from tulip.core.messages import Message, Role
from tulip.memory.backends.file import FileCheckpointer
from tulip.memory.backends.memory import MemoryCheckpointer
from tulip.testing import FunctionModel, text


if TYPE_CHECKING:
    from pathlib import Path


def _user_texts(messages: list[Message]) -> list[str]:
    return [str(m.content) for m in messages if m.role == Role.USER]


@pytest.fixture(params=["memory", "file"])
def checkpointer(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    if request.param == "memory":
        return MemoryCheckpointer()
    return FileCheckpointer(tmp_path / "checkpoints")


def _agent(checkpointer: Any, seen: list[list[Message]] | None = None) -> Agent:
    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
        if seen is not None:
            seen.append(list(messages))
        return text("ok")

    return Agent(
        model=FunctionModel(handler),
        tools=[],
        checkpointer=checkpointer,
        max_iterations=5,
        reflexion=False,
        grounding=False,
    )


async def _two_segments(agent: Agent, thread_id: str = "t") -> None:
    async for _ in agent.run("first request", thread_id=thread_id):
        pass
    async for _ in agent.run("second request", thread_id=thread_id):
        pass


@pytest.mark.asyncio
async def test_history_is_newest_first_and_each_state_is_readable(checkpointer: Any) -> None:
    agent = _agent(checkpointer)
    await _two_segments(agent)

    history = await agent.get_state_history("t")

    assert len(history) >= 2
    newest_id, newest = history[0]
    oldest_id, oldest = history[-1]
    assert "second request" in _user_texts(newest.messages)
    assert "second request" not in _user_texts(oldest.messages)
    assert _user_texts((await agent.get_state("t", oldest_id)).messages) == _user_texts(
        oldest.messages
    )
    assert (await agent.get_state("t")).messages == (await agent.get_state("t", newest_id)).messages


@pytest.mark.asyncio
async def test_an_edit_writes_a_new_checkpoint_and_the_next_run_sees_it(checkpointer: Any) -> None:
    seen: list[list[Message]] = []
    agent = _agent(checkpointer, seen)
    await _two_segments(agent)
    [(before_id, before), *_] = await agent.get_state_history("t")

    new_id = await agent.update_state(
        "t", messages=[Message.user("note: the order is ord-9")], metadata={"edited_by": "alice"}
    )

    assert new_id != before_id
    assert (await agent.get_state_history("t"))[0][0] == new_id
    unchanged = await agent.get_state("t", before_id)
    assert _user_texts(unchanged.messages) == _user_texts(before.messages), (
        "old checkpoint untouched"
    )
    assert (await agent.get_state("t")).metadata["edited_by"] == "alice"

    async for _ in agent.run("third request", thread_id="t"):
        pass
    assert "note: the order is ord-9" in _user_texts(seen[-1])


@pytest.mark.asyncio
async def test_a_fork_diverges_without_touching_its_parent(checkpointer: Any) -> None:
    seen: list[list[Message]] = []
    agent = _agent(checkpointer, seen)
    await _two_segments(agent)
    oldest_id, _ = (await agent.get_state_history("t"))[-1]
    parent_before = _user_texts((await agent.get_state("t")).messages)

    forked = await agent.fork("t", oldest_id, new_thread_id="t-experiment")
    async for _ in agent.run("a different second request", thread_id=forked):
        pass

    assert forked == "t-experiment"
    assert _user_texts((await agent.get_state("t")).messages) == parent_before
    fork_texts = _user_texts((await agent.get_state(forked)).messages)
    assert "a different second request" in fork_texts
    assert "second request" not in fork_texts


@pytest.mark.asyncio
async def test_a_fork_without_a_name_gets_a_fresh_thread_id(checkpointer: Any) -> None:
    agent = _agent(checkpointer)
    await _two_segments(agent)

    forked = await agent.fork("t")

    assert forked.startswith("t-fork-")
    assert _user_texts((await agent.get_state(forked)).messages) == _user_texts(
        (await agent.get_state("t")).messages
    )


@pytest.mark.asyncio
async def test_history_needs_a_checkpointer_and_a_real_checkpoint() -> None:
    bare = Agent(
        model=FunctionModel(lambda m, t: text("ok")), tools=[], reflexion=False, grounding=False
    )
    with pytest.raises(RuntimeError, match="checkpointer"):
        await bare.get_state_history("t")

    agent = _agent(MemoryCheckpointer())
    await _two_segments(agent)
    with pytest.raises(LookupError):
        await agent.get_state("t", "no-such-checkpoint")
    with pytest.raises(LookupError):
        await agent.get_state("no-such-thread")
    with pytest.raises(ValueError, match="different"):
        await agent.fork("t", new_thread_id="t")
