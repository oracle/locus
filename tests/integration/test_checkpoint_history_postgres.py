# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint history against a real Postgres checkpointer.

The unit tests cover history on the memory and file checkpointers. These run the
same scenarios through ``StorageBackendAdapter`` on ``PostgreSQLBackend``, which
keeps each checkpoint under its own key and a newest-first index per thread:
history order, an edit that writes a new checkpoint, a fork that leaves its parent
alone, and a checkpoint per graph step.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING, Any

import pytest

from tulip.agent import Agent
from tulip.core.messages import Message, Role
from tulip.memory.backends.adapters import StorageBackendAdapter
from tulip.multiagent.graph import END, START, StateGraph
from tulip.testing import FunctionModel, text


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.fixture
async def checkpointer() -> AsyncIterator[StorageBackendAdapter]:
    from tulip.memory.backends import PostgreSQLBackend

    backend = PostgreSQLBackend(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "tulip_test"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        table_name="test_history_" + uuid.uuid4().hex[:8],
    )
    yield StorageBackendAdapter(backend)
    for thread in await backend.list_threads():
        await backend.delete(thread)
    await backend.close()


def _user_texts(messages: list[Message]) -> list[str]:
    return [str(m.content) for m in messages if m.role == Role.USER]


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


async def _two_segments(agent: Agent, thread_id: str) -> None:
    async for _ in agent.run("first request", thread_id=thread_id):
        pass
    async for _ in agent.run("second request", thread_id=thread_id):
        pass


@pytest.mark.requires_postgres
@pytest.mark.asyncio
async def test_agent_history_edit_and_fork_on_postgres(checkpointer: StorageBackendAdapter) -> None:
    seen: list[list[Message]] = []
    agent = _agent(checkpointer, seen)
    await _two_segments(agent, "pg-agent")

    history = await agent.get_state_history("pg-agent")
    assert len(history) >= 2
    newest_id, newest = history[0]
    oldest_id, oldest = history[-1]
    assert "second request" in _user_texts(newest.messages)
    assert "second request" not in _user_texts(oldest.messages)

    edited_id = await agent.update_state(
        "pg-agent", messages=[Message.user("note: the order is ord-9")]
    )
    assert (await agent.get_state_history("pg-agent"))[0][0] == edited_id
    assert _user_texts((await agent.get_state("pg-agent", newest_id)).messages) == _user_texts(
        newest.messages
    ), "the edit must not rewrite the checkpoint it started from"

    async for _ in agent.run("third request", thread_id="pg-agent"):
        pass
    assert "note: the order is ord-9" in _user_texts(seen[-1])

    forked = await agent.fork("pg-agent", oldest_id, new_thread_id="pg-agent-alt")
    fork_texts = _user_texts((await agent.get_state(forked)).messages)
    assert "second request" not in fork_texts
    assert "third request" in _user_texts((await agent.get_state("pg-agent")).messages)


@pytest.mark.requires_postgres
@pytest.mark.asyncio
async def test_graph_steps_history_and_fork_on_postgres(
    checkpointer: StorageBackendAdapter,
) -> None:
    graph = StateGraph()

    async def a(inputs: dict[str, Any]) -> dict[str, Any]:
        return {"count": inputs.get("count", 0) + 1, "path": "a"}

    async def b(inputs: dict[str, Any]) -> dict[str, Any]:
        return {"count": inputs["count"] + 1, "path": "b"}

    graph.add_node("a", a)
    graph.add_node("b", b)
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)
    graph.compile(checkpointer=checkpointer)
    graph.config.thread_id = "pg-graph"

    await graph.execute({"count": 0})
    history = await graph.aget_state_history("pg-graph")

    assert [state["path"] for _, state in history] == ["b", "a"]
    first_id, first = history[-1]
    assert first["count"] == 1
    assert (await graph.aget_state("pg-graph", checkpoint_id=first_id))["count"] == 1

    forked = await graph.afork("pg-graph", first_id, new_thread_id="pg-graph-alt")
    assert (await graph.aget_state(forked))["count"] == 1
    assert (await graph.aget_state("pg-graph"))["count"] == 2
