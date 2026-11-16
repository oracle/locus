# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Durability modes and the tool-effect ledger on PostgreSQL.

The same cases as ``tests/unit/test_durability_modes.py`` on a real
``postgresql_checkpointer``: a process killed after a tool ran but before any
checkpoint landed (``durability="async"`` / ``"exit"``) is picked up with the
tool run exactly once overall, and the three modes end a turn in the same
state. Needs ``POSTGRES_USER`` and ``POSTGRES_DB`` (``POSTGRES_HOST`` and
``POSTGRES_PORT`` optional).
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import TYPE_CHECKING, Any

import pytest

from locus.agent import Agent
from locus.core.events import TerminateEvent, ToolCompleteEvent
from locus.core.messages import Message, Role, ToolCall
from locus.memory.backends import PostgreSQLBackend, StorageBackendAdapter
from locus.memory.checkpointer import BaseCheckpointer
from locus.memory.ledger import ToolEffectLedger, ledger_thread_id
from locus.models.base import ModelResponse
from locus.tools.decorator import tool


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from locus.core.state import AgentState


pytestmark = [pytest.mark.integration, pytest.mark.requires_postgres]


class _KilledError(Exception):
    """Stands in for the process dying under the run."""


class _Killable(BaseCheckpointer):
    """Wraps a real checkpointer; its writes stop once the 'process' dies."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.dead = False
        self.saves: dict[str, int] = {}

    @property
    def deletes_single_checkpoints(self) -> bool:
        return bool(getattr(self.inner, "deletes_single_checkpoints", False))

    async def save(
        self,
        state: AgentState,
        thread_id: str,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        # A real write takes a round trip; give an in-flight write time to be
        # overtaken by the kill, as it would be.
        await asyncio.sleep(0.005)
        if self.dead:
            return "lost"
        self.saves[thread_id] = self.saves.get(thread_id, 0) + 1
        saved: str = await self.inner.save(state, thread_id, checkpoint_id, metadata)
        return saved

    async def load(self, thread_id: str, checkpoint_id: str | None = None) -> AgentState | None:
        loaded: AgentState | None = await self.inner.load(thread_id, checkpoint_id)
        return loaded

    async def list_checkpoints(self, thread_id: str, limit: int = 10) -> list[str]:
        listed: list[str] = await self.inner.list_checkpoints(thread_id, limit)
        return listed

    async def delete(self, thread_id: str, checkpoint_id: str | None = None) -> bool:
        if self.dead:
            return False
        deleted: bool = await self.inner.delete(thread_id, checkpoint_id)
        return deleted


@pytest.fixture
async def backend() -> AsyncIterator[PostgreSQLBackend]:
    pg = PostgreSQLBackend(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "locus_test"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        table_name="test_durability_" + uuid.uuid4().hex[:8],
    )
    yield pg
    for thread in await pg.list_threads():
        await pg.delete(thread)
    await pg.close()


class _Model:
    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **_: Any
    ) -> ModelResponse:
        step = self.script.pop(0)
        if callable(step) and not isinstance(step, ModelResponse):
            step()
        response: ModelResponse = step
        return response


calls: dict[str, int] = {}


@tool
def charge(order_id: str) -> str:
    """Charge an order."""
    calls[order_id] = calls.get(order_id, 0) + 1
    return f"charged {order_id}"


def _call(name: str, call_id: str, **arguments: Any) -> ModelResponse:
    return ModelResponse(
        message=Message.assistant(
            content=None, tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)]
        ),
        usage={"prompt_tokens": 1, "completion_tokens": 1},
    )


def _answer(text: str) -> ModelResponse:
    return ModelResponse(
        message=Message.assistant(text), usage={"prompt_tokens": 1, "completion_tokens": 1}
    )


def _agent(model: Any, checkpointer: Any, **kwargs: Any) -> Agent:
    return Agent(
        model=model,
        tools=[charge],
        checkpointer=checkpointer,
        max_iterations=10,
        reflexion=False,
        grounding=False,
        **kwargs,
    )


async def _drain(gen: Any) -> list[Any]:
    return [event async for event in gen]


@pytest.fixture(autouse=True)
def _reset_calls() -> None:
    calls.clear()


@pytest.mark.parametrize("mode", ["async", "exit"])
async def test_killed_after_the_tool_ran_on_postgres_the_tool_runs_once(
    backend: PostgreSQLBackend, mode: str
) -> None:
    store = _Killable(StorageBackendAdapter(backend))
    thread = "t-" + uuid.uuid4().hex[:8]

    def die() -> None:
        store.dead = True
        raise _KilledError("killed")

    first = _agent(_Model([_call("charge", "c1", order_id="o1"), die]), store, durability=mode)  # type: ignore[arg-type]
    with pytest.raises(_KilledError):
        await _drain(first.run("charge o1", thread_id=thread, invocation_id="job-1"))

    assert calls == {"o1": 1}
    assert await store.load(thread) is None, "no checkpoint of the turn landed"
    entries = await ToolEffectLedger(store).entries(thread)
    assert [(e.step, e.tool) for e in entries] == [(1, "charge")]

    # A new process, the store alive again.
    store.dead = False
    second = _agent(
        _Model([_call("charge", "c9", order_id="o1"), _answer("charged once")]),
        store,
        durability=mode,  # type: ignore[arg-type]
    )
    events = await _drain(second.resume(invocation_id="job-1", thread_id=thread))

    assert calls == {"o1": 1}, "the recorded call is replayed, not run again"
    completed = [e for e in events if isinstance(e, ToolCompleteEvent)]
    assert completed[0].result == "charged o1"
    assert [e for e in events if isinstance(e, TerminateEvent)][-1].final_message == "charged once"

    final = await backend.load(f"{thread}:latest")
    assert final is not None
    assert [m["content"] for m in final["messages"] if m["role"] == "tool"] == ["charged o1"]
    assert final["messages"][-1]["content"] == "charged once"
    # The ledger is cleared once the final checkpoint holds every result.
    assert await store.list_checkpoints(ledger_thread_id(thread)) == []


@pytest.mark.parametrize("mode", ["sync", "async", "exit"])
async def test_every_mode_ends_in_the_same_state_on_postgres(
    backend: PostgreSQLBackend, mode: str
) -> None:
    store = _Killable(StorageBackendAdapter(backend))
    thread = f"{mode}-" + uuid.uuid4().hex[:8]
    script = [
        _call("charge", "c1", order_id="o1"),
        _call("charge", "c2", order_id="o2"),
        _answer("ok"),
    ]
    await _drain(_agent(_Model(script), store, durability=mode).run("go", thread_id=thread))  # type: ignore[arg-type]

    final = await store.load(thread)
    assert final is not None
    assert [(m.role, m.content) for m in final.messages if m.role == Role.TOOL] == [
        (Role.TOOL, "charged o1"),
        (Role.TOOL, "charged o2"),
    ]
    assert final.messages[-1].content == "ok"
    assert [e.tool_name for e in final.tool_executions] == ["charge", "charge"]
    assert store.saves[thread] == (1 if mode == "exit" else 3)
    assert len(await store.list_checkpoints(thread, limit=50)) == 1
    assert calls == {"o1": 1, "o2": 1}
