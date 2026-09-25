# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""One Agent instance, many concurrent runs: per-run state must not leak.

Before 2.17 ``arun`` read the final state off the shared agent
(``self._last_run_state``) after its awaits, so under ``asyncio.gather`` run B
returned run A's state and tool executions. The cancel signal and the
composable termination condition were shared the same way.
"""

from __future__ import annotations

import asyncio
from typing import Any

from tulip.agent import Agent
from tulip.core.events import TerminateEvent
from tulip.core.state import AgentState
from tulip.core.termination import TerminationCondition
from tulip.memory.backends.memory import MemoryCheckpointer
from tulip.testing import _RecordingModel, text, tool_call
from tulip.tools.decorator import tool


class _EchoFirstModel(_RecordingModel):
    """Calls ``echo`` with the user prompt, then answers; A is slower than B."""

    async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> Any:
        self._record(messages, tools)
        user = [m.content for m in messages if m.role.value == "user"][-1]
        await asyncio.sleep(0.05 if user == "A" else 0.01)
        if not any(m.role.value == "tool" for m in messages):
            return tool_call("echo", call_id=f"c_{user}", who=user)
        return text(f"answer-for-{user}")


@tool
async def echo(who: str) -> str:
    """Echo."""
    await asyncio.sleep(0.01)
    return f"echo {who}"


class _SlowSaveCheckpointer(MemoryCheckpointer):
    """Delays thread B's final save so run A finishes in between."""

    async def save(self, state: AgentState, thread_id: str, *a: Any, **k: Any) -> str:
        await asyncio.sleep(0.2 if thread_id.endswith("B") else 0.0)
        saved: str = await super().save(state, thread_id, *a, **k)
        return saved


def _agent(**kwargs: Any) -> Agent:
    return Agent(
        model=_EchoFirstModel(),
        tools=[echo],
        reflexion=False,
        grounding=False,
        **kwargs,
    )


async def test_gathered_aruns_each_return_their_own_state() -> None:
    agent = _agent(checkpointer=_SlowSaveCheckpointer())

    ra, rb = await asyncio.gather(
        agent.arun("A", thread_id="t-A"),
        agent.arun("B", thread_id="t-B"),
    )

    for result, who in ((ra, "A"), (rb, "B")):
        assert result.message == f"answer-for-{who}"
        users = [m.content for m in result.state.messages if m.role.value == "user"]
        assert users == [who]
        assert [e.arguments for e in result.tool_executions] == [{"who": who}]


async def test_gathered_aruns_without_checkpointer_are_isolated() -> None:
    agent = _agent()
    results = await asyncio.gather(*(agent.arun(p, thread_id=f"x{p}") for p in "ABAB"))
    for result, who in zip(results, "ABAB", strict=True):
        assert [e.arguments["who"] for e in result.tool_executions] == [who]


async def test_arun_ignores_a_nested_run_of_the_same_agent() -> None:
    """A run started inside another run's tool never steals the outer result."""
    inner_agent = Agent(model=_EchoFirstModel(), tools=[echo], reflexion=False, grounding=False)

    @tool
    async def delegate(who: str) -> str:
        """Run the inner agent."""
        return (await inner_agent.arun("B")).message

    outer = Agent(
        model=_ScriptThenAnswer(),
        tools=[delegate],
        reflexion=False,
        grounding=False,
    )
    result = await outer.arun("outer")
    assert result.message == "outer done"
    assert [e.tool_name for e in result.tool_executions] == ["delegate"]


class _ScriptThenAnswer(_RecordingModel):
    async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> Any:
        self._record(messages, tools)
        if not any(m.role.value == "tool" for m in messages):
            return tool_call("delegate", call_id="d1", who="B")
        return text("outer done")


# ---------------------------------------------------------------------------
# cancel(thread_id=...)
# ---------------------------------------------------------------------------


class _LoopingModel(_RecordingModel):
    """Calls ``echo`` forever (distinct args, so no tool-loop stop)."""

    def __init__(self) -> None:
        super().__init__()
        self._n = 0

    async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> Any:
        self._record(messages, tools)
        self._n += 1
        await asyncio.sleep(0.01)
        return tool_call("echo", call_id=f"c{self._n}", who=str(self._n))


async def test_cancel_by_thread_stops_only_that_run() -> None:
    agent = Agent(
        model=_LoopingModel(),
        tools=[echo],
        reflexion=False,
        grounding=False,
        max_iterations=40,
    )

    async def reason(thread: str) -> str | None:
        last = None
        async for event in agent.run("go", thread_id=thread):
            if isinstance(event, TerminateEvent):
                last = event.reason
        return last

    task_a = asyncio.create_task(reason("A"))
    task_b = asyncio.create_task(reason("B"))
    await asyncio.sleep(0.05)
    assert agent.cancel(thread_id="A") == 1
    assert await task_a == "cancelled"
    assert not task_b.done()
    assert agent.cancel(thread_id="nobody") == 0
    # The no-argument form still cancels everything in flight.
    assert agent.cancel() == 1
    assert await task_b == "cancelled"


async def test_cancel_without_runs_arms_the_next_run() -> None:
    agent = Agent(model=_LoopingModel(), tools=[echo], reflexion=False, grounding=False)
    assert agent.cancel() == 0
    assert agent.is_cancelled
    result = await agent.arun("go")
    assert result.stop_reason == "cancelled"
    assert not agent.is_cancelled


# ---------------------------------------------------------------------------
# termination condition is copied per run
# ---------------------------------------------------------------------------


class _RecordingCondition(TerminationCondition):
    reset_ids: list[int] = []

    def check(self, state: AgentState, **context: Any) -> tuple[bool, str | None]:
        return False, None

    def reset(self) -> None:
        type(self).reset_ids.append(id(self))


async def test_runs_reset_their_own_copy_of_the_termination_condition() -> None:
    """Resetting the shared instance restarted every other run's TimeLimit clock."""
    shared = _RecordingCondition()
    agent = Agent(
        model=_EchoFirstModel(),
        tools=[echo],
        reflexion=False,
        grounding=False,
        termination=shared,
    )
    _RecordingCondition.reset_ids = []
    await asyncio.gather(agent.arun("A"), agent.arun("B"))
    assert len(_RecordingCondition.reset_ids) == 2
    assert id(shared) not in _RecordingCondition.reset_ids
    assert len(set(_RecordingCondition.reset_ids)) == 2


class _UncopyableCondition(TerminationCondition):
    """Holds a lock, so ``deepcopy`` fails: the run falls back to sharing it."""

    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()
        self.resets = 0

    def check(self, state: AgentState, **context: Any) -> tuple[bool, str | None]:
        return False, None

    def reset(self) -> None:
        self.resets += 1


async def test_uncopyable_termination_condition_is_shared_not_fatal() -> None:
    condition = _UncopyableCondition()
    agent = Agent(
        model=_EchoFirstModel(),
        tools=[echo],
        reflexion=False,
        grounding=False,
        termination=condition,
    )
    result = await agent.arun("A")
    assert result.message == "answer-for-A"
    assert condition.resets == 1
