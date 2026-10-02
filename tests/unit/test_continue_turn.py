# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-iteration checkpoints and ``Agent.continue_turn``.

A long run killed mid-turn must lose at most the iteration in flight: the
checkpoint written after each iteration's tool results is enough to pick the
SAME turn up in a new process, without a second user message and without
re-running a call whose result was saved.
"""

from __future__ import annotations

from typing import Any

import pytest

from tulip.agent import Agent
from tulip.agent.agent import _UNFINISHED_CALL_ERROR
from tulip.agent.runtime_loop import ITERATION_CHECKPOINTS_KEY
from tulip.core.events import TerminateEvent, ToolStartEvent
from tulip.core.messages import Message, Role, ToolCall, ToolResult
from tulip.core.state import AgentState
from tulip.memory.backends.memory import MemoryCheckpointer
from tulip.memory.checkpointer import BaseCheckpointer
from tulip.models.base import ModelResponse
from tulip.tools.decorator import tool


class _KilledError(Exception):
    """Stands in for the process dying under the run."""


class _Model:
    """Replays scripted responses; a ``_KilledError`` entry raises instead."""

    def __init__(self, script: list[ModelResponse | type[BaseException]]) -> None:
        self.script = list(script)
        self.seen: list[list[Message]] = []

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **_: Any
    ) -> ModelResponse:
        self.seen.append(list(messages))
        step = self.script.pop(0)
        if isinstance(step, type):
            raise step("killed")
        return step


class _KillableCheckpointer(MemoryCheckpointer):
    """A memory checkpointer whose writes stop once the 'process' dies.

    A real kill never reaches the run's ``finally``, so the turn's final save
    never happens; freezing the store reproduces that.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dead = False
        self.saves = 0

    async def save(
        self,
        state: AgentState,
        thread_id: str,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if self.dead:
            return "lost"
        self.saves += 1
        return await super().save(state, thread_id, checkpoint_id, metadata)

    async def delete(self, thread_id: str, checkpoint_id: str | None = None) -> bool:
        # A dead process deletes nothing either: the final save's clean-up of
        # the turn's iteration checkpoints never runs.
        if self.dead:
            return False
        return await super().delete(thread_id, checkpoint_id)


calls: dict[str, int] = {}


@tool
def step_one() -> str:
    """First step."""
    calls["step_one"] = calls.get("step_one", 0) + 1
    return "one done"


@tool
def step_two() -> str:
    """Second step."""
    calls["step_two"] = calls.get("step_two", 0) + 1
    return "two done"


def _call(name: str, call_id: str) -> ModelResponse:
    return ModelResponse(
        message=Message.assistant(
            content=None, tool_calls=[ToolCall(id=call_id, name=name, arguments={})]
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
        tools=[step_one, step_two],
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


async def test_killed_turn_continues_from_the_last_iteration_checkpoint() -> None:
    store = _KillableCheckpointer()

    class _DieError(_KilledError):
        def __init__(self, msg: str) -> None:
            store.dead = True
            super().__init__(msg)

    first = _Model([_call("step_one", "c1"), _call("step_two", "c2"), _DieError])
    agent = _agent(first, store, checkpoint_every_n_iterations=1)
    with pytest.raises(_KilledError):
        await _drain(agent.run("do both steps", thread_id="t"))

    # One save per finished iteration; the turn's final save never happened.
    assert store.saves == 2
    assert calls == {"step_one": 1, "step_two": 1}

    # A new process: fresh agent, same store.
    store.dead = False
    second = _Model([_answer("both steps done")])
    events = await _drain(_agent(second, store).continue_turn("t"))

    assert calls == {"step_one": 1, "step_two": 1}, "no finished call runs twice"
    assert not any(isinstance(e, ToolStartEvent) for e in events)
    done = [e for e in events if isinstance(e, TerminateEvent)]
    assert done[-1].reason == "complete"
    assert done[-1].final_message == "both steps done"

    sent = second.seen[0]
    assert [m.content for m in sent if m.role == Role.USER] == ["do both steps"]
    assert [m.content for m in sent if m.role == Role.TOOL] == ["one done", "two done"]

    final = await store.load("t")
    assert final is not None
    assert final.iteration == 3, "the iteration count carries on"
    assert final.messages[-1].content == "both steps done"
    assert sum(m.role == Role.USER for m in final.messages) == 1


async def test_a_call_without_a_result_is_answered_with_an_error_not_rerun() -> None:
    store = MemoryCheckpointer()
    paused = AgentState(
        messages=(
            Message.system("sys"),
            Message.user("go"),
            Message.assistant(
                content=None,
                tool_calls=[
                    ToolCall(id="c1", name="step_one", arguments={}),
                    ToolCall(id="c2", name="step_two", arguments={}),
                ],
            ),
            Message.tool(ToolResult(tool_call_id="c1", name="step_one", content="one done")),
        ),
        iteration=1,
        max_iterations=10,
    )
    await store.save(paused, "t")
    model = _Model([_answer("stopping here")])

    events = await _drain(_agent(model, store).continue_turn("t"))

    assert calls == {}, "the unanswered call is not re-run"
    assert not any(isinstance(e, ToolStartEvent) for e in events)
    tool_msgs = [m for m in model.seen[0] if m.role == Role.TOOL]
    assert [m.tool_call_id for m in tool_msgs] == ["c1", "c2"]
    assert tool_msgs[1].content == f"Error: {_UNFINISHED_CALL_ERROR}"


async def test_continued_segment_checkpoints_every_iteration() -> None:
    store = _KillableCheckpointer()
    await store.save(
        AgentState(
            messages=(Message.system("sys"), Message.user("go")),
            iteration=1,
            max_iterations=10,
        ),
        "t",
    )
    store.saves = 0
    model = _Model([_call("step_one", "c1"), _answer("done")])

    await _drain(_agent(model, store, checkpoint_every_n_iterations=1).continue_turn("t"))

    assert calls == {"step_one": 1}
    # The iteration that ran the tool, then the turn's final save.
    assert store.saves == 2


async def test_default_saves_every_iteration_and_keeps_only_the_final_save() -> None:
    store = _KillableCheckpointer()
    model = _Model([_call("step_one", "c1"), _call("step_two", "c2"), _answer("done")])

    await _drain(_agent(model, store).run("go", thread_id="t"))

    # Two iterations ran tools, then the turn's final save ...
    assert store.saves == 3
    # ... which superseded the two iteration saves, so the thread's history
    # is what a per-turn checkpointer would have left.
    assert len(await store.list_checkpoints("t")) == 1
    final = await store.load("t")
    assert final is not None
    assert ITERATION_CHECKPOINTS_KEY not in final.metadata


class _NoDeleteCheckpointer(BaseCheckpointer):
    """A backend without ``delete``: iteration saves could never be removed."""

    def __init__(self) -> None:
        self.inner = MemoryCheckpointer()
        self.saves = 0

    async def save(
        self,
        state: AgentState,
        thread_id: str,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        self.saves += 1
        return await self.inner.save(state, thread_id, checkpoint_id, metadata)

    async def load(self, thread_id: str, checkpoint_id: str | None = None) -> AgentState | None:
        return await self.inner.load(thread_id, checkpoint_id)

    async def list_checkpoints(self, thread_id: str, limit: int = 10) -> list[str]:
        return await self.inner.list_checkpoints(thread_id, limit)


async def test_default_saves_only_at_the_end_where_saves_cannot_be_deleted() -> None:
    store = _NoDeleteCheckpointer()
    assert not store.deletes_single_checkpoints
    model = _Model([_call("step_one", "c1"), _call("step_two", "c2"), _answer("done")])

    await _drain(_agent(model, store).run("go", thread_id="t"))

    assert store.saves == 1


async def test_an_explicit_interval_on_a_backend_that_cannot_delete_keeps_every_save() -> None:
    store = _NoDeleteCheckpointer()
    model = _Model([_call("step_one", "c1"), _call("step_two", "c2"), _answer("done")])

    await _drain(_agent(model, store, checkpoint_every_n_iterations=1).run("go", thread_id="t"))

    assert store.saves == 3
    assert len(await store.list_checkpoints("t")) == 3
    for checkpoint_id in await store.list_checkpoints("t"):
        saved = await store.load("t", checkpoint_id)
        assert saved is not None
        assert ITERATION_CHECKPOINTS_KEY not in saved.metadata


async def test_a_run_without_a_thread_saves_nothing_by_default() -> None:
    store = _KillableCheckpointer()
    model = _Model([_call("step_one", "c1"), _answer("done")])

    await _drain(_agent(model, store).run("go"))

    assert store.saves == 0


async def test_keep_iteration_checkpoints_leaves_them_in_the_history() -> None:
    store = MemoryCheckpointer()
    model = _Model([_call("step_one", "c1"), _call("step_two", "c2"), _answer("done")])
    agent = _agent(model, store, keep_iteration_checkpoints=True)

    await _drain(agent.run("go", thread_id="t"))

    history = await agent.get_state_history("t")
    assert len(history) == 3
    assert all(ITERATION_CHECKPOINTS_KEY not in state.metadata for _, state in history)


async def test_saves_from_a_killed_process_are_deleted_when_the_turn_is_continued() -> None:
    store = _KillableCheckpointer()

    class _DieError(_KilledError):
        def __init__(self, msg: str) -> None:
            store.dead = True
            super().__init__(msg)

    first = _Model([_call("step_one", "c1"), _call("step_two", "c2"), _DieError])
    with pytest.raises(_KilledError):
        await _drain(_agent(first, store).run("do both steps", thread_id="t"))
    assert len(await store.list_checkpoints("t")) == 2

    store.dead = False
    await _drain(_agent(_Model([_answer("both done")]), store).continue_turn("t"))

    # The new process never saved those two, but the checkpoint it continued
    # from named them, so the turn's final save removed them.
    [remaining] = await store.list_checkpoints("t")
    final = await store.load("t", remaining)
    assert final is not None
    assert final.messages[-1].content == "both done"
    assert ITERATION_CHECKPOINTS_KEY not in final.metadata


async def test_a_new_turn_on_a_killed_thread_closes_its_dangling_calls() -> None:
    store = MemoryCheckpointer()
    await store.save(
        AgentState(
            messages=(
                Message.system("sys"),
                Message.user("go"),
                Message.assistant(
                    content=None, tool_calls=[ToolCall(id="c1", name="step_one", arguments={})]
                ),
            ),
            metadata={ITERATION_CHECKPOINTS_KEY: ["gone-1"]},
        ),
        "t",
        "gone-1",
    )
    model = _Model([_answer("fresh answer")])

    result = await _agent(model, store).arun("something else", thread_id="t")

    sent = model.seen[0]
    tool_msgs = [m for m in sent if m.role == Role.TOOL]
    assert tool_msgs[0].tool_call_id == "c1"
    assert tool_msgs[0].content == f"Error: {_UNFINISHED_CALL_ERROR}"
    assert [m.content for m in sent if m.role == Role.USER] == ["go", "something else"]
    # The killed turn's iteration save is superseded by this turn's final one.
    assert await store.list_checkpoints("t") != ["gone-1"]
    assert "gone-1" not in await store.list_checkpoints("t")
    assert ITERATION_CHECKPOINTS_KEY not in result.state.metadata


async def test_a_failed_delete_is_logged_not_raised(caplog: pytest.LogCaptureFixture) -> None:
    class _BrokenDelete(MemoryCheckpointer):
        async def delete(self, thread_id: str, checkpoint_id: str | None = None) -> bool:
            raise OSError("disk went away")

    store = _BrokenDelete()
    model = _Model([_call("step_one", "c1"), _answer("done")])

    events = await _drain(_agent(model, store).run("go", thread_id="t"))

    assert [e for e in events if isinstance(e, TerminateEvent)][-1].reason == "complete"
    assert "could not delete superseded iteration checkpoint" in caplog.text


async def test_continue_turn_reinjects_memory() -> None:
    class _Memory:
        def __init__(self) -> None:
            self.started = 0

        async def on_session_start(self, state: AgentState) -> AgentState:
            self.started += 1
            return state

        async def on_session_end(self, state: AgentState) -> None:
            return None

    memory = _Memory()
    store = MemoryCheckpointer()
    await store.save(
        AgentState(messages=(Message.system("sys"), Message.user("go")), max_iterations=10), "t"
    )
    agent = _agent(_Model([_answer("done")]), store, memory_manager=memory)

    await _drain(agent.continue_turn("t", metadata={"k": "v"}))

    assert memory.started == 1


async def test_a_finished_turn_is_not_continued() -> None:
    store = MemoryCheckpointer()
    await store.save(AgentState(messages=(Message.user("go"), Message.assistant("answered"))), "t")
    with pytest.raises(RuntimeError, match="already finished"):
        await _drain(_agent(_Model([]), store).continue_turn("t"))


async def test_continue_turn_needs_a_checkpoint() -> None:
    with pytest.raises(RuntimeError, match="needs a checkpointer"):
        await _drain(_agent(_Model([]), None).continue_turn("t"))
    with pytest.raises(RuntimeError, match="No checkpoint found"):
        await _drain(_agent(_Model([]), MemoryCheckpointer()).continue_turn("t"))
    empty = MemoryCheckpointer()
    await empty.save(AgentState(), "t")
    model = _Model([_answer("done")])
    # A checkpoint with no messages is not a finished turn.
    await _drain(_agent(model, empty).continue_turn("t"))
    assert model.seen


async def test_a_paused_thread_is_resumed_not_continued() -> None:
    store = MemoryCheckpointer()
    agent = _agent(_Model([]), store)
    agent._interrupts["t"] = object()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="resume"):
        await _drain(agent.continue_turn("t"))
