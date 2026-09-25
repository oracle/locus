# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""A new user message on a checkpointed thread starts a new turn.

Before 2.17 the loaded state was continued verbatim: the iteration counter kept
climbing across turns, so with ``max_iterations=3`` every turn from the third
on ended ``max_iterations``; tool-loop detection counted earlier turns; and the
first turn's metadata and (callable) system prompt were frozen for the life of
the thread. Resume-after-interrupt, by contrast, must continue the SAME turn.
"""

from __future__ import annotations

from typing import Any

from tulip.agent import Agent
from tulip.core.events import InterruptEvent, TerminateEvent, TulipEvent
from tulip.memory.backends.memory import MemoryCheckpointer
from tulip.testing import ScriptedModel, text, tool_call
from tulip.tools.decorator import tool


@tool
def ping(n: int = 0) -> str:
    """Ping."""
    return "pong"


async def _turn(agent: Agent, prompt: str, **kwargs: Any) -> list[TulipEvent]:
    return [e async for e in agent.run(prompt, thread_id="T", **kwargs)]


async def test_iteration_budget_is_per_turn() -> None:
    checkpointer = MemoryCheckpointer()
    agent = Agent(
        model=ScriptedModel([text(f"reply {i}") for i in range(6)]),
        checkpointer=checkpointer,
        max_iterations=3,
        reflexion=False,
        grounding=False,
    )
    for i in range(6):
        events = await _turn(agent, f"turn {i}")
        terminate = events[-1]
        assert isinstance(terminate, TerminateEvent)
        assert terminate.reason == "complete", f"turn {i}: {terminate.reason}"
        assert terminate.final_message == f"reply {i}"
        state = await checkpointer.load("T")
        assert state is not None
        assert state.iteration == 1
    # History is kept: 1 system + 6 user + 6 assistant messages.
    assert len(state.messages) == 13


async def test_tool_loop_window_does_not_span_turns() -> None:
    # Each turn makes the SAME call once, then answers. Three turns used to
    # add up to a "loop" (threshold 3) and end the third turn early.
    turns = []
    for i in range(4):
        turns += [tool_call("ping", call_id=f"p{i}", n=1), text(f"done {i}")]
    agent = Agent(
        model=ScriptedModel(turns),
        tools=[ping],
        checkpointer=MemoryCheckpointer(),
        tool_loop_threshold=3,
        reflexion=False,
        grounding=False,
    )
    for i in range(4):
        events = await _turn(agent, f"turn {i}")
        assert isinstance(events[-1], TerminateEvent)
        assert events[-1].reason == "complete"
        assert events[-1].final_message == f"done {i}"


async def test_idempotent_reuse_is_scoped_to_the_turn() -> None:
    hits: list[int] = []

    @tool(idempotent=True)
    def quote(city: str) -> str:
        """Quote."""
        hits.append(1)
        return f"quote {len(hits)}"

    agent = Agent(
        model=ScriptedModel(
            [
                tool_call("quote", call_id="q1", city="Paris"),
                text("a"),
                tool_call("quote", call_id="q2", city="Paris"),
                text("b"),
            ]
        ),
        tools=[quote],
        checkpointer=MemoryCheckpointer(),
        reflexion=False,
        grounding=False,
    )
    await _turn(agent, "price?")
    result = await agent.arun("price again?", thread_id="T")
    # A fresh turn asks again rather than replaying last turn's answer.
    assert len(hits) == 2
    assert [e.tool_name for e in result.tool_executions] == ["quote"]
    assert result.metrics.iterations == 2


async def test_new_turn_applies_new_metadata_and_system_prompt() -> None:
    checkpointer = MemoryCheckpointer()
    model = ScriptedModel([text("ok")], repeat_last=True)
    agent = Agent(
        model=model,
        checkpointer=checkpointer,
        system_prompt=lambda ctx: f"user={ctx['metadata'].get('user')}",  # type: ignore[arg-type]
        reflexion=False,
        grounding=False,
    )
    for i in range(3):
        await _turn(agent, f"turn {i}", metadata={"user": f"u{i}"})
        state = await checkpointer.load("T")
        assert state is not None
        assert state.metadata["user"] == f"u{i}"
        assert state.messages[0].content == f"user=u{i}"
        # Exactly one system prompt, re-evaluated in place.
        assert sum(m.role.value == "system" for m in state.messages) == 1
        assert model.received_messages[-1][0].content == f"user=u{i}"


async def test_terminal_tool_in_one_turn_does_not_end_the_next() -> None:
    agent = Agent(
        model=ScriptedModel([tool_call("done", call_id="d1"), text("second answer")]),
        tools=[_done_tool()],
        checkpointer=MemoryCheckpointer(),
        reflexion=False,
        grounding=False,
    )
    first = await _turn(agent, "one")
    assert isinstance(first[-1], TerminateEvent)
    assert first[-1].reason == "terminal_tool"
    second = await _turn(agent, "two")
    assert isinstance(second[-1], TerminateEvent)
    assert second[-1].reason == "complete"
    assert second[-1].final_message == "second answer"


def _done_tool() -> Any:
    @tool(name="done")
    def done() -> str:
        """Finish."""
        return "finished"

    return done


async def test_new_turn_gets_a_new_run_id_but_resume_keeps_the_turn() -> None:
    checkpointer = MemoryCheckpointer()
    agent = Agent(
        model=ScriptedModel(
            [
                text("hello"),
                tool_call("ask_user", call_id="a1", question="Which city?"),
                tool_call("ping", call_id="p1"),
                text("booked"),
            ]
        ),
        tools=[ping],
        checkpointer=checkpointer,
        completion_mode="auto",
        reflexion=False,
        grounding=False,
    )
    agent._initialize()
    from tulip.agent.initializer import register_builtin_tools

    register_builtin_tools(agent)

    await _turn(agent, "hi")
    first = await checkpointer.load("T")
    events = await _turn(agent, "book something")
    assert isinstance(events[-1], InterruptEvent)
    paused = await checkpointer.load("T")
    assert first is not None
    assert paused is not None
    assert paused.run_id != first.run_id
    assert paused.iteration == 1

    resumed = [e async for e in agent.resume("Paris", thread_id="T")]
    assert isinstance(resumed[-1], TerminateEvent)
    assert resumed[-1].final_message == "booked"
    done = await checkpointer.load("T")
    assert done is not None
    # Same turn: same run id, iteration carried on from the pause.
    assert done.run_id == paused.run_id
    assert done.iteration == 3
