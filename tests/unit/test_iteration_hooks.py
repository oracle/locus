# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""The two lifecycle hooks that were documented and never fired.

`HookProvider` documents eight callbacks. Six fired. `on_iteration_start` and
`on_iteration_end` did not, and the reason was not a missing implementation —
`HookRegistry.emit_iteration_start` / `emit_iteration_end` were there the whole
time, and nothing called them. The agent dispatches through
`HookOrchestrator`, which implemented six methods.

A hook that never fires is worse than one that does not exist: you write the
callback, attach it, see no error, and conclude the run never reached that
phase. The docs elsewhere say "six lifecycle phases", which is what happens
when documentation is written by reading behaviour instead of the contract.

The pairing is what these mostly cover. `run()`'s loop has seven exits and
`_run_from_state()`'s has four, so an end fired at each of them is a contract
that breaks when someone adds a twelfth. The iteration is closed at the top of
the following pass instead, and once after the loop.
"""

from __future__ import annotations

import pytest

from tulip.agent import Agent
from tulip.core.state import AgentState
from tulip.hooks.provider import HookProvider
from tulip.testing import ScriptedModel, text, tool_call
from tulip.tools.decorator import tool


@tool
def get_weather(city: str) -> str:
    """Look up the weather."""
    return "sunny"


@tool
def failing_tool(city: str) -> str:
    """A tool that raises."""
    raise RuntimeError("boom")


class Spy(HookProvider):
    """Records every lifecycle callback in the order it arrives."""

    def __init__(self) -> None:
        self.seen: list[str] = []
        self.iterations: list[int] = []

    @property
    def priority(self) -> int:
        return 100

    async def on_before_invocation(self, prompt: str, state: AgentState) -> AgentState:
        self.seen.append("before_invocation")
        return state

    async def on_after_invocation(self, state: AgentState, success: bool) -> None:
        self.seen.append("after_invocation")

    async def on_before_model_call(self, event: object) -> None:
        self.seen.append("before_model")

    async def on_after_model_call(self, event: object) -> None:
        self.seen.append("after_model")

    async def on_before_tool_call(self, event: object) -> None:
        self.seen.append("before_tool")

    async def on_after_tool_call(self, event: object) -> None:
        self.seen.append("after_tool")

    async def on_iteration_start(self, iteration: int, state: AgentState) -> None:
        self.seen.append(f"iteration_start:{iteration}")
        self.iterations.append(iteration)

    async def on_iteration_end(self, iteration: int, state: AgentState) -> None:
        self.seen.append(f"iteration_end:{iteration}")


def _agent(spy: Spy, turns: list, tools: list | None = None) -> Agent:
    return Agent(
        model=ScriptedModel(turns, repeat_last=True),
        tools=tools if tools is not None else [get_weather],
        hooks=[spy],
    )


# --------------------------------------------------------------------------
# They fire at all
# --------------------------------------------------------------------------


def test_the_iteration_hooks_fire() -> None:
    """The regression. Before this, both were silent for every run."""
    spy = Spy()
    _agent(spy, [tool_call("get_weather", city="Paris"), text("Sunny.")]).run_sync("weather?")

    assert any(s.startswith("iteration_start") for s in spy.seen)
    assert any(s.startswith("iteration_end") for s in spy.seen)


def test_all_eight_documented_callbacks_are_reachable() -> None:
    """``HookProvider`` documents eight. Six used to be the real number."""
    spy = Spy()
    _agent(spy, [tool_call("get_weather", city="Paris"), text("Sunny.")]).run_sync("weather?")

    phases = {s.split(":")[0] for s in spy.seen}
    assert phases == {
        "before_invocation",
        "after_invocation",
        "before_model",
        "after_model",
        "before_tool",
        "after_tool",
        "iteration_start",
        "iteration_end",
    }


# --------------------------------------------------------------------------
# They pair — the part that survives a twelfth loop exit
# --------------------------------------------------------------------------


def test_every_start_has_exactly_one_end() -> None:
    spy = Spy()
    _agent(spy, [tool_call("get_weather", city="Paris"), text("Sunny.")]).run_sync("weather?")

    starts = [s for s in spy.seen if s.startswith("iteration_start")]
    ends = [s for s in spy.seen if s.startswith("iteration_end")]
    assert len(starts) == len(ends)
    assert [s.split(":")[1] for s in starts] == [e.split(":")[1] for e in ends]


def test_the_iteration_number_counts_from_zero_and_increases() -> None:
    spy = Spy()
    _agent(spy, [tool_call("get_weather", city="Paris"), text("Sunny.")]).run_sync("weather?")

    assert spy.iterations == list(range(len(spy.iterations)))
    assert spy.iterations[0] == 0


def test_a_start_is_never_left_open() -> None:
    """The loop has many exits; the last one must still close its iteration."""
    spy = Spy()
    _agent(spy, [tool_call("get_weather", city="Paris"), text("Sunny.")]).run_sync("weather?")

    assert spy.seen[-2].startswith("iteration_end")
    assert spy.seen[-1] == "after_invocation"


def test_a_single_turn_run_still_pairs() -> None:
    """One pass, one start, one end — the shortest path through the loop."""
    spy = Spy()
    _agent(spy, [text("No tools needed.")]).run_sync("hello")

    assert [s for s in spy.seen if s.startswith("iteration_")] == [
        "iteration_start:0",
        "iteration_end:0",
    ]


def test_the_boundary_wraps_the_model_call() -> None:
    """An iteration that started after its own model call would be useless for
    timing it, which is most of what these hooks are for."""
    spy = Spy()
    _agent(spy, [text("done")]).run_sync("hello")

    assert spy.seen.index("iteration_start:0") < spy.seen.index("before_model")
    assert spy.seen.index("after_model") < spy.seen.index("iteration_end:0")


def test_a_failing_tool_still_closes_the_iteration() -> None:
    """A tool error is an ordinary outcome; it must not strand the boundary."""
    spy = Spy()
    _agent(
        spy,
        [tool_call("failing_tool", city="Paris"), text("Could not.")],
        tools=[failing_tool],
    ).run_sync("weather?")

    starts = [s for s in spy.seen if s.startswith("iteration_start")]
    ends = [s for s in spy.seen if s.startswith("iteration_end")]
    assert len(starts) == len(ends)


def test_hitting_the_iteration_cap_still_pairs() -> None:
    """max_iterations exits through a different break than a clean finish."""
    spy = Spy()
    agent = Agent(
        model=ScriptedModel([tool_call("get_weather", city="Paris")], repeat_last=True),
        tools=[get_weather],
        hooks=[spy],
        max_iterations=3,
    )
    agent.run_sync("weather?")

    starts = [s for s in spy.seen if s.startswith("iteration_start")]
    ends = [s for s in spy.seen if s.startswith("iteration_end")]
    assert len(starts) == len(ends), f"{len(starts)} starts, {len(ends)} ends"


# --------------------------------------------------------------------------
# It must not require the callbacks
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_hook_without_the_iteration_callbacks_is_fine() -> None:
    """Dispatch is by ``hasattr``; a six-method hook must keep working."""

    class Partial:
        @property
        def priority(self) -> int:
            return 50

        async def on_before_invocation(self, prompt: str, state: AgentState) -> AgentState:
            return state

    agent = Agent(model=ScriptedModel([text("fine")]), hooks=[Partial()])

    assert (await agent.arun("hi")).message == "fine"
