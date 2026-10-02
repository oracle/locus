# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Tool-loop detection that stops loops and leaves working runs alone.

The traces here are shaped on real coding-agent runs: a model that re-reads a
test file between other work is working, and must never be stopped; a model
that reads the same file four times back to back and gets the same text each
time is stuck, and is warned before it is stopped.
"""

from __future__ import annotations

from typing import Any

import pytest

from tulip.agent import Agent
from tulip.core.events import CustomEvent, TerminateEvent
from tulip.core.loops import DEFAULT_READ_ONLY_TOOLS, detect_tool_loop, warning_text
from tulip.core.messages import Message, Role, ToolCall
from tulip.core.state import AgentState, ReasoningStep, ToolExecution
from tulip.models.base import ModelResponse
from tulip.testing import FunctionModel, ScriptedModel, text, tool_call
from tulip.tools.decorator import tool


# --------------------------------------------------------------- helpers --


def _step(i: int, *calls: tuple[str, dict[str, Any], str]) -> ReasoningStep:
    """One step: ``(tool, arguments, result)`` per call."""
    tool_calls = [
        ToolCall(id=f"s{i}c{n}", name=name, arguments=args)
        for n, (name, args, _) in enumerate(calls)
    ]
    results = [
        ToolExecution(tool_name=name, tool_call_id=f"s{i}c{n}", arguments=args, result=result)
        for n, (name, args, result) in enumerate(calls)
    ]
    return ReasoningStep(iteration=i, tool_calls=tool_calls, tool_results=results)


def _read(path: str, result: str = "contents") -> tuple[str, dict[str, Any], str]:
    return ("read", {"path": path}, result)


def _grep(pattern: str, result: str = "hits") -> tuple[str, dict[str, Any], str]:
    return ("grep", {"pattern": pattern}, result)


def _bash(command: str, result: str = "ok") -> tuple[str, dict[str, Any], str]:
    return ("bash", {"command": command}, result)


def _todo(n: int) -> tuple[str, dict[str, Any], str]:
    return ("todo_write", {"items": [{"n": n}]}, "saved")


def _never_fires(steps: list[ReasoningStep]) -> None:
    """The detector stays quiet at every point of the run, not only at its end."""
    for end in range(1, len(steps) + 1):
        assert detect_tool_loop(steps[:end]) is None, f"fired after step {end}"


# ---------------------------------------------------- traces that must pass --


def test_rereading_a_test_file_between_other_work_is_not_a_loop() -> None:
    """The luach-email-questions run: 30 calls in 18 steps, with the test file
    read at calls 14, 18, 19 and 30 — 18 and 19 back to back — and reads, greps
    and todo updates in between. It was working, and must not be stopped."""
    test_file = _read("tests/test_units.py", "def test_units(): ...")
    steps = [
        _step(1, ("ls", {"path": "."}, "src tests")),
        _step(2, _read("README.md"), _read("pyproject.toml")),
        _step(3, _grep("email"), _grep("chat")),
        _step(4, _read("src/luach/chat.py")),
        _step(5, _read("src/luach/mail.py"), _todo(1)),
        _step(6, _grep("def ask"), _read("src/luach/tools.py")),
        _step(7, _read("src/luach/units.py")),
        _step(8, _grep("add_finding"), _read("tests/conftest.py")),
        _step(9, test_file, _read("tests/test_chat.py")),  # call 14
        _step(10, _grep("summary")),
        _step(11, _read("src/luach/store.py")),
        _step(12, test_file),  # call 18
        _step(13, test_file),  # call 19: the same read, back to back
        _step(14, _grep("inbox"), _read("src/luach/inbox.py")),
        _step(15, _read("src/luach/chat.py", "updated"), _todo(3)),
        _step(16, _bash("pytest -q tests/test_units.py", "1 failed")),
        _step(17, _grep("finding"), _read("src/luach/units.py"), _todo(4)),
        _step(18, _grep("ask_mail"), _read("src/luach/chat.py", "updated"), test_file),  # call 30
    ]
    assert sum(len(s.tool_calls) for s in steps) == 30
    _never_fires(steps)


def test_the_same_read_with_another_call_in_between_is_not_a_loop() -> None:
    steps = [
        _step(1, _read("a.py")),
        _step(2, _read("a.py")),
        _step(3, _grep("x")),
        _step(4, _read("a.py")),
        _step(5, _read("a.py")),
        _step(6, _todo(1)),
        _step(7, _read("a.py")),
        _step(8, _read("a.py")),
    ]
    _never_fires(steps)


def test_a_repeat_whose_result_changed_is_progress() -> None:
    """Re-running the tests after each edit returns something new each time."""
    steps = [_step(i, _bash("pytest -q", f"{5 - i} failed")) for i in range(5)]
    _never_fires(steps)


def test_a_reread_after_the_file_changed_is_progress() -> None:
    steps = [_step(i, _read("a.py", f"version {i}")) for i in range(6)]
    _never_fires(steps)


def test_the_same_tool_with_other_arguments_is_progress() -> None:
    steps = [_step(i, _read(f"src/m{i}.py")) for i in range(8)]
    _never_fires(steps)


def test_a_step_without_a_tool_call_breaks_the_streak() -> None:
    steps = [
        _step(1, _bash("make")),
        _step(2, _bash("make")),
        ReasoningStep(iteration=3, thought="sent back"),
        _step(4, _bash("make")),
        _step(5, _bash("make")),
    ]
    _never_fires(steps)


def test_a_cycle_seen_twice_is_not_yet_a_loop() -> None:
    steps = [_step(1, _bash("make")), _step(2, _bash("pytest"))] * 2
    _never_fires(steps)


def test_identical_calls_fanned_out_in_one_step_are_not_a_loop() -> None:
    steps = [_step(1, _bash("make"), _bash("make"), _bash("make"))]
    _never_fires(steps)


# ----------------------------------------------------- loops that must fire --


def test_four_identical_reads_back_to_back_warn_and_a_fifth_stops() -> None:
    steps = [_step(i, _read("tests/test_units.py")) for i in range(1, 6)]

    assert detect_tool_loop(steps[:3]) is None, "a read gets one repeat more than the threshold"
    warned = detect_tool_loop(steps[:4])
    assert warned is not None
    assert (warned.repeats, warned.threshold, warned.read_only) == (4, 4, True)
    assert not warned.past_warning
    stopped = detect_tool_loop(steps[:5])
    assert stopped is not None
    assert stopped.past_warning
    assert stopped.signature == warned.signature


def test_three_identical_commands_with_identical_output_are_a_loop() -> None:
    steps = [_step(i, _bash("pytest -q", "1 failed")) for i in range(3)]
    loop = detect_tool_loop(steps)
    assert loop is not None
    assert (loop.period, loop.repeats, loop.read_only) == (1, 3, False)
    assert "bash" in loop.describe()
    assert "3 times in a row" in loop.describe()


def test_an_alternating_cycle_repeated_whole_is_a_loop() -> None:
    edit = ("edit", {"path": "a.py", "old": "x", "new": "y"}, "old string not found")
    steps = [_step(i, edit if i % 2 == 0 else _bash("pytest", "1 failed")) for i in range(6)]
    loop = detect_tool_loop(steps)
    assert loop is not None
    assert (loop.period, loop.repeats) == (2, 3)
    assert "cycle" in loop.describe()
    # Seen from the other step of the cycle it is the same loop: one warning.
    rotated = detect_tool_loop([*steps, steps[0]])
    assert rotated is not None
    assert rotated.signature == loop.signature
    assert not rotated.past_warning
    after = detect_tool_loop([*steps, steps[0], steps[1]])
    assert after is not None
    assert after.past_warning


def test_thresholds_and_the_read_only_set_are_configurable() -> None:
    steps = [_step(i, _read("a.py")) for i in range(3)]
    assert detect_tool_loop(steps, read_only_threshold=3) is not None
    assert detect_tool_loop(steps, read_only_tools=()) is not None
    assert detect_tool_loop(steps, threshold=4, read_only_tools=()) is None
    assert "read" in DEFAULT_READ_ONLY_TOOLS


def test_steps_without_recorded_results_compare_by_calls() -> None:
    steps = [
        ReasoningStep(iteration=i, tool_calls=[ToolCall(name="search", arguments={"q": "x"})])
        for i in range(3)
    ]
    assert detect_tool_loop(steps) is not None


def test_arguments_json_cannot_encode_still_compare() -> None:
    steps = []
    for i in range(3):
        call = ToolCall(name="bash", arguments={"a": 1})
        call.arguments["self"] = call.arguments  # circular: json.dumps refuses it
        steps.append(ReasoningStep(iteration=i, tool_calls=[call]))
    assert detect_tool_loop(steps) is not None


def test_the_warning_names_the_call_and_says_what_happens_next() -> None:
    loop = detect_tool_loop([_step(i, _bash("pytest -q", "1 failed")) for i in range(3)])
    assert loop is not None
    note = warning_text(loop)
    assert "pytest -q" in note
    assert "will be stopped" in note
    assert loop.as_dict()["repeats"] == 3


def test_a_long_argument_is_clipped_in_the_warning() -> None:
    loop = detect_tool_loop([_step(i, _bash("x" * 500)) for i in range(3)])
    assert loop is not None
    assert len(loop.steps[0]) < 120


# ------------------------------------------------------------ the state API --


def test_state_warns_first_and_stops_only_past_the_warning() -> None:
    state = AgentState()
    for i in range(3):
        state = state.with_reasoning_step(_step(i, _bash("make", "error 2")))
    loop = state.tool_loop
    assert loop is not None
    assert state.has_tool_loop
    assert not state.tool_loop_warned(loop)
    assert state.should_terminate == (False, None)

    state = state.with_tool_loop_warning(loop)
    assert state.tool_loop_warned(loop)
    assert state.should_terminate == (False, None)

    state = state.with_reasoning_step(_step(4, _bash("make", "error 2")))
    assert state.tool_loop_persists
    assert state.should_terminate == (True, "tool_loop")


def test_state_round_trips_the_loop_settings_through_a_checkpoint() -> None:
    state = AgentState(tool_loop_read_only_threshold=6, tool_loop_read_only_tools=frozenset({"x"}))
    state = state.with_reasoning_step(_step(1, _read("a.py")))
    loop_free = AgentState.from_checkpoint(state.to_checkpoint())
    assert loop_free.tool_loop_read_only_threshold == 6
    assert loop_free.tool_loop_read_only_tools == frozenset({"x"})


# ----------------------------------------------------------- the agent loop --


@tool
def read(path: str) -> str:
    """Read a file."""
    return f"contents of {path}"


@tool
def grep(pattern: str) -> str:
    """Search the files."""
    return f"no match for {pattern}"


def _agent(model: Any, **kwargs: Any) -> Agent:
    return Agent(
        model=model,
        tools=[read, grep],
        reflexion=False,
        grounding=False,
        max_iterations=kwargs.pop("max_iterations", 20),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_an_agent_reading_the_same_file_forever_is_warned_then_stopped() -> None:
    model = ScriptedModel([tool_call("read", path="tests/test_units.py")], repeat_last=True)
    events = [e async for e in _agent(model).run("fix the units")]

    warnings = [e for e in events if isinstance(e, CustomEvent) and e.name == "tool_loop_warning"]
    terminate = next(e for e in events if isinstance(e, TerminateEvent))
    assert len(warnings) == 1
    assert warnings[0].data["repeats"] == 4
    assert terminate.reason == "tool_loop"
    # Four reads, the warning, one more read, and the stop: the model saw the
    # note before the call that stopped it.
    assert model.call_count == 5
    seen = model.received_messages[-1]
    assert any(
        m.role == Role.SYSTEM and (m.content or "").startswith("[Repeated tool call]") for m in seen
    )


@pytest.mark.asyncio
async def test_an_agent_that_changes_approach_after_the_warning_finishes() -> None:
    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        warned = any((m.content or "").startswith("[Repeated tool call]") for m in messages)
        if not warned:
            return tool_call("read", path="a.py")
        if not any(m.role == Role.TOOL and "no match" in (m.content or "") for m in messages):
            return tool_call("grep", pattern="units")
        return text("Done: the units are handled.")

    events = [e async for e in _agent(FunctionModel(handler)).run("fix it")]
    terminate = next(e for e in events if isinstance(e, TerminateEvent))
    assert terminate.reason == "complete"
    assert terminate.final_message == "Done: the units are handled."


@pytest.mark.asyncio
async def test_an_agent_rereading_between_work_runs_to_its_answer() -> None:
    script: list[ModelResponse] = []
    for i in range(16):
        if i in (8, 11, 12, 15):
            script.append(tool_call("read", call_id=f"r{i}", path="tests/test_units.py"))
        elif i % 2:
            script.append(tool_call("grep", call_id=f"g{i}", pattern=f"p{i}"))
        else:
            script.append(tool_call("read", call_id=f"r{i}", path=f"src/m{i}.py"))
    script.append(text("Added the chat entry point and its tests."))
    events = [e async for e in _agent(ScriptedModel(script)).run("add it")]

    assert not [e for e in events if isinstance(e, CustomEvent) and e.name == "tool_loop_warning"]
    terminate = next(e for e in events if isinstance(e, TerminateEvent))
    assert terminate.reason == "complete"


@pytest.mark.asyncio
async def test_explicit_mode_warns_but_never_stops_on_a_loop() -> None:
    model = ScriptedModel([tool_call("read", path="a.py")], repeat_last=True)
    agent = _agent(model, completion_mode="explicit", max_iterations=7)
    events = [e async for e in agent.run("go")]
    assert any(isinstance(e, CustomEvent) and e.name == "tool_loop_warning" for e in events)
    assert next(e for e in events if isinstance(e, TerminateEvent)).reason == "max_iterations"


@pytest.mark.asyncio
async def test_the_agent_config_sets_the_thresholds() -> None:
    model = ScriptedModel([tool_call("read", path="a.py")], repeat_last=True)
    agent = _agent(model, tool_loop_read_only_threshold=2)
    events = [e async for e in agent.run("go")]
    assert next(e for e in events if isinstance(e, TerminateEvent)).reason == "tool_loop"
    assert model.call_count == 3
