# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Every way a run ends is reported as what it was.

A supervisor reading ``TerminateEvent.reason`` (or ``AgentResult.stop_reason``)
branches on it: ``complete`` is success, everything else is not. A run that is
stopped and reported as ``complete`` is the worst outcome — it looks like the
work was done. One test per termination path, each driven through the real
loop.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest

from tulip.agent import Agent
from tulip.agent.result import STOP_REASONS, normalize_stop_reason
from tulip.agent.runtime_loop import _normalize_stop_reason as runtime_normalize
from tulip.core.events import TerminateEvent
from tulip.core.messages import Message, Role, ToolCall, ToolResult
from tulip.models.base import ModelResponse
from tulip.models.metadata import ModelMetadata, register_metadata
from tulip.testing import FunctionModel, ScriptedModel, text, tool_call
from tulip.tools.decorator import tool


_PRICED = "tulip-test-stop-reason-priced"
register_metadata(
    ModelMetadata(
        model_id=_PRICED,
        family="test",
        context_length=200_000,
        max_output_tokens=8_192,
        input_price_per_mtok="10.00",
        output_price_per_mtok="30.00",
    )
)

_AGENTS: list[Agent] = []


@tool
def step(n: int) -> str:
    """Do one step of work."""
    return f"step {n} done"


@tool
def stop_me() -> str:
    """Cancel the running agent from inside a tool."""
    _AGENTS[-1].cancel()
    return "cancelling"


@tool
def edit(path: str) -> str:
    """Edit a file."""
    return f"edited {path}"


def _agent(model: Any, **kwargs: Any) -> Agent:
    agent = Agent(
        model=model,
        tools=[step, stop_me, edit],
        reflexion=False,
        grounding=False,
        **kwargs,
    )
    _AGENTS.append(agent)
    return agent


def _working_forever() -> FunctionModel:
    """Calls ``step`` with a new argument every turn: progress, never a loop."""

    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        n = len(messages)
        return tool_call("step", call_id=f"c{n}", n=n)

    return FunctionModel(handler)


async def _terminate(agent: Agent, prompt: str = "go") -> TerminateEvent:
    events = [e async for e in agent.run(prompt)]
    terminates = [e for e in events if isinstance(e, TerminateEvent)]
    assert len(terminates) == 1, "one TerminateEvent per run"
    return terminates[0]


@pytest.mark.asyncio
async def test_completion_is_complete() -> None:
    agent = _agent(ScriptedModel([tool_call("step", n=1), text("All done.")]))
    assert (await _terminate(agent)).reason == "complete"
    result = await _agent(ScriptedModel([text("Done.")])).arun("go")
    assert result.stop_reason == "complete"


@pytest.mark.asyncio
async def test_a_tool_loop_is_tool_loop() -> None:
    agent = _agent(ScriptedModel([tool_call("step", n=1)], repeat_last=True))
    end = await _terminate(agent)
    assert end.reason == "tool_loop"
    result = await _agent(ScriptedModel([tool_call("step", n=1)], repeat_last=True)).arun("go")
    assert result.stop_reason == "tool_loop"


@pytest.mark.asyncio
async def test_the_iteration_limit_is_max_iterations() -> None:
    end = await _terminate(_agent(_working_forever(), max_iterations=3))
    assert end.reason == "max_iterations"
    assert end.iterations_used == 3


@pytest.mark.asyncio
async def test_the_token_budget_is_token_budget() -> None:
    end = await _terminate(_agent(_working_forever(), token_budget=50))
    assert end.reason == "token_budget"


@pytest.mark.asyncio
async def test_the_spend_budget_is_cost_budget() -> None:
    model = _working_forever()
    model.config = SimpleNamespace(model=_PRICED)  # type: ignore[attr-defined]
    agent = _agent(model, max_cost_usd=0.000_5, max_tokens=10)
    end = await _terminate(agent)
    assert end.reason == "cost_budget"
    result = await _agent(model, max_cost_usd=0.000_5, max_tokens=10).arun("go")
    assert result.stop_reason == "cost_budget"


@pytest.mark.asyncio
async def test_the_time_budget_is_time_budget() -> None:
    def slow(messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        time.sleep(0.02)
        return tool_call("step", call_id=f"c{len(messages)}", n=len(messages))

    end = await _terminate(_agent(FunctionModel(slow), time_budget_seconds=0.01))
    assert end.reason == "time_budget"


@pytest.mark.asyncio
async def test_a_context_that_cannot_fit_is_context_exhausted() -> None:
    agent = _agent(ScriptedModel([text("never")]), system_prompt="x" * 40_000, context_window=8_000)
    assert (await _terminate(agent)).reason == "context_exhausted"


@pytest.mark.asyncio
async def test_a_cancel_is_cancelled() -> None:
    agent = _agent(ScriptedModel([tool_call("stop_me")], repeat_last=True))
    assert (await _terminate(agent)).reason == "cancelled"


@pytest.mark.asyncio
async def test_a_failure_is_error() -> None:
    def broken(messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        raise ValueError("the provider said no")

    agent = _agent(FunctionModel(broken), model_retry=None)
    events: list[Any] = []

    async def drain() -> None:
        async for event in agent.run("go"):
            events.append(event)

    with pytest.raises(ValueError, match="provider said no"):
        await drain()
    end = next(e for e in events if isinstance(e, TerminateEvent))
    assert end.reason == "error"
    assert end.error is not None


# ------------------------------------------------- an empty reply mid-task --


@pytest.mark.asyncio
async def test_an_empty_reply_mid_task_is_sent_back_with_the_tools() -> None:
    """The luach shape: an empty reply between tool calls. It used to get a
    tool-less "give your final answer" call, which ended the run before the
    edits and reported it complete."""
    empty = ModelResponse(message=Message.assistant(content=None), usage={}, stop_reason="stop")
    model = ScriptedModel(
        [
            tool_call("step", n=1),
            empty,
            tool_call("edit", path="src/chat.py"),
            text("Edited src/chat.py."),
        ]
    )
    result = await _agent(model).arun("add it")

    assert result.stop_reason == "complete"
    assert [e.tool_name for e in result.state.tool_executions] == ["step", "edit"]
    assert result.message == "Edited src/chat.py."
    # The send-back offered the tools; nothing asked for a final answer.
    assert model.offered_tools[2]
    notes = [m.content or "" for m in model.received_messages[2] if m.role == Role.SYSTEM]
    assert any(n.startswith("[Empty reply]") for n in notes)
    assert not any("[Final answer requested]" in n for n in notes)


@pytest.mark.asyncio
async def test_a_second_empty_reply_still_gets_a_final_answer() -> None:
    empty = ModelResponse(message=Message.assistant(content=None), usage={}, stop_reason="stop")
    model = ScriptedModel([tool_call("step", n=1), empty, empty, text("Summary of the work.")])
    result = await _agent(model).arun("go")
    assert result.stop_reason == "complete"
    assert result.message == "Summary of the work."
    assert model.offered_tools[3] == [], "the last resort is a tool-less call"


@pytest.mark.asyncio
async def test_an_empty_first_reply_keeps_the_direct_final_answer_call() -> None:
    """No tools called yet: the reasoning-model case the safety net was for."""
    empty = ModelResponse(message=Message.assistant(content=None), usage={}, stop_reason="stop")
    model = ScriptedModel([empty, text("Paris.")])
    result = await _agent(model).arun("capital of France?")
    assert result.message == "Paris."
    assert model.call_count == 2


@pytest.mark.asyncio
async def test_an_empty_reply_in_a_resumed_turn_is_sent_back_too() -> None:
    from tulip.core.state import AgentState, ToolExecution

    model = ScriptedModel(
        [
            ModelResponse(message=Message.assistant(content=None), usage={}, stop_reason="stop"),
            tool_call("edit", path="a.py"),
            text("Edited a.py."),
        ]
    )
    agent = _agent(model)
    state = AgentState(max_iterations=10).with_messages(
        [
            Message.system("sys"),
            Message.user("edit a.py"),
            Message.assistant(
                content=None, tool_calls=[ToolCall(id="c1", name="step", arguments={"n": 1})]
            ),
            Message.tool(ToolResult(tool_call_id="c1", name="step", content="step 1 done")),
        ]
    )
    state = state.with_tool_execution(
        ToolExecution(tool_name="step", tool_call_id="c1", arguments={"n": 1}, result="step 1 done")
    )
    events = [e async for e in agent._run_from_state(state, "edit a.py", None, None)]
    end = next(e for e in events if isinstance(e, TerminateEvent))
    assert end.reason == "complete"
    assert end.final_message == "Edited a.py."


# ---------------------------------------------------------- normalisation --


@pytest.mark.parametrize("reason", STOP_REASONS)
def test_every_stop_reason_survives_normalisation(reason: str) -> None:
    """``cost_budget`` was missing from one of two hand-kept lists and came
    back from a subagent as ``complete``."""
    assert normalize_stop_reason(reason) == reason
    assert runtime_normalize(reason) == reason


def test_a_subagent_stopped_by_its_spend_budget_says_so() -> None:
    assert runtime_normalize("cost_budget") == "cost_budget"
    assert normalize_stop_reason("subagent hit cost_budget") == "cost_budget"
