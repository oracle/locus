# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""A run that ends through ``task_complete`` carries its answer.

In ``completion_mode="explicit"`` the loop stops on the terminal tool; the
``TerminateEvent`` and ``AgentResult.message`` carry the ``summary`` the model
passed (or the assistant text around the call), and the model is told once
that a reply with no tool call ends nothing in that mode.
"""

from __future__ import annotations

from typing import Any

import pytest

import locus.integrations.fastmcp as fastmcp_mod
from locus.agent import Agent
from locus.core.events import TerminateEvent
from locus.core.messages import Role
from locus.testing import FunctionModel, ScriptedModel, text, tool_call
from locus.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First.
        b: Second.
    """
    return a + b


def _agent(model: Any, **overrides: Any) -> Agent:
    settings: dict[str, Any] = {
        "tools": [add],
        "completion_mode": "explicit",
        "reflexion": False,
        "grounding": False,
        "max_iterations": 10,
    }
    settings.update(overrides)
    return Agent(model=model, **settings)


async def test_the_task_complete_summary_is_the_final_message() -> None:
    model = ScriptedModel(
        [tool_call("add", a=2, b=3), tool_call("task_complete", summary="2 + 3 = 5.")]
    )
    agent = _agent(model)
    terminate: TerminateEvent | None = None
    async for event in agent.run("What is 2 + 3?"):
        if isinstance(event, TerminateEvent):
            terminate = event
    assert terminate is not None
    assert terminate.reason == "terminal_tool"
    assert terminate.final_message == "2 + 3 = 5."
    assert model.call_count == 2


async def test_arun_and_run_sync_carry_the_summary() -> None:
    result = await _agent(ScriptedModel([tool_call("task_complete", summary="Hello, Ada.")])).arun(
        "Greet me"
    )
    assert result.message == "Hello, Ada."

    sync = _agent(ScriptedModel([tool_call("task_complete", summary="Hello, Ada.")])).run_sync(
        "Greet me"
    )
    assert sync.message == "Hello, Ada."


async def test_a_result_argument_counts_and_an_empty_summary_falls_back_to_the_text() -> None:
    @tool(name="done")
    def done(result: str = "") -> str:
        """Finish the task.

        Args:
            result: The answer.
        """
        return "ok"

    agent = _agent(ScriptedModel([tool_call("done", result="42")]), tools=[done])
    assert (await agent.arun("answer")).message == "42"

    with_text = _agent(ScriptedModel([tool_call("task_complete", content="Done: 5.", summary="")]))
    assert (await with_text.arun("answer")).message == "Done: 5."


async def test_a_text_reply_in_explicit_mode_is_told_once_what_ends_the_run() -> None:
    """Before, the same context went back unchanged after a plain answer — the
    same answer came back, iteration after iteration, until the budget warning."""

    def decide(messages: list[Any], tools: list[Any]) -> Any:
        noted = any(
            m.role == Role.SYSTEM and "[Explicit completion]" in (m.content or "") for m in messages
        )
        if noted:
            return tool_call("task_complete", summary="5")
        return text("2 + 3 = 5")

    model = FunctionModel(decide)
    result = await _agent(model).arun("What is 2 + 3?")
    assert result.message == "5"
    assert model.call_count == 2
    notes = [
        m
        for m in result.state.messages
        if m.role == Role.SYSTEM and "[Explicit completion]" in (m.content or "")
    ]
    assert len(notes) == 1


async def test_the_note_is_sent_once_even_if_the_model_keeps_answering_in_text() -> None:
    model = ScriptedModel([text("5")], repeat_last=True)
    result = await _agent(model, max_iterations=4).arun("What is 2 + 3?")
    notes = [
        m
        for m in result.state.messages
        if m.role == Role.SYSTEM and "[Explicit completion]" in (m.content or "")
    ]
    assert len(notes) == 1
    assert result.stop_reason == "max_iterations"


async def test_the_mcp_relay_returns_the_summary() -> None:
    agent = _agent(ScriptedModel([tool_call("task_complete", summary="Hello, Ada.")]))
    assert await fastmcp_mod._run_agent_answering_questions(agent, "Greet me") == "Hello, Ada."


async def test_the_mcp_relay_answers_a_question_then_returns_the_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The producer side of elicitation: ``ask_user`` relayed to the MCP client,
    the run resumed with the reply, the greeting returned — not
    "Agent completed without response"."""

    async def answer(question: str, options: list[str] | None) -> str | None:
        assert question == "What is your first name?"
        return "Ada"

    monkeypatch.setattr(fastmcp_mod, "_ask_through_mcp", answer)

    def decide(messages: list[Any], tools: list[Any]) -> Any:
        answers = [m for m in messages if m.role == Role.TOOL]
        if not answers:
            return tool_call("ask_user", question="What is your first name?")
        return tool_call("task_complete", summary=f"Hello, {answers[-1].content}!")

    agent = _agent(FunctionModel(decide))
    assert await fastmcp_mod._run_agent_answering_questions(agent, "Greet me") == "Hello, Ada!"
