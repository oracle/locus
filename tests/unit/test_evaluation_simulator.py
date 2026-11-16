# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""A simulated user plays a multi-turn case against the agent.

These tests pin the budget (the conversation stops on the goal, a stop phrase,
the turn budget or an agent error), how the agent is driven (one thread with a
checkpointer, the transcript carried without one), what the runner grades (every
agent reply and every tool call across the turns, the whole transcript for the
judge), and that a simulator that is down stops the suite rather than failing
the agent.
"""

from __future__ import annotations

from typing import Any

import pytest

from locus.agent import Agent
from locus.core.messages import Message
from locus.evaluation import (
    Conversation,
    ConversationTurn,
    EvalCase,
    EvalRunner,
    JudgeUnavailableError,
    LLMJudge,
    SimulatedUser,
    SimulatorUnavailableError,
    ToolSelection,
    UserSimulator,
)
from locus.evaluation.simulator import render_transcript
from locus.memory.backends.memory import MemoryCheckpointer
from locus.models.base import ModelResponse
from locus.testing import FunctionModel, ScriptedModel, text, tool_call
from locus.tools.decorator import tool


class _UserModel:
    """A simulated-user model that replays scripted lines and records its prompts."""

    def __init__(self, *lines: str) -> None:
        self.lines = list(lines)
        self.prompts: list[str] = []

    async def complete(
        self, messages: list[Message], tools: Any = None, **kwargs: Any
    ) -> ModelResponse:
        self.prompts.append(messages[-1].content or "")
        line = self.lines.pop(0) if self.lines else "[DONE]"
        return ModelResponse(message=Message.assistant(content=line), usage={})

    async def stream(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


class _DeadModel(_UserModel):
    async def complete(self, *a: Any, **k: Any) -> ModelResponse:
        raise ConnectionError("simulator is down")


@tool
def issue_refund(order_id: str) -> str:
    """Refund an order."""
    return f"refunded {order_id}"


def _support_agent(*, checkpointer: Any = None) -> Agent:
    """Asks for the order id, then refunds it, counting the user turns it can see."""

    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        user_turns = sum(1 for m in messages if m.role == "user")
        if any(m.role == "tool" for m in messages):
            return text(f"Refund issued for your order. Anything else? (turns seen: {user_turns})")
        if "ord-" in (messages[-1].content or "").lower():
            return tool_call("issue_refund", order_id="ord-4821")
        return text("Which order should I refund?")

    return Agent(model=FunctionModel(handler), tools=[issue_refund], checkpointer=checkpointer)


def _spec(**overrides: Any) -> SimulatedUser:
    fields: dict[str, Any] = {"persona": "a terse customer", "goal": "get ord-4821 refunded"}
    fields.update(overrides)
    return SimulatedUser(**fields)


# --------------------------------------------------------------------------
# UserSimulator
# --------------------------------------------------------------------------


async def test_next_message_is_none_once_the_user_says_done() -> None:
    model = _UserModel("It's ord-4821.", "[DONE]")
    simulator = UserSimulator(model)
    turns = [ConversationTurn(role="user", content="hi")]
    assert await simulator.next_message(_spec(), turns) == "It's ord-4821."
    assert await simulator.next_message(_spec(), turns) is None
    assert "WHO YOU ARE" in model.prompts[0]
    assert "user: hi" in model.prompts[0]


async def test_a_dead_simulator_model_is_unavailable_not_a_finding() -> None:
    with pytest.raises(SimulatorUnavailableError, match="could not be reached"):
        await UserSimulator(_DeadModel()).next_message(_spec(), [])
    assert issubclass(SimulatorUnavailableError, JudgeUnavailableError)


async def test_the_conversation_ends_when_the_goal_is_met() -> None:
    model = _UserModel("Order ord-4821 please.", "[DONE]")
    conversation = await UserSimulator(model).converse(
        _support_agent(checkpointer=MemoryCheckpointer()),
        _spec(max_turns=5),
        opening="I want a refund",
    )
    assert conversation.stopped_by == "goal_met"
    assert conversation.ended
    assert conversation.user_turns == 2
    assert [turn.role for turn in conversation.turns] == ["user", "agent", "user", "agent"]
    assert conversation.turns[-1].tool_calls == ["issue_refund"]
    assert [e.tool_name for e in conversation.tool_executions] == ["issue_refund"]
    assert "Refund issued" in conversation.message
    assert "Which order" in conversation.message


async def test_a_checkpointer_continues_one_thread_across_turns() -> None:
    """The agent sees both user turns on the second call: the thread was continued."""
    model = _UserModel("Order ord-4821 please.")
    conversation = await UserSimulator(model).converse(
        _support_agent(checkpointer=MemoryCheckpointer()), _spec(), opening="I want a refund"
    )
    assert "turns seen: 2" in conversation.turns[-1].content


async def test_without_a_checkpointer_the_transcript_travels_in_the_prompt() -> None:
    seen: list[str] = []

    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        seen.append(messages[-1].content or "")
        return text("Noted.")

    agent = Agent(model=FunctionModel(handler))
    model = _UserModel("Second message.")
    await UserSimulator(model).converse(agent, _spec(max_turns=2), opening="First message.")
    assert seen[0] == "First message."
    assert seen[1].startswith("Conversation so far:\nuser: First message.\nassistant: Noted.")
    assert seen[1].endswith("user: Second message.")


async def test_the_turn_budget_stops_the_conversation() -> None:
    model = _UserModel("more", "more", "more", "more")
    conversation = await UserSimulator(model).converse(
        Agent(model=ScriptedModel([text("Tell me more.")], repeat_last=True)),
        _spec(max_turns=2),
        opening="hello",
    )
    assert conversation.stopped_by == "max_turns"
    assert conversation.user_turns == 2
    assert not conversation.ended
    assert len(model.prompts) == 1  # one model turn after the opening, then the budget


async def test_a_stop_phrase_in_the_agent_reply_ends_the_conversation() -> None:
    model = _UserModel("more", "more")
    conversation = await UserSimulator(model).converse(
        Agent(model=ScriptedModel([text("Done. Ticket closed."), text("still here")])),
        _spec(max_turns=5, stop_on=["ticket closed"]),
        opening="hello",
    )
    assert conversation.stopped_by == "stop_phrase"
    assert conversation.ended
    assert model.prompts == []


async def test_an_agent_that_raises_ends_the_conversation_with_the_error() -> None:
    class _Broken:
        config = None

        async def arun(self, prompt: str, **_: Any) -> Any:
            raise RuntimeError("tool server down")

    conversation = await UserSimulator(_UserModel()).converse(_Broken(), _spec(), opening="hello")
    assert conversation.stopped_by == "agent_error"
    assert conversation.error == "tool server down"
    assert conversation.turns[-1].role == "user"


def test_the_transcript_renders_with_roles_and_the_metrics_sum() -> None:
    conversation = Conversation(
        turns=[
            ConversationTurn(role="user", content="hi"),
            ConversationTurn(role="agent", content="hello"),
        ],
        stopped_by="goal_met",
        total_tokens=120,
        cost_usd=0.5,
    )
    assert conversation.transcript == "user: hi\nassistant: hello"
    assert conversation.metrics.total_tokens == 120
    assert conversation.metrics.cost_usd == 0.5
    assert render_transcript([]) == "(empty)"


async def test_tokens_and_cost_are_summed_over_the_turns_and_unknown_when_any_is() -> None:
    class _Result:
        def __init__(self, tokens: int | None, cost: float | None) -> None:
            self.message = "ok"
            self.iterations = 1
            self.tool_executions: list[Any] = []
            self.metrics = type("M", (), {"total_tokens": tokens, "cost_usd": cost})()

    class _Agent:
        config = None

        def __init__(self, results: list[_Result]) -> None:
            self.results = results

        async def arun(self, prompt: str, **_: Any) -> Any:
            return self.results.pop(0)

    priced = await UserSimulator(_UserModel("again")).converse(
        _Agent([_Result(100, 0.1), _Result(50, 0.2)]), _spec(max_turns=2), opening="go"
    )
    assert priced.total_tokens == 150
    assert priced.cost_usd == pytest.approx(0.3)
    assert priced.iterations == 2
    unpriced = await UserSimulator(_UserModel("again")).converse(
        _Agent([_Result(100, 0.1), _Result(50, None)]), _spec(max_turns=2), opening="go"
    )
    assert unpriced.cost_usd is None


# --------------------------------------------------------------------------
# On a case
# --------------------------------------------------------------------------


def _case(**overrides: Any) -> EvalCase:
    fields: dict[str, Any] = {
        "name": "refund_conversation",
        "prompt": "I want a refund",
        "simulated_user": _spec(max_turns=4),
        "expected_tools": ["issue_refund"],
        "expected_output_contains": ["refund issued"],
    }
    fields.update(overrides)
    return EvalCase(**fields)


def test_the_sync_path_refuses_a_simulated_case_rather_than_running_its_opening_alone() -> None:
    report = EvalRunner(agent=_support_agent()).run([_case()])
    result = report.results[0]
    assert not result.passed
    assert result.checks == {"simulated_user:requires_arun": False}
    assert result.output == ""


async def test_without_a_simulator_or_a_judge_the_case_fails_loudly() -> None:
    report = await EvalRunner(agent=_support_agent(), concurrency=1).arun([_case()])
    assert report.results[0].checks == {"simulated_user:no_model_configured": False}
    assert not report.results[0].passed


async def test_the_runner_grades_the_whole_conversation() -> None:
    model = _UserModel("Order ord-4821 please.", "[DONE]")
    runner = EvalRunner(
        agent=_support_agent(checkpointer=MemoryCheckpointer()),
        simulator=UserSimulator(model),
        concurrency=1,
    )
    case = _case(evaluators=[ToolSelection(expected=["issue_refund"])])
    report = await runner.arun([case])
    result = report.results[0]
    assert result.passed, result.checks
    assert result.checks == {
        "tool_called:issue_refund": True,
        "output_contains:refund issued": True,
        "tool_selection": True,
        "simulated_user:goal_met": True,
    }
    assert result.stopped_by == "goal_met"
    assert [turn.role for turn in result.transcript] == ["user", "agent", "user", "agent"]
    assert result.tools_called == ["issue_refund"]
    assert result.iterations == 3


async def test_a_conversation_that_hit_the_budget_fails_goal_met() -> None:
    runner = EvalRunner(
        agent=Agent(model=ScriptedModel([text("Tell me more.")], repeat_last=True)),
        simulator=UserSimulator(_UserModel("more", "more", "more")),
        concurrency=1,
    )
    case = _case(
        simulated_user=_spec(max_turns=2), expected_tools=[], expected_output_contains=["more"]
    )
    result = (await runner.arun([case])).results[0]
    assert result.checks["simulated_user:goal_met"] is False
    assert not result.passed
    assert result.stopped_by == "max_turns"
    relaxed = _case(
        simulated_user=_spec(max_turns=2, require_goal_met=False),
        expected_tools=[],
        expected_output_contains=["more"],
    )
    runner = EvalRunner(
        agent=Agent(model=ScriptedModel([text("Tell me more.")], repeat_last=True)),
        simulator=UserSimulator(_UserModel("more", "more", "more")),
        concurrency=1,
    )
    assert (await runner.arun([relaxed])).results[0].passed


async def test_the_judges_model_plays_the_user_by_default_and_reads_the_transcript() -> None:
    class _JudgeAndUser(_UserModel):
        """One model for both roles: user lines when asked for a user turn, else a verdict."""

        async def complete(
            self, messages: list[Message], tools: Any = None, **kwargs: Any
        ) -> ModelResponse:
            request = messages[-1].content or ""
            self.prompts.append(request)
            if "WHO YOU ARE" in request:
                return ModelResponse(
                    message=Message.assistant(
                        content=self.lines.pop(0) if self.lines else "[DONE]"
                    ),
                    usage={},
                )
            return ModelResponse(
                message=Message.assistant(
                    content='{"passed": true, "score": 0.9, "reason": "polite"}'
                ),
                usage={},
            )

    model = _JudgeAndUser("Order ord-4821 please.")
    judge = LLMJudge(model, channel="text")
    runner = EvalRunner(
        agent=_support_agent(checkpointer=MemoryCheckpointer()), judge=judge, concurrency=1
    )
    assert runner.simulator is not None
    assert runner.simulator.model is model
    case = _case(rubric="Passes if the agent asked for the order before refunding.")
    result = (await runner.arun([case])).results[0]
    assert result.passed, result.checks
    assert result.score == pytest.approx(0.9)
    grading = next(p for p in model.prompts if "RUBRIC" in p)
    assert "assistant: Which order should I refund?" in grading
    assert "user: Order ord-4821 please." in grading
    assert "Their goal: get ord-4821 refunded" in grading


async def test_a_dead_simulator_stops_the_suite() -> None:
    runner = EvalRunner(
        agent=_support_agent(), simulator=UserSimulator(_DeadModel()), concurrency=1
    )
    with pytest.raises(SimulatorUnavailableError):
        await runner.arun([_case()])


async def test_an_agent_error_mid_conversation_is_a_failed_case_with_the_error() -> None:
    calls = 0

    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("model exploded")
        return text("Which order?")

    runner = EvalRunner(
        agent=Agent(model=FunctionModel(handler)),
        simulator=UserSimulator(_UserModel("ord-4821")),
        concurrency=1,
    )
    result = (await runner.arun([_case(expected_tools=[])])).results[0]
    assert not result.passed
    assert result.stopped_by == "agent_error"
    assert result.checks["simulated_user:agent_answered"] is False
    assert result.error is not None
    assert "model exploded" in result.error


async def test_a_simulated_case_respects_the_duration_budget() -> None:
    import asyncio

    class _Slow:
        config = None

        async def arun(self, prompt: str, **_: Any) -> Any:
            await asyncio.sleep(5)

    runner = EvalRunner(agent=_Slow(), simulator=UserSimulator(_UserModel()), concurrency=1)
    result = (await runner.arun([_case(max_duration_ms=50, expected_tools=[])])).results[0]
    assert result.timed_out
    assert result.checks == {"within_duration_budget": False}


def test_a_simulated_user_enters_the_fingerprint_and_a_case_file() -> None:
    from locus.evaluation import suite_fingerprint

    short = _case(simulated_user=_spec(max_turns=2))
    long = _case(simulated_user=_spec(max_turns=6))
    assert suite_fingerprint([short]) != suite_fingerprint([long])
    data = short.model_dump(mode="json")
    assert data["simulated_user"]["persona"] == "a terse customer"
    assert EvalCase.model_validate(data) == short
