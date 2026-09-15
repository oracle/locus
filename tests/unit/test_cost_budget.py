# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""A spend budget that stops the call that would cross it.

``token_budget`` is checked at the top of each iteration, so it stops a run from
continuing but lets one oversized turn through. These tests pin a cost budget
that is checked before each call against that call's worst case, reports what the
run spent, and refuses to exist for a model whose prices it cannot know.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tulip.agent import Agent
from tulip.agent.agent import _normalize_stop_reason
from tulip.core.events import TerminateEvent
from tulip.core.messages import Message, ToolCall
from tulip.core.state import AgentState
from tulip.models.base import ModelResponse
from tulip.models.metadata import ModelMetadata, register_metadata
from tulip.testing import FunctionModel
from tulip.tools.decorator import tool


_PRICED = "tulip-test-priced"
_UNPRICED = "tulip-test-unpriced"

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
register_metadata(
    ModelMetadata(
        model_id=_UNPRICED, family="test", context_length=200_000, max_output_tokens=8_192
    )
)


@tool
def lookup(item: int) -> str:
    """Look something up."""
    return f"item {item}"


def _looping_model(model_id: str, calls: list[int]) -> FunctionModel:
    """Calls ``lookup`` forever, reporting 1,000 input and 1,000 output tokens a turn."""

    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        calls.append(len(calls))
        return ModelResponse(
            message=Message.assistant(
                content=None,
                tool_calls=[
                    ToolCall(id=f"c{len(calls)}", name="lookup", arguments={"item": len(calls)})
                ],
            ),
            usage={"prompt_tokens": 1_000, "completion_tokens": 1_000},
            stop_reason="tool_calls",
        )

    model = FunctionModel(handler)
    model.config = SimpleNamespace(model=model_id)  # type: ignore[attr-defined]
    return model


def test_cost_accumulates_from_known_prices_only() -> None:
    priced = AgentState(input_price_per_mtok=10.0, output_price_per_mtok=30.0)
    unpriced = AgentState()

    assert priced.with_token_usage(1_000, 1_000).cost_usd_used == pytest.approx(0.04)
    assert unpriced.with_token_usage(1_000, 1_000).cost_usd_used == 0.0
    assert unpriced.cost_of(1_000, 1_000) is None


def test_crossing_the_budget_after_a_call_stops_the_run() -> None:
    state = AgentState(input_price_per_mtok=10.0, output_price_per_mtok=30.0, cost_budget_usd=0.05)

    assert state.with_token_usage(1_000, 1_000).should_terminate == (False, None)
    assert state.with_token_usage(2_000, 1_000).should_terminate == (True, "cost_budget")


def test_the_worst_case_of_the_next_call_is_checked_before_it() -> None:
    state = AgentState(
        input_price_per_mtok=10.0, output_price_per_mtok=30.0, cost_budget_usd=0.10
    ).with_message(Message.user("x" * 4_000))  # ~1,000 input tokens = $0.01

    assert not state.would_exceed_cost_budget(max_output_tokens=2_000)  # 0.01 + 0.06
    assert state.would_exceed_cost_budget(max_output_tokens=4_000)  # 0.01 + 0.12
    assert not AgentState().would_exceed_cost_budget(max_output_tokens=10**9), "no budget, no stop"


@pytest.mark.asyncio
async def test_the_run_stops_before_the_call_that_could_cross_the_budget() -> None:
    calls: list[int] = []
    agent = Agent(
        model=_looping_model(_PRICED, calls),
        tools=[lookup],
        max_cost_usd=0.20,
        max_tokens=4_000,
        max_iterations=50,
        reflexion=False,
        grounding=False,
    )

    events = [event async for event in agent.run("look everything up")]

    terminate = next(e for e in events if isinstance(e, TerminateEvent))
    assert terminate.reason == "cost_budget"
    # Each call spends $0.04 and its worst case is ~$0.12: after two calls
    # ($0.08), a third could reach ~$0.20+, so it is never made.
    assert len(calls) == 2
    assert agent._last_run_state is not None
    assert agent._last_run_state.cost_usd_used == pytest.approx(0.08)
    assert agent._last_run_state.cost_usd_used <= 0.20


def test_the_result_reports_what_the_run_cost() -> None:
    calls: list[int] = []
    agent = Agent(
        model=_looping_model(_PRICED, calls),
        tools=[lookup],
        max_cost_usd=0.20,
        max_tokens=4_000,
        max_iterations=50,
        reflexion=False,
        grounding=False,
    )

    result = agent.run_sync("look everything up")

    assert result.stop_reason == "cost_budget"
    assert result.metrics.cost_usd == pytest.approx(0.08)


def test_an_unpriced_run_reports_no_cost_rather_than_zero() -> None:
    calls: list[int] = []
    agent = Agent(
        model=_looping_model(_UNPRICED, calls),
        tools=[lookup],
        max_iterations=2,
        reflexion=False,
        grounding=False,
    )

    assert agent.run_sync("look it up").metrics.cost_usd is None


def test_a_budget_on_an_unpriced_model_is_refused_when_the_agent_is_built() -> None:
    with pytest.raises(ValueError, match="max_cost_usd needs prices"):
        Agent(
            model=_looping_model(_UNPRICED, []),
            tools=[lookup],
            max_cost_usd=1.0,
            reflexion=False,
            grounding=False,
        )


def test_cost_budget_is_a_stop_reason() -> None:
    assert _normalize_stop_reason("cost_budget") == "cost_budget"
