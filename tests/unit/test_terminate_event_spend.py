# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""``TerminateEvent`` says what the segment cost and, on failure, why.

A streaming consumer — a headless CLI writing a result record, a socket
front end — reads the terminate event and stops. Before these fields it had
to keep its own price table to report money, saw no cache counts, and got
``reason="error"`` with no message because the exception is raised after it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tulip.agent import Agent
from tulip.agent.runtime_loop import _error_text
from tulip.core.events import TerminateEvent
from tulip.core.messages import Message, ToolCall
from tulip.core.state import AgentState
from tulip.memory.backends.memory import MemoryCheckpointer
from tulip.models.base import ModelResponse
from tulip.models.metadata import ModelMetadata, register_metadata
from tulip.testing import FunctionModel


_PRICED = "tulip-test-terminate-priced"
_UNPRICED = "tulip-test-terminate-unpriced"

register_metadata(
    ModelMetadata(
        model_id=_PRICED,
        family="test",
        context_length=100_000,
        max_output_tokens=4_096,
        input_price_per_mtok="10.00",
        output_price_per_mtok="30.00",
    )
)
register_metadata(
    ModelMetadata(
        model_id=_UNPRICED, family="test", context_length=100_000, max_output_tokens=4_096
    )
)


def _model(model_id: str, usage: dict[str, int]) -> FunctionModel:
    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        return ModelResponse(message=Message.assistant("done"), usage=usage)

    model = FunctionModel(handler)
    model.config = SimpleNamespace(model=model_id)  # type: ignore[attr-defined]
    return model


async def _terminate(agent: Agent, **kwargs: Any) -> TerminateEvent:
    events = [e async for e in agent.run("go", **kwargs)]
    return next(e for e in events if isinstance(e, TerminateEvent))


async def test_a_priced_run_reports_its_cost() -> None:
    agent = Agent(
        model=_model(_PRICED, {"prompt_tokens": 1_000, "completion_tokens": 1_000}),
        reflexion=False,
        grounding=False,
    )
    done = await _terminate(agent)
    assert done.reason == "complete"
    assert done.cost_usd == pytest.approx(0.04)
    assert done.error is None


async def test_an_unpriced_run_reports_no_cost_rather_than_zero() -> None:
    agent = Agent(
        model=_model(_UNPRICED, {"prompt_tokens": 1_000, "completion_tokens": 1_000}),
        reflexion=False,
        grounding=False,
    )
    assert (await _terminate(agent)).cost_usd is None


async def test_cache_tokens_ride_on_the_usage() -> None:
    agent = Agent(
        model=_model(
            _PRICED,
            {
                "prompt_tokens": 1_000,
                "completion_tokens": 10,
                "cache_read_input_tokens": 800,
                "cache_creation_input_tokens": 150,
            },
        ),
        reflexion=False,
        grounding=False,
    )
    usage = (await _terminate(agent)).usage
    assert usage is not None
    assert usage["cache_read_input_tokens"] == 800
    assert usage["cache_creation_input_tokens"] == 150


async def test_a_provider_without_caching_reports_the_same_three_keys() -> None:
    agent = Agent(
        model=_model(_PRICED, {"prompt_tokens": 5, "completion_tokens": 5}),
        reflexion=False,
        grounding=False,
    )
    usage = (await _terminate(agent)).usage
    assert usage is not None
    assert set(usage) == {"prompt_tokens", "completion_tokens", "total_tokens"}


class _BoomError(RuntimeError):
    pass


async def _collect(events: Any, into: list[Any]) -> None:
    async for event in events:
        into.append(event)


def _failing_model() -> FunctionModel:
    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        raise _BoomError("provider said 400: model not found")

    model = FunctionModel(handler)
    model.config = SimpleNamespace(model=_PRICED)  # type: ignore[attr-defined]
    return model


async def test_an_error_termination_carries_the_message() -> None:
    agent = Agent(model=_failing_model(), reflexion=False, grounding=False)
    seen: list[Any] = []
    with pytest.raises(_BoomError):
        await _collect(agent.run("go"), seen)
    done = [e for e in seen if isinstance(e, TerminateEvent)]
    assert done[-1].reason == "error"
    assert done[-1].error == "_BoomError: provider said 400: model not found"


async def test_a_continued_turn_that_fails_says_why_too() -> None:
    store = MemoryCheckpointer()
    await store.save(
        AgentState(
            messages=(
                Message.system("sys"),
                Message.user("go"),
                Message.assistant(
                    content=None, tool_calls=[ToolCall(id="c1", name="nothing", arguments={})]
                ),
            ),
            iteration=1,
            max_iterations=5,
        ),
        "t",
    )
    agent = Agent(
        model=_failing_model(),
        checkpointer=store,
        reflexion=False,
        grounding=False,
    )
    seen: list[Any] = []
    with pytest.raises(_BoomError):
        await _collect(agent.continue_turn("t"), seen)
    done = [e for e in seen if isinstance(e, TerminateEvent)]
    assert done
    assert done[-1].reason == "error"
    assert "model not found" in (done[-1].error or "")


def test_error_text_is_one_bounded_line() -> None:
    assert _error_text(ValueError("")) == "ValueError"
    assert _error_text(ValueError("bad")) == "ValueError: bad"
    long = _error_text(RuntimeError("x" * 10_000))
    assert len(long) == 2_000
    assert long.endswith("…")


def test_the_new_fields_are_optional() -> None:
    event = TerminateEvent(
        reason="complete", iterations_used=1, final_confidence=1.0, total_tool_calls=0
    )
    assert event.cost_usd is None
    assert event.error is None
