# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Coverage tests for the self-contained helper methods of
``AgentRuntimeMixin`` plus the ``_bus_bridge`` decorator.

These exercise the no-op / fallback / error branches of the reasoning,
grounding, structured-output, and GSAR helpers by calling them directly on
a constructed ``Agent`` with scripted stub models.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from tulip.agent import Agent
from tulip.agent.config import GSARConfig
from tulip.core.events import GroundingEvent, TerminateEvent, ThinkEvent
from tulip.core.messages import Message
from tulip.core.state import AgentState, ToolExecution
from tulip.models.base import ModelResponse


# ---------------------------------------------------------------------------
# Stub models / schema
# ---------------------------------------------------------------------------


class _OneShotModel:
    def __init__(self, content: str) -> None:
        self.content = content

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        return ModelResponse(message=Message.assistant(self.content), usage={})

    async def stream(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


class _StructuredModel:
    """Advertises native structured output and records the kwargs it got."""

    supports_structured_output = True

    def __init__(self, content: str) -> None:
        self.content = content
        self.last_kwargs: dict[str, Any] | None = None

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.last_kwargs = kwargs
        return ModelResponse(message=Message.assistant(self.content), usage={})

    async def stream(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


class _ResponseFormatRejectingModel:
    """Raises ``TypeError`` when handed a ``response_format`` kwarg."""

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        if "response_format" in kwargs:
            raise TypeError("response_format not supported")
        return ModelResponse(message=Message.assistant('{"answer": "ok"}'), usage={})

    async def stream(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


class _RaisingModel:
    async def complete(self, **kwargs: Any) -> ModelResponse:
        raise RuntimeError("model unavailable")

    async def stream(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


class _Answer(BaseModel):
    answer: str


def _agent(model: Any, **kwargs: Any) -> Agent:
    return Agent(model=model, reflexion=False, grounding=False, **kwargs)


# ---------------------------------------------------------------------------
# _bus_bridge swallows telemetry failures (lines 83-84)
# ---------------------------------------------------------------------------


async def test_bus_bridge_swallows_telemetry_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _boom(event: Any) -> None:
        raise RuntimeError("telemetry boom")

    monkeypatch.setattr("tulip.observability.agent_bridge.bridge_tulip_event", _boom)

    agent = _agent(_OneShotModel("hi"), max_iterations=5)
    events: list[Any] = []
    async for ev in agent.run("go"):
        events.append(ev)
    # Telemetry raised on every event but the loop still produced them.
    assert any(isinstance(e, ThinkEvent) for e in events)
    assert any(isinstance(e, TerminateEvent) for e in events)


# ---------------------------------------------------------------------------
# _create_initial_state with a callable system prompt (line 1338)
# ---------------------------------------------------------------------------


async def test_create_initial_state_callable_system_prompt() -> None:
    agent = _agent(
        _OneShotModel("hi"),
        system_prompt=lambda ctx: f"dynamic::{ctx['prompt']}",
    )
    state = await agent._create_initial_state("hello", None, {})
    system_msgs = [m for m in state.messages if m.role.value == "system"]
    assert any("dynamic::hello" in (m.content or "") for m in system_msgs)


# ---------------------------------------------------------------------------
# _get_final_state reconstructs a fresh state (lines 1366-1367)
# ---------------------------------------------------------------------------


async def test_get_final_state_returns_initial_state() -> None:
    agent = _agent(_OneShotModel("hi"))
    state = await agent._get_final_state("prompt", None, None)
    assert isinstance(state, AgentState)
    assert any(m.role.value == "user" for m in state.messages)


# ---------------------------------------------------------------------------
# Native response_format threaded through _get_model_response (1539-1557)
# ---------------------------------------------------------------------------


async def test_native_response_format_passed_to_model() -> None:
    model = _StructuredModel('{"answer": "42"}')
    agent = Agent(
        model=model,
        output_schema=_Answer,
        reflexion=False,
        grounding=False,
        max_iterations=5,
    )
    async for _ev in agent.run("answer?"):
        pass
    assert model.last_kwargs is not None
    assert "response_format" in model.last_kwargs


# ---------------------------------------------------------------------------
# _structure_output: no schema is a no-op (line 1683)
# ---------------------------------------------------------------------------


async def test_structure_output_no_schema_is_noop() -> None:
    agent = _agent(_OneShotModel("hi"))
    state = AgentState()
    parsed, err, out_state = await agent._structure_output(state, "anything")
    assert parsed is None
    assert err is None
    assert out_state is state


# ---------------------------------------------------------------------------
# _structure_output: provider rejects response_format -> retry (1715-1717)
# ---------------------------------------------------------------------------


async def test_structure_output_typeerror_retries_without_response_format() -> None:
    agent = Agent(
        model=_ResponseFormatRejectingModel(),
        output_schema=_Answer,
        output_schema_retries=2,
        reflexion=False,
        grounding=False,
    )
    state = await agent._create_initial_state("p", None, None)
    parsed, err, _state = await agent._structure_output(state, "not valid json")
    assert err is None
    assert parsed is not None
    assert parsed.answer == "ok"


# ---------------------------------------------------------------------------
# _apply_reflexion fallback when no reflector (line 1750)
# ---------------------------------------------------------------------------


async def test_apply_reflexion_without_reflector_is_noop() -> None:
    agent = _agent(_OneShotModel("hi"))
    assert agent._reflector is None
    state = AgentState(confidence=0.4)
    event, out_state = await agent._apply_reflexion(state)
    assert event.assessment == "on_track"
    assert event.confidence_delta == 0.0
    assert out_state is state


# ---------------------------------------------------------------------------
# _apply_grounding default returns (lines 1794 + 1809)
# ---------------------------------------------------------------------------


async def test_apply_grounding_without_evaluator_returns_default() -> None:
    agent = _agent(_OneShotModel("hi"))
    assert agent._grounding_evaluator is None
    event, out_state = await agent._apply_grounding(AgentState(), "Some grounded claim here.")
    assert isinstance(event, GroundingEvent)
    assert event.claims_evaluated == 0
    assert event.requires_replan is False
    assert out_state is not None


async def test_apply_grounding_with_no_claims_returns_default() -> None:
    agent = Agent(model=_OneShotModel("hi"), grounding=True, reflexion=False)
    assert agent._grounding_evaluator is not None
    # Empty response -> no claims extracted -> default event.
    event, _state = await agent._apply_grounding(AgentState(), "")
    assert event.claims_evaluated == 0
    assert event.score == 1.0


# ---------------------------------------------------------------------------
# _gather_evidence truncates long results (line 1863)
# ---------------------------------------------------------------------------


def test_gather_evidence_truncates_long_results() -> None:
    state = AgentState()
    state = state.with_tool_execution(
        ToolExecution(
            tool_name="search",
            tool_call_id="c1",
            arguments={"q": "x"},
            result="y" * 600,
        )
    )
    evidence = Agent._gather_evidence(state)
    assert len(evidence) == 1
    assert evidence[0].endswith("...")
    # 500-char cap + the "..." suffix + the "[search]: " prefix.
    assert len(evidence[0]) < 600


# ---------------------------------------------------------------------------
# _run_gsar_judgment default-judge / error-skip / weight-map (1923-1940)
# ---------------------------------------------------------------------------


async def test_run_gsar_judgment_default_judge_and_weight_map() -> None:
    agent = Agent(
        model=_RaisingModel(),
        gsar=GSARConfig(weight_map={"tool_match": 1.0}),
        reflexion=False,
        grounding=False,
    )
    state = AgentState()
    state = state.with_tool_execution(
        ToolExecution(
            tool_name="ok_tool",
            tool_call_id="c1",
            arguments={"q": "x"},
            result="found data",
        )
    )
    state = state.with_tool_execution(
        ToolExecution(
            tool_name="bad_tool",
            tool_call_id="c2",
            arguments={},
            result=None,
            error="boom",
        )
    )
    judgment, score, decision = await agent._run_gsar_judgment(state, "final synthesis text")
    # The default judge ran; its model raised internally so it returned the
    # safe-default judgment, which still scores + decides.
    assert decision in {"proceed", "regenerate", "replan", "abstain"}
    assert score is not None


async def test_run_gsar_judgment_disabled_returns_none() -> None:
    agent = _agent(MagicMock())
    judgment, score, decision = await agent._run_gsar_judgment(AgentState(), "msg")
    assert judgment is None
    assert score is None
    assert decision is None
