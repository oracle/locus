# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``locus.hooks.builtin.steering`` (LLM-powered steering).

The steering hook delegates decisions to a separate LLM. Tests use a
stub model that returns canned responses to exercise each branch:
PROCEED / GUIDE / INTERRUPT, exception path, allow-list interrupts,
context tracking, and the response evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from locus.core.messages import Message
from locus.hooks.builtin.steering import (
    SteeringAction,
    SteeringContext,
    SteeringDecision,
    SteeringHook,
)
from locus.models.base import ModelResponse


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubModel:
    """Returns a configured ``content`` string from ``complete``."""

    def __init__(self, content: str = "PROCEED", *, raise_exc: bool = False) -> None:
        self._content = content
        self._raise = raise_exc
        self.calls: list[Any] = []

    async def complete(self, **kwargs: Any) -> ModelResponse:
        self.calls.append(kwargs)
        if self._raise:
            raise RuntimeError("model down")
        return ModelResponse(message=Message.assistant(self._content))


@dataclass
class _BeforeToolEvent:
    tool_name: str
    arguments: dict[str, Any]
    cancel: bool | str = False


@dataclass
class _AfterModelEvent:
    response: ModelResponse


@dataclass
class _BeforeModelEvent:
    messages: list[Message] = field(default_factory=list)
    tools: list[dict[str, Any]] | None = None


# ---------------------------------------------------------------------------
# SteeringContext / SteeringDecision
# ---------------------------------------------------------------------------


class TestSteeringContext:
    def test_record_tool_call(self) -> None:
        ctx = SteeringContext()
        ctx.record_tool_call("search", {"q": "hi"})
        assert len(ctx.tool_calls) == 1
        assert ctx.tool_calls[0]["tool"] == "search"

    def test_to_prompt_with_policy(self) -> None:
        ctx = SteeringContext(policy="Read-only.")
        out = ctx.to_prompt()
        assert "## Policy" in out
        assert "Read-only." in out
        assert "Model calls: 0" in out

    def test_to_prompt_without_policy(self) -> None:
        ctx = SteeringContext()
        out = ctx.to_prompt()
        assert "## Policy" not in out

    def test_to_prompt_truncates_to_last_five_calls(self) -> None:
        ctx = SteeringContext()
        for i in range(10):
            ctx.record_tool_call(f"tool_{i}", {"i": i})
        out = ctx.to_prompt()
        # Only the last 5 are listed.
        assert "tool_5" in out
        assert "tool_9" in out
        assert "tool_0" not in out


# ---------------------------------------------------------------------------
# Construction + properties
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_priority_in_security_band(self) -> None:
        from locus.hooks.provider import HookPriority

        hook = SteeringHook(model=_StubModel())
        assert hook.priority == HookPriority.SECURITY_DEFAULT

    def test_explicit_priority(self) -> None:
        hook = SteeringHook(model=_StubModel(), priority=42)
        assert hook.priority == 42

    def test_name(self) -> None:
        assert SteeringHook(model=_StubModel()).name == "SteeringHook"


# ---------------------------------------------------------------------------
# _evaluate_tool_call decision parsing
# ---------------------------------------------------------------------------


class TestEvaluateToolCallParsing:
    @pytest.mark.asyncio
    async def test_proceed_default(self) -> None:
        hook = SteeringHook(model=_StubModel(content="PROCEED"))
        decision = await hook._evaluate_tool_call("search", {"q": "x"})
        assert decision.action == SteeringAction.PROCEED

    @pytest.mark.asyncio
    async def test_unknown_response_treated_as_proceed(self) -> None:
        # Defensive: unparseable model output should NOT block execution.
        hook = SteeringHook(model=_StubModel(content="¯\\_(ツ)_/¯"))
        decision = await hook._evaluate_tool_call("search", {"q": "x"})
        assert decision.action == SteeringAction.PROCEED

    @pytest.mark.asyncio
    async def test_guide_response_carries_reason(self) -> None:
        # The hook upper-cases the response; case-insensitive assertion below.
        hook = SteeringHook(model=_StubModel(content="GUIDE: scope outside policy"))
        decision = await hook._evaluate_tool_call("delete", {"id": 1})
        assert decision.action == SteeringAction.GUIDE
        assert "SCOPE OUTSIDE POLICY" in decision.reason
        # The user-facing guidance message wraps the reason.
        assert "Steering blocked" in decision.guidance

    @pytest.mark.asyncio
    async def test_interrupt_response(self) -> None:
        hook = SteeringHook(model=_StubModel(content="INTERRUPT: needs human"))
        decision = await hook._evaluate_tool_call("write", {"path": "/etc"})
        assert decision.action == SteeringAction.INTERRUPT
        assert "NEEDS HUMAN" in decision.reason

    @pytest.mark.asyncio
    async def test_model_failure_defaults_to_proceed(self) -> None:
        # A failed steering eval must not stall the agent — fail-open is
        # documented behaviour. The exception is logged, decision = PROCEED.
        hook = SteeringHook(model=_StubModel(raise_exc=True))
        decision = await hook._evaluate_tool_call("search", {})
        assert decision.action == SteeringAction.PROCEED


# ---------------------------------------------------------------------------
# on_before_tool_call dispatch
# ---------------------------------------------------------------------------


class TestOnBeforeToolCall:
    @pytest.mark.asyncio
    async def test_proceed_does_not_cancel(self) -> None:
        hook = SteeringHook(model=_StubModel(content="PROCEED"))
        event = _BeforeToolEvent(tool_name="search", arguments={"q": "x"})
        await hook.on_before_tool_call(event)
        assert event.cancel is False
        assert hook.decisions[0].action == SteeringAction.PROCEED

    @pytest.mark.asyncio
    async def test_guide_decision_cancels_with_message(self) -> None:
        hook = SteeringHook(model=_StubModel(content="GUIDE: nope"))
        event = _BeforeToolEvent(tool_name="delete", arguments={"id": 1})
        await hook.on_before_tool_call(event)
        assert isinstance(event.cancel, str)
        assert "Steering blocked" in event.cancel

    @pytest.mark.asyncio
    async def test_interrupt_decision_cancels_with_approval_message(self) -> None:
        hook = SteeringHook(model=_StubModel(content="INTERRUPT: confirm"))
        event = _BeforeToolEvent(tool_name="write", arguments={})
        await hook.on_before_tool_call(event)
        assert isinstance(event.cancel, str)
        assert "REQUIRES APPROVAL" in event.cancel

    @pytest.mark.asyncio
    async def test_interrupt_tool_short_circuits_without_llm(self) -> None:
        # When a tool is in the interrupt allow-list, the steering LLM
        # is NOT consulted — INTERRUPT is forced.
        stub = _StubModel(content="PROCEED")
        hook = SteeringHook(model=stub, interrupt_tools={"shell_exec"})
        event = _BeforeToolEvent(tool_name="shell_exec", arguments={"cmd": "ls"})
        await hook.on_before_tool_call(event)
        assert "REQUIRES APPROVAL" in event.cancel
        # Stub model was never called.
        assert stub.calls == []

    @pytest.mark.asyncio
    async def test_evaluate_tools_disabled_skips_evaluation(self) -> None:
        stub = _StubModel(content="GUIDE: should not run")
        hook = SteeringHook(model=stub, evaluate_tools=False)
        event = _BeforeToolEvent(tool_name="search", arguments={})
        await hook.on_before_tool_call(event)
        assert event.cancel is False
        assert stub.calls == []

    @pytest.mark.asyncio
    async def test_records_tool_call_in_context(self) -> None:
        hook = SteeringHook(model=_StubModel(content="PROCEED"))
        event = _BeforeToolEvent(tool_name="search", arguments={"q": "x"})
        await hook.on_before_tool_call(event)
        assert hook._context.tool_calls[-1]["tool"] == "search"


# ---------------------------------------------------------------------------
# on_before_model_call / on_after_model_call
# ---------------------------------------------------------------------------


class TestModelCallTracking:
    @pytest.mark.asyncio
    async def test_before_model_call_increments_counter(self) -> None:
        hook = SteeringHook(model=_StubModel())
        await hook.on_before_model_call(_BeforeModelEvent())
        await hook.on_before_model_call(_BeforeModelEvent())
        assert hook._context.model_calls == 2

    @pytest.mark.asyncio
    async def test_after_model_call_skipped_when_evaluation_disabled(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        hook = SteeringHook(model=_StubModel(), evaluate_responses=False)
        event = _AfterModelEvent(
            response=ModelResponse(message=Message.assistant("the password is hunter2"))
        )
        with caplog.at_level("WARNING"):
            await hook.on_after_model_call(event)
        assert "sensitive info" not in caplog.text

    @pytest.mark.asyncio
    async def test_after_model_call_warns_on_sensitive_response(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        hook = SteeringHook(
            model=_StubModel(),
            policy="Never expose secrets.",
            evaluate_responses=True,
        )
        event = _AfterModelEvent(
            response=ModelResponse(message=Message.assistant("the password is hunter2"))
        )
        with caplog.at_level("WARNING"):
            await hook.on_after_model_call(event)
        assert "sensitive info" in caplog.text

    @pytest.mark.asyncio
    async def test_after_model_call_empty_content_short_circuits(self) -> None:
        hook = SteeringHook(model=_StubModel(), evaluate_responses=True)
        event = _AfterModelEvent(response=ModelResponse(message=Message.assistant("")))
        # No exception, no warning — empty string returns early.
        await hook.on_after_model_call(event)


# ---------------------------------------------------------------------------
# Decision dataclass plumbing
# ---------------------------------------------------------------------------


class TestSteeringDecision:
    def test_default_action_field(self) -> None:
        dec = SteeringDecision(action=SteeringAction.PROCEED)
        assert dec.reason == ""
        assert dec.guidance == ""
