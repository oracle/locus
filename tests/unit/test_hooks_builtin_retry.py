# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``locus.hooks.builtin.retry`` (ModelRetryHook).

Drives the retry budget, the empty-response gate, and the
exponential-backoff delay calculation. Uses ``monkeypatch`` to skip
real ``asyncio.sleep`` so the test suite stays fast.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from locus.core.messages import Message
from locus.hooks.builtin.retry import ModelRetryHook
from locus.hooks.provider import HookPriority
from locus.models.base import ModelResponse


@dataclass
class _StubAfterEvent:
    """Duck-types ``AfterModelCallEvent`` — only the fields the hook reads."""

    response: ModelResponse
    retry: bool = False


def _empty_response() -> ModelResponse:
    return ModelResponse(message=Message.assistant(""))


def _content_response() -> ModelResponse:
    return ModelResponse(message=Message.assistant("hello"))


def _tool_call_response() -> ModelResponse:
    # Empty content but with a tool call — should NOT trigger retry.
    from locus.core.messages import ToolCall

    return ModelResponse(
        message=Message.assistant(
            "",
            tool_calls=[ToolCall(id="t1", name="search", arguments={"q": "x"})],
        ),
    )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip real ``asyncio.sleep`` calls inside the hook."""

    async def fake_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("locus.hooks.builtin.retry.asyncio.sleep", fake_sleep)


class TestConstruction:
    def test_default_priority_is_default(self) -> None:
        hook = ModelRetryHook()
        assert hook.priority == HookPriority.DEFAULT

    def test_explicit_priority(self) -> None:
        hook = ModelRetryHook(priority=42)
        assert hook.priority == 42

    def test_name(self) -> None:
        assert ModelRetryHook().name == "ModelRetryHook"


class TestRetryBudget:
    @pytest.mark.asyncio
    async def test_resets_attempt_on_before_call(self) -> None:
        hook = ModelRetryHook()
        hook._attempt = 5
        await hook.on_before_model_call(event=object())
        assert hook._attempt == 0

    @pytest.mark.asyncio
    async def test_empty_response_triggers_retry(self) -> None:
        hook = ModelRetryHook(max_retries=3, initial_delay=0)
        event = _StubAfterEvent(response=_empty_response())
        await hook.on_after_model_call(event)
        assert event.retry is True
        assert hook._attempt == 1
        assert hook.retries_total == 1

    @pytest.mark.asyncio
    async def test_content_response_does_not_retry(self) -> None:
        hook = ModelRetryHook(initial_delay=0)
        event = _StubAfterEvent(response=_content_response())
        await hook.on_after_model_call(event)
        assert event.retry is False
        assert hook._attempt == 0

    @pytest.mark.asyncio
    async def test_tool_call_response_does_not_retry(self) -> None:
        # Empty content but a tool call — model is still making progress.
        hook = ModelRetryHook(initial_delay=0)
        event = _StubAfterEvent(response=_tool_call_response())
        await hook.on_after_model_call(event)
        assert event.retry is False

    @pytest.mark.asyncio
    async def test_retry_disabled_when_retry_on_empty_false(self) -> None:
        hook = ModelRetryHook(retry_on_empty=False, initial_delay=0)
        event = _StubAfterEvent(response=_empty_response())
        await hook.on_after_model_call(event)
        # ``should_retry`` stayed False → fast-return path; no retry.
        assert event.retry is False
        assert hook._attempt == 0

    @pytest.mark.asyncio
    async def test_exhausted_budget_accepts_response(self) -> None:
        hook = ModelRetryHook(max_retries=2, initial_delay=0)
        # Pump the attempt counter to the cap.
        hook._attempt = 2
        event = _StubAfterEvent(response=_empty_response())
        await hook.on_after_model_call(event)
        assert event.retry is False
        # Counter resets so the next model call gets a fresh budget.
        assert hook._attempt == 0


class TestBackoffCalculation:
    @pytest.mark.asyncio
    async def test_delay_clamped_to_max(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorded: list[float] = []

        async def capture_sleep(delay: float) -> None:
            recorded.append(delay)

        monkeypatch.setattr("locus.hooks.builtin.retry.asyncio.sleep", capture_sleep)

        hook = ModelRetryHook(
            max_retries=10,
            initial_delay=10.0,
            max_delay=15.0,
            backoff_factor=10.0,
        )
        event = _StubAfterEvent(response=_empty_response())
        await hook.on_after_model_call(event)
        # 10.0 * 10**0 = 10.0 → first delay is 10s.
        assert recorded == [10.0]

        # Second retry: 10.0 * 10**1 = 100s → clamped to max_delay (15s).
        await hook.on_after_model_call(event)
        assert recorded[1] == 15.0
