# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""``AgentConfig.model_retry``: the loop retries transient model-call failures.

One 429 or dropped connection among the hundreds of calls a long run makes
used to end the run with ``TerminateEvent(reason="error")``. These tests pin
which failures are retried (429, 5xx, transport, timeout) and which are not
(context overflow, validation, auth, unclassified), the backoff and
``retry-after`` handling, the time budget, the per-retry event, and the
streaming rule: no retry once a chunk has reached the caller.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from types import SimpleNamespace
from typing import Any

import pytest

from tulip.agent import Agent, ModelRetryConfig
from tulip.agent import model_retry as retry_mod
from tulip.agent.model_retry import call_with_retry, retry_after_seconds
from tulip.core.events import (
    ModelChunkEvent,
    ModelRetryEvent,
    TerminateEvent,
    TulipEvent,
)
from tulip.core.messages import Message
from tulip.testing import FunctionModel, text


class FakeHTTPError(Exception):
    """The shape provider SDKs raise: a status code and a response with headers."""

    def __init__(self, status: int, message: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.status_code = status
        self.response = SimpleNamespace(headers=headers or {})


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record backoff sleeps instead of waiting; jitter takes the ceiling."""
    recorded: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(retry_mod, "_sleep", fake_sleep)
    monkeypatch.setattr(retry_mod, "_uniform", lambda low, high: high)
    return recorded


def _failing_then(*failures: BaseException, answer: str = "done") -> FunctionModel:
    pending = list(failures)

    def handler(messages: list[Message], tools: list[dict[str, Any]]) -> str:
        if pending:
            raise pending.pop(0)
        return answer

    return FunctionModel(handler)


def _agent(model: Any, **kwargs: Any) -> Agent:
    return Agent(model=model, reflexion=False, grounding=False, **kwargs)


async def _events(agent: Agent, **kwargs: Any) -> list[TulipEvent]:
    return [e async for e in agent.run("hi", **kwargs)]


async def _events_until(
    agent: Agent, error: type[BaseException], match: str | None = None
) -> list[TulipEvent]:
    """The events a run yields before it raises ``error``."""
    seen: list[TulipEvent] = []

    async def drive() -> None:
        async for event in agent.run("hi"):
            seen.append(event)

    with pytest.raises(error, match=match):
        await drive()
    return seen


def _retries(events: list[TulipEvent]) -> list[ModelRetryEvent]:
    return [e for e in events if isinstance(e, ModelRetryEvent)]


# ---------------------------------------------------------------------------
# Through the agent loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (FakeHTTPError(429, "Too Many Requests"), "rate_limit"),
        (FakeHTTPError(503, "Service Unavailable"), "overloaded"),
        (FakeHTTPError(500, "Internal Server Error"), "server_error"),
        (ConnectionError("connection reset by peer"), "timeout"),
        (TimeoutError("read timed out"), "timeout"),
    ],
)
async def test_transient_failure_is_retried_and_the_run_completes(
    sleeps: list[float], failure: BaseException, reason: str
) -> None:
    model = _failing_then(failure)
    events = await _events(_agent(model))

    retries = _retries(events)
    assert [r.reason for r in retries] == [reason]
    assert retries[0].attempt == 1
    assert retries[0].delay_seconds == 1.0
    assert type(failure).__name__ in retries[0].error
    terminate = events[-1]
    assert isinstance(terminate, TerminateEvent)
    assert terminate.reason != "error"
    assert terminate.final_message == "done"
    assert model.call_count == 2
    assert sleeps == [1.0]


async def test_backoff_doubles_up_to_max_delay(sleeps: list[float]) -> None:
    failures = [FakeHTTPError(503, "overloaded") for _ in range(4)]
    policy = ModelRetryConfig(initial_delay=2.0, max_delay=5.0)
    events = await _events(_agent(_failing_then(*failures), model_retry=policy))

    assert sleeps == [2.0, 4.0, 5.0, 5.0]
    assert [r.attempt for r in _retries(events)] == [1, 2, 3, 4]


async def test_full_jitter_draws_below_the_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    draws: list[tuple[float, float]] = []
    slept: list[float] = []

    def fake_uniform(low: float, high: float) -> float:
        draws.append((low, high))
        return high / 4

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(retry_mod, "_uniform", fake_uniform)
    monkeypatch.setattr(retry_mod, "_sleep", fake_sleep)
    await _events(_agent(_failing_then(FakeHTTPError(502, "bad gateway"))))

    assert draws == [(0.0, 1.0)]
    assert slept == [0.25]


async def test_retry_after_header_sets_the_delay(sleeps: list[float]) -> None:
    model = _failing_then(FakeHTTPError(429, "slow down", headers={"retry-after": "7"}))
    events = await _events(_agent(model))

    (retry,) = _retries(events)
    assert retry.delay_seconds == 7.0
    assert retry.from_retry_after is True
    assert retry.status_code == 429
    assert sleeps == [7.0]


@pytest.mark.parametrize(
    "failure",
    [
        FakeHTTPError(
            400, "This model's maximum context length is 128000 tokens; reduce the length"
        ),
        FakeHTTPError(422, "Unprocessable Entity: messages.0.content is required"),
        FakeHTTPError(401, "Invalid API key"),
        FakeHTTPError(402, "Payment required: insufficient credits"),
        ValueError("a bug in a hook, not a provider failure"),
    ],
)
async def test_non_transient_failure_is_not_retried(
    sleeps: list[float], failure: BaseException
) -> None:
    model = _failing_then(failure)
    seen = await _events_until(_agent(model), type(failure))

    assert _retries(seen) == []
    assert model.call_count == 1
    assert sleeps == []
    assert isinstance(seen[-1], TerminateEvent)
    assert seen[-1].reason == "error"


async def test_spent_retries_reraise_the_last_failure_after_reporting_each(
    sleeps: list[float],
) -> None:
    failures = [FakeHTTPError(503, f"overloaded {i}") for i in range(3)]
    model = _failing_then(*failures)
    agent = _agent(model, model_retry=ModelRetryConfig(max_retries=2))
    seen = await _events_until(agent, FakeHTTPError, match="overloaded 2")

    # Non-streaming: the notices still reach the caller, before the error.
    assert [r.attempt for r in _retries(seen)] == [1, 2]
    kinds = [type(e).__name__ for e in seen]
    assert kinds.index("ModelRetryEvent") < kinds.index("TerminateEvent")
    assert model.call_count == 3


async def test_retry_after_past_the_budget_fails_at_once(sleeps: list[float]) -> None:
    failure = FakeHTTPError(429, "slow down", headers={"Retry-After": "600"})
    model = _failing_then(failure)
    agent = _agent(model, model_retry=ModelRetryConfig(total_budget_seconds=60))

    with pytest.raises(FakeHTTPError):
        await _events(agent)

    assert sleeps == []
    assert model.call_count == 1


async def test_budget_counts_elapsed_time(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    clock = iter([0.0, 50.0, 95.0])
    monkeypatch.setattr(retry_mod, "_monotonic", lambda: next(clock))
    failures = [FakeHTTPError(503, "overloaded") for _ in range(3)]
    policy = ModelRetryConfig(initial_delay=10.0, max_delay=10.0, total_budget_seconds=100)
    model = _failing_then(*failures)

    with pytest.raises(FakeHTTPError):
        await _events(_agent(model, model_retry=policy))

    # 50 s in, a 10 s wait still fits; 95 s in, it would end at 105 s.
    assert sleeps == [10.0]
    assert model.call_count == 2


@pytest.mark.parametrize("setting", [False, None, ModelRetryConfig(enabled=False)])
async def test_retry_can_be_turned_off(sleeps: list[float], setting: Any) -> None:
    model = _failing_then(FakeHTTPError(503, "overloaded"))

    with pytest.raises(FakeHTTPError):
        await _events(_agent(model, model_retry=setting))

    assert model.call_count == 1


def test_true_restores_the_defaults() -> None:
    agent = _agent(_failing_then(), model_retry=True)
    assert agent.config.model_retry == ModelRetryConfig()


async def test_streaming_delivers_the_notice_live_and_retries(sleeps: list[float]) -> None:
    model = _failing_then(FakeHTTPError(429, "rate limit"), answer="streamed answer")
    events = await _events(_agent(model), stream_tokens=True)

    kinds = [type(e).__name__ for e in events]
    assert kinds.count("ModelRetryEvent") == 1
    # Live: the notice comes before the retried call's first chunk.
    assert kinds.index("ModelRetryEvent") < kinds.index("ModelChunkEvent")
    streamed = "".join(e.content or "" for e in events if isinstance(e, ModelChunkEvent))
    assert streamed == "streamed answer"


async def test_streaming_notice_is_not_held_with_the_tokens(sleeps: list[float]) -> None:
    model = _failing_then(FakeHTTPError(503, "overloaded"))
    agent = _agent(model, hold_final_answer_tokens=True)
    events = await _events(agent, stream_tokens=True)

    assert len(_retries(events)) == 1


class _BreaksMidStream:
    """Streams one chunk, then the connection drops."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages: list[Message], **kwargs: Any) -> Any:
        return text("unused")

    async def stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[Any]:
        self.calls += 1
        yield ModelChunkEvent(content="The answer is")
        raise ConnectionError("connection reset by peer")


async def test_streaming_is_not_retried_after_a_chunk_reached_the_caller(
    sleeps: list[float],
) -> None:
    model = _BreaksMidStream()
    agent = _agent(model)

    with pytest.raises(ConnectionError):
        await _events(agent, stream_tokens=True)

    assert model.calls == 1
    assert sleeps == []


# ---------------------------------------------------------------------------
# The helpers on their own
# ---------------------------------------------------------------------------


async def test_cancelled_run_is_not_retried(sleeps: list[float]) -> None:
    calls = 0

    async def attempt() -> str:
        nonlocal calls
        calls += 1
        raise FakeHTTPError(503, "overloaded")

    async def notify(event: ModelRetryEvent) -> None:
        raise AssertionError("no retry expected")

    with pytest.raises(FakeHTTPError):
        await call_with_retry(attempt, ModelRetryConfig(), notify=notify, may_retry=lambda: False)
    assert calls == 1


def test_retry_after_forms() -> None:
    assert retry_after_seconds(FakeHTTPError(429, "x", {"retry-after": "3"})) == 3.0
    assert retry_after_seconds(FakeHTTPError(429, "x", {"Retry-After": "2.5"})) == 2.5
    assert retry_after_seconds(FakeHTTPError(429, "x", {"retry-after-ms": "1500"})) == 1.5
    assert retry_after_seconds(FakeHTTPError(429, "x", {"retry-after": "soon"})) is None
    assert retry_after_seconds(FakeHTTPError(429, "x")) is None

    later = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)
    waited = retry_after_seconds(FakeHTTPError(429, "x", {"retry-after": later}))
    assert waited is not None
    assert 25 <= waited <= 31

    attr = RuntimeError("throttled")
    attr.retry_after = 4  # type: ignore[attr-defined]
    assert retry_after_seconds(attr) == 4.0


def test_retry_after_is_found_on_a_wrapped_cause() -> None:
    outer = RuntimeError("all tiers failed")
    outer.__cause__ = FakeHTTPError(429, "slow down", {"retry-after": "9"})
    assert retry_after_seconds(outer) == 9.0


def test_retry_after_ignores_odd_header_containers() -> None:
    odd = FakeHTTPError(429, "x")
    odd.response = SimpleNamespace(headers=["not", "a", "mapping"])
    assert retry_after_seconds(odd) is None
