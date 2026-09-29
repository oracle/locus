# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Provider failover: ``CredentialPoolModel`` streaming and ``FallbackChain``.

Before this, ``CredentialPoolModel.stream`` wrapped only the *call* to
``model.stream(...)`` in its try — and calling an async generator raises
nothing, so a 429 on the opening request surfaced on first iteration, outside
the try, and rotation never happened on the streaming path. And there was no
cross-provider failover at all: ``failover.classify`` said ``should_fallback``
and nothing acted on it.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import SecretStr

from tulip.agent import Agent
from tulip.core.events import ModelChunkEvent, TerminateEvent
from tulip.core.messages import Message
from tulip.models.credentials import Credential, CredentialPool
from tulip.models.fallback import (
    CircuitBreaker,
    CircuitState,
    FallbackChain,
    FallbackEvent,
    FallbackExhaustedError,
)
from tulip.models.pooled import CredentialPoolModel
from tulip.testing import ScriptedModel, text


class APIError(Exception):
    def __init__(self, status: int, message: str = "boom") -> None:
        super().__init__(message)
        self.status_code = status
        self.headers: dict[str, str] = {}


class Failing:
    """A model that fails every call — optionally after streaming a chunk."""

    def __init__(self, exc: BaseException, *, partial: bool = False) -> None:
        self.exc = exc
        self.partial = partial
        self.calls = 0

    async def complete(self, messages: list[Message], tools: Any = None, **kw: Any) -> Any:
        self.calls += 1
        raise self.exc

    async def stream(
        self, messages: list[Message], tools: Any = None, **kw: Any
    ) -> AsyncIterator[ModelChunkEvent]:
        self.calls += 1
        if self.partial:
            yield ModelChunkEvent(content="partial-from-primary ")
        raise self.exc


class Slow:
    """A model that takes ``delay`` seconds to produce its first chunk."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.calls = 0

    async def complete(self, messages: list[Message], tools: Any = None, **kw: Any) -> Any:
        self.calls += 1
        await asyncio.sleep(self.delay)
        return text("slow")

    async def stream(
        self, messages: list[Message], tools: Any = None, **kw: Any
    ) -> AsyncIterator[ModelChunkEvent]:
        self.calls += 1
        await asyncio.sleep(self.delay)
        yield ModelChunkEvent(content="slow")


def _backup(reply: str = "from-backup") -> ScriptedModel:
    return ScriptedModel([text(reply)], repeat_last=True)


def _msgs() -> list[Message]:
    return [Message.user("q")]


async def _stream_text(model: Any) -> str:
    return "".join([c.content or "" async for c in model.stream(_msgs())])


# ---------------------------------------------------------------------------
# CredentialPoolModel streaming
# ---------------------------------------------------------------------------


def _pool() -> CredentialPool:
    return CredentialPool(
        [
            Credential(label="primary", api_key=SecretStr("x")),
            Credential(label="backup", api_key=SecretStr("y")),
        ]
    )


async def test_pool_stream_rotates_on_an_error_before_the_first_chunk() -> None:
    primary, backup = Failing(APIError(429)), _backup()
    model = CredentialPoolModel(
        pool=_pool(), build_model=lambda c: primary if c.label == "primary" else backup
    )
    assert await _stream_text(model) == "from-backup"
    assert (primary.calls, backup.call_count, model.attempts) == (1, 1, 2)


async def test_pool_stream_never_splices_after_the_first_chunk() -> None:
    primary, backup = Failing(APIError(429), partial=True), _backup()
    model = CredentialPoolModel(
        pool=_pool(), build_model=lambda c: primary if c.label == "primary" else backup
    )
    seen: list[str] = []

    async def consume() -> None:
        async for chunk in model.stream(_msgs()):
            seen.append(chunk.content or "")

    with pytest.raises(APIError):
        await consume()
    assert seen == ["partial-from-primary "]
    assert backup.call_count == 0


async def test_pool_stream_does_not_rotate_a_non_credential_error() -> None:
    primary, backup = Failing(APIError(400, "bad request")), _backup()
    model = CredentialPoolModel(
        pool=_pool(), build_model=lambda c: primary if c.label == "primary" else backup
    )
    with pytest.raises(APIError):
        await _stream_text(model)
    assert backup.call_count == 0


async def test_pool_streaming_agent_survives_a_429() -> None:
    primary, backup = Failing(APIError(429)), _backup()
    model = CredentialPoolModel(
        pool=_pool(), build_model=lambda c: primary if c.label == "primary" else backup
    )
    agent = Agent(model=model, reflexion=False, grounding=False)
    events = [e async for e in agent.run("q", stream_tokens=True)]
    assert [e.final_message for e in events if isinstance(e, TerminateEvent)] == ["from-backup"]


# ---------------------------------------------------------------------------
# FallbackChain — policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 502, 503, 529])
async def test_complete_falls_back_on_provider_errors(status: int) -> None:
    primary, backup = Failing(APIError(status)), _backup()
    chain = FallbackChain([primary, backup], names=["primary", "backup"])
    response = await chain.complete(_msgs())
    assert response.content == "from-backup"
    assert chain.last_tier == 1


@pytest.mark.parametrize("exc", [TimeoutError(), ConnectionError("reset by peer")])
async def test_complete_falls_back_on_transport_errors(exc: BaseException) -> None:
    chain = FallbackChain([Failing(exc), _backup()])
    assert (await chain.complete(_msgs())).content == "from-backup"


async def test_a_request_error_is_raised_not_failed_over() -> None:
    """Every tier would reject an oversized prompt the same way."""
    primary = Failing(APIError(400, "prompt is too long: 250000 tokens"))
    backup = _backup()
    chain = FallbackChain([primary, backup])
    with pytest.raises(APIError):
        await chain.complete(_msgs())
    assert backup.call_count == 0
    assert chain.tiers[0].breaker.state is CircuitState.CLOSED


async def test_should_fallback_overrides_the_policy() -> None:
    backup = _backup()
    chain = FallbackChain(
        [Failing(ValueError("anything")), backup], should_fallback=lambda exc, d: True
    )
    assert (await chain.complete(_msgs())).content == "from-backup"


async def test_all_tiers_failing_raises_exhausted_with_every_error() -> None:
    chain = FallbackChain([Failing(APIError(503)), Failing(APIError(500))], names=["a", "b"])
    with pytest.raises(FallbackExhaustedError) as info:
        await chain.complete(_msgs())
    assert [name for name, _ in info.value.errors] == ["a", "b"]
    assert isinstance(info.value.__cause__, APIError)


async def test_cancellation_is_not_failed_over() -> None:
    class Cancelled:
        async def complete(self, *a: Any, **k: Any) -> Any:
            raise asyncio.CancelledError

    backup = _backup()
    chain = FallbackChain([Cancelled(), backup])
    with pytest.raises(asyncio.CancelledError):
        await chain.complete(_msgs())
    assert backup.call_count == 0


async def test_complete_timeout_moves_on() -> None:
    chain = FallbackChain([Slow(5), _backup()], complete_timeout=0.05)
    assert (await chain.complete(_msgs())).content == "from-backup"


# ---------------------------------------------------------------------------
# FallbackChain — streaming
# ---------------------------------------------------------------------------


async def test_stream_falls_back_before_the_first_chunk() -> None:
    primary, backup = Failing(APIError(529)), _backup()
    chain = FallbackChain([primary, backup])
    assert await _stream_text(chain) == "from-backup"


async def test_stream_never_splices_providers_mid_reply() -> None:
    primary, backup = Failing(APIError(529), partial=True), _backup()
    chain = FallbackChain([primary, backup])
    seen: list[str] = []

    async def consume() -> None:
        async for chunk in chain.stream(_msgs()):
            seen.append(chunk.content or "")

    with pytest.raises(APIError):
        await consume()
    assert seen == ["partial-from-primary "]
    assert backup.call_count == 0


async def test_first_chunk_timeout_moves_on() -> None:
    slow, backup = Slow(5), _backup()
    chain = FallbackChain([slow, backup], first_chunk_timeout=0.05)
    assert await _stream_text(chain) == "from-backup"
    assert slow.calls == 1


async def test_a_streaming_agent_fails_over() -> None:
    chain = FallbackChain([Failing(APIError(503)), _backup("answered by backup")])
    agent = Agent(model=chain, reflexion=False, grounding=False)
    events = [e async for e in agent.run("q", stream_tokens=True)]
    assert [e.final_message for e in events if isinstance(e, TerminateEvent)] == [
        "answered by backup"
    ]


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_breaker_opens_after_threshold_within_window() -> None:
    clock = _Clock()
    breaker = CircuitBreaker(failure_threshold=2, window_s=10, cooldown_s=30, clock=clock)
    breaker.record_failure()
    clock.now += 11  # first failure ages out of the window
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert breaker.allow() is False


def test_breaker_half_open_admits_one_probe() -> None:
    clock = _Clock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_s=30, clock=clock)
    breaker.record_failure()
    clock.now += 30
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.allow() is True  # the probe
    assert breaker.allow() is False  # only one
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    clock.now += 30
    assert breaker.allow() is True
    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED
    assert breaker.allow() is True


async def test_an_open_tier_is_skipped_until_its_probe() -> None:
    clock = _Clock()
    primary, backup = Failing(APIError(503)), _backup()
    events: list[FallbackEvent] = []
    chain = FallbackChain(
        [primary, backup],
        names=["primary", "backup"],
        failure_threshold=2,
        cooldown_s=60,
        clock=clock,
        on_event=events.append,
    )
    for _ in range(2):
        await chain.complete(_msgs())
    assert primary.calls == 2
    assert chain.tiers[0].breaker.state is CircuitState.OPEN

    await chain.complete(_msgs())  # skipped: breaker open
    assert primary.calls == 2

    clock.now += 60  # cooldown over: one probe, which fails and re-opens
    await chain.complete(_msgs())
    assert primary.calls == 3
    assert chain.tiers[0].breaker.state is CircuitState.OPEN

    kinds = [e.kind for e in events]
    assert {"skipped", "served", "fallback"} <= set(kinds)
    transitions = [(e.previous_state, e.state) for e in events if e.kind == "breaker"]
    assert transitions == [("closed", "open"), ("open", "half_open"), ("half_open", "open")]
    fallback = next(e for e in events if e.kind == "fallback")
    assert (fallback.tier, fallback.next_tier, fallback.reason) == (0, 1, "overloaded")
    metrics = chain.metrics
    assert metrics["primary"]["state"] == "open"
    assert metrics["backup"]["served"] == 4
    assert metrics["primary"]["skipped"] == 1


async def test_when_every_breaker_is_open_the_primary_is_tried() -> None:
    primary = Failing(APIError(503))
    chain = FallbackChain([primary], failure_threshold=1, cooldown_s=3600)
    with pytest.raises(FallbackExhaustedError):
        await chain.complete(_msgs())
    with pytest.raises(FallbackExhaustedError):
        await chain.complete(_msgs())
    assert primary.calls == 2


async def test_fallback_events_reach_the_telemetry_bus() -> None:
    from tulip.observability.context import run_context
    from tulip.observability.emit import EV_MODEL_BREAKER, EV_MODEL_FALLBACK
    from tulip.observability.event_bus import get_event_bus

    bus = get_event_bus()
    seen: list[Any] = []
    original = bus.publish

    async def _capture(event: Any) -> None:
        seen.append(event)
        await original(event)

    bus.publish = _capture  # type: ignore[method-assign]
    try:
        chain = FallbackChain([Failing(APIError(503)), _backup()], failure_threshold=1)
        async with run_context("run-fallback-test"):
            await chain.complete(_msgs())
    finally:
        bus.publish = original  # type: ignore[method-assign]
    types = [e.event_type for e in seen]
    assert EV_MODEL_FALLBACK in types
    assert EV_MODEL_BREAKER in types


def test_config_mirrors_the_primary() -> None:
    class WithConfig:
        config = object()

    primary = WithConfig()
    assert FallbackChain([primary, _backup()]).config is primary.config


class Scripted:
    """Fails or answers per call, from a list of outcomes (None = answer)."""

    def __init__(self, outcomes: list[BaseException | None], reply: str) -> None:
        self.outcomes = list(outcomes)
        self.reply = reply
        self.calls = 0

    async def complete(self, messages: list[Message], tools: Any = None, **kw: Any) -> Any:
        self.calls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else None
        if outcome is not None:
            raise outcome
        return text(self.reply)


async def test_a_probe_is_claimed_only_when_the_call_reaches_the_tier() -> None:
    """Asking every breaker up front claimed half-open probes that were never
    sent, and a claimed-but-unsent probe wedged the tier open for good."""
    clock = _Clock()
    a = Scripted([APIError(503), None, APIError(503)], "a")
    b = Scripted([APIError(503), None], "b")
    chain = FallbackChain([a, b], failure_threshold=1, cooldown_s=10, clock=clock)
    with pytest.raises(FallbackExhaustedError):
        await chain.complete(_msgs())  # both open
    clock.now += 10  # both half-open
    assert (await chain.complete(_msgs())).content == "a"  # a's probe succeeds
    assert chain.tiers[1].breaker.state is CircuitState.HALF_OPEN
    # b's probe was never claimed, so when a fails again b is still tried.
    assert (await chain.complete(_msgs())).content == "b"
    assert chain.tiers[1].breaker.state is CircuitState.CLOSED


async def test_a_cancelled_probe_is_released() -> None:
    clock = _Clock()

    class Hangs:
        async def complete(self, *a: Any, **k: Any) -> Any:
            await asyncio.sleep(3600)

    chain = FallbackChain([Hangs()], failure_threshold=1, cooldown_s=10, clock=clock)
    chain.tiers[0].breaker.record_failure()
    clock.now += 10
    task = asyncio.ensure_future(chain.complete(_msgs()))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert chain.tiers[0].breaker.allow() is True  # the probe slot is free again
