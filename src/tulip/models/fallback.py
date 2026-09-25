# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Provider failover: an ordered chain of models behind circuit breakers.

:mod:`tulip.models.failover` *classifies* an error; :mod:`tulip.models.pooled`
rotates *credentials* of one provider. Neither moves a request to a different
provider when the one serving it is overloaded, erroring or slow. That is what
:class:`FallbackChain` does::

    from tulip.models import get_model
    from tulip.models.fallback import FallbackChain

    model = FallbackChain(
        [
            get_model("anthropic:claude-sonnet-4-6"),
            get_model("bedrock:us.anthropic.claude-sonnet-4-6-v1:0"),
            get_model("openai:gpt-4o"),
        ],
        first_chunk_timeout=20.0,
        on_event=lambda ev: log.info("model fallback: %s", ev),
    )
    agent = Agent(model=model, ...)

**Per call.** Tiers are tried in order. A tier whose breaker is open is
skipped. An error that :func:`~tulip.models.failover.classify` puts down to
the provider — 429, 5xx, 529/overloaded, a timeout or dropped connection, an
auth/billing failure, an unknown model — moves the call to the next tier
(:data:`HEALTH_REASONS`). An error that is the *request's* fault (a context
overflow, a 400 the classifier does not think another provider would accept)
is raised at once: every tier would reject it the same way. Pass
``should_fallback`` to change the policy.

**Streaming.** A tier is committed once it yields its first chunk. A failure
before that falls back; a failure after it is re-raised — splicing a second
provider's answer onto the first's half-reply would produce text neither
model wrote. ``first_chunk_timeout`` bounds how long a tier may take to start.

**Circuit breakers.** Each tier has a :class:`CircuitBreaker`: after
``failure_threshold`` provider failures within ``window_s`` seconds it
*opens* and the tier is skipped for ``cooldown_s``; then it goes *half-open*
and admits a single probe call — success closes it, failure re-opens it.
When every tier is open the primary is tried anyway rather than failing
without an attempt.

**Observability.** Every tier switch and breaker transition is reported three
ways: a :class:`FallbackEvent` to the ``on_event`` callback, an event on the
Tulip telemetry bus (:data:`~tulip.observability.emit.EV_MODEL_FALLBACK`,
:data:`~tulip.observability.emit.EV_MODEL_BREAKER`) when a run context is
active, and counters on :attr:`FallbackChain.metrics`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from tulip.models.failover import FailoverDecision, FailoverReason, classify
from tulip.observability.emit import EV_MODEL_BREAKER, EV_MODEL_FALLBACK, emit


if TYPE_CHECKING:
    from tulip.core.events import ModelChunkEvent
    from tulip.core.messages import Message
    from tulip.models.base import ModelResponse


logger = logging.getLogger(__name__)

__all__ = [
    "HEALTH_REASONS",
    "CircuitBreaker",
    "CircuitState",
    "FallbackChain",
    "FallbackEvent",
    "FallbackExhaustedError",
]


#: Failure reasons that say the *provider* is unwell (not the request). They
#: move a call to the next tier and count against the tier's breaker.
HEALTH_REASONS: frozenset[FailoverReason] = frozenset(
    {
        FailoverReason.RATE_LIMIT,
        FailoverReason.OVERLOADED,
        FailoverReason.SERVER_ERROR,
        FailoverReason.TIMEOUT,
        FailoverReason.AUTH_TRANSIENT,
        FailoverReason.AUTH_PERMANENT,
        FailoverReason.BILLING,
        FailoverReason.MODEL_NOT_FOUND,
    }
)


class CircuitState(StrEnum):
    """A breaker's state."""

    CLOSED = "closed"
    """Healthy: calls go through."""
    OPEN = "open"
    """Tripped: calls skip this tier until the cooldown ends."""
    HALF_OPEN = "half_open"
    """Cooling down finished: one probe call decides."""


class CircuitBreaker:
    """Failure-threshold-within-window breaker with a single half-open probe.

    Args:
        failure_threshold: Failures within ``window_s`` that open the breaker.
        window_s: Sliding window for counting failures, in seconds.
        cooldown_s: How long an open breaker stays open.
        on_change: Called ``(old_state, new_state)`` on every transition.
        clock: Monotonic clock, injectable for tests.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        window_s: float = 60.0,
        cooldown_s: float = 30.0,
        on_change: Callable[[CircuitState, CircuitState], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if window_s <= 0 or cooldown_s < 0:
            raise ValueError("window_s must be positive and cooldown_s non-negative")
        self.failure_threshold = failure_threshold
        self.window_s = window_s
        self.cooldown_s = cooldown_s
        self._on_change = on_change
        self._clock = clock
        self._failures: deque[float] = deque()
        self._state = CircuitState.CLOSED
        self._opened_at = 0.0
        self._probe_in_flight = False

    @property
    def state(self) -> CircuitState:
        """The current state; an open breaker past its cooldown reads half-open."""
        if self._state is CircuitState.OPEN and self._clock() - self._opened_at >= self.cooldown_s:
            self._transition(CircuitState.HALF_OPEN)
        return self._state

    def allow(self) -> bool:
        """Whether a call may go to this tier now. Claims the half-open probe."""
        state = self.state
        if state is CircuitState.CLOSED:
            return True
        if state is CircuitState.HALF_OPEN and not self._probe_in_flight:
            self._probe_in_flight = True
            return True
        return False

    def record_success(self) -> None:
        """A call succeeded: close the breaker and forget past failures."""
        self._probe_in_flight = False
        self._failures.clear()
        if self._state is not CircuitState.CLOSED:
            self._transition(CircuitState.CLOSED)

    def record_failure(self) -> None:
        """A call failed for a provider-health reason."""
        now = self._clock()
        if self._state is CircuitState.HALF_OPEN:
            self._probe_in_flight = False
            self._open(now)
            return
        self._failures.append(now)
        while self._failures and now - self._failures[0] > self.window_s:
            self._failures.popleft()
        if self._state is CircuitState.CLOSED and len(self._failures) >= self.failure_threshold:
            self._open(now)

    def release(self) -> None:
        """A claimed probe ended without a verdict (e.g. a request error)."""
        self._probe_in_flight = False

    def _open(self, now: float) -> None:
        self._opened_at = now
        self._failures.clear()
        self._transition(CircuitState.OPEN)

    def _transition(self, new: CircuitState) -> None:
        old, self._state = self._state, new
        if old is not new and self._on_change is not None:
            try:
                self._on_change(old, new)
            except Exception:  # noqa: BLE001 — an observer must not break the breaker
                logger.debug("circuit breaker observer failed", exc_info=True)


@dataclass(frozen=True)
class FallbackEvent:
    """Something the chain did that an operator wants to see.

    Attributes:
        kind: ``"fallback"`` — a tier failed and the call moves on;
            ``"served"`` — a tier other than the first answered;
            ``"breaker"`` — a tier's breaker changed state;
            ``"skipped"`` — a tier was passed over because its breaker is open;
            ``"exhausted"`` — every tier failed.
        tier: Index of the tier concerned.
        model: That tier's name.
        next_tier: For ``"fallback"``, the tier tried next (None if none).
        reason: The classified failure reason, when there was a failure.
        error: ``"Type: message"`` of the failure, truncated.
        state: For ``"breaker"``, the new state.
        previous_state: For ``"breaker"``, the old state.
    """

    kind: Literal["fallback", "served", "breaker", "skipped", "exhausted"]
    tier: int
    model: str
    next_tier: int | None = None
    reason: str | None = None
    error: str | None = None
    state: str | None = None
    previous_state: str | None = None


class FallbackExhaustedError(RuntimeError):
    """Every tier of a :class:`FallbackChain` failed.

    ``errors`` holds ``(tier_name, exception)`` for each attempt, in order; the
    last one is also the ``__cause__``.
    """

    def __init__(self, errors: list[tuple[str, BaseException]]) -> None:
        self.errors = errors
        detail = "; ".join(f"{name}: {type(exc).__name__}: {exc}"[:200] for name, exc in errors)
        super().__init__(f"all {len(errors)} model tier(s) failed — {detail}")


@dataclass
class _Tier:
    index: int
    name: str
    model: Any
    breaker: CircuitBreaker
    counters: dict[str, int] = field(
        default_factory=lambda: {"attempts": 0, "failures": 0, "served": 0, "skipped": 0}
    )


def _model_name(model: Any) -> str:
    config = getattr(model, "config", None)
    name = getattr(config, "model", None) or getattr(model, "model", None)
    if isinstance(name, str) and name:
        return name
    return type(model).__name__


class FallbackChain:
    """A ``ModelProtocol`` that fails over across an ordered list of models.

    Args:
        models: The tiers, primary first. At least one.
        names: Display names for the tiers (default: each model's
            ``config.model`` or class name).
        failure_threshold: Provider failures within ``window_s`` that open a
            tier's breaker.
        window_s: Failure-counting window, seconds.
        cooldown_s: How long an open breaker skips its tier before a probe.
        first_chunk_timeout: Streaming: seconds a tier may take to produce
            its first chunk before the chain moves on. None waits.
        complete_timeout: Non-streaming: seconds a tier may take to answer
            before the chain moves on. None waits.
        should_fallback: ``(exc, decision) -> bool`` overriding the default
            policy (fall back on :data:`HEALTH_REASONS` or when the classifier
            sets ``should_fallback``).
        on_event: Called with a :class:`FallbackEvent` on every tier switch,
            breaker transition and exhaustion. Exceptions are swallowed.
        clock: Monotonic clock for the breakers, injectable for tests.
    """

    name = "FallbackChain"

    def __init__(
        self,
        models: Sequence[Any],
        *,
        names: Sequence[str] | None = None,
        failure_threshold: int = 3,
        window_s: float = 60.0,
        cooldown_s: float = 30.0,
        first_chunk_timeout: float | None = None,
        complete_timeout: float | None = None,
        should_fallback: Callable[[BaseException, FailoverDecision], bool] | None = None,
        on_event: Callable[[FallbackEvent], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not models:
            raise ValueError("FallbackChain needs at least one model")
        if names is not None and len(names) != len(models):
            raise ValueError("names must have one entry per model")
        self._on_event = on_event
        self._should_fallback = should_fallback
        self.first_chunk_timeout = first_chunk_timeout
        self.complete_timeout = complete_timeout
        self.tiers: list[_Tier] = []
        for index, model in enumerate(models):
            tier_name = names[index] if names is not None else _model_name(model)
            breaker = CircuitBreaker(
                failure_threshold=failure_threshold,
                window_s=window_s,
                cooldown_s=cooldown_s,
                on_change=self._breaker_observer(index, tier_name),
                clock=clock,
            )
            self.tiers.append(_Tier(index=index, name=tier_name, model=model, breaker=breaker))
        self.last_tier: int | None = None
        self._pending: list[FallbackEvent] = []

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def config(self) -> Any:
        """The primary tier's config, so code that reads ``model.config``
        (pricing, context window, provider hints) keeps working."""
        return getattr(self.tiers[0].model, "config", None)

    @property
    def metrics(self) -> dict[str, dict[str, Any]]:
        """Per-tier counters and breaker state, keyed by tier name."""
        return {
            tier.name: {**tier.counters, "state": tier.breaker.state.value} for tier in self.tiers
        }

    # ------------------------------------------------------------------
    # ModelProtocol
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Complete on the first healthy tier that answers."""
        errors: list[tuple[str, BaseException]] = []
        for tier in self._candidates():
            tier.counters["attempts"] += 1
            try:
                if self.complete_timeout is not None:
                    async with asyncio.timeout(self.complete_timeout):
                        response = await tier.model.complete(messages, tools, **kwargs)
                else:
                    response = await tier.model.complete(messages, tools, **kwargs)
            except Exception as exc:  # noqa: BLE001 — classified below; non-fallback errors re-raise
                errors.append((tier.name, exc))
                if not await self._handle_failure(tier, exc):
                    raise
                continue
            except BaseException:
                tier.breaker.release()  # cancelled: no verdict on the tier
                raise
            await self._served(tier)
            return response  # type: ignore[no-any-return]
        await self._exhausted(errors)
        raise FallbackExhaustedError(errors) from errors[-1][1]

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelChunkEvent]:
        """Stream from the first healthy tier that starts answering.

        Fails over only before a tier's first chunk; see the module docs.
        """
        errors: list[tuple[str, BaseException]] = []
        for tier in self._candidates():
            tier.counters["attempts"] += 1
            source = tier.model.stream(messages, tools, **kwargs)
            iterator = source.__aiter__()
            try:
                if self.first_chunk_timeout is not None:
                    async with asyncio.timeout(self.first_chunk_timeout):
                        first = await iterator.__anext__()
                else:
                    first = await iterator.__anext__()
            except StopAsyncIteration:
                await self._served(tier)
                return
            except Exception as exc:  # noqa: BLE001 — classified below; non-fallback errors re-raise
                await _aclose(iterator)
                errors.append((tier.name, exc))
                if not await self._handle_failure(tier, exc):
                    raise
                continue
            except BaseException:
                tier.breaker.release()  # cancelled: no verdict on the tier
                await _aclose(iterator)
                raise

            await self._served(tier)
            try:
                yield first
                async for chunk in iterator:
                    yield chunk
            except Exception as exc:
                # Committed to this tier: never splice in another provider.
                if self._decide(exc)[1]:
                    tier.breaker.record_failure()
                    tier.counters["failures"] += 1
                    await self._flush()
                raise
            finally:
                await _aclose(iterator)
            return
        await self._exhausted(errors)
        raise FallbackExhaustedError(errors) from errors[-1][1]

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def _candidates(self) -> Iterator[_Tier]:
        """Tiers to try, in order, skipping (and reporting) open breakers.

        Lazy on purpose: ``allow()`` claims a half-open tier's single probe,
        so a tier is asked only when the call actually reaches it — asking
        up front would claim probes that are never sent and wedge them.
        """
        attempted = False
        for tier in self.tiers:
            if tier.breaker.allow():
                attempted = True
                yield tier
            else:
                tier.counters["skipped"] += 1
                self._record(FallbackEvent(kind="skipped", tier=tier.index, model=tier.name))
        if not attempted:
            # Every breaker is open. Refusing without trying helps nobody;
            # the primary is the best guess at what recovered first.
            yield self.tiers[0]

    def _decide(self, exc: BaseException) -> tuple[FailoverDecision, bool, bool]:
        """``(decision, counts_against_health, falls_back)`` for ``exc``."""
        decision = classify(exc)
        health = decision.reason in HEALTH_REASONS
        if self._should_fallback is not None:
            falls_back = bool(self._should_fallback(exc, decision))
        else:
            falls_back = health or decision.should_fallback
        return decision, health, falls_back

    async def _handle_failure(self, tier: _Tier, exc: BaseException) -> bool:
        """Record a tier failure; return whether to try the next tier."""
        decision, health, falls_back = self._decide(exc)
        tier.counters["failures"] += 1
        if health:
            tier.breaker.record_failure()
        else:
            tier.breaker.release()
        if falls_back:
            nxt = self._next_index(tier.index)
            self._record(
                FallbackEvent(
                    kind="fallback",
                    tier=tier.index,
                    model=tier.name,
                    next_tier=nxt,
                    reason=decision.reason.value,
                    error=f"{type(exc).__name__}: {exc}"[:300],
                )
            )
            logger.warning(
                "model tier %s (%s) failed with %s; %s",
                tier.index,
                tier.name,
                decision.reason.value,
                "trying the next tier" if nxt is not None else "no tier left",
            )
        await self._flush()
        return falls_back

    def _next_index(self, index: int) -> int | None:
        for tier in self.tiers[index + 1 :]:
            if tier.breaker.state is not CircuitState.OPEN:
                return tier.index
        return None

    async def _served(self, tier: _Tier) -> None:
        tier.breaker.record_success()
        tier.counters["served"] += 1
        self.last_tier = tier.index
        if tier.index != 0:
            self._record(FallbackEvent(kind="served", tier=tier.index, model=tier.name))
        await self._flush()

    async def _exhausted(self, errors: list[tuple[str, BaseException]]) -> None:
        last_name, last_exc = errors[-1]
        self._record(
            FallbackEvent(
                kind="exhausted",
                tier=len(self.tiers) - 1,
                model=last_name,
                error=f"{type(last_exc).__name__}: {last_exc}"[:300],
            )
        )
        await self._flush()

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def _breaker_observer(
        self, index: int, tier_name: str
    ) -> Callable[[CircuitState, CircuitState], None]:
        def _observe(old: CircuitState, new: CircuitState) -> None:
            if new is CircuitState.OPEN:
                logger.warning("model tier %s (%s) circuit opened", index, tier_name)
            self._record(
                FallbackEvent(
                    kind="breaker",
                    tier=index,
                    model=tier_name,
                    state=new.value,
                    previous_state=old.value,
                )
            )

        return _observe

    def _record(self, event: FallbackEvent) -> None:
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:  # noqa: BLE001 — an observer must not break the call
                logger.debug("FallbackChain on_event failed", exc_info=True)
        self._pending.append(event)

    async def _flush(self) -> None:
        """Publish recorded events on the telemetry bus (no-op without a run)."""
        pending, self._pending = self._pending, []
        for event in pending:
            name = EV_MODEL_BREAKER if event.kind == "breaker" else EV_MODEL_FALLBACK
            await emit(
                name,
                kind=event.kind,
                tier=event.tier,
                model=event.model,
                next_tier=event.next_tier,
                reason=event.reason,
                error=event.error,
                state=event.state,
                previous_state=event.previous_state,
            )


async def _aclose(iterator: Any) -> None:
    aclose = getattr(iterator, "aclose", None)
    if aclose is not None:
        with contextlib.suppress(Exception):
            await aclose()
