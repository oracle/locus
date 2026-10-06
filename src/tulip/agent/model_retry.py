# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Loop-level retry of transient model-call failures.

The agent loop wraps every model call in :func:`call_with_retry`. Which
failures are worth another attempt is the failover classifier's call
(:func:`tulip.models.failover.classify`); this module adds the policy on top:
how long to wait, how many times, and within what wall-clock budget. See
:class:`~tulip.agent.config.ModelRetryConfig` for the knobs.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from email.utils import parsedate_to_datetime
from typing import Any, TypeVar

from tulip.agent.config import ModelRetryConfig
from tulip.core.events import ModelRetryEvent
from tulip.models.failover import FailoverReason, classify


__all__ = ["RETRYABLE_REASONS", "call_with_retry", "retry_after_seconds"]

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Reasons a retry can fix. Everything else either fails the same way again
#: (context overflow, malformed request, auth, billing, unknown model) or is
#: unclassified — and an unclassified exception is as likely a bug in a hook
#: or a binding as a provider hiccup, so it surfaces at once.
RETRYABLE_REASONS: frozenset[FailoverReason] = frozenset(
    {
        FailoverReason.RATE_LIMIT,
        FailoverReason.OVERLOADED,
        FailoverReason.SERVER_ERROR,
        FailoverReason.TIMEOUT,
    }
)

# Module-level so tests can replace them without real waiting or randomness.
_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
_uniform: Callable[[float, float], float] = random.uniform
_monotonic: Callable[[], float] = time.monotonic


def _headers_of(exc: BaseException) -> Any:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    return headers


def _header(headers: Any, name: str) -> Any:
    if headers is None:
        return None
    try:
        value = headers.get(name)
        if value is None:
            # A plain dict is case-sensitive; httpx.Headers is not.
            value = headers.get(name.title())
    except (AttributeError, TypeError):
        return None
    return value


def _parse_retry_after(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return max(0.0, float(value))
    text = str(value).strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time())


def retry_after_seconds(exc: BaseException) -> float | None:
    """The wait the provider asked for, in seconds, or ``None``.

    Looks at a ``retry_after`` attribute, then the ``retry-after-ms`` and
    ``retry-after`` response headers (seconds or an HTTP date), along the
    exception's ``__cause__`` / ``__context__`` chain — a wrapper such as a
    fallback chain keeps the provider's error there.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        attr = _parse_retry_after(getattr(current, "retry_after", None))
        if attr is not None:
            return attr
        headers = _headers_of(current)
        ms = _parse_retry_after(_header(headers, "retry-after-ms"))
        if ms is not None:
            return ms / 1000.0
        seconds = _parse_retry_after(_header(headers, "retry-after"))
        if seconds is not None:
            return seconds
        current = current.__cause__ or current.__context__
    return None


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


async def call_with_retry(
    attempt: Callable[[], Awaitable[T]],
    policy: ModelRetryConfig | None,
    *,
    notify: Callable[[ModelRetryEvent], Awaitable[None]],
    may_retry: Callable[[], bool] = lambda: True,
) -> T:
    """Await ``attempt()``, retrying transient provider failures.

    Args:
        attempt: Makes one model call. Called again for each retry.
        policy: The agent's retry policy; ``None`` or disabled means one try.
        notify: Receives one :class:`ModelRetryEvent` per retry, before the
            backoff sleep.
        may_retry: Consulted after a failure; ``False`` re-raises it. The loop
            uses it to refuse a retry once a streamed attempt has already put
            chunks in front of the caller, or once the run is cancelled.

    Raises:
        The last failure, once it is not retryable, the retries are spent, or
        the next wait would end past ``policy.total_budget_seconds``.
    """
    if policy is None or not policy.enabled or policy.max_retries == 0:
        return await attempt()

    started = _monotonic()
    retries = 0
    while True:
        try:
            return await attempt()
        except Exception as exc:
            if retries >= policy.max_retries or not may_retry():
                raise
            decision = classify(exc)
            retryable = decision.reason in RETRYABLE_REASONS or (
                policy.retry_unclassified and decision.reason is FailoverReason.UNKNOWN
            )
            if not retryable:
                raise
            requested = retry_after_seconds(exc)
            if requested is not None:
                delay = requested
            else:
                ceiling = min(policy.max_delay, policy.initial_delay * (2**retries))
                delay = _uniform(0.0, ceiling)
            elapsed = _monotonic() - started
            if elapsed + delay > policy.total_budget_seconds:
                logger.warning(
                    "Model call failed (%s); a retry in %.1fs would pass the %.0fs "
                    "retry budget, giving up",
                    decision.reason.value,
                    delay,
                    policy.total_budget_seconds,
                )
                raise
            retries += 1
            event = ModelRetryEvent(
                attempt=retries,
                delay_seconds=delay,
                reason=decision.reason.value,
                status_code=decision.status_code,
                error=_describe(exc),
                from_retry_after=requested is not None,
            )
            logger.warning(
                "Model call failed (%s, %s); retry %d/%d in %.1fs",
                decision.reason.value,
                event.error,
                retries,
                policy.max_retries,
                delay,
            )
            await notify(event)
            await _sleep(delay)
