# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Structured rate-limit state parsed from HTTP response headers.

Most model providers expose per-request rate-limit headroom through
``x-ratelimit-*`` response headers. Locus captures these into frozen
:class:`RateLimitBucket` / :class:`RateLimitState` models so hooks,
orchestrators, and credential-pool rotation can reason about headroom
without every consumer re-parsing strings.

Header conventions supported:

* **OpenAI / OpenRouter / Nous Portal**: 12 headers — ``limit``,
  ``remaining``, ``reset`` across ``requests`` / ``tokens`` and
  minute / ``-1h`` hour windows. Reset values may be numeric seconds
  (``58``) or OpenAI's human-readable duration (``1m60s`` / ``200ms``).

Providers whose headers use a different naming scheme (Anthropic's
``anthropic-ratelimit-*``, Bedrock's ``x-amzn-*``, …) are not parsed
by this module. :func:`parse_rate_limit_headers` returns ``None``
instead of a half-filled state so callers can attach their own
provider-specific parser.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field


__all__ = [
    "RateLimitBucket",
    "RateLimitState",
    "parse_rate_limit_headers",
]


# ---------------------------------------------------------------------------
# Value models
# ---------------------------------------------------------------------------


class RateLimitBucket(BaseModel):
    """One rate-limit window (e.g. requests per minute)."""

    model_config = {"frozen": True}

    limit: int = Field(default=0, ge=0, description="Total capacity in this window.")
    remaining: int = Field(default=0, ge=0, description="Remaining headroom in this window.")
    reset_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Seconds until this window resets, as reported by the "
            "provider at ``captured_at``. Call "
            ":meth:`seconds_until_reset` for an elapsed-adjusted value."
        ),
    )
    captured_at: datetime = Field(description="Wall-clock time the headers were observed (UTC).")

    @property
    def used(self) -> int:
        """How many units of this bucket have been consumed."""
        return max(0, self.limit - self.remaining)

    @property
    def usage_pct(self) -> float:
        """Consumption as a percentage of the limit (``0`` if unknown)."""
        if self.limit <= 0:
            return 0.0
        return (self.used / self.limit) * 100.0

    def seconds_until_reset(self, *, now: datetime | None = None) -> float:
        """Estimate reset seconds from ``now``, adjusted for elapsed time."""
        at = now if now is not None else datetime.now(UTC)
        elapsed = (at - self.captured_at).total_seconds()
        return max(0.0, self.reset_seconds - elapsed)

    def reset_at(self) -> datetime:
        """Absolute wall-clock time the bucket is expected to reset."""
        return self.captured_at + timedelta(seconds=self.reset_seconds)


class RateLimitState(BaseModel):
    """Full rate-limit state parsed from a single response.

    A bucket is ``None`` when the corresponding header wasn't present —
    not every provider surfaces all four dimensions, and consumers
    should tolerate missing data rather than assume zero.
    """

    model_config = {"frozen": True}

    requests_min: RateLimitBucket | None = None
    requests_hour: RateLimitBucket | None = None
    tokens_min: RateLimitBucket | None = None
    tokens_hour: RateLimitBucket | None = None
    captured_at: datetime
    provider: str = ""

    @property
    def has_any_bucket(self) -> bool:
        return any(
            b is not None
            for b in (
                self.requests_min,
                self.requests_hour,
                self.tokens_min,
                self.tokens_hour,
            )
        )

    def age_seconds(self, *, now: datetime | None = None) -> float:
        at = now if now is not None else datetime.now(UTC)
        return (at - self.captured_at).total_seconds()


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------


# Matches OpenAI-style duration strings: optional minutes + optional
# seconds (with decimals) + optional milliseconds. Examples:
#   "60"        -> 60.0
#   "60s"       -> 60.0
#   "1m60s"     -> 120.0
#   "1.5s"      -> 1.5
#   "200ms"     -> 0.2
#   "1m200ms"   -> 60.2
_DURATION_RE = re.compile(
    r"""
    ^\s*
    (?:(?P<min>\d+(?:\.\d+)?)\s*m(?!s))?      # minutes: '1m' but not '1ms'
    \s*
    (?:(?P<sec>\d+(?:\.\d+)?)\s*s(?!$\|(?<=m)s))? # seconds
    \s*
    (?:(?P<ms>\d+(?:\.\d+)?)\s*ms)?            # milliseconds
    \s*$
    """,
    re.VERBOSE,
)


def _parse_duration(value: Any) -> float:
    """Parse a reset-time value into seconds.

    Accepts:
    - plain numbers (``58``, ``58.3``) — treated as seconds
    - OpenAI duration strings (``1m60s``, ``200ms``, ``1m``, ``60s``)
    - floats as strings (``"58.5"``)
    - None / empty / unparseable → ``0.0``
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return max(0.0, float(value))

    s = str(value).strip()
    if not s:
        return 0.0

    # Plain numeric path.
    try:
        return max(0.0, float(s))
    except ValueError:
        pass

    m = _DURATION_RE.match(s)
    if not m or not any(m.groupdict().values()):
        return 0.0
    minutes = float(m.group("min") or 0)
    seconds = float(m.group("sec") or 0)
    millis = float(m.group("ms") or 0)
    return minutes * 60.0 + seconds + millis / 1000.0


def _parse_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return 0


def parse_rate_limit_headers(
    headers: Mapping[str, str],
    *,
    provider: str = "",
    now: datetime | None = None,
) -> RateLimitState | None:
    """Parse ``x-ratelimit-*`` headers into a :class:`RateLimitState`.

    Args:
        headers: Response headers. Case-insensitive lookup is applied.
        provider: Optional provider identifier to stamp on the state.
        now: Override the capture time (useful for deterministic tests).

    Returns:
        A :class:`RateLimitState`, or ``None`` if no ``x-ratelimit-*``
        headers are present at all.
    """
    lowered = {k.lower(): v for k, v in headers.items()}
    if not any(k.startswith("x-ratelimit-") for k in lowered):
        return None

    captured_at = now if now is not None else datetime.now(UTC)

    def _bucket(resource: str, suffix: str = "") -> RateLimitBucket | None:
        tag = f"{resource}{suffix}"
        limit_key = f"x-ratelimit-limit-{tag}"
        remaining_key = f"x-ratelimit-remaining-{tag}"
        reset_key = f"x-ratelimit-reset-{tag}"
        if not any(k in lowered for k in (limit_key, remaining_key, reset_key)):
            return None
        return RateLimitBucket(
            limit=_parse_int(lowered.get(limit_key)),
            remaining=_parse_int(lowered.get(remaining_key)),
            reset_seconds=_parse_duration(lowered.get(reset_key)),
            captured_at=captured_at,
        )

    return RateLimitState(
        requests_min=_bucket("requests"),
        requests_hour=_bucket("requests", "-1h"),
        tokens_min=_bucket("tokens"),
        tokens_hour=_bucket("tokens", "-1h"),
        captured_at=captured_at,
        provider=provider,
    )
