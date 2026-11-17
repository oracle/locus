# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Checkpoint expiry: the time a checkpoint stops being loadable, and how backends keep it.

Every checkpointer takes ``ttl=`` (seconds or a ``timedelta``). A save made
with one records ``expires_at = now + ttl`` beside the checkpoint — in the
record's own fields in memory and in a file's header, in the adapter's
index entry and document, in a bucket's ``.meta.json`` and payload, in an
``expires_at`` column on SQL — and every read filters on it: an expired
checkpoint is not loaded, listed or forked, whether or not the store has
removed it yet. :meth:`~locus.memory.checkpointer.BaseCheckpointer.expire`
removes what has expired and returns how many checkpoints went; a backend
with native expiry (Redis ``EXPIRE``) also lets the store drop the keys.

The helpers here are what the backends share: the parsing of a ``ttl``, the
expiry time of a save and the test a read applies.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

from locus.memory.time_travel import parse_time


__all__ = ["EXPIRES_AT_KEY", "expiry_time", "is_expired", "parse_ttl", "ttl_seconds"]

#: The field a backend records a checkpoint's expiry time under (ISO-8601, UTC).
EXPIRES_AT_KEY = "expires_at"


def parse_ttl(ttl: float | timedelta | None) -> timedelta | None:
    """``ttl`` as a positive ``timedelta``; ``None`` for no expiry.

    Raises:
        TypeError: ``ttl`` is neither a number of seconds nor a ``timedelta``.
        ValueError: ``ttl`` is zero or negative.
    """
    if ttl is None:
        return None
    if isinstance(ttl, timedelta):
        span = ttl
    elif isinstance(ttl, int | float) and not isinstance(ttl, bool):
        span = timedelta(seconds=float(ttl))
    else:
        raise TypeError(f"ttl must be seconds or a timedelta; got {type(ttl).__name__}")
    if span <= timedelta(0):
        raise ValueError("ttl must be positive")
    return span


def ttl_seconds(ttl: timedelta) -> int:
    """``ttl`` as whole seconds for a store that expires keys in seconds, at least one."""
    return max(1, math.ceil(ttl.total_seconds()))


def expiry_time(ttl: timedelta | None, now: datetime | None = None) -> datetime | None:
    """When a save made ``now`` with ``ttl`` expires; ``None`` without a ttl."""
    if ttl is None:
        return None
    return (now or datetime.now(UTC)) + ttl


def is_expired(expires_at: Any, now: datetime | None = None) -> bool:
    """Whether a recorded expiry time (a datetime, an ISO string or ``None``) has passed.

    A record without one, or with one that does not parse, never expires:
    expiry is never guessed.
    """
    when = parse_time(expires_at)
    if when is None:
        return False
    return when <= (now or datetime.now(UTC))
