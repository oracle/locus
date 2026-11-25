# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Where the agent loop reads the time, makes ids and draws random numbers.

The loop stamps events and state, measures durations and time budgets, makes
run, tool-call and checkpoint ids, and jitters its retry delays. It reads all
of that from here instead of from :mod:`time`, :mod:`datetime`, :mod:`uuid`
and :mod:`random`, so a host that must control them can supply its own.

The case this exists for is a durable-execution engine that replays the loop,
such as a Temporal workflow: on replay the loop must read the same times and
make the same ids as it did the first time, or it can take a different branch
than the one that was recorded. Such a host installs sources backed by the
engine's deterministic clock, ids and random generator::

    from locus.core import sources


    class WorkflowSources(sources.SystemSources):
        def now(self) -> datetime:
            return workflow.now() if workflow.in_workflow() else super().now()

        ...


    sources.set_sources(WorkflowSources())

The sources are process-wide, and are asked on every read. Sources that
should apply only in some contexts decide that per call, as above. A test can
install fixed sources the same way and restore the previous ones with
:func:`set_sources` or :func:`reset_sources`.

Without a call to :func:`set_sources` nothing changes: :class:`SystemSources`
reads the system clock and makes random ids, as the loop always has.
"""

from __future__ import annotations

import random as _random
import time as _time
import uuid as _uuid
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable


__all__ = [
    "RuntimeSources",
    "SystemSources",
    "get_sources",
    "monotonic",
    "now",
    "reset_sources",
    "set_sources",
    "time",
    "uniform",
    "uuid4",
]


@runtime_checkable
class RuntimeSources(Protocol):
    """What the agent loop needs from its host: a clock, ids and randomness."""

    def now(self) -> datetime:
        """The current time, timezone-aware, in UTC."""
        ...

    def time(self) -> float:
        """The current time in seconds since the epoch."""
        ...

    def monotonic(self) -> float:
        """A clock in seconds for measuring durations and budgets.

        Only differences between two reads are used.
        """
        ...

    def uuid4(self) -> _uuid.UUID:
        """A new UUID."""
        ...

    def uniform(self, a: float, b: float) -> float:
        """A random float between ``a`` and ``b``."""
        ...


class SystemSources:
    """The system clock, random UUIDs and the process's random generator."""

    def now(self) -> datetime:
        """The current time, timezone-aware, in UTC."""
        return datetime.now(UTC)

    def time(self) -> float:
        """The current time in seconds since the epoch."""
        return _time.time()

    def monotonic(self) -> float:
        """The system's monotonic clock, in seconds."""
        return _time.perf_counter()

    def uuid4(self) -> _uuid.UUID:
        """A new random UUID."""
        return _uuid.uuid4()

    def uniform(self, a: float, b: float) -> float:
        """A random float between ``a`` and ``b``."""
        return _random.uniform(a, b)  # noqa: S311 - retry jitter, not a secret


_SYSTEM: RuntimeSources = SystemSources()
_current: RuntimeSources = _SYSTEM


def get_sources() -> RuntimeSources:
    """Return the sources the agent loop reads from."""
    return _current


def set_sources(sources: RuntimeSources) -> RuntimeSources:
    """Install ``sources`` process-wide and return the ones they replace."""
    global _current  # noqa: PLW0603
    previous = _current
    _current = sources
    return previous


def reset_sources() -> None:
    """Go back to :class:`SystemSources`."""
    set_sources(_SYSTEM)


def now() -> datetime:
    """The current time, timezone-aware, in UTC."""
    return _current.now()


def time() -> float:
    """The current time in seconds since the epoch."""
    return _current.time()


def monotonic() -> float:
    """A clock in seconds for measuring durations and budgets."""
    return _current.monotonic()


def uuid4() -> _uuid.UUID:
    """A new UUID."""
    return _current.uuid4()


def uniform(a: float, b: float) -> float:
    """A random float between ``a`` and ``b``."""
    return _current.uniform(a, b)
