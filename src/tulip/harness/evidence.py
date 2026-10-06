# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""A record of every command the harness ran, for the audit trail.

An approval says a command was allowed; it does not say what the command
did. :class:`ExecRecord` is the second half: which command (by hash), where
it ran, how it ended, how long it took and what it printed (by hash and
size). It carries digests rather than text on purpose. A command line can
hold a token and its output can hold anything; a record that copies them
into telemetry is a leak with a timestamp. The digests still bind the
record to the exact bytes, so whoever kept the transcript can prove which
command and which output a record is about.

Records go out on the SDK's event bus as ``harness.exec``
(:data:`~tulip.observability.emit.EV_HARNESS_EXEC`) — a no-op outside a run —
and to the harness's own ``on_exec`` sink when one is given, which is how
the gateway attaches them to a run's evidence.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass

from tulip.harness.backend import ExecResult
from tulip.observability.emit import EV_HARNESS_EXEC, emit_sync


__all__ = ["ExecRecord", "exec_record", "publish"]


@dataclass(frozen=True)
class ExecRecord:
    """One command the harness ran.

    Attributes:
        command_sha256: SHA-256 of the command line, UTF-8.
        exit_code: Its exit status; ``None`` when it did not finish, or when
            it was started in the background and nobody has waited on it.
        duration_s: Seconds from start to the end, or to when the caller
            stopped waiting.
        timed_out: Whether it ran out of time.
        output_sha256: SHA-256 of the output the harness kept.
        output_bytes: Size of that output.
        truncated: Whether output was dropped before it was kept.
        backend_label: Where it ran (``BackendCapabilities.label``).
        background: Whether it was started without waiting for it.
    """

    command_sha256: str
    exit_code: int | None
    duration_s: float
    timed_out: bool
    output_sha256: str
    output_bytes: int
    truncated: bool
    backend_label: str
    background: bool = False


def exec_record(
    command: str,
    result: ExecResult,
    *,
    backend_label: str,
    background: bool = False,
) -> ExecRecord:
    """The record for ``command`` and what it did."""
    return ExecRecord(
        command_sha256=hashlib.sha256(command.encode("utf-8")).hexdigest(),
        exit_code=result.exit_code,
        duration_s=round(result.duration_s, 3),
        timed_out=result.timed_out,
        output_sha256=hashlib.sha256(result.output).hexdigest(),
        output_bytes=len(result.output),
        truncated=result.truncated,
        backend_label=backend_label,
        background=background,
    )


def publish(record: ExecRecord, sink: Callable[[ExecRecord], None] | None = None) -> None:
    """Emit ``record`` on the event bus and hand it to ``sink``."""
    emit_sync(EV_HARNESS_EXEC, **asdict(record))
    if sink is not None:
        sink(record)
