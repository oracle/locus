# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""One segment of a durable agent run, independent of the engine.

A durable run is a sequence of segments: the first run until the agent finishes
or pauses on a held call, then a resume after each decision. Temporal runs a
segment in an activity (:mod:`tulip.durable.temporal`), DBOS in a step
(:mod:`tulip.durable.dbos`); both call :func:`run_agent_segment`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tulip.core.events import InterruptEvent, TerminateEvent


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from tulip.agent import Agent


class SegmentError(Exception):
    """A segment cannot run at all, so retrying it will not help."""


@dataclass
class SegmentOutcome:
    """Where a segment left the run: ``paused`` on a person, or ``done``."""

    status: str
    final_message: str | None = None
    stop_reason: str | None = None
    question: str | None = None
    approval_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _json_safe(value: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = json.loads(json.dumps(value, default=str))
    return safe


async def run_agent_segment(
    agents: Mapping[str, Callable[[], Agent]],
    agent: str,
    *,
    thread_id: str,
    kind: str,
    prompt: str = "",
    answer: str = "",
    perform: bool = True,
) -> SegmentOutcome:
    """Run (``kind="run"``) or resume a registered agent until it finishes or pauses.

    Raises:
        SegmentError: No agent is registered under ``agent``, or it has no
            checkpointer to keep the conversation between segments.
    """
    factory = agents.get(agent)
    if factory is None:
        raise SegmentError(f"no agent registered as {agent!r} on this worker")
    instance = factory()
    if instance.config.checkpointer is None:
        raise SegmentError(f"agent {agent!r} needs a checkpointer shared by every worker")
    if kind == "run":
        events = instance.run(prompt, thread_id=thread_id)
    else:
        events = instance.resume(answer, thread_id=thread_id, perform_dangling=perform)

    paused: InterruptEvent | None = None
    final: TerminateEvent | None = None
    async for event in events:
        if isinstance(event, InterruptEvent):
            paused = event
        elif isinstance(event, TerminateEvent):
            final = event
    if paused is not None:
        return SegmentOutcome(
            status="paused",
            question=paused.question,
            approval_id=paused.metadata.get("approval_id"),
            metadata=_json_safe(dict(paused.metadata)),
        )
    return SegmentOutcome(
        status="done",
        final_message=final.final_message if final else None,
        stop_reason=final.reason if final else None,
    )


__all__ = ["SegmentError", "SegmentOutcome", "run_agent_segment"]
