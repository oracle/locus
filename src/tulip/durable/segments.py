# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""One segment of a durable agent or graph run, independent of the engine.

A durable run is a sequence of segments: the first run until the agent finishes
or pauses on a held call, then a resume after each decision. Temporal runs a
segment in an activity (:mod:`tulip.durable.temporal`), DBOS in a step
(:mod:`tulip.durable.dbos`); both call :func:`run_agent_segment`.

A :class:`~tulip.multiagent.graph.StateGraph` runs the same way: a segment runs
the graph until it ends or a node pauses on
:func:`~tulip.core.interrupt.interrupt`, then resumes it from its checkpoint
with the value the pause was waiting for (:func:`run_graph`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from tulip.core.events import InterruptEvent, TerminateEvent


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from tulip.agent import Agent
    from tulip.multiagent.graph import StateGraph


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


@dataclass
class GraphSegmentOutcome:
    """Where a segment left a graph run: ``paused`` on an interrupt, ``done`` or ``failed``.

    ``state`` is the graph's state at that point, without its internal
    ``__``-prefixed keys. ``interrupt`` is the payload the pausing node passed
    to :func:`~tulip.core.interrupt.interrupt`, and ``node`` the node that
    paused (or failed). ``error`` is the failed node's error.
    """

    status: Literal["done", "paused", "failed"]
    state: dict[str, Any] = field(default_factory=dict)
    interrupt: Any = None
    node: str | None = None
    error: str | None = None


def _plain(value: Any) -> Any:
    """``value`` as plain JSON data: models dumped, anything else unknown as text."""

    def default(obj: Any) -> Any:
        dump = getattr(obj, "model_dump", None)
        if callable(dump):
            return dump(mode="json")
        return str(obj)

    return json.loads(json.dumps(value, default=default))


async def run_graph(
    graphs: Mapping[str, Callable[[], StateGraph]],
    graph: str,
    *,
    thread_id: str,
    inputs: dict[str, Any] | None = None,
    resume: Any = None,
    resuming: bool = False,
) -> GraphSegmentOutcome:
    """Run a registered graph from ``inputs``, or resume it with ``resume``, until it
    ends or pauses on an interrupt.

    The graph runs with its own config copied and ``thread_id`` set, so one
    registered graph serves any number of runs at once without sharing state.

    Raises:
        SegmentError: No graph is registered under ``graph``, or it has no
            checkpointer to keep its state between segments.
    """
    from tulip.core.command import Command  # noqa: PLC0415

    factory = graphs.get(graph)
    if factory is None:
        raise SegmentError(f"no graph registered as {graph!r} on this worker")
    instance = factory()
    if instance.config.checkpointer is None:
        raise SegmentError(f"graph {graph!r} needs a checkpointer shared by every worker")
    config = instance.config.model_copy(update={"thread_id": thread_id})
    if resuming:
        result = await instance.execute(Command(resume=resume), config=config)
    else:
        result = await instance.execute(dict(inputs or {}), config=config)
    state = _plain({k: v for k, v in result.final_state.items() if not k.startswith("__")})
    if result.interrupt is not None:
        value = result.interrupt.interrupt
        payload = getattr(value, "payload", value)
        return GraphSegmentOutcome(
            status="paused",
            state=state,
            interrupt=_plain(payload),
            node=result.interrupt.node_id,
        )
    failed = next((r for r in result.node_results.values() if r.status == "failed"), None)
    if failed is not None or not result.success:
        return GraphSegmentOutcome(
            status="failed",
            state=state,
            node=failed.node_id if failed else None,
            error=(failed.error if failed else None) or "the graph did not complete",
        )
    return GraphSegmentOutcome(status="done", state=state)


__all__ = [
    "GraphSegmentOutcome",
    "SegmentError",
    "SegmentOutcome",
    "run_agent_segment",
    "run_graph",
]
