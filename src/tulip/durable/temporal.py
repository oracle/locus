# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Run an agent on Temporal: durable waits for people, across worker restarts.

A run is a Temporal workflow. The agent's work runs in activities, in segments:
the first run until the agent finishes or pauses on a held call, then a resume
after each decision. Between segments the workflow waits for a ``decide``
signal, for as long as it takes; workers can restart meanwhile::

    worker = create_worker(
        client, task_queue="agents", agents={"refunds": make_agent}
    )
    async with worker:
        handle = await start_agent_run(
            client,
            agent="refunds",
            prompt="refund order 4821",
            thread_id="t1",
            task_queue="agents",
        )
        # ... a person approves in the ApprovalStore, then:
        await signal_decided(client, handle.id)
        outcome = await handle.result()

The agent must have a checkpointer that every worker reaches (a file on shared
storage, Postgres): the conversation lives there between segments, and approval
decisions live in the approval store, exactly as without Temporal.

Each segment is attempted once. A segment that dies halfway may already have run
tool calls, and running it again would repeat them, so a failed segment fails
the workflow; the checkpoint shows where it stopped.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from temporalio import activity
from temporalio.exceptions import ApplicationError
from temporalio.worker import Worker

from tulip.core.events import InterruptEvent, TerminateEvent
from tulip.durable.temporal_workflow import (
    RUN_SEGMENT,
    AgentRequest,
    AgentWorkflow,
    Decision,
    Segment,
    SegmentResult,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from temporalio.client import Client, WorkflowHandle

    from tulip.agent import Agent


_AGENTS: dict[str, Callable[[], Agent]] = {}


def _json_safe(value: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = json.loads(json.dumps(value, default=str))
    return safe


@activity.defn(name=RUN_SEGMENT)
async def run_segment(segment: Segment) -> SegmentResult:
    """Run or resume a registered agent until it finishes or pauses."""
    factory = _AGENTS.get(segment.agent)
    if factory is None:
        raise ApplicationError(
            f"no agent registered as {segment.agent!r} on this worker", non_retryable=True
        )
    agent = factory()
    if agent.config.checkpointer is None:
        raise ApplicationError(
            f"agent {segment.agent!r} needs a checkpointer shared by every worker",
            non_retryable=True,
        )
    if segment.kind == "run":
        events = agent.run(segment.prompt, thread_id=segment.thread_id)
    else:
        events = agent.resume(
            segment.answer, thread_id=segment.thread_id, perform_dangling=segment.perform
        )

    paused: InterruptEvent | None = None
    final: TerminateEvent | None = None
    async for event in events:
        if isinstance(event, InterruptEvent):
            paused = event
        elif isinstance(event, TerminateEvent):
            final = event
    if paused is not None:
        return SegmentResult(
            status="paused",
            question=paused.question,
            approval_id=paused.metadata.get("approval_id"),
            metadata=_json_safe(dict(paused.metadata)),
        )
    return SegmentResult(
        status="done",
        final_message=final.final_message if final else None,
        stop_reason=final.reason if final else None,
    )


def create_worker(
    client: Client,
    *,
    task_queue: str,
    agents: Mapping[str, Callable[[], Agent]],
    **worker_options: Any,
) -> Worker:
    """A worker that runs :class:`AgentWorkflow` and the agents registered by name."""
    _AGENTS.update(agents)
    return Worker(
        client,
        task_queue=task_queue,
        workflows=[AgentWorkflow],
        activities=[run_segment],
        **worker_options,
    )


async def start_agent_run(
    client: Client,
    *,
    agent: str,
    prompt: str,
    thread_id: str,
    task_queue: str,
    workflow_id: str | None = None,
    segment_timeout_seconds: float = 600.0,
) -> WorkflowHandle[Any, SegmentResult]:
    """Start a durable run of a registered agent; the workflow id defaults to the thread."""
    handle: WorkflowHandle[Any, SegmentResult] = await client.start_workflow(
        AgentWorkflow.run,
        AgentRequest(
            agent=agent,
            prompt=prompt,
            thread_id=thread_id,
            segment_timeout_seconds=segment_timeout_seconds,
        ),
        id=workflow_id or f"tulip-{thread_id}",
        task_queue=task_queue,
    )
    return handle


async def signal_decided(
    client: Client, workflow_id: str, *, answer: str = "approved", perform: bool = True
) -> None:
    """Tell a paused run that its approval has been decided, so it resumes."""
    await client.get_workflow_handle(workflow_id).signal(
        AgentWorkflow.decide, Decision(answer=answer, perform=perform)
    )


__all__ = [
    "AgentRequest",
    "AgentWorkflow",
    "Decision",
    "SegmentResult",
    "create_worker",
    "run_segment",
    "signal_decided",
    "start_agent_run",
]
