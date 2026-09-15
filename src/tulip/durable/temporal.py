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

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from temporalio import activity
from temporalio.exceptions import ApplicationError
from temporalio.worker import Worker

from tulip.durable.segments import SegmentError, run_agent_segment
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


@activity.defn(name=RUN_SEGMENT)
async def run_segment(segment: Segment) -> SegmentResult:
    """Run or resume a registered agent until it finishes or pauses."""
    try:
        outcome = await run_agent_segment(
            _AGENTS,
            segment.agent,
            thread_id=segment.thread_id,
            kind=segment.kind,
            prompt=segment.prompt,
            answer=segment.answer,
            perform=segment.perform,
        )
    except SegmentError as exc:
        raise ApplicationError(str(exc), non_retryable=True) from exc
    return SegmentResult(**asdict(outcome))


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
