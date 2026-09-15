# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""The Temporal workflow for an agent run. Workflow code only.

Temporal replays workflow code and requires it to be deterministic, and the
agent loop is not (it reads the clock, makes ids, starts tasks). So the agent
never runs here: every segment of it runs in the ``tulip_run_segment`` activity
(:mod:`tulip.durable.temporal`), and this module only sequences segments and
waits. It imports nothing from ``tulip`` so Temporal's sandbox can load it
cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy


RUN_SEGMENT = "tulip_run_segment"


@dataclass
class AgentRequest:
    """What to run: a registered agent, a prompt, and the thread that persists it."""

    agent: str
    prompt: str
    thread_id: str
    segment_timeout_seconds: float = 600.0


@dataclass
class Segment:
    """One activity's worth of agent work: the first run, or a resume."""

    agent: str
    thread_id: str
    kind: str
    prompt: str = ""
    answer: str = ""
    perform: bool = True


@dataclass
class SegmentResult:
    """Where a segment left the run: ``paused`` on a person, or ``done``."""

    status: str
    final_message: str | None = None
    stop_reason: str | None = None
    question: str | None = None
    approval_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    """Sent once a person has decided; the run resumes with it."""

    answer: str = "approved"
    perform: bool = True


@workflow.defn(name="TulipAgentRun")
class AgentWorkflow:
    """Run segments until the agent is done, waiting on :meth:`decide` at each pause."""

    def __init__(self) -> None:
        self._pending: SegmentResult | None = None
        self._decision: Decision | None = None

    @workflow.run
    async def run(self, request: AgentRequest) -> SegmentResult:
        timeout = timedelta(seconds=request.segment_timeout_seconds)
        # One attempt per segment: a segment that died halfway may have run tool
        # calls, and replaying it would add the prompt again and repeat them.
        once = RetryPolicy(maximum_attempts=1)
        result: SegmentResult = await workflow.execute_activity(
            RUN_SEGMENT,
            Segment(
                agent=request.agent, thread_id=request.thread_id, kind="run", prompt=request.prompt
            ),
            result_type=SegmentResult,
            start_to_close_timeout=timeout,
            retry_policy=once,
        )
        while result.status == "paused":
            self._pending = result
            await workflow.wait_condition(lambda: self._decision is not None)
            decision = self._decision or Decision()
            self._decision = None
            self._pending = None
            result = await workflow.execute_activity(
                RUN_SEGMENT,
                Segment(
                    agent=request.agent,
                    thread_id=request.thread_id,
                    kind="resume",
                    answer=decision.answer,
                    perform=decision.perform,
                ),
                result_type=SegmentResult,
                start_to_close_timeout=timeout,
                retry_policy=once,
            )
        return result

    @workflow.signal
    def decide(self, decision: Decision) -> None:
        """A person has decided; resume the run."""
        self._decision = decision

    @workflow.query
    def pending(self) -> SegmentResult | None:
        """The pause the run is waiting on, or ``None``."""
        return self._pending
