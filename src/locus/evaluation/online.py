# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Online evaluation: sample live runs off the event bus and grade them in the background.

A regression suite grades the cases somebody wrote. Production grades the
cases nobody wrote. :class:`OnlineEvaluator` subscribes to the
:class:`~locus.observability.event_bus.EventBus`, follows each run's
``agent.*`` events to its ``agent.terminate``, keeps the runs a sampler
picks within a budget, turns each into an :class:`~locus.evaluation.EvalCase`
from a template of expectations, and grades it with
:meth:`EvalRunner.agrade <locus.evaluation.EvalRunner.agrade>`: the same
checks, the same evaluators, the same judge as the suite. The results fold
into an ordinary :class:`~locus.evaluation.EvalReport`, so the gate, the
diff and the stores read it unchanged.

What the bus carries, the evaluator grades. The bridge publishes every tool
call with its name, outcome and a preview of its result, the iteration
count, the stop reason and a preview of the final message; it does not
publish the prompt, the arguments or the full answer, which keeps user
data off the bus by design. So the trajectory evaluators, the budgets and
the forbidden-tool checks grade a bus-only run completely, while a rubric
or a receipt check sees previews until the application attaches what it
knows with :meth:`OnlineEvaluator.attach`, or hands the whole result to
:meth:`OnlineEvaluator.submit`.

    evaluator = OnlineEvaluator(
        template=EvalCase(
            name="live",
            prompt="",
            evaluators=[ToolSelection(expected=["lookup_order"], forbidden=["issue_refund"])],
            max_iterations=6,
        ),
        sampler=fraction(0.1, seed=7),
        budget=OnlineBudget(max_cases=200, max_per_minute=20),
    )
    await evaluator.start()
    ...
    await evaluator.stop()
    print(evaluator.report().summary())
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from locus.core.state import ToolExecution
from locus.evaluation.framework import (
    EvalCase,
    EvalReport,
    EvalResult,
    EvalRunner,
    suite_fingerprint,
)
from locus.observability.emit import (
    EV_AGENT_TERMINATE,
    EV_AGENT_THINK,
    EV_AGENT_TOKENS_USED,
    EV_AGENT_TOOL_COMPLETED,
    EV_AGENT_TOOL_STARTED,
)
from locus.observability.event_bus import EventBus, StreamEvent, get_event_bus


__all__ = [
    "OnlineBudget",
    "OnlineEvaluator",
    "RecordedRun",
    "RecordedToolCall",
    "Sampler",
    "every_nth",
    "fraction",
    "template_fingerprint",
]


logger = logging.getLogger(__name__)

#: A sampler: given a finished run, whether to grade it.
Sampler = Callable[["RecordedRun"], bool]

_MAX_OPEN_RUNS = 1000


class RecordedToolCall(BaseModel):
    """One tool call seen on the bus, started and (when it finished) completed."""

    tool_name: str
    span_id: str | None = None
    success: bool | None = None
    error: str | None = None
    output_preview: str = ""
    duration_ms: float | None = None
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Attached by the application; the bus carries none"
    )


class RecordedRun(BaseModel):
    """One run reconstructed from its ``agent.*`` events, plus what the app attached."""

    run_id: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    prompt: str | None = Field(default=None, description="Attached; the bus carries no prompt")
    output: str | None = Field(default=None, description="Attached; the bus carries a preview")
    final_message_preview: str = ""
    reason: str | None = None
    iterations: int = 0
    total_tokens: int = 0
    tool_calls: list[RecordedToolCall] = Field(default_factory=list)
    events: int = 0

    @property
    def finished(self) -> bool:
        return self.finished_at is not None

    @property
    def message(self) -> str:
        """The answer to grade: the attached output, else the bus preview."""
        return self.output if self.output is not None else self.final_message_preview

    @property
    def tool_executions(self) -> list[ToolExecution]:
        """The calls as ``ToolExecution`` rows, for the evidence ledger."""
        return [
            ToolExecution(
                tool_name=call.tool_name,
                tool_call_id=call.span_id or f"call-{index}",
                arguments=dict(call.arguments),
                result=call.output_preview or None,
                error=call.error,
                duration_ms=call.duration_ms,
            )
            for index, call in enumerate(self.tool_calls)
        ]

    @property
    def metrics(self) -> Any:
        return _RunMetrics(total_tokens=self.total_tokens)

    @property
    def duration_ms(self) -> float:
        """From the first event seen to the terminate: a lower bound on the run.

        The bus carries no run-started event, and a run that makes one model
        call yields its events together once the call returns, so a short
        run can read as 0 ms here.
        """
        if self.finished_at is None:
            return 0.0
        return (self.finished_at - self.started_at).total_seconds() * 1000


class _RunMetrics:
    """Token total from the bus; cost is never known there, so it stays ``None``."""

    def __init__(self, *, total_tokens: int) -> None:
        self.total_tokens = total_tokens
        self.cost_usd: float | None = None


def every_nth(n: int) -> Sampler:
    """Grade every ``n``-th finished run; ``1`` grades them all."""
    if n < 1:
        raise ValueError("every_nth needs n >= 1")
    seen = 0

    def sample(_run: RecordedRun) -> bool:
        nonlocal seen
        seen += 1
        return seen % n == 0

    return sample


def fraction(share: float, *, seed: int | None = None) -> Sampler:
    """Grade a random ``share`` of finished runs; ``seed`` pins the choice."""
    if not 0.0 <= share <= 1.0:
        raise ValueError("fraction needs a share between 0 and 1")
    generator = random.Random(seed)  # noqa: S311 — sampling, not security

    def sample(_run: RecordedRun) -> bool:
        return generator.random() < share

    return sample


class OnlineBudget(BaseModel):
    """How much online grading may cost.

    Attributes:
        max_cases: Runs graded over the evaluator's lifetime; further sampled
            runs are counted in ``over_budget`` and dropped.
        max_per_minute: Runs that may start grading in any sliding minute.
        max_concurrency: Gradings in flight at once.
    """

    max_cases: int | None = Field(default=None, ge=0)
    max_per_minute: float | None = Field(default=None, gt=0)
    max_concurrency: int = Field(default=2, ge=1)


class OnlineEvaluator:
    """Grade a sample of live runs in the background with the suite's checks.

    Args:
        template: The expectations every sampled run is graded against. Its
            ``name`` is the report's suite name and its ``prompt`` is unused;
            the case's name is ``<template name>:<run id>`` and its prompt the
            one attached for the run, if any.
        judge: The judge for a template with a rubric, claims or a grounding
            floor; as on :class:`~locus.evaluation.EvalRunner`.
        sampler: Which finished runs to grade. Every run by default.
        budget: Caps on gradings; see :class:`OnlineBudget`.
        bus: The bus to subscribe to; the process-wide one by default.
        clock: Time source for the per-minute budget, for tests.
    """

    def __init__(
        self,
        template: EvalCase,
        *,
        judge: Any = None,
        sampler: Sampler | None = None,
        budget: OnlineBudget | None = None,
        bus: EventBus | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.template = template
        self.budget = budget or OnlineBudget()
        self.sampler: Sampler = sampler or every_nth(1)
        self.runner = EvalRunner(agent=None, judge=judge, concurrency=self.budget.max_concurrency)
        self._bus = bus
        self._clock = clock
        self._open: dict[str, RecordedRun] = {}
        self._attached: dict[str, dict[str, Any]] = {}
        self._results: list[EvalResult] = []
        self._cases: list[EvalCase] = []
        self._tasks: set[asyncio.Task[None]] = set()
        self._semaphore = asyncio.Semaphore(self.budget.max_concurrency)
        self._starts: deque[float] = deque()
        self._subscription: asyncio.Task[None] | None = None
        self.seen = 0
        self.sampled = 0
        self.over_budget = 0
        self.errors: list[str] = []

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to the bus and grade in the background until :meth:`stop`."""
        if self._subscription is not None:
            return
        bus = self._bus or get_event_bus()

        async def follow() -> None:
            async for event in bus.subscribe_global():
                await self.observe(event)

        self._subscription = asyncio.ensure_future(follow())
        # Let the subscriber register before returning: a run that never
        # suspends (a scripted model, say) would otherwise finish before the
        # bus knew anyone was listening.
        await asyncio.sleep(0)

    async def stop(self, *, drain: bool = True) -> None:
        """Unsubscribe; with ``drain`` wait for the gradings in flight."""
        if self._subscription is not None:
            self._subscription.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._subscription
            self._subscription = None
        if drain and self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        else:
            for task in self._tasks:
                task.cancel()

    async def drain(self) -> None:
        """Wait for every grading in flight; the subscription stays open."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # -- feeding -------------------------------------------------------------

    def attach(
        self,
        run_id: str,
        *,
        prompt: str | None = None,
        output: str | None = None,
        arguments: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Give the evaluator what the bus does not carry for ``run_id``.

        ``arguments`` maps a tool call's ``span_id`` (its ``tool_call_id``) to
        the arguments it was called with, for
        :class:`~locus.evaluation.evaluators.ArgumentCorrectness`. Attach
        before the run finishes, or at least before it is graded.
        """
        entry = self._attached.setdefault(run_id, {})
        if prompt is not None:
            entry["prompt"] = prompt
        if output is not None:
            entry["output"] = output
        if arguments:
            entry.setdefault("arguments", {}).update(arguments)
        run = self._open.get(run_id)
        if run is not None:
            self._apply_attached(run)

    async def observe(self, event: StreamEvent) -> None:
        """Feed one event, as the subscription does; a finished run is sampled here."""
        if not event.run_id or not event.event_type.startswith("agent."):
            return
        run = self._open.get(event.run_id)
        if run is None:
            run = RecordedRun(run_id=event.run_id, started_at=event.timestamp)
            self._open[event.run_id] = run
            self._evict()
        run.events += 1
        data = event.data
        if event.event_type == EV_AGENT_THINK:
            iteration = data.get("iteration")
            if isinstance(iteration, int):
                run.iterations = max(run.iterations, iteration)
        elif event.event_type == EV_AGENT_TOOL_STARTED:
            run.tool_calls.append(
                RecordedToolCall(
                    tool_name=str(data.get("tool_name") or ""),
                    span_id=str(data["span_id"]) if data.get("span_id") is not None else None,
                )
            )
        elif event.event_type == EV_AGENT_TOOL_COMPLETED:
            call = self._call_for(run, data)
            call.success = bool(data.get("success")) if data.get("success") is not None else None
            call.error = data.get("error") if isinstance(data.get("error"), str) else None
            call.output_preview = str(data.get("output_preview") or "")
            duration = data.get("duration_ms")
            call.duration_ms = float(duration) if isinstance(duration, int | float) else None
        elif event.event_type == EV_AGENT_TOKENS_USED:
            total = data.get("total_tokens")
            if isinstance(total, int):
                run.total_tokens += total
        elif event.event_type == EV_AGENT_TERMINATE:
            run.reason = data.get("reason") if isinstance(data.get("reason"), str) else None
            used = data.get("iterations_used")
            if isinstance(used, int):
                run.iterations = max(run.iterations, used)
            run.final_message_preview = str(data.get("final_message_preview") or "")
            run.finished_at = event.timestamp
            self._open.pop(event.run_id, None)
            self._apply_attached(run)
            self._attached.pop(event.run_id, None)
            self._consider(run)

    async def submit(self, prompt: str, result: Any, *, run_id: str | None = None) -> bool:
        """Offer a finished run the application holds in full; returns whether it was graded.

        Goes through the same sampler and budget as runs seen on the bus,
        and grades the real ``AgentResult`` rather than previews.
        """
        run = RecordedRun(
            run_id=run_id or f"submitted-{self.seen + 1}",
            prompt=prompt,
            output=str(getattr(result, "message", "") or ""),
            iterations=int(getattr(result, "iterations", 0) or 0),
            finished_at=datetime.now(UTC),
        )
        return self._consider(run, result=result)

    # -- results -------------------------------------------------------------

    @property
    def results(self) -> list[EvalResult]:
        """Every graded run so far, in grading order."""
        return list(self._results)

    @property
    def in_flight(self) -> int:
        return len(self._tasks)

    def report(self) -> EvalReport:
        """The graded runs as a report.

        Its fingerprint is the template's, not the sampled cases': every
        report this evaluator produces measured the same expectations, so
        two of them diff and trend as one suite.
        """
        report = self.runner._report(  # noqa: SLF001 — same package, same aggregation
            list(self._results), len(self._results), cases=self._cases or None
        )
        report.fingerprint = template_fingerprint(self.template)
        return report

    # -- internals -----------------------------------------------------------

    def _apply_attached(self, run: RecordedRun) -> None:
        entry = self._attached.get(run.run_id)
        if not entry:
            return
        if "prompt" in entry:
            run.prompt = entry["prompt"]
        if "output" in entry:
            run.output = entry["output"]
        for call in run.tool_calls:
            if call.span_id in entry.get("arguments", {}):
                call.arguments = dict(entry["arguments"][call.span_id])

    @staticmethod
    def _call_for(run: RecordedRun, data: dict[str, Any]) -> RecordedToolCall:
        span = data.get("span_id")
        for call in reversed(run.tool_calls):
            if span is not None and call.span_id == str(span) and call.success is None:
                return call
        for call in reversed(run.tool_calls):
            if call.success is None and call.tool_name == str(data.get("tool_name") or ""):
                return call
        call = RecordedToolCall(
            tool_name=str(data.get("tool_name") or ""),
            span_id=str(span) if span is not None else None,
        )
        run.tool_calls.append(call)
        return call

    def _evict(self) -> None:
        overflow = len(self._open) - _MAX_OPEN_RUNS
        for run_id in list(self._open)[: max(0, overflow)]:
            del self._open[run_id]

    def _within_budget(self) -> bool:
        if self.budget.max_cases is not None and (
            len(self._results) + len(self._tasks) >= self.budget.max_cases
        ):
            return False
        if self.budget.max_per_minute is not None:
            now = self._clock()
            while self._starts and now - self._starts[0] >= 60:
                self._starts.popleft()
            if len(self._starts) >= self.budget.max_per_minute:
                return False
        return True

    def _consider(self, run: RecordedRun, *, result: Any = None) -> bool:
        self.seen += 1
        if not self.sampler(run):
            return False
        self.sampled += 1
        if not self._within_budget():
            self.over_budget += 1
            return False
        self._starts.append(self._clock())
        case = self.template.model_copy(
            update={"name": f"{self.template.name}:{run.run_id}", "prompt": run.prompt or ""}
        )
        task = asyncio.ensure_future(self._grade(case, run if result is None else result, run))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    async def _grade(self, case: EvalCase, result: Any, run: RecordedRun) -> None:
        async with self._semaphore:
            try:
                graded = await self.runner.agrade(case, result, duration_ms=run.duration_ms)
            except Exception as exc:  # noqa: BLE001 — a background grader must not die quietly
                self.errors.append(f"{case.name}: {exc}")
                logger.warning("online evaluation of %s failed: %s", case.name, exc)
                return
            self._cases.append(case)
            self._results.append(graded)


def template_fingerprint(template: EvalCase) -> str:
    """The fingerprint online reports carry: the template's expectations, no prompt."""
    return suite_fingerprint([template.model_copy(update={"prompt": ""})])
