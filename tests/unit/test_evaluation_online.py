# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Online evaluation: live runs sampled off the event bus and graded in the background.

These tests pin how a run is rebuilt from its ``agent.*`` events (tool calls
paired by span, tokens summed, the terminate closing it), what the sampler and
the budget let through, what the application can attach that the bus does not
carry, that the report's fingerprint is the template's so online reports trend
as one suite, and that a grading failure is recorded rather than lost.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from locus.agent import Agent
from locus.evaluation import (
    ArgumentCorrectness,
    EvalCase,
    LLMJudge,
    OnlineBudget,
    OnlineEvaluator,
    ToolSelection,
    TrajectoryMatch,
    every_nth,
    fraction,
)
from locus.evaluation.online import RecordedRun, template_fingerprint
from locus.models.base import ModelResponse
from locus.observability import run_context
from locus.observability.emit import (
    EV_AGENT_TERMINATE,
    EV_AGENT_THINK,
    EV_AGENT_TOKENS_USED,
    EV_AGENT_TOOL_COMPLETED,
    EV_AGENT_TOOL_STARTED,
)
from locus.observability.event_bus import StreamEvent, reset_event_bus
from locus.testing import ScriptedModel, text, tool_call
from locus.tools.decorator import tool


@pytest.fixture(autouse=True)
def _fresh_bus() -> Any:
    reset_event_bus()
    yield
    reset_event_bus()


def _event(run_id: str, kind: str, at: datetime | None = None, **data: Any) -> StreamEvent:
    event = StreamEvent(run_id=run_id, event_type=kind, data=data)
    if at is not None:
        event.timestamp = at
    return event


def _run_events(run_id: str, *, tools: list[str], at: datetime) -> list[StreamEvent]:
    events = [_event(run_id, EV_AGENT_THINK, at, iteration=1, has_tool_calls=bool(tools))]
    for index, name in enumerate(tools):
        span = f"{run_id}-{index}"
        events.append(_event(run_id, EV_AGENT_TOOL_STARTED, at, tool_name=name, span_id=span))
        events.append(
            _event(
                run_id,
                EV_AGENT_TOOL_COMPLETED,
                at,
                tool_name=name,
                span_id=span,
                success=True,
                duration_ms=3.0,
                output_preview=f"{name} result: amount=49.90",
                error=None,
            )
        )
    events.append(_event(run_id, EV_AGENT_TOKENS_USED, at, total_tokens=40))
    events.append(
        _event(
            run_id,
            EV_AGENT_TERMINATE,
            at + timedelta(milliseconds=250),
            reason="complete",
            iterations_used=2,
            final_message_preview="Refunded.",
        )
    )
    return events


TEMPLATE = EvalCase(
    name="live",
    prompt="",
    evaluators=[ToolSelection(expected=["lookup_order"], forbidden=["delete_order"])],
    max_iterations=5,
)


async def _feed(evaluator: OnlineEvaluator, events: list[StreamEvent]) -> None:
    for event in events:
        await evaluator.observe(event)
    await evaluator.drain()


# --------------------------------------------------------------------------
# Rebuilding a run from its events
# --------------------------------------------------------------------------


async def test_a_run_is_rebuilt_from_its_events_and_graded_on_terminate() -> None:
    evaluator = OnlineEvaluator(TEMPLATE)
    at = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)
    await _feed(evaluator, _run_events("r1", tools=["lookup_order"], at=at))
    assert evaluator.seen == 1
    assert evaluator.sampled == 1
    result = evaluator.results[0]
    assert result.case_name == "live:r1"
    assert result.passed, result.checks
    assert result.tools_called == ["lookup_order"]
    assert result.iterations == 2
    assert result.duration_ms == pytest.approx(250.0)
    assert result.output == "Refunded."


async def test_a_forbidden_tool_on_the_bus_fails_the_sampled_run() -> None:
    evaluator = OnlineEvaluator(TEMPLATE)
    at = datetime.now(UTC)
    await _feed(evaluator, _run_events("r2", tools=["lookup_order", "delete_order"], at=at))
    result = evaluator.results[0]
    assert not result.passed
    assert result.checks["tool_selection"] is False
    assert result.evaluations[0].details["forbidden_selected"] == ["delete_order"]


async def test_events_outside_the_agent_namespace_and_unfinished_runs_are_not_graded() -> None:
    evaluator = OnlineEvaluator(TEMPLATE)
    at = datetime.now(UTC)
    await evaluator.observe(_event("r3", "composition.stage.started", at))
    await evaluator.observe(_event("r3", EV_AGENT_THINK, at, iteration=1))
    await evaluator.drain()
    assert evaluator.seen == 0
    assert evaluator.results == []


async def test_a_completed_event_without_a_start_still_counts_the_call() -> None:
    evaluator = OnlineEvaluator(TEMPLATE)
    at = datetime.now(UTC)
    events = [
        _event("r4", EV_AGENT_TOOL_COMPLETED, at, tool_name="lookup_order", success=True),
        _event("r4", EV_AGENT_TERMINATE, at, reason="complete", iterations_used=1),
    ]
    await _feed(evaluator, events)
    assert evaluator.results[0].tools_called == ["lookup_order"]


def test_the_recorded_run_exposes_what_the_runner_reads() -> None:
    run = RecordedRun(run_id="x", final_message_preview="preview", total_tokens=12)
    assert run.message == "preview"
    assert run.metrics.total_tokens == 12
    assert run.metrics.cost_usd is None
    assert run.duration_ms == 0.0
    run.output = "the full answer"
    assert run.message == "the full answer"


# --------------------------------------------------------------------------
# Sampler and budget
# --------------------------------------------------------------------------


async def test_every_nth_grades_one_run_in_n() -> None:
    evaluator = OnlineEvaluator(TEMPLATE, sampler=every_nth(3))
    at = datetime.now(UTC)
    for index in range(6):
        await _feed(evaluator, _run_events(f"s{index}", tools=["lookup_order"], at=at))
    assert evaluator.seen == 6
    assert evaluator.sampled == 2
    assert [r.case_name for r in evaluator.results] == ["live:s2", "live:s5"]


def test_samplers_reject_bad_settings() -> None:
    with pytest.raises(ValueError, match="n >= 1"):
        every_nth(0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        fraction(1.5)


async def test_a_seeded_fraction_is_reproducible() -> None:
    at = datetime.now(UTC)
    picks = []
    for _ in range(2):
        evaluator = OnlineEvaluator(TEMPLATE, sampler=fraction(0.5, seed=11))
        for index in range(10):
            await _feed(evaluator, _run_events(f"f{index}", tools=["lookup_order"], at=at))
        picks.append([r.case_name for r in evaluator.results])
    assert picks[0] == picks[1]
    assert 0 < len(picks[0]) < 10


async def test_max_cases_caps_the_gradings_and_counts_the_rest() -> None:
    evaluator = OnlineEvaluator(TEMPLATE, budget=OnlineBudget(max_cases=2))
    at = datetime.now(UTC)
    for index in range(4):
        await _feed(evaluator, _run_events(f"b{index}", tools=["lookup_order"], at=at))
    assert len(evaluator.results) == 2
    assert evaluator.sampled == 4
    assert evaluator.over_budget == 2


async def test_max_per_minute_is_a_sliding_window() -> None:
    now = [1000.0]
    evaluator = OnlineEvaluator(
        TEMPLATE, budget=OnlineBudget(max_per_minute=2), clock=lambda: now[0]
    )
    at = datetime.now(UTC)
    for index in range(3):
        await _feed(evaluator, _run_events(f"m{index}", tools=["lookup_order"], at=at))
    assert len(evaluator.results) == 2
    assert evaluator.over_budget == 1
    now[0] += 61
    await _feed(evaluator, _run_events("m3", tools=["lookup_order"], at=at))
    assert len(evaluator.results) == 3


# --------------------------------------------------------------------------
# What the application attaches
# --------------------------------------------------------------------------


async def test_attached_prompt_output_and_arguments_reach_the_grading() -> None:
    template = EvalCase(
        name="live",
        prompt="",
        expected_output_contains=["49.90"],
        require_receipts=True,
        evaluators=[ArgumentCorrectness(tool="issue_refund", expected={"amount": 49.9})],
    )
    evaluator = OnlineEvaluator(template)
    at = datetime.now(UTC)
    events = _run_events("a1", tools=["issue_refund"], at=at)
    for event in events[:-1]:
        await evaluator.observe(event)
    evaluator.attach(
        "a1",
        prompt="Refund ord-4821",
        output="Refunded 49.90 on ord-4821.",
        arguments={"a1-0": {"order_id": "ord-4821", "amount": 49.9}},
    )
    await evaluator.observe(events[-1])
    await evaluator.drain()
    result = evaluator.results[0]
    assert result.passed, result.checks
    assert result.output == "Refunded 49.90 on ord-4821."
    assert result.receipts is not None
    assert result.receipts.passed
    assert result.evaluations[0].passed


async def test_attaching_before_the_first_event_is_kept_until_the_run_finishes() -> None:
    evaluator = OnlineEvaluator(EvalCase(name="live", prompt="", expected_output_contains=["full"]))
    evaluator.attach("early", output="the full answer")
    await _feed(evaluator, _run_events("early", tools=[], at=datetime.now(UTC)))
    assert evaluator.results[0].passed


async def test_submit_grades_a_result_the_application_holds() -> None:
    @tool
    def lookup_order(order_id: str) -> str:
        """Look up."""
        return "found"

    agent = Agent(
        model=ScriptedModel([tool_call("lookup_order", order_id="o1"), text("Found it.")]),
        tools=[lookup_order],
    )
    result = await agent.arun("Where is o1?")
    evaluator = OnlineEvaluator(TEMPLATE, sampler=every_nth(1))
    assert await evaluator.submit("Where is o1?", result, run_id="held")
    await evaluator.drain()
    graded = evaluator.results[0]
    assert graded.case_name == "live:held"
    assert graded.passed
    assert graded.output == "Found it."


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


async def test_the_report_carries_the_template_fingerprint_whatever_was_sampled() -> None:
    at = datetime.now(UTC)
    first = OnlineEvaluator(TEMPLATE)
    await _feed(first, _run_events("p1", tools=["lookup_order"], at=at))
    second = OnlineEvaluator(TEMPLATE)
    await _feed(second, _run_events("q1", tools=["lookup_order"], at=at))
    await _feed(second, _run_events("q2", tools=["delete_order"], at=at))
    assert (
        first.report().fingerprint == second.report().fingerprint == template_fingerprint(TEMPLATE)
    )
    other = EvalCase(
        name="live", prompt="", evaluators=[TrajectoryMatch(expected=["lookup_order"])]
    )
    assert template_fingerprint(other) != template_fingerprint(TEMPLATE)
    report = second.report()
    assert report.total_cases == 2
    assert report.passed == 1
    assert report.failed == 1
    assert "live:q2" in report.summary()


async def test_an_empty_report_is_still_a_report_of_the_template() -> None:
    report = OnlineEvaluator(TEMPLATE).report()
    assert report.total_cases == 0
    assert report.fingerprint == template_fingerprint(TEMPLATE)


# --------------------------------------------------------------------------
# On the bus
# --------------------------------------------------------------------------


async def test_runs_under_a_run_context_are_sampled_from_the_bus() -> None:
    @tool
    def lookup_order(order_id: str) -> str:
        """Look up."""
        return "found"

    evaluator = OnlineEvaluator(TEMPLATE)
    await evaluator.start()
    agent = Agent(
        model=ScriptedModel(
            [tool_call("lookup_order", order_id="o1"), text("Found it.")], repeat_last=True
        ),
        tools=[lookup_order],
    )
    for _ in range(2):
        agent = Agent(
            model=ScriptedModel([tool_call("lookup_order", order_id="o1"), text("Found it.")]),
            tools=[lookup_order],
        )
        async with run_context() as run_id:
            result = await agent.arun("Where is o1?")
            evaluator.attach(run_id, prompt="Where is o1?", output=result.message)
    await asyncio.sleep(0.05)
    await evaluator.stop()
    assert evaluator.seen == 2
    assert all(r.passed for r in evaluator.results), [r.checks for r in evaluator.results]
    assert all(r.tools_called == ["lookup_order"] for r in evaluator.results)
    assert evaluator.results[0].output == "Found it."
    assert evaluator.in_flight == 0


async def test_start_twice_is_one_subscription_and_stop_without_drain_cancels() -> None:
    evaluator = OnlineEvaluator(TEMPLATE)
    await evaluator.start()
    first = evaluator._subscription
    await evaluator.start()
    assert evaluator._subscription is first
    await evaluator.stop(drain=False)
    assert evaluator._subscription is None


# --------------------------------------------------------------------------
# A grading that fails
# --------------------------------------------------------------------------


async def test_a_judge_that_cannot_be_reached_is_recorded_not_raised_from_the_background() -> None:
    class _Dead:
        async def complete(self, *a: Any, **k: Any) -> ModelResponse:
            raise ConnectionError("judge down")

        async def stream(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
            raise NotImplementedError

    template = EvalCase(name="live", prompt="", rubric="Passes if polite.")
    evaluator = OnlineEvaluator(template, judge=LLMJudge(_Dead()))
    await _feed(evaluator, _run_events("j1", tools=[], at=datetime.now(UTC)))
    assert evaluator.results == []
    assert len(evaluator.errors) == 1
    assert "live:j1" in evaluator.errors[0]
    assert "judge down" in evaluator.errors[0]
