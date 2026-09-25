# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Grading semantics of ``EvalRunner`` that a report must not misstate.

Each test here pins a way the runner used to report something other than
what happened: a case that checked nothing passing, a dead judge reported as
a bad answer, a failed case scoring 1.0, a hung agent hanging the suite, and
a trajectory failure naming the wrong missing steps.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from tulip.agent import Agent
from tulip.core.messages import Message
from tulip.evaluation import (
    EvalCase,
    EvalRunner,
    JudgeUnavailableError,
    LLMJudge,
    check_trajectory,
)
from tulip.models.base import ModelResponse
from tulip.testing import ScriptedModel, text


class _ReplyModel:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def complete(self, *a: Any, **k: Any) -> ModelResponse:
        return ModelResponse(message=Message.assistant(content=self.reply), usage={})

    async def stream(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


class _DeadModel(_ReplyModel):
    async def complete(self, *a: Any, **k: Any) -> ModelResponse:
        raise ConnectionError("judge is down")


def _agent(reply: str = "an answer") -> Agent:
    return Agent(model=ScriptedModel([text(reply)], repeat_last=True))


# --------------------------------------------------------------------------
# 1. A case that checks nothing must not pass
# --------------------------------------------------------------------------


def test_a_case_with_no_expectations_fails_on_the_sync_path() -> None:
    report = EvalRunner(agent=_agent()).run([EvalCase(name="vacuous", prompt="p")])
    result = report.results[0]

    assert not result.passed
    assert result.score == 0.0
    assert result.checks == {"has_expectations": False}
    assert report.passed == 0
    assert report.failed == 1
    assert "has_expectations: FAILED" in report.summary()


@pytest.mark.asyncio
async def test_a_case_with_no_expectations_fails_on_the_async_path() -> None:
    report = await EvalRunner(agent=_agent(), concurrency=1).arun(
        [EvalCase(name="vacuous", prompt="p")]
    )
    assert not report.results[0].passed
    assert report.results[0].checks == {"has_expectations": False}
    assert report.avg_score == 0.0


def test_a_rubric_is_not_silently_ignored_by_the_sync_path() -> None:
    """``run()`` cannot call a judge; a rubric-only case used to pass there."""
    report = EvalRunner(agent=_agent()).run([EvalCase(name="graded", prompt="p", rubric="r")])
    result = report.results[0]

    assert not result.passed
    assert result.checks == {"rubric:requires_arun": False}


# --------------------------------------------------------------------------
# 2. An unreachable judge stops the suite; it is not a model failure
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unreachable_judge_propagates_out_of_arun() -> None:
    runner = EvalRunner(agent=_agent(), judge=LLMJudge(_DeadModel("")), concurrency=1)
    with pytest.raises(JudgeUnavailableError, match="could not be reached"):
        await runner.arun([EvalCase(name="graded", prompt="p", rubric="r")])


@pytest.mark.asyncio
async def test_an_unreachable_judge_cancels_the_cases_still_running() -> None:
    """Abandoned cases must not keep spending tokens after the suite stopped."""
    cancelled = asyncio.Event()

    class _Hanging:
        async def arun(self, prompt: str, **_: Any) -> Any:
            if prompt == "graded":
                return await _agent().arun(prompt)
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancelled.set()
                raise

    runner = EvalRunner(agent=_Hanging(), judge=LLMJudge(_DeadModel("")), concurrency=2)
    cases = [
        EvalCase(name="slow", prompt="slow", expected_output_contains=["x"]),
        EvalCase(name="graded", prompt="graded", rubric="r"),
    ]
    with pytest.raises(JudgeUnavailableError):
        await asyncio.wait_for(runner.arun(cases), timeout=10)
    assert cancelled.is_set()


def test_judge_unavailable_is_still_a_runtime_error() -> None:
    """Callers catching the documented ``RuntimeError`` keep working."""
    assert issubclass(JudgeUnavailableError, RuntimeError)


# --------------------------------------------------------------------------
# 3. Trajectory failures name the steps that are actually missing
# --------------------------------------------------------------------------


def test_a_repeated_step_reports_the_true_missing_tail() -> None:
    ok, reason = check_trajectory(["a", "b"], ["a", "b", "a"])
    assert not ok
    assert "['a'] did not follow" in reason


def test_the_missing_tail_is_located_by_position_not_by_name() -> None:
    ok, reason = check_trajectory(["x", "y"], ["x", "y", "x", "y"])
    assert not ok
    assert "['x', 'y'] did not follow" in reason


# --------------------------------------------------------------------------
# 4. A failed case never scores 1.0
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_structural_failure_caps_the_judged_score() -> None:
    judge = LLMJudge(_ReplyModel('{"passed": true, "score": 1.0, "reason": "great"}'))
    case = EvalCase(name="c", prompt="p", rubric="r", expected_output_contains=["missing"])

    result = (await EvalRunner(agent=_agent(), judge=judge, concurrency=1).arun([case])).results[0]

    assert not result.passed
    # min(judge 1.0, 1 of 2 checks passed)
    assert result.score == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_a_failing_verdict_caps_its_own_score() -> None:
    """A judge that says "failed" but scores 1.0 must not read as perfect."""
    judge = LLMJudge(_ReplyModel('{"passed": false, "score": 1.0, "reason": "no"}'))
    case = EvalCase(name="c", prompt="p", rubric="r")

    result = (await EvalRunner(agent=_agent(), judge=judge, concurrency=1).arun([case])).results[0]

    assert not result.passed
    assert result.score < 1.0


@pytest.mark.asyncio
async def test_a_lower_judged_score_still_wins_when_everything_passed() -> None:
    judge = LLMJudge(_ReplyModel('{"passed": true, "score": 0.6, "reason": "ok"}'))
    case = EvalCase(name="c", prompt="p", rubric="r", expected_output_contains=["answer"])

    result = (await EvalRunner(agent=_agent(), judge=judge, concurrency=1).arun([case])).results[0]

    assert result.passed
    assert result.score == pytest.approx(0.6)


# --------------------------------------------------------------------------
# 5. max_duration_ms is a real timeout
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_hung_agent_is_cancelled_at_its_budget_on_the_async_path() -> None:
    cancelled = asyncio.Event()

    class _Hung:
        async def arun(self, *a: Any, **k: Any) -> Any:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancelled.set()
                raise

    case = EvalCase(name="hang", prompt="p", expected_output_contains=["x"], max_duration_ms=50)
    started = time.perf_counter()
    report = await asyncio.wait_for(EvalRunner(agent=_Hung(), concurrency=1).arun([case]), 10)
    elapsed = time.perf_counter() - started
    result = report.results[0]

    assert elapsed < 5
    assert cancelled.is_set()
    assert result.timed_out
    assert not result.passed
    assert result.score == 0.0
    assert result.checks["within_duration_budget"] is False
    assert report.timed_out == 1
    assert "TIMEOUT" in report.summary()


def test_a_hung_agent_does_not_hang_the_sync_suite() -> None:
    release = threading.Event()

    class _Hung:
        def run_sync(self, *a: Any, **k: Any) -> Any:
            release.wait(60)
            raise RuntimeError("released")

    cases = [
        EvalCase(name="hang", prompt="p", expected_output_contains=["x"], max_duration_ms=50),
        EvalCase(name="next", prompt="p", expected_output_contains=["x"], max_duration_ms=50),
    ]
    started = time.perf_counter()
    try:
        report = EvalRunner(agent=_Hung()).run(cases)
    finally:
        release.set()

    assert time.perf_counter() - started < 5
    assert [r.timed_out for r in report.results] == [True, True]
    assert report.failed == 2


@pytest.mark.asyncio
async def test_an_agent_raising_timeout_error_is_an_error_not_a_budget_timeout() -> None:
    """An HTTP timeout inside the agent is the agent's error, not our budget."""

    class _Raises:
        async def arun(self, *a: Any, **k: Any) -> Any:
            raise TimeoutError("upstream timed out")

    case = EvalCase(name="c", prompt="p", expected_output_contains=["x"], max_duration_ms=60_000)
    result = (await EvalRunner(agent=_Raises(), concurrency=1).arun([case])).results[0]

    assert not result.timed_out
    assert result.error == "upstream timed out"


def test_a_fast_run_within_budget_is_unaffected() -> None:
    case = EvalCase(
        name="c", prompt="p", expected_output_contains=["answer"], max_duration_ms=60_000
    )
    result = EvalRunner(agent=_agent()).run([case]).results[0]

    assert result.passed
    assert not result.timed_out
    assert result.checks["within_duration_budget"] is True


def test_a_sync_agent_error_under_a_budget_is_still_reported() -> None:
    class _Crash:
        def run_sync(self, *a: Any, **k: Any) -> Any:
            raise ValueError("boom")

    case = EvalCase(name="c", prompt="p", expected_output_contains=["x"], max_duration_ms=60_000)
    result = EvalRunner(agent=_Crash()).run([case]).results[0]

    assert result.error == "boom"
    assert not result.timed_out
