# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LLM-as-judge scoring and trajectory assertions.

``tulip/evaluation`` advertised "LLM-as-judge scoring" in its module
docstring and contained no judge, no model call, and no rubric — a 250-line
harness of boolean checks. These cover the judge that now backs the claim,
and the ordering assertion that ``expected_tools`` could never express.
"""

from __future__ import annotations

from typing import Any

import pytest

from tulip.agent import Agent
from tulip.core.messages import Message
from tulip.evaluation import EvalCase, EvalRunner, LLMJudge, Verdict, check_trajectory
from tulip.models.base import ModelResponse
from tulip.testing import ScriptedModel, text, tool_call
from tulip.tools.decorator import tool


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order."""
    return "delivered damaged, within window"


@tool
def issue_refund(order_id: str) -> str:
    """Refund an order."""
    return "refunded"


class _ReplyModel:
    """A model that returns one fixed reply, for judge parsing tests."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.max_tokens_seen: int | None = None

    async def complete(
        self, messages: list[Message], tools: Any = None, **kwargs: Any
    ) -> ModelResponse:
        self.max_tokens_seen = kwargs.get("max_tokens")
        return ModelResponse(message=Message.assistant(content=self.reply), usage={})

    async def stream(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


class _DeadModel(_ReplyModel):
    async def complete(self, *a: Any, **k: Any) -> ModelResponse:
        raise ConnectionError("judge is down")


# --------------------------------------------------------------------------
# check_trajectory — order, not membership
# --------------------------------------------------------------------------


def test_correct_order_passes() -> None:
    ok, _ = check_trajectory(["lookup", "refund"], ["lookup", "refund"])
    assert ok


def test_reversed_order_fails() -> None:
    """The whole point: expected_tools cannot express this."""
    ok, reason = check_trajectory(["refund", "lookup"], ["lookup", "refund"])
    assert not ok
    assert "lookup" in reason
    assert "refund" in reason


def test_extra_calls_are_tolerated_by_default() -> None:
    ok, _ = check_trajectory(["auth", "lookup", "retry", "refund"], ["lookup", "refund"])
    assert ok


def test_exact_mode_rejects_extra_calls() -> None:
    ok, reason = check_trajectory(["lookup", "retry", "refund"], ["lookup", "refund"], exact=True)
    assert not ok
    assert "exactly" in reason


def test_a_missing_step_fails() -> None:
    ok, _ = check_trajectory(["lookup"], ["lookup", "refund"])
    assert not ok


def test_an_empty_expectation_passes() -> None:
    assert check_trajectory(["anything"], [])[0]


def test_repeated_tools_are_matched_in_sequence() -> None:
    """Two refunds expected means two must actually have happened."""
    assert check_trajectory(["refund", "refund"], ["refund", "refund"])[0]
    assert not check_trajectory(["refund"], ["refund", "refund"])[0]


# --------------------------------------------------------------------------
# LLMJudge — parsing
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_clean_verdict_is_parsed() -> None:
    model = _ReplyModel('{"passed": true, "score": 0.9, "reason": "states the decision"}')
    verdict = await LLMJudge(model).score(prompt="p", output="o", rubric="r")
    assert verdict.passed
    assert verdict.score == pytest.approx(0.9)
    assert verdict.reason == "states the decision"
    assert not verdict.unparseable


@pytest.mark.asyncio
async def test_a_fenced_verdict_is_parsed() -> None:
    """Models wrap JSON in fences often enough that refusing it is churn."""
    model = _ReplyModel('Here is my grade:\n```json\n{"passed": false, "score": 0.1}\n```')
    verdict = await LLMJudge(model).score(prompt="p", output="o", rubric="r")
    assert not verdict.passed
    assert verdict.score == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_a_missing_score_follows_the_verdict() -> None:
    """A judge that says passed must not be recorded as scoring zero."""
    model = _ReplyModel('{"passed": true, "reason": "fine"}')
    verdict = await LLMJudge(model).score(prompt="p", output="o", rubric="r")
    assert verdict.passed
    assert verdict.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_an_out_of_range_score_is_clamped() -> None:
    model = _ReplyModel('{"passed": true, "score": 7}')
    assert (await LLMJudge(model).score(prompt="p", output="o", rubric="r")).score == 1.0


@pytest.mark.asyncio
async def test_an_empty_reply_names_the_likely_cause() -> None:
    """A reasoning model out of budget returns "" — not an error.

    Reported as a content failure it looks like the rubric's fault, which
    sends the reader to debug the wrong thing.
    """
    verdict = await LLMJudge(_ReplyModel("")).score(prompt="p", output="o", rubric="r")
    assert verdict.unparseable
    assert not verdict.passed
    assert "max_tokens" in verdict.reason


@pytest.mark.asyncio
async def test_a_non_json_reply_fails_closed() -> None:
    verdict = await LLMJudge(_ReplyModel("Looks good to me!")).score(
        prompt="p", output="o", rubric="r"
    )
    assert verdict.unparseable
    assert not verdict.passed


@pytest.mark.asyncio
async def test_malformed_json_fails_closed() -> None:
    verdict = await LLMJudge(_ReplyModel('{"passed": tru')).score(
        prompt="p", output="o", rubric="r"
    )
    assert verdict.unparseable


@pytest.mark.asyncio
async def test_an_unreachable_judge_raises_rather_than_scoring_zero() -> None:
    """A "failure" that means the judge was down is worse than no eval."""
    with pytest.raises(RuntimeError, match="could not be reached"):
        await LLMJudge(_DeadModel("")).score(prompt="p", output="o", rubric="r")


@pytest.mark.asyncio
async def test_the_token_budget_is_generous_by_default() -> None:
    """Measured: a reasoning judge returns empty below ~1024."""
    model = _ReplyModel('{"passed": true}')
    await LLMJudge(model).score(prompt="p", output="o", rubric="r")
    assert model.max_tokens_seen is not None
    assert model.max_tokens_seen >= 2048


# --------------------------------------------------------------------------
# EvalRunner integration
# --------------------------------------------------------------------------


def _agent(turns: list[Any]) -> Agent:
    return Agent(model=ScriptedModel(turns, repeat_last=True), tools=[lookup_order, issue_refund])


@pytest.mark.asyncio
async def test_trajectory_is_checked_by_the_runner() -> None:
    agent = _agent(
        [
            tool_call("issue_refund", order_id="o1"),
            tool_call("lookup_order", order_id="o1"),
            text("Done."),
        ]
    )
    case = EvalCase(
        name="wrong_order",
        prompt="refund o1",
        expected_tool_sequence=["lookup_order", "issue_refund"],
    )
    report = await EvalRunner(agent=agent, concurrency=1).arun([case])
    assert report.results[0].checks["tool_sequence"] is False


@pytest.mark.asyncio
async def test_a_rubric_without_a_judge_fails_rather_than_passing() -> None:
    """A suite must not look green because nobody supplied the judge."""
    agent = _agent([text("anything")])
    case = EvalCase(name="graded", prompt="p", rubric="must be excellent")
    report = await EvalRunner(agent=agent, concurrency=1).arun([case])

    assert not report.results[0].passed
    assert any("no_judge_configured" in k for k in report.results[0].checks)


@pytest.mark.asyncio
async def test_the_judge_score_becomes_the_case_score() -> None:
    """Graded cases report the judge's score, not the fraction of boxes."""
    agent = _agent([text("a considered answer")])
    judge = LLMJudge(_ReplyModel('{"passed": true, "score": 0.75, "reason": "ok"}'))
    case = EvalCase(name="graded", prompt="p", rubric="r")

    report = await EvalRunner(agent=agent, judge=judge, concurrency=1).arun([case])
    assert report.results[0].score == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_an_unparseable_verdict_is_labelled_distinctly() -> None:
    """A judge that never answered is a different finding from a bad answer."""
    agent = _agent([text("answer")])
    judge = LLMJudge(_ReplyModel(""))
    report = await EvalRunner(agent=agent, judge=judge, concurrency=1).arun(
        [EvalCase(name="graded", prompt="p", rubric="r")]
    )
    assert any(k.startswith("judge_unparseable") for k in report.results[0].checks)


@pytest.mark.asyncio
async def test_arun_reports_every_case() -> None:
    agents = [_agent([text("one")]), _agent([text("two")])]
    reports = [
        await EvalRunner(agent=a, concurrency=1).arun([EvalCase(name=f"c{i}", prompt="p")])
        for i, a in enumerate(agents)
    ]
    assert [r.total_cases for r in reports] == [1, 1]
    assert all(r.passed == 1 for r in reports)


def test_verdict_defaults_are_safe() -> None:
    """An unset verdict must not read as a pass."""
    assert Verdict(passed=False).score == 0.0
    assert Verdict(passed=False).unparseable is False


# --------------------------------------------------------------------------
# The async path must apply every structural check the sync path does.
# They share _structural_checks precisely so the two cannot grade the same
# case differently; these pin that down.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arun_applies_every_structural_check() -> None:
    agent = _agent([tool_call("lookup_order", order_id="o1"), text("The order is refundable.")])
    case = EvalCase(
        name="all_checks",
        prompt="is o1 refundable?",
        expected_tools=["lookup_order"],
        expected_output_contains=["refundable"],
        expected_output_not_contains=["error"],
        max_iterations=10,
        max_duration_ms=60_000,
    )
    result = (await EvalRunner(agent=agent, concurrency=1).arun([case])).results[0]

    assert result.checks["tool_called:lookup_order"] is True
    assert result.checks["output_contains:refundable"] is True
    assert result.checks["output_not_contains:error"] is True
    assert result.checks["within_iteration_budget"] is True
    assert result.checks["within_duration_budget"] is True
    assert result.passed


@pytest.mark.asyncio
async def test_arun_reports_a_failing_structural_check() -> None:
    agent = _agent([text("nothing useful")])
    case = EvalCase(
        name="misses",
        prompt="p",
        expected_tools=["lookup_order"],
        expected_output_contains=["refundable"],
        max_iterations=0,
    )
    result = (await EvalRunner(agent=agent, concurrency=1).arun([case])).results[0]

    assert not result.passed
    assert result.checks["tool_called:lookup_order"] is False
    assert result.checks["output_contains:refundable"] is False
    assert result.checks["within_iteration_budget"] is False


@pytest.mark.asyncio
async def test_a_crashing_agent_becomes_a_failed_case_not_a_crashed_run() -> None:
    """One broken case must not take the rest of the suite with it."""

    class _Exploding:
        async def arun(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("agent exploded")

    report = await EvalRunner(agent=_Exploding(), concurrency=1).arun(
        [EvalCase(name="boom", prompt="p")]
    )
    result = report.results[0]

    assert not result.passed
    assert result.score == 0.0
    assert result.error is not None
    assert "exploded" in result.error
    assert report.failed == 1


@pytest.mark.asyncio
async def test_a_failing_case_appears_in_the_summary() -> None:
    class _Exploding:
        async def arun(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("agent exploded")

    report = await EvalRunner(agent=_Exploding(), concurrency=1).arun(
        [EvalCase(name="boom", prompt="p")]
    )
    assert "exploded" in report.summary()


def test_the_sync_path_checks_the_duration_budget() -> None:
    """An impossible budget must fail rather than being ignored."""
    agent = _agent([text("done")])
    report = EvalRunner(agent=agent).run([EvalCase(name="slow", prompt="p", max_duration_ms=0.0)])
    assert report.results[0].checks["within_duration_budget"] is False
