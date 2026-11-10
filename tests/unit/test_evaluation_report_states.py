# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""What the report says when the judge abstains, grades, or is the model under test.

Three states the report used to flatten into pass/fail: a judge that
answered with no verdict (not graded, not failed, and never part of the
average), a rubric with letters (the grade lands on the result and the
distribution), and a judge that is the model under test (stamped and warned,
not failed).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from locus.agent import Agent
from locus.core.messages import Message, ToolCall
from locus.evaluation import Dimension, EvalCase, EvalRunner, LLMJudge, Rubric
from locus.evaluation.judge import SUBMIT_TOOL
from locus.models.base import ModelResponse
from locus.testing import ScriptedModel, text


RUBRIC = Rubric(
    dimensions=[
        Dimension(key="root_cause", weight=0.5),
        Dimension(key="evidence", weight=0.3),
        Dimension(key="actionable", weight=0.2),
    ],
    grade_thresholds=Rubric.default_grades(),
)

GOOD = {
    "dimensions": [
        {"key": "root_cause", "score": 80, "reason": "named"},
        {"key": "evidence", "score": 100, "reason": "cited"},
        {"key": "actionable", "score": 60, "reason": "vague"},
    ],
    "issues": [],
    "passed": True,
}


class _JudgeModel:
    """Submits ``GOOD`` unless the prompt asks it to abstain, in which case it says nothing."""

    def __init__(self, model_id: str | None = "judge-model") -> None:
        self.config = SimpleNamespace(model=model_id) if model_id else None

    async def complete(self, messages: list[Message], tools: Any = None, **_: Any) -> ModelResponse:
        prompt = messages[-1].content or ""
        if "abstain" in prompt:
            return ModelResponse(message=Message.assistant(content=""), usage={})
        return ModelResponse(
            message=Message.assistant(
                tool_calls=[ToolCall(id="call_1", name=SUBMIT_TOOL, arguments=dict(GOOD))]
            ),
            usage={},
        )

    async def stream(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


def _agent(reply: str = "an answer") -> Agent:
    return Agent(model=ScriptedModel([text(reply)], repeat_last=True))


class _FakeAgent:
    """An agent whose model carries a configured id, for the self-graded stamp."""

    def __init__(self, model_id: str) -> None:
        self.model = SimpleNamespace(config=SimpleNamespace(model=model_id))

    async def arun(self, prompt: str, **_: Any) -> Any:
        return SimpleNamespace(message="an answer", iterations=1, tool_executions=())


# --------------------------------------------------------------------------
# not graded
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_abstained_judge_leaves_the_case_not_graded() -> None:
    runner = EvalRunner(agent=_agent(), judge=LLMJudge(_JudgeModel()), concurrency=1)
    report = await runner.arun([EvalCase(name="c", prompt="please abstain", rubric=RUBRIC)])
    result = report.results[0]

    assert result.not_graded_reason is not None
    assert "max_tokens" in result.not_graded_reason
    assert not result.passed
    assert result.score == 0.0
    assert result.grade is None
    assert any(k.startswith("judge_unparseable") for k in result.checks)

    assert report.total_cases == 1
    assert report.not_graded == 1
    assert report.passed == 0
    assert report.failed == 0
    assert report.avg_score == 0.0
    summary = report.summary()
    assert "[NOT GRADED] c" in summary
    assert "1 not graded" in summary
    assert "not graded: " in summary


@pytest.mark.asyncio
async def test_not_graded_cases_do_not_lower_the_average() -> None:
    runner = EvalRunner(agent=_agent(), judge=LLMJudge(_JudgeModel()), concurrency=1)
    report = await runner.arun(
        [
            EvalCase(name="graded", prompt="grade this", rubric=RUBRIC),
            EvalCase(name="silent", prompt="please abstain", rubric=RUBRIC),
        ]
    )

    assert report.passed == 1
    assert report.failed == 0
    assert report.not_graded == 1
    assert report.avg_score == pytest.approx(0.82)


# --------------------------------------------------------------------------
# grades
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_rubric_grade_lands_on_the_result_and_the_distribution() -> None:
    runner = EvalRunner(agent=_agent(), judge=LLMJudge(_JudgeModel()), concurrency=1)
    report = await runner.arun([EvalCase(name="c", prompt="grade this", rubric=RUBRIC)])
    result = report.results[0]

    assert result.passed
    assert result.score == pytest.approx(0.82)
    assert result.grade == "B"
    assert result.verdict is not None
    assert result.verdict.dimensions["evidence"] == 100
    assert report.grade_distribution == {"B": 1}
    assert "grade B" in report.summary()


@pytest.mark.asyncio
async def test_a_failing_structural_check_still_caps_a_rubric_score() -> None:
    runner = EvalRunner(agent=_agent(), judge=LLMJudge(_JudgeModel()), concurrency=1)
    case = EvalCase(
        name="c", prompt="grade this", rubric=RUBRIC, expected_output_contains=["missing"]
    )
    result = (await runner.arun([case])).results[0]

    assert not result.passed
    assert result.score == pytest.approx(0.5)  # min(0.82, 1 of 2 checks)
    assert result.grade == "B"


def test_the_sync_path_still_refuses_a_rubric_object() -> None:
    report = EvalRunner(agent=_agent()).run([EvalCase(name="c", prompt="p", rubric=RUBRIC)])
    assert report.results[0].checks == {"rubric:requires_arun": False}


@pytest.mark.asyncio
async def test_a_report_with_verdicts_serialises_to_json() -> None:
    runner = EvalRunner(agent=_agent(), judge=LLMJudge(_JudgeModel()), concurrency=1)
    report = await runner.arun([EvalCase(name="c", prompt="grade this", rubric=RUBRIC)])
    dumped = report.model_dump_json()
    assert '"judge_model":"judge-model"' in dumped
    assert '"grade_distribution":{"B":1}' in dumped


# --------------------------------------------------------------------------
# self-graded
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_self_graded_is_stamped_and_warned_not_failed() -> None:
    runner = EvalRunner(
        agent=_FakeAgent("same-model"),
        judge=LLMJudge(_JudgeModel("same-model")),
        concurrency=1,
    )
    report = await runner.arun([EvalCase(name="c", prompt="grade this", rubric=RUBRIC)])
    result = report.results[0]

    assert result.passed
    assert result.verdict is not None
    assert result.verdict.provenance is not None
    assert result.verdict.provenance.self_graded is True
    assert report.self_graded
    assert "self-graded" in report.summary()


@pytest.mark.asyncio
async def test_self_graded_is_unknown_when_an_id_is_missing() -> None:
    runner = EvalRunner(agent=_agent(), judge=LLMJudge(_JudgeModel()), concurrency=1)
    report = await runner.arun([EvalCase(name="c", prompt="grade this", rubric=RUBRIC)])
    provenance = report.results[0].verdict.provenance  # type: ignore[union-attr]

    assert provenance.self_graded is None
    assert not report.self_graded
    assert "self-graded" not in report.summary()


@pytest.mark.asyncio
async def test_different_models_are_not_self_graded() -> None:
    runner = EvalRunner(
        agent=_FakeAgent("agent-model"),
        judge=LLMJudge(_JudgeModel("judge-model")),
        concurrency=1,
    )
    report = await runner.arun([EvalCase(name="c", prompt="grade this", rubric=RUBRIC)])
    provenance = report.results[0].verdict.provenance  # type: ignore[union-attr]
    assert provenance.self_graded is False


# --------------------------------------------------------------------------
# a judge that is not LLMJudge
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_duck_typed_judge_without_provenance_still_grades() -> None:
    """Any object with ``score()`` is a judge; provenance is optional on it."""
    from locus.evaluation import Verdict

    class _FixedJudge:
        async def score(self, *, prompt: str, output: str, rubric: Any) -> Verdict:
            return Verdict(passed=True, score=0.7, reason="fixed")

    runner = EvalRunner(agent=_agent(), judge=_FixedJudge(), concurrency=1)
    report = await runner.arun([EvalCase(name="c", prompt="p", rubric="r")])
    result = report.results[0]

    assert result.passed
    assert result.score == pytest.approx(0.7)
    assert result.verdict is not None
    assert result.verdict.provenance is None
    assert not report.self_graded
