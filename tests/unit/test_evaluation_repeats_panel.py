# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Repeats, judge panels, second opinions and pairwise comparison in the runner.

A suite that runs each case once reports the judge's variance as the
agent's. These tests pin the runner reading a case several times (pass^k,
the pessimistic reading shown, consistency flagged when the readings
disagree), grading an answer with a panel (agreement, and a trust decision
between the first two judges), and comparing two answers in both orders so
a position preference reads as a tie.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from locus.agent import Agent
from locus.core.messages import Message, ToolCall
from locus.evaluation import (
    Dimension,
    EvalCase,
    EvalRunner,
    JudgeUnavailableError,
    LLMJudge,
    Rubric,
)
from locus.evaluation.judge import SUBMIT_COMPARISON_TOOL, SUBMIT_TOOL
from locus.models.base import ModelResponse
from locus.testing import ScriptedModel, text


RUBRIC = Rubric(
    dimensions=[Dimension(key="root_cause", weight=0.5), Dimension(key="evidence", weight=0.5)],
    grade_thresholds=Rubric.default_grades(),
)


def _verdict_args(score: int) -> dict[str, Any]:
    return {
        "dimensions": [
            {"key": "root_cause", "score": score, "reason": "r"},
            {"key": "evidence", "score": score, "reason": "e"},
        ],
        "issues": [],
        "passed": score >= 60,
    }


class _JudgeModel:
    """Returns the scripted verdict scores in order; an empty reply where the script says None."""

    def __init__(self, scores: list[int | None], model_id: str = "judge") -> None:
        self.scores = list(scores)
        self.config = SimpleNamespace(model=model_id)
        self.calls = 0

    async def complete(self, messages: list[Message], tools: Any = None, **_: Any) -> ModelResponse:
        score = self.scores[min(self.calls, len(self.scores) - 1)]
        self.calls += 1
        if score is None:
            return ModelResponse(message=Message.assistant(content=""), usage={})
        call = ToolCall(id="v", name=SUBMIT_TOOL, arguments=_verdict_args(score))
        return ModelResponse(message=Message.assistant(tool_calls=[call]), usage={})

    async def stream(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


def _agent(*replies: str, repeat_last: bool = True) -> Agent:
    return Agent(model=ScriptedModel([text(r) for r in replies], repeat_last=repeat_last))


# --------------------------------------------------------------------------
# repeats
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeats_are_pass_k_and_show_the_pessimistic_reading() -> None:
    agent = _agent("yes", "no", "yes", repeat_last=False)
    case = EvalCase(name="flaky", prompt="p", expected_output_contains=["yes"])
    report = await EvalRunner(agent=agent, concurrency=1).arun([case], repeats=3)
    result = report.results[0]

    assert not result.passed  # one reading failed
    assert result.score == 0.0  # the failing reading is the one shown
    assert result.pass_rate == pytest.approx(2 / 3)
    assert [s.passed for s in result.samples] == [True, False, True]
    assert result.consistency is not None
    assert result.consistency.samples == 3
    assert result.consistency.flagged_unstable  # scores 1, 0, 1 spread far past the gate
    assert report.repeats == 3
    assert report.unstable == 1
    assert report.failed == 1
    summary = report.summary()
    assert "3 readings per case, 1 unstable" in summary
    assert "pass 2/3, unstable" in summary


@pytest.mark.asyncio
async def test_consistent_readings_pass_and_are_not_flagged() -> None:
    case = EvalCase(name="steady", prompt="p", expected_output_contains=["yes"])
    runner = EvalRunner(agent=_agent("yes"), concurrency=1, repeats=3)
    result = (await runner.arun([case])).results[0]

    assert result.passed
    assert result.pass_rate == 1.0
    assert len(result.samples) == 3
    assert result.consistency is not None
    assert not result.consistency.flagged_unstable
    assert result.duration_ms >= sum(s.duration_ms for s in result.samples) * 0.99


@pytest.mark.asyncio
async def test_a_single_reading_has_no_samples() -> None:
    case = EvalCase(name="once", prompt="p", expected_output_contains=["yes"])
    result = (await EvalRunner(agent=_agent("yes"), concurrency=1).arun([case])).results[0]
    assert result.samples == []
    assert result.pass_rate is None
    assert result.consistency is None


@pytest.mark.asyncio
async def test_a_reading_the_judge_abstained_on_does_not_count_against_the_case() -> None:
    judge = LLMJudge(_JudgeModel([80, None, 80]))
    case = EvalCase(name="c", prompt="p", rubric=RUBRIC)
    runner = EvalRunner(agent=_agent("answer"), judge=judge, concurrency=1)
    result = (await runner.arun([case], repeats=3)).results[0]

    assert result.passed
    assert result.pass_rate == 1.0
    assert result.not_graded_reason is None
    assert result.consistency is not None
    assert result.consistency.samples == 2
    assert result.consistency.not_graded == 1
    assert [s.not_graded_reason is None for s in result.samples] == [True, False, True]


@pytest.mark.asyncio
async def test_no_graded_reading_leaves_the_case_not_graded() -> None:
    judge = LLMJudge(_JudgeModel([None]))
    case = EvalCase(name="c", prompt="p", rubric=RUBRIC)
    runner = EvalRunner(agent=_agent("answer"), judge=judge, concurrency=1)
    report = await runner.arun([case], repeats=2)

    assert report.results[0].not_graded_reason is not None
    assert report.results[0].not_graded_reason.startswith("no graded reading in 2")
    assert report.not_graded == 1


@pytest.mark.asyncio
async def test_the_judge_moving_a_letter_is_flagged_on_the_judge() -> None:
    judge = LLMJudge(_JudgeModel([41, 69, 76]))
    case = EvalCase(name="c", prompt="p", rubric=RUBRIC)
    runner = EvalRunner(agent=_agent("the same answer"), judge=judge, concurrency=1)
    result = (await runner.arun([case], repeats=3)).results[0]

    assert result.consistency is not None
    assert result.consistency.grade_distribution == {"F": 1, "C": 1, "B": 1}
    assert result.consistency.flagged_unstable
    assert result.grade == "F"  # the pessimistic reading
    assert not result.passed


# --------------------------------------------------------------------------
# panels and second opinions
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_panel_records_agreement_and_a_trust_decision() -> None:
    primary = LLMJudge(_JudgeModel([82], model_id="judge-1"))
    second = LLMJudge(_JudgeModel([80], model_id="judge-2"))
    case = EvalCase(name="c", prompt="p", rubric=RUBRIC)
    runner = EvalRunner(agent=_agent("answer"), judges=[primary, second], concurrency=1)
    result = (await runner.arun([case])).results[0]

    assert result.grade == "B"
    assert len(result.panel_verdicts) == 2
    assert result.panel is not None
    assert result.panel.judges == 2
    assert result.panel.consensus_grade == "B"
    assert result.trust is not None
    assert result.trust.decision == "trusted"
    assert "trusted" in runner._report([result], 1).summary()


@pytest.mark.asyncio
async def test_a_conflicting_second_opinion_is_rejected_but_the_primary_still_scores() -> None:
    primary = LLMJudge(_JudgeModel([82]))
    second = LLMJudge(_JudgeModel([41]))
    case = EvalCase(name="c", prompt="p", rubric=RUBRIC)
    runner = EvalRunner(agent=_agent("answer"), judges=[primary, second], concurrency=1)
    result = (await runner.arun([case])).results[0]

    assert result.score == pytest.approx(0.82)  # the primary scores the case
    assert result.trust is not None
    assert result.trust.decision == "rejected"
    assert result.panel is not None
    assert result.panel.flagged_low_confidence is True


@pytest.mark.asyncio
async def test_a_second_judge_that_abstains_leaves_the_primary_uncorroborated() -> None:
    primary = LLMJudge(_JudgeModel([82]))
    second = LLMJudge(_JudgeModel([None]))
    case = EvalCase(name="c", prompt="p", rubric=RUBRIC)
    runner = EvalRunner(agent=_agent("answer"), judges=[primary, second], concurrency=1)
    result = (await runner.arun([case])).results[0]

    assert result.passed
    assert result.trust is not None
    assert result.trust.decision == "calibration_required"
    assert result.trust.reason == "no independent opinion"
    assert result.panel is not None
    assert result.panel.judges == 1


@pytest.mark.asyncio
async def test_judge_and_judges_combine_with_the_single_judge_first() -> None:
    primary = LLMJudge(_JudgeModel([95], model_id="primary"))
    second = LLMJudge(_JudgeModel([95], model_id="second"))
    runner = EvalRunner(agent=_agent("answer"), judge=primary, judges=[second], concurrency=1)

    assert runner.judge is primary
    assert runner.judges == [primary, second]
    result = (await runner.arun([EvalCase(name="c", prompt="p", rubric=RUBRIC)])).results[0]
    assert result.verdict is not None
    assert result.verdict.provenance is not None
    assert result.verdict.provenance.judge_model == "primary"


@pytest.mark.asyncio
async def test_one_judge_is_not_a_panel() -> None:
    runner = EvalRunner(agent=_agent("answer"), judge=LLMJudge(_JudgeModel([82])), concurrency=1)
    result = (await runner.arun([EvalCase(name="c", prompt="p", rubric=RUBRIC)])).results[0]
    assert result.panel is None
    assert result.trust is None
    assert result.panel_verdicts == []


# --------------------------------------------------------------------------
# pairwise comparison
# --------------------------------------------------------------------------


class _CompareModel:
    """Picks by policy: ``prefers`` names the text that should win wherever it sits."""

    def __init__(
        self, *, prefers: str | None = None, always: str | None = None, content: str | None = None
    ) -> None:
        self.prefers = prefers
        self.always = always
        self.content = content
        self.config = SimpleNamespace(model="judge")
        self.prompts: list[str] = []

    async def complete(self, messages: list[Message], tools: Any = None, **_: Any) -> ModelResponse:
        prompt = messages[-1].content or ""
        self.prompts.append(prompt)
        if self.content is not None:
            return ModelResponse(message=Message.assistant(content=self.content), usage={})
        if self.always is not None:
            pick = self.always
        else:
            first = prompt.split("<first>\n", 1)[1].split("\n</first>", 1)[0]
            pick = "first" if self.prefers in first else "second"
        if tools is None:
            return ModelResponse(
                message=Message.assistant(content=json.dumps({"winner": pick, "reason": "r"})),
                usage={},
            )
        call = ToolCall(
            id="c", name=SUBMIT_COMPARISON_TOOL, arguments={"winner": pick, "reason": "r"}
        )
        return ModelResponse(message=Message.assistant(tool_calls=[call]), usage={})

    async def stream(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


@pytest.mark.asyncio
async def test_a_preference_that_survives_the_swap_wins() -> None:
    model = _CompareModel(prefers="ANSWER-B")
    result = await LLMJudge(model).compare(
        prompt="p", output_a="ANSWER-A text", output_b="ANSWER-B text", rubric="r"
    )
    assert result.winner == "b"
    assert result.consistent
    assert result.picks == ["b", "b"]
    assert len(model.prompts) == 2
    assert "<first>\nANSWER-A text\n</first>" in model.prompts[0]
    assert "<first>\nANSWER-B text\n</first>" in model.prompts[1]
    assert result.provenance is not None
    assert result.provenance.channel == "tool"


@pytest.mark.asyncio
async def test_a_position_preference_is_a_tie() -> None:
    model = _CompareModel(always="first")
    result = await LLMJudge(model).compare(prompt="p", output_a="A", output_b="B", rubric="r")
    assert result.winner == "tie"
    assert not result.consistent
    assert result.picks == ["a", "b"]
    assert "flipped" in result.reason


@pytest.mark.asyncio
async def test_a_tie_in_both_orders_is_a_consistent_tie() -> None:
    result = await LLMJudge(_CompareModel(always="tie")).compare(
        prompt="p", output_a="A", output_b="B", rubric="r"
    )
    assert result.winner == "tie"
    assert result.consistent


@pytest.mark.asyncio
async def test_a_reading_without_a_pick_abstains() -> None:
    result = await LLMJudge(_CompareModel(content="")).compare(
        prompt="p", output_a="A", output_b="B", rubric="r"
    )
    assert result.abstained
    assert result.winner == "tie"
    assert result.picks == ["abstained", "abstained"]


@pytest.mark.asyncio
async def test_compare_takes_a_rubric_object_and_the_text_channel() -> None:
    model = _CompareModel(prefers="ANSWER-A")
    judge = LLMJudge(model, rubric=RUBRIC, channel="text")
    result = await judge.compare(prompt="p", output_a="ANSWER-A", output_b="ANSWER-B")
    assert result.winner == "a"
    assert "root_cause (weight 50%)" in model.prompts[0]
    assert result.provenance is not None
    assert result.provenance.channel == "text"


@pytest.mark.asyncio
async def test_compare_without_a_rubric_raises() -> None:
    with pytest.raises(ValueError, match="needs a rubric"):
        await LLMJudge(_CompareModel(always="tie")).compare(prompt="p", output_a="A", output_b="B")


@pytest.mark.asyncio
async def test_an_unreachable_comparison_judge_raises() -> None:
    class _Dead:
        async def complete(self, *a: Any, **k: Any) -> Any:
            raise ConnectionError("down")

    with pytest.raises(JudgeUnavailableError):
        await LLMJudge(_Dead()).compare(prompt="p", output_a="A", output_b="B", rubric="r")
