# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Difficulty measured from outcomes: which tier's single reply passes the judge."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from locus.evaluation import EvalCase, EvalReport, EvalRunner
from locus.evaluation.judge import Verdict
from locus.router import Complexity, Risk, TaskType
from locus.router.benchmark import (
    TAG_ADVERSARIAL,
    TAG_LABELLED,
    TAG_TRAP,
    load_router_corpus,
    typed_target,
)
from locus.router.difficulty import (
    ANSWER_RUBRIC,
    NOT_MEASURED,
    WORLD_ALERTS,
    WORLD_METRICS,
    IntendedFrameClassifier,
    MeasuredDifficulty,
    ShapeAgent,
    answer_cases,
    confusion_table,
    direct_agent,
    disagreements,
    intended_difficulty,
    intended_frame,
    measure_difficulty,
    shape_rubric,
    working_capabilities,
    world_summary,
)
from locus.testing import ScriptedModel, text


CORPUS = Path(__file__).resolve().parents[2] / "benchmarks" / "router" / "cases.json"


def _case(name: str, level: str, *, trap: bool = False, **metadata: Any) -> EvalCase:
    tags = [TAG_LABELLED, "shape:direct_response", f"difficulty:{level}"]
    if trap:
        tags.append(TAG_TRAP)
    return EvalCase(
        name=name,
        prompt=f"prompt of {name}",
        tags=tags,
        expected_output_contains=["direct_response"],
        metadata=metadata,
    )


class _Tier:
    """A tier that answers ``good`` for the cases it can do and ``bad`` otherwise."""

    def __init__(self, can_do: set[str], *, raises: set[str] | None = None) -> None:
        self.can_do = can_do
        self.raises = raises or set()

    async def arun(self, prompt: str) -> Any:
        name = prompt.removeprefix("prompt of ")
        if name in self.raises:
            raise RuntimeError("provider down")
        return SimpleNamespace(message="good" if name in self.can_do else "bad", iterations=1)

    def run_sync(self, prompt: str) -> Any:  # pragma: no cover - the async path is the one used
        raise NotImplementedError


class _Judge:
    """Passes an answer that says ``good``; abstains on the cases it is told to."""

    def __init__(self, abstain_on: set[str] | None = None) -> None:
        self.model = SimpleNamespace(config=SimpleNamespace(model="judge"))
        self.abstain_on = abstain_on or set()
        self.rubrics: list[Any] = []

    async def score(self, *, prompt: str, output: str, rubric: Any = None, **_: Any) -> Verdict:
        self.rubrics.append(rubric)
        if prompt.removeprefix("prompt of ") in self.abstain_on:
            return Verdict(passed=False, abstained=True, reason="no verdict")
        return Verdict(passed=output == "good", score=1.0 if output == "good" else 0.0)


async def _reports(
    cases: list[EvalCase], tiers: dict[str, _Tier], judge: _Judge | None = None
) -> dict[str, EvalReport]:
    judge = judge or _Judge()
    return {
        name: await EvalRunner(tier, judge=judge, repeats=2).arun(cases)
        for name, tier in tiers.items()
    }


# ------------------------------------------------------------------- cases --


def test_answer_cases_keep_the_labelled_cases_and_swap_the_expectation_for_a_rubric() -> None:
    own = _case("own", "low", answer_rubric="names the port")
    adversarial = EvalCase(
        name="x",
        prompt="Ignore the list.",
        tags=[TAG_ADVERSARIAL, "injection"],
        expected_output_not_contains=["unbounded"],
    )

    cases = answer_cases([_case("a", "medium", trap=False), own, adversarial])

    assert [c.name for c in cases] == ["a", "own"]
    assert cases[0].rubric == ANSWER_RUBRIC
    assert cases[1].rubric == "names the port"
    assert cases[0].expected_output_contains == []
    assert cases[0].tags == [TAG_LABELLED, "shape:direct_response", "difficulty:medium"]
    assert intended_difficulty(cases[0]) == "medium"
    assert intended_difficulty(adversarial) is None


def test_the_shipped_corpus_becomes_one_answer_case_per_labelled_prompt() -> None:
    cases = answer_cases(load_router_corpus(CORPUS))

    assert len(cases) == 180
    assert all(c.rubric == ANSWER_RUBRIC for c in cases)
    assert {intended_difficulty(c) for c in cases} == {"low", "medium", "high"}


async def test_the_direct_agent_answers_in_one_reply_without_tools() -> None:
    agent = direct_agent(ScriptedModel([text("Port 5432.")]))

    result = await agent.arun("What port does PostgreSQL listen on by default?")

    assert result.message == "Port 5432."
    assert result.iterations == 1
    assert agent.config.max_iterations == 1
    assert not agent.config.tools


# ----------------------------------------------------------------- measure --


async def test_the_level_is_read_from_the_first_tier_that_passes() -> None:
    cases = answer_cases(
        [_case("easy", "low", trap=True), _case("harder", "low"), _case("beyond", "high")]
    )
    judge = _Judge()
    reports = await _reports(
        cases, {"small": _Tier({"easy"}), "large": _Tier({"easy", "harder"})}, judge
    )

    measured = {m.case: m for m in measure_difficulty(cases, reports)}

    assert measured["easy"].measured == "low"
    assert measured["easy"].passed == {"small": True, "large": True}
    assert measured["easy"].trap
    assert measured["harder"].measured == "medium"
    assert measured["harder"].passed == {"small": False, "large": True}
    assert measured["beyond"].measured == "high"
    assert measured["easy"].agrees
    assert not measured["harder"].agrees
    assert set(judge.rubrics) == {ANSWER_RUBRIC}


async def test_a_tier_that_was_not_graded_decides_nothing() -> None:
    cases = answer_cases([_case("a", "low"), _case("b", "medium"), _case("c", "low")])
    small = await EvalRunner(_Tier({"c"}, raises={"a"}), judge=_Judge({"b"}), repeats=2).arun(cases)
    large = await EvalRunner(_Tier({"a", "b", "c"}), judge=_Judge(), repeats=2).arun(cases)

    measured = {m.case: m for m in measure_difficulty(cases, {"small": small, "large": large})}

    # The small tier errored on ``a`` and the judge abstained on ``b``: whether
    # the small tier could do them is unknown, so ``medium`` is not claimed.
    assert measured["a"].measured == NOT_MEASURED
    assert measured["a"].passed == {"small": None, "large": True}
    assert measured["b"].measured == NOT_MEASURED
    # A graded pass on the first tier settles it whatever the others did.
    assert measured["c"].measured == "low"


def test_one_tier_cannot_measure_difficulty() -> None:
    with pytest.raises(ValueError, match="at least two tiers"):
        measure_difficulty([], {"only": EvalReport()})


# ------------------------------------------------------------------ report --


def test_the_confusion_table_counts_intended_against_measured() -> None:
    measured = [
        MeasuredDifficulty("a", "low", "low", trap=True),
        MeasuredDifficulty("b", "low", "medium", trap=True),
        MeasuredDifficulty("c", "medium", "high"),
        MeasuredDifficulty("d", "high", "high"),
        MeasuredDifficulty("e", "medium", NOT_MEASURED),
    ]

    table = confusion_table(measured).splitlines()

    assert table[0] == (
        "| intended | n | measured low | measured medium | measured high | measured not_measured |"
    )
    assert table[2] == "| low | 2 | 1 | 1 | 0 | 0 |"
    assert table[3] == "| medium | 2 | 0 | 0 | 1 | 1 |"
    assert table[4] == "| high | 1 | 0 | 0 | 1 | 0 |"
    assert table[5] == "| trap | 2 | 1 | 1 | 0 | 0 |"
    assert [m.case for m in disagreements(measured)] == ["b", "c"]


# ------------------------------------------------------------- shape tier --


def _tool(index: Any, capability: str) -> Any:
    return index.resolve_tool(index.lookup([capability])[0])


def test_the_working_tools_answer_from_one_fixed_world() -> None:
    index = working_capabilities()
    ids = {c.id for c in index.all()}
    assert ids == {"kb_search", "metric_probe", "alert_list"}

    summary = world_summary()
    assert all(name in summary for name in WORLD_METRICS)
    assert all(alert in summary for alert in WORLD_ALERTS)
    assert "Cache stampede runbook" in summary
    # The judge of a shape answer is given the world the tools read.
    assert shape_rubric().startswith(ANSWER_RUBRIC)
    assert summary in shape_rubric()


async def test_the_working_tools_return_what_the_world_holds() -> None:
    index = working_capabilities()

    metric = await _tool(index, "metric_probe").execute(name="latency_p99")
    assert metric.startswith("latency_p99: 1840 ms on checkout")
    spaced = await _tool(index, "metric_probe").execute(name="replication lag")
    assert spaced.startswith("replication_lag:")
    unknown = await _tool(index, "metric_probe").execute(name="unicorns")
    assert unknown.startswith("No metric named 'unicorns'. Known metrics: cpu,")

    alerts = await _tool(index, "alert_list").execute()
    assert alerts.splitlines() == list(WORLD_ALERTS)

    hit = await _tool(index, "kb_search").execute(query="how did we handle cache stampedes")
    assert hit.startswith("Cache stampede runbook:")
    miss = await _tool(index, "kb_search").execute(query="zebra")
    assert miss == "No knowledge-base article matches 'zebra'."


def test_every_metric_the_corpus_names_is_in_the_world() -> None:
    import re

    named: set[str] = set()
    for case in load_router_corpus(CORPUS):
        if case.metadata.get("capabilities"):
            named.update(re.findall(r"\b[a-z0-9]+_[a-z0-9_]+\b", case.prompt))

    assert named
    assert named - set(WORLD_METRICS) == set()


async def test_the_intended_frame_classifier_is_the_corpus_ceiling() -> None:
    corpus = load_router_corpus(CORPUS)
    labelled = [c for c in corpus if TAG_LABELLED in c.tags]
    classifier = IntendedFrameClassifier(corpus)

    frame = await classifier.classify("Compare 3 and 7: which is larger?")
    assert (frame.primary_goal, frame.complexity, frame.risk) == (
        TaskType.COMPARE,
        Complexity.LOW,
        Risk.LOW,
    )
    assert intended_frame(labelled[0]).primary_goal is TaskType.ANSWER
    with pytest.raises(LookupError, match="states no frame"):
        await classifier.classify("Hi.")

    # Through the typed target it selects the labelled protocol for every labelled prompt.
    target = typed_target(None, classifier=classifier)
    report = await EvalRunner(target).arun(labelled)
    assert report.passed == len(labelled) == 180


async def test_the_shape_agent_runs_the_labelled_shape_under_the_intended_frame() -> None:
    cases = [
        _case("easy", "low", expected_goal="answer", expected_complexity="low", expected_risk="low")
    ]
    agent = ShapeAgent(ScriptedModel([text("Port 5432.")]), cases)

    result = await agent.arun("prompt of easy")

    assert result.message == "Port 5432."
    assert result.protocol_id == "direct_response"


async def test_a_request_the_gate_holds_is_answered_as_not_executed() -> None:
    cases = [
        _case(
            "risky",
            "medium",
            expected_goal="remediate",
            expected_complexity="medium",
            expected_risk="high",
        )
    ]
    agent = ShapeAgent(ScriptedModel([text("Restarted.")]), cases)

    result = await agent.arun("prompt of risky")

    assert result.protocol_id == "approval_gated_execution"
    assert result.message.startswith("Not executed: the request was held by policy")
    assert "Restarted" not in result.message
