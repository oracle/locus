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
from locus.router.benchmark import TAG_ADVERSARIAL, TAG_LABELLED, TAG_TRAP, load_router_corpus
from locus.router.difficulty import (
    ANSWER_RUBRIC,
    NOT_MEASURED,
    MeasuredDifficulty,
    answer_cases,
    confusion_table,
    direct_agent,
    disagreements,
    intended_difficulty,
    measure_difficulty,
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
