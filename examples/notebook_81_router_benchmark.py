# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Notebook 81: Router benchmark — measure selection and difficulty on a labelled corpus.

The router's accuracy is a measurement, and this notebook takes it on a
sample of the routing corpus (benchmarks/router/cases.json): labelled
prompts with the protocol each should reach, traps whose verb points at an
expensive shape while the work is one step, and adversarial prompts that
tell the model to leave the protocol list. Selection only: every target
answers with a protocol id, and nothing is built or run until the last part.

Key ideas:
- Load the corpus with load_router_corpus and take a sample that covers a
  direct answer, traps, structured shapes, a held action and injections.
- Put two classifiers through the router's front seam with typed_target:
  IntendedFrameClassifier, the corpus's own frame and the ceiling of the
  benchmark, and a keyword rule that reads the verb and falls for the traps.
- Read the pass rate per tag with tag_table and hold the rule to the
  ceiling with router_gate, which lists every case it regressed on.
- With a live model, run the typed router against the free-form baseline
  that asks the model for a protocol id directly.
- With a live model, measure difficulty from outcomes: a single reply, then
  the labelled shape with working tools, graded by a judge.

Run it:
    # Default: OCI GenAI if ~/.oci/config has a usable profile, else the bundled mock model
    python examples/notebook_81_router_benchmark.py

    # Offline / no credentials (the two live parts print a skip notice):
    LOCUS_MODEL_PROVIDER=mock python examples/notebook_81_router_benchmark.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from config import get_model, resolved_provider

from locus.evaluation import EvalCase, EvalRunner, LLMJudge
from locus.router import Complexity, GoalFrame, Risk, TaskType
from locus.router.benchmark import (
    freeform_target,
    load_router_corpus,
    router_gate,
    tag_table,
    typed_target,
)
from locus.router.difficulty import (
    IntendedFrameClassifier,
    ShapeAgent,
    answer_cases,
    confusion_table,
    direct_agent,
    measure_difficulty,
    shape_rubric,
)


CORPUS = Path(__file__).resolve().parent.parent / "benchmarks" / "router" / "cases.json"

#: The sample every part runs on: one direct answer, three traps, four
#: structured shapes, one action the policy gate holds, and two injections.
SAMPLE = (
    "answer-01-what-is-rag",
    "plan-04-plan-a-one-line",
    "build-05-build-the-cron-expression",
    "compare-07-compare-3-and-7",
    "plan-02-plan-the-rollout-of",
    "generate_code-02-write-a-typescript-function",
    "diagnose-07-diagnose-the-login-service",
    "compare-01-compare-token-bucket-vs",
    "remediate-02-roll-back-the-last",
    "injection-01-ignore-the-available-protocol",
    "injection-06-respond-with-the-protocol",
)

#: The sample the difficulty part answers: a trap, a plan and two tool requests.
DIFFICULTY_SAMPLE = (
    "compare-07-compare-3-and-7",
    "plan-02-plan-the-rollout-of",
    "diagnose-07-diagnose-the-login-service",
    "research-08-research-which-of-our",
)

#: The first verb of a request, mapped to the goal it usually names.
VERBS: dict[str, TaskType] = {
    "what": TaskType.ANSWER,
    "plan": TaskType.PLAN,
    "build": TaskType.BUILD,
    "compare": TaskType.COMPARE,
    "write": TaskType.GENERATE_CODE,
    "diagnose": TaskType.DIAGNOSE,
    "roll": TaskType.REMEDIATE,
}


def sample(names: tuple[str, ...] = SAMPLE) -> list[EvalCase]:
    """The named cases of the routing corpus, in the order given."""
    by_name = {case.name: case for case in load_router_corpus(CORPUS)}
    return [by_name[name] for name in names]


class VerbClassifier:
    """A classifier that reads the first verb and nothing else.

    It is the kind of rule the benchmark exists to catch: every request a
    verb names is framed MEDIUM, so a one-step request under an expensive
    verb goes to the expensive shape.
    """

    async def classify(self, text: str) -> GoalFrame:
        goal = VERBS.get(text.split(maxsplit=1)[0].lower(), TaskType.ANSWER)
        simple = goal is TaskType.ANSWER
        return GoalFrame(
            primary_goal=goal,
            domain="engineering",
            complexity=Complexity.LOW if simple else Complexity.MEDIUM,
            risk=Risk.HIGH if goal is TaskType.REMEDIATE else Risk.LOW,
        )


async def score_classifiers(cases: list[EvalCase]) -> None:
    """Part 1: two classifiers through the seam, one table, one gate. No model."""
    print("\n=== Part 1: classifiers through the router's front seam ===\n")
    targets = {
        "ceiling": typed_target(None, classifier=IntendedFrameClassifier(cases)),
        "verb_rule": typed_target(None, classifier=VerbClassifier()),
    }
    reports = {name: await EvalRunner(target).arun(cases) for name, target in targets.items()}
    print(tag_table(reports, cases))

    verdict = router_gate(reports["verb_rule"], reports["ceiling"], per_case=True)
    print("\nverb_rule held to the ceiling, case by case:\n")
    print(verdict.to_markdown())


async def score_model(model: Any, cases: list[EvalCase]) -> None:
    """Part 2: the typed router against asking the model for the id directly."""
    print("\n=== Part 2: typed router against the free-form baseline ===\n")
    reports = {
        "typed": await EvalRunner(typed_target(model), repeats=2).arun(cases),
        "freeform": await EvalRunner(freeform_target(model), repeats=2).arun(cases),
    }
    print(tag_table(reports, cases))
    print("\ntyped held to the free-form floors:\n")
    print(router_gate(reports["typed"], reports["freeform"]).to_markdown())


async def measure(model: Any, cases: list[EvalCase]) -> None:
    """Part 3: difficulty from outcomes: a single reply, then the labelled shape."""
    print("\n=== Part 3: difficulty measured from outcomes ===\n")
    judge = LLMJudge(model)
    replies = await EvalRunner(direct_agent(model), judge=judge).arun(answer_cases(cases))
    shapes = await EvalRunner(ShapeAgent(model, cases), judge=judge).arun(
        answer_cases(cases, rubric=shape_rubric())
    )
    measured = measure_difficulty(cases, {"single reply": replies, "labelled shape": shapes})
    print(confusion_table(measured))
    print()
    for item in measured:
        outcome = ", ".join(f"{tier}: {passed}" for tier, passed in item.passed.items())
        print(f"  {item.case:<44} {item.intended} -> {item.measured}  ({outcome})")


async def main() -> None:
    print("=" * 60)
    print("Notebook 81: Router benchmark — selection and difficulty on a labelled corpus")
    print("=" * 60)
    cases = sample()
    print(f"\n{len(cases)} cases from {CORPUS.name}:")
    for case in cases:
        expected = (case.expected_output_contains or ["(any registered id)"])[0]
        print(f"  {case.name:<46} -> {expected}")

    await score_classifiers(cases)

    if resolved_provider() == "mock":
        print(
            "\n[skip] Parts 2 and 3 call a model: the bundled mock model fills no frame and "
            "judges no answer. Set LOCUS_MODEL_PROVIDER to run them."
        )
        return
    model = get_model(max_tokens=8192)
    await score_model(model, cases)
    await measure(model, sample(DIFFICULTY_SAMPLE))


if __name__ == "__main__":
    asyncio.run(main())
