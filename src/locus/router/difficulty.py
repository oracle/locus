# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Difficulty measured from outcomes, for the router corpus.

A labelled case of the router corpus carries the difficulty its author
intends (its ``difficulty:<level>`` tag). This module measures it instead:
each prompt is answered in one reply, with no tools, by models of increasing
strength (the *tiers*), and a judge grades every answer against a rubric.
The level is read from which tier's answer passes:

- ``low``: the cheapest tier's single reply does the work;
- ``medium``: only a stronger tier's single reply does;
- ``high``: no tier's single reply does, so the request needs more than one
  reply can give (tools, several steps, a check, an approval).

The parts are the evaluation primitives: :func:`answer_cases` turns the
corpus into judged cases, :func:`direct_agent` is the single-reply agent,
an :class:`~locus.evaluation.EvalRunner` with a judge produces one report
per tier, and :func:`measure_difficulty` reads the reports. A measurement is
per model family: the same prompt can be ``low`` on one tier ladder and
``medium`` on another.

Example:
    >>> from locus.evaluation import EvalRunner, LLMJudge
    >>> from locus.router.benchmark import load_router_corpus
    >>> from locus.router.difficulty import (
    ...     answer_cases,
    ...     direct_agent,
    ...     measure_difficulty,
    ... )
    >>> cases = answer_cases(load_router_corpus("benchmarks/router/cases.json"))
    >>> judge = LLMJudge(judge_model)
    >>> reports = {
    ...     name: await EvalRunner(direct_agent(model), judge=judge, repeats=3).arun(
    ...         cases
    ...     )
    ...     for name, model in (("small", small_model), ("large", large_model))
    ... }
    >>> measured = measure_difficulty(cases, reports)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from locus.evaluation import EvalCase, EvalReport
from locus.router.benchmark import DIFFICULTIES, DIFFICULTY_TAG_PREFIX, TAG_LABELLED, TAG_TRAP


if TYPE_CHECKING:
    from locus.agent.agent import Agent

__all__ = [
    "ANSWER_RUBRIC",
    "NOT_MEASURED",
    "MeasuredDifficulty",
    "answer_cases",
    "confusion_table",
    "direct_agent",
    "disagreements",
    "intended_difficulty",
    "measure_difficulty",
]

#: The pass criterion a single reply is judged against. A case may carry its
#: own under ``metadata["answer_rubric"]``.
ANSWER_RUBRIC = (
    "The response, on its own, completes what the request asks for. It passes only if "
    "the work is done in the response: correct, complete, and usable as given. It fails "
    "if it invents data it could not have seen (metric values, alerts, documents, the "
    "state of a system), if it describes or outlines work the request asks to be "
    "performed, if it claims an action on a real system that a text reply cannot take, "
    "if it defers to tools, people or later steps for the substance of the request, or "
    "if it is wrong or materially incomplete."
)

#: The level of a case no decision could be read for (a tier was not graded).
NOT_MEASURED = "not_measured"

_DIRECT_PROMPT = (
    "Answer the request in one reply. You have no tools and cannot look anything up, "
    "run anything or change any system. Do the work in the reply itself when a reply "
    "can do it. Never invent data you cannot see; when the request needs something a "
    "single reply cannot provide, say plainly what is missing."
)


def intended_difficulty(case: EvalCase) -> str | None:
    """The level a case's ``difficulty:<level>`` tag names, or ``None`` without one."""
    for tag in case.tags:
        if tag.startswith(DIFFICULTY_TAG_PREFIX):
            return tag[len(DIFFICULTY_TAG_PREFIX) :]
    return None


def answer_cases(cases: Sequence[EvalCase], *, rubric: str = ANSWER_RUBRIC) -> list[EvalCase]:
    """The labelled cases of the router corpus as judged answer cases.

    Each keeps its name, prompt, tags and metadata and swaps the protocol
    expectation for a rubric: ``metadata["answer_rubric"]`` when the case
    has one, else ``rubric``. Adversarial cases have no right answer to
    grade and are left out.
    """
    out: list[EvalCase] = []
    for case in cases:
        if TAG_LABELLED not in case.tags:
            continue
        own = case.metadata.get("answer_rubric")
        out.append(
            EvalCase(
                name=case.name,
                prompt=case.prompt,
                tags=list(case.tags),
                metadata=dict(case.metadata),
                rubric=own if isinstance(own, str) and own else rubric,
            )
        )
    return out


def direct_agent(model: Any) -> Agent:
    """The single-reply agent a tier answers with: ``model``, no tools, one iteration."""
    from locus.agent.agent import Agent  # noqa: PLC0415 — keep the agent package off this import

    return Agent(model=model, system_prompt=_DIRECT_PROMPT, max_iterations=1)


@dataclass(frozen=True)
class MeasuredDifficulty:
    """What the tiers' answers say about one case."""

    case: str
    #: The level the case's tag names.
    intended: str | None
    #: ``low``, ``medium``, ``high`` or :data:`NOT_MEASURED`.
    measured: str
    #: Per tier, cheapest first: ``True`` passed, ``False`` failed, ``None`` not graded.
    passed: dict[str, bool | None] = field(default_factory=dict)
    trap: bool = False

    @property
    def agrees(self) -> bool:
        """Whether the measurement is the level the author intended."""
        return self.measured == self.intended


def _outcome(report: EvalReport, name: str) -> bool | None:
    for result in report.results:
        if result.case_name == name:
            if result.not_graded_reason is not None or result.error is not None:
                return None
            return result.passed
    return None


def measure_difficulty(
    cases: Sequence[EvalCase], reports: Mapping[str, EvalReport]
) -> list[MeasuredDifficulty]:
    """Read each case's level from the tiers' reports.

    Args:
        cases: The cases the reports were run over.
        reports: One report per tier, cheapest tier first. With repeats a
            case passes a tier only when every reading passed.

    A case is ``low`` when the first tier passed it, ``medium`` when a later
    tier did, and ``high`` when every tier failed it. A tier that was not
    graded for a case (the judge abstained, the answer errored) decides
    nothing: the case is :data:`NOT_MEASURED` unless a graded tier already
    settled it.

    Raises:
        ValueError: Fewer than two tiers; one tier cannot tell ``medium`` from ``high``.
    """
    if len(reports) < 2:
        raise ValueError("measuring difficulty needs at least two tiers, cheapest first")
    tiers = list(reports)
    out: list[MeasuredDifficulty] = []
    for case in cases:
        passed = {tier: _outcome(reports[tier], case.name) for tier in tiers}
        if passed[tiers[0]]:
            level = DIFFICULTIES[0]
        elif passed[tiers[0]] is not None and any(passed[t] for t in tiers[1:]):
            level = DIFFICULTIES[1]
        elif all(passed[t] is False for t in tiers):
            level = DIFFICULTIES[2]
        else:
            level = NOT_MEASURED
        out.append(
            MeasuredDifficulty(
                case=case.name,
                intended=intended_difficulty(case),
                measured=level,
                passed=passed,
                trap=TAG_TRAP in case.tags,
            )
        )
    return out


def disagreements(measured: Sequence[MeasuredDifficulty]) -> list[MeasuredDifficulty]:
    """The measured cases whose level is not the one their author intended."""
    return [m for m in measured if m.measured != NOT_MEASURED and not m.agrees]


def confusion_table(measured: Sequence[MeasuredDifficulty]) -> str:
    """A Markdown table of intended level (rows) against measured level (columns).

    The last row counts the traps on their own: a trap is intended ``low``
    by definition, so its row is where an author's reading of "one step" is
    checked against outcomes.
    """
    columns = [*DIFFICULTIES, NOT_MEASURED]

    def row(label: str, items: Sequence[MeasuredDifficulty]) -> str:
        counts = [sum(1 for m in items if m.measured == column) for column in columns]
        return f"| {label} | {len(items)} | " + " | ".join(str(n) for n in counts) + " |"

    lines = [
        "| intended | n | " + " | ".join(f"measured {c}" for c in columns) + " |",
        "|---|---:|" + "---:|" * len(columns),
    ]
    lines.extend(row(level, [m for m in measured if m.intended == level]) for level in DIFFICULTIES)
    lines.append(row("trap", [m for m in measured if m.trap]))
    return "\n".join(lines)
