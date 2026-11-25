# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""The training split for a router classifier, and the profile it learns.

The routing corpus (``benchmarks/router/cases.json``) is the held-out set: a
classifier is gated on it, so it never trains on it. The training split
(``benchmarks/router/train.json``) is a second file of labelled cases built
the same way, each stating the frame its author intends, with the label the
protocol the compiler selects for that frame. :func:`load_router_training_set`
reads it, holds it to the corpus's rules, and refuses a prompt that repeats
or closely paraphrases a held-out one.

What a classifier is trained to emit is a :class:`RoutingProfile`: a task
type, a difficulty, a risk and a set of needs, every one from a closed
vocabulary. It never names a protocol. :meth:`RoutingProfile.to_frame` turns
a profile into the :class:`~locus.router.GoalFrame` the compiler resolves,
so the path from a profile to a shape is the deterministic one the router
already runs. :func:`training_examples` pairs each prompt with its profile,
and :func:`write_jsonl` writes the pairs one JSON object a line.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

from locus.router.benchmark import (
    DIFFICULTIES,
    TAG_LABELLED,
    TAG_TRAP,
    load_router_corpus,
)
from locus.router.difficulty import intended_frame
from locus.router.goal_frame import Complexity, GoalFrame, Risk, TaskType
from locus.router.policy import PolicyGate
from locus.router.protocol import builtin_protocols


if TYPE_CHECKING:
    from locus.evaluation.framework import EvalCase
    from locus.router.protocol import Protocol


__all__ = [
    "MULTI_AGENT_PROTOCOLS",
    "NEEDS",
    "RoutingProfile",
    "TrainingExample",
    "cell_table",
    "load_router_training_set",
    "near_duplicates",
    "prompt_overlap",
    "routing_profile",
    "training_examples",
    "write_jsonl",
]


#: The closed vocabulary of ``RoutingProfile.needs``, in the order a profile lists them.
NEEDS: tuple[str, ...] = ("tools", "memory", "code_generation", "multi_agent", "approval")

#: The built-in protocols that run more than one agent. A case labelled with
#: one of them needs ``multi_agent``.
MULTI_AGENT_PROTOCOLS: frozenset[str] = frozenset({"specialist_fanout", "handoff_chain", "debate"})


class RoutingProfile(BaseModel):
    """What a router classifier emits for a request: closed vocabulary, no shape id.

    ``risk`` is part of the profile because the compiler needs it, but it is
    meant to come from rules over the request (the environment named, the
    kind of action), not from the learned part of a classifier.
    """

    task_type: TaskType
    difficulty: Complexity
    risk: Risk
    needs: tuple[str, ...] = Field(
        default=(), description=f"A subset of {', '.join(NEEDS)}, in that order."
    )

    model_config = {"frozen": True}

    @field_validator("needs")
    @classmethod
    def _closed_needs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = sorted(set(value) - set(NEEDS))
        if unknown:
            raise ValueError(f"needs outside the vocabulary: {', '.join(unknown)}")
        return tuple(need for need in NEEDS if need in value)

    def to_frame(self, *, domain: str = "general", capabilities: Iterable[str] = ()) -> GoalFrame:
        """The frame the compiler resolves for this profile.

        ``domain`` and ``capabilities`` are not part of the profile: they
        come from the deployment (which capability pool to bind) and do not
        decide the shape.
        """
        return GoalFrame(
            primary_goal=self.task_type,
            domain=domain,
            complexity=self.difficulty,
            risk=self.risk,
            requires_tools="tools" in self.needs,
            requires_memory="memory" in self.needs,
            requires_code_generation="code_generation" in self.needs,
            requires_multi_agent="multi_agent" in self.needs,
            approval_required="approval" in self.needs,
            required_capabilities=list(capabilities),
        )


class TrainingExample(BaseModel):
    """One training row: a prompt, the profile to emit for it, and the shape that follows."""

    name: str
    prompt: str
    profile: RoutingProfile
    protocol: str = Field(description="The labelled protocol; a check, never a training target.")
    trap: bool = False
    measured_difficulty: str | None = Field(
        default=None,
        description="The level an outcome measurement found, when one was supplied.",
    )

    model_config = {"frozen": True}


def _labelled_protocol(case: EvalCase) -> str:
    return case.expected_output_contains[0]


def routing_profile(
    case: EvalCase,
    *,
    policy: PolicyGate | None = None,
    protocols: Iterable[Protocol] | None = None,
) -> RoutingProfile:
    """The profile a labelled case stands for, read from the frame it states.

    The task type, difficulty and risk are the stated ones. The needs follow
    by rule: ``tools`` when the case names capabilities, ``code_generation``
    for a code-generation goal, ``multi_agent`` when the labelled protocol is
    one of :data:`MULTI_AGENT_PROTOCOLS`, and ``approval`` when the policy
    gate (the default one when not given) holds the frame for approval on the
    labelled protocol. No case in the corpus states a
    need for memory, so ``memory`` is never set.

    Raises:
        ValueError: The case is not labelled, or its label names no registered protocol.
    """
    if TAG_LABELLED not in case.tags:
        raise ValueError(f"case {case.name!r} is not labelled, so it states no profile")
    frame = intended_frame(case)
    gate = policy if policy is not None else PolicyGate()
    label = _labelled_protocol(case)
    catalogue = {p.id: p for p in (protocols if protocols is not None else builtin_protocols())}
    if label not in catalogue:
        raise ValueError(f"case {case.name!r} names unregistered protocol {label!r}")
    needs: list[str] = []
    if frame.required_capabilities:
        needs.append("tools")
    if frame.primary_goal is TaskType.GENERATE_CODE:
        needs.append("code_generation")
    if label in MULTI_AGENT_PROTOCOLS:
        needs.append("multi_agent")
    if gate.check(frame, catalogue[label]).require_approval:
        needs.append("approval")
    return RoutingProfile(
        task_type=frame.primary_goal,
        difficulty=frame.complexity,
        risk=frame.risk,
        needs=tuple(needs),
    )


def _words(prompt: str) -> frozenset[str]:
    cleaned = "".join(c if c.isalnum() or c == "_" else " " for c in prompt.lower())
    return frozenset(cleaned.split())


def prompt_overlap(first: str, second: str) -> float:
    """How much two prompts share: the Jaccard index of their word sets, 0 to 1."""
    a, b = _words(first), _words(second)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def near_duplicates(
    cases: Sequence[EvalCase], held_out: Sequence[EvalCase], *, threshold: float = 0.5
) -> list[tuple[str, str, float]]:
    """Every pair, one case from each set, whose prompts overlap at or above ``threshold``.

    Returns ``(case name, held-out case name, overlap)``, highest overlap first.
    """
    held = [(case.name, _words(case.prompt)) for case in held_out]
    pairs: list[tuple[str, str, float]] = []
    for case in cases:
        words = _words(case.prompt)
        if not words:
            continue
        for name, other in held:
            if not other:
                continue
            overlap = len(words & other) / len(words | other)
            if overlap >= threshold:
                pairs.append((case.name, name, overlap))
    return sorted(pairs, key=lambda pair: (-pair[2], pair[0], pair[1]))


def load_router_training_set(
    path: str | Path,
    *,
    held_out: Sequence[EvalCase] | None = None,
    threshold: float = 0.5,
    protocols: Iterable[Protocol] | None = None,
) -> list[EvalCase]:
    """Read the training split and check it can be trained on.

    Every case passes :func:`~locus.router.benchmark.load_router_corpus`'s
    rules and is ``labelled`` (a profile cannot be read from an adversarial
    case), no two cases share a prompt, and, when ``held_out`` is given, no
    prompt overlaps a held-out prompt at or above ``threshold``
    (:func:`prompt_overlap`). Pass the routing corpus as ``held_out``: a
    classifier gated on prompts it trained on has not been tested.

    Raises:
        ValueError: A case that does not fit, or the overlapping pairs, named.
    """
    cases = load_router_corpus(path, protocols=protocols)
    file = Path(path).name
    seen: dict[frozenset[str], str] = {}
    for case in cases:
        if TAG_LABELLED not in case.tags:
            raise ValueError(f"{file}: case {case.name!r} is not labelled")
        words = _words(case.prompt)
        if words in seen:
            raise ValueError(f"{file}: cases {seen[words]!r} and {case.name!r} share a prompt")
        seen[words] = case.name
    if held_out is not None:
        leaks = near_duplicates(cases, held_out, threshold=threshold)
        if leaks:
            shown = "; ".join(f"{a!r} and {b!r} ({o:.2f})" for a, b, o in leaks[:5])
            more = f" and {len(leaks) - 5} more" if len(leaks) > 5 else ""
            raise ValueError(
                f"{file}: {len(leaks)} prompt(s) overlap the held-out set at or above "
                f"{threshold}: {shown}{more}"
            )
    return cases


def training_examples(
    cases: Sequence[EvalCase],
    *,
    policy: PolicyGate | None = None,
    protocols: Iterable[Protocol] | None = None,
    measured: Mapping[str, str] | None = None,
) -> list[TrainingExample]:
    """Each labelled case as a training row: its prompt and the profile to emit for it.

    ``measured`` maps a case name to the level an outcome measurement found
    (:func:`~locus.router.difficulty.measure_difficulty`). A row then carries
    it as ``measured_difficulty`` next to the stated one; a case the
    measurement did not settle carries none. The profile keeps the stated
    difficulty, because the label follows from it.
    """
    levels = measured or {}
    catalogue = list(protocols if protocols is not None else builtin_protocols())
    return [
        TrainingExample(
            name=case.name,
            prompt=case.prompt,
            profile=routing_profile(case, policy=policy, protocols=catalogue),
            protocol=_labelled_protocol(case),
            trap=TAG_TRAP in case.tags,
            measured_difficulty=(
                levels[case.name] if levels.get(case.name) in DIFFICULTIES else None
            ),
        )
        for case in cases
        if TAG_LABELLED in case.tags
    ]


def write_jsonl(examples: Sequence[TrainingExample], path: str | Path) -> None:
    """Write the rows one JSON object a line, the layout fine-tuning tools read."""
    lines = [
        json.dumps(example.model_dump(mode="json"), ensure_ascii=False) for example in examples
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def cell_table(cases: Sequence[EvalCase]) -> str:
    """Cases per task type and difficulty as a Markdown table, with the traps counted."""
    examples = training_examples(cases)
    header = (
        "| task type | " + " | ".join(DIFFICULTIES) + " | traps | total |\n"
        "|---|" + "---:|" * (len(DIFFICULTIES) + 2)
    )
    rows = [header]
    totals = dict.fromkeys(DIFFICULTIES, 0)
    traps = 0
    for task in TaskType:
        mine = [e for e in examples if e.profile.task_type is task]
        if not mine:
            continue
        counts = {
            level: sum(e.profile.difficulty.value == level for e in mine) for level in DIFFICULTIES
        }
        trapped = sum(e.trap for e in mine)
        for level in DIFFICULTIES:
            totals[level] += counts[level]
        traps += trapped
        cells = " | ".join(str(counts[level]) for level in DIFFICULTIES)
        rows.append(f"| {task.value} | {cells} | {trapped} | {len(mine)} |")
    cells = " | ".join(str(totals[level]) for level in DIFFICULTIES)
    rows.append(f"| **all** | {cells} | {traps} | {len(examples)} |")
    return "\n".join(rows)
