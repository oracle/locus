# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Evaluators that score the agent's path over the evidence ledger.

The structural checks and the judge read the answer. For an agent that acts,
the path to the answer is most of what correctness means: which tools it
chose, in what order, with what arguments. Three evaluators read that path
off the :class:`~locus.evaluation.evidence.EvidenceLedger`, with no model
call, and each returns a typed :class:`EvaluatorVerdict` that says what
matched, what was missing, what was unexpected, and which settings produced
the number.

- :class:`TrajectoryMatch` compares the sequence of tool calls with an
  expected one: ``strict`` (the same calls in the same order), ``unordered``
  (the same calls in any order) or ``subset`` (the expected calls appear in
  order, other calls allowed between them).
- :class:`ToolSelection` scores which tools were chosen against which should
  have been, as precision and recall over the set, with tools that must not
  be chosen.
- :class:`ArgumentCorrectness` compares the arguments of one tool's call with
  the expected ones, reading the tool's JSON schema where it has one so that
  ``"42"`` and ``42`` agree on an integer argument, ``5.0`` and ``5.0001`` agree
  within a tolerance, and a list compares element by element.

Every evaluator is data, so it lives on an :class:`~locus.evaluation.EvalCase`
(``evaluators=[...]``), loads from a case file, and enters the suite
fingerprint. The same object evaluates on its own::

    verdict = TrajectoryMatch(
        expected=["lookup_order", "issue_refund"], mode="subset"
    ).evaluate(ledger)
    assert verdict.passed, verdict.reason
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from locus.evaluation.evidence import EvidenceLedger, EvidenceRow


__all__ = [
    "ArgumentCorrectness",
    "ArgumentMismatch",
    "EvaluatorProvenance",
    "EvaluatorVerdict",
    "LedgerEvaluator",
    "ToolSelection",
    "TrajectoryMatch",
    "evaluator_label",
]


_REVISION_CHARS = 12
_WS = re.compile(r"\s+")


def _revision(name: str, settings: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {"evaluator": name, **settings}, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:_REVISION_CHARS]


class EvaluatorProvenance(BaseModel):
    """Which evaluator produced a verdict, under which settings.

    Deterministic evaluators have no model, temperature or seed; what makes
    two of their verdicts comparable is the evaluator and its settings, so
    ``revision`` hashes both. ``evidence_ids`` names the ledger rows read.
    """

    evaluator: str = Field(description="The evaluator's kind, e.g. trajectory_match")
    revision: str = Field(description="Short hash of the evaluator kind and its settings")
    settings: dict[str, Any] = Field(default_factory=dict, description="The settings used")
    evidence_ids: list[str] = Field(default_factory=list, description="Ledger rows read")
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ArgumentMismatch(BaseModel):
    """One argument whose value did not agree with the expected one."""

    argument: str
    expected: Any = None
    actual: Any = None
    reason: str = ""


class EvaluatorVerdict(BaseModel):
    """One deterministic evaluator's reading of a run."""

    evaluator: str
    passed: bool
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    matched: list[str] = Field(default_factory=list, description="Expected items found")
    missing: list[str] = Field(default_factory=list, description="Expected items not found")
    unexpected: list[str] = Field(
        default_factory=list, description="Items found that were not expected"
    )
    mismatches: list[ArgumentMismatch] = Field(
        default_factory=list, description="Arguments whose values disagreed"
    )
    details: dict[str, Any] = Field(default_factory=dict, description="Evaluator-specific figures")
    provenance: EvaluatorProvenance | None = None


class LedgerEvaluator(BaseModel):
    """Common shape of the evaluators: data in, an :class:`EvaluatorVerdict` out."""

    kind: str

    @property
    def label(self) -> str:
        """The check name the runner records the verdict under."""
        return self.kind

    def settings(self) -> dict[str, Any]:
        """The evaluator's configuration, as the provenance records it."""
        return self.model_dump(mode="json", exclude={"kind"})

    def evaluate(self, evidence: EvidenceLedger | Any) -> EvaluatorVerdict:
        """Score ``evidence``: a ledger, or anything with ``tool_executions``."""
        ledger = (
            evidence
            if isinstance(evidence, EvidenceLedger)
            else EvidenceLedger.from_result(evidence)
        )
        verdict = self._evaluate(ledger)
        verdict.provenance = EvaluatorProvenance(
            evaluator=self.kind,
            revision=_revision(self.kind, self.settings()),
            settings=self.settings(),
            evidence_ids=[row.id for row in ledger.tool_rows],
        )
        return verdict

    def _evaluate(self, ledger: EvidenceLedger) -> EvaluatorVerdict:  # pragma: no cover
        raise NotImplementedError

    def _rows(self, ledger: EvidenceLedger, *, executed_only: bool) -> list[EvidenceRow]:
        return [row for row in ledger.tool_rows if row.executed or not executed_only]


def _longest_common_subsequence(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for item in a:
        current = [0]
        for j, other in enumerate(b, start=1):
            current.append(
                previous[j - 1] + 1 if item == other else max(previous[j], current[j - 1])
            )
        previous = current
    return previous[-1]


class TrajectoryMatch(LedgerEvaluator):
    """The sequence of tool calls against the sequence expected.

    Args:
        expected: Tool names in the order expected.
        mode: ``strict``: the run made exactly these calls in this order.
            ``unordered``: the same calls, as a multiset, in any order.
            ``subset``: the expected calls appear in this order, with other
            calls allowed before, between and after them.
        executed_only: Count only calls that ran: a call the gate refused or
            that errored is left out of the trajectory. Off by default, so the
            trajectory is the path the agent took, refusals included.

    The score gives partial credit: in ``strict`` mode the longest common
    subsequence over the longer of the two trajectories, in ``unordered`` the
    overlap of the two multisets over the larger one, in ``subset`` the share
    of expected steps found in order. ``passed`` requires a full match.
    """

    kind: Literal["trajectory_match"] = "trajectory_match"
    expected: list[str] = Field(min_length=1)
    mode: Literal["strict", "unordered", "subset"] = "strict"
    executed_only: bool = False

    @property
    def label(self) -> str:
        return f"trajectory:{self.mode}"

    def _evaluate(self, ledger: EvidenceLedger) -> EvaluatorVerdict:
        actual = [row.tool_name for row in self._rows(ledger, executed_only=self.executed_only)]
        expected = list(self.expected)
        details: dict[str, Any] = {"actual": actual, "expected": expected, "mode": self.mode}

        if self.mode == "strict":
            common = _longest_common_subsequence(actual, expected)
            longest = max(len(actual), len(expected))
            score = common / longest if longest else 1.0
            passed = actual == expected
            matched, missing, unexpected = self._in_order(actual, expected)
            if passed:
                reason = "the trajectory matches exactly"
            else:
                reason = f"expected exactly {expected}, got {actual}"
        elif self.mode == "unordered":
            have, want = Counter(actual), Counter(expected)
            overlap = sum((have & want).values())
            largest = max(sum(have.values()), sum(want.values()))
            score = overlap / largest if largest else 1.0
            passed = have == want
            matched = sorted((have & want).elements())
            missing = sorted((want - have).elements())
            unexpected = sorted((have - want).elements())
            if passed:
                reason = "the same calls were made, in some order"
            else:
                reason = (
                    f"missing {missing}, unexpected {unexpected}" if missing or unexpected else ""
                )
        else:
            matched, missing, unexpected = self._in_order(actual, expected)
            score = len(matched) / len(expected)
            passed = not missing
            if passed:
                reason = "the expected calls appear in order"
            else:
                reason = f"expected {expected} in order; {missing} did not follow, got {actual}"

        return EvaluatorVerdict(
            evaluator=self.kind,
            passed=passed,
            score=min(1.0, max(0.0, score)),
            reason=reason,
            matched=matched,
            missing=missing,
            unexpected=unexpected,
            details=details,
        )

    @staticmethod
    def _in_order(
        actual: Sequence[str], expected: Sequence[str]
    ) -> tuple[list[str], list[str], list[str]]:
        """Walk ``actual`` once: which expected steps followed in order, which did not."""
        matched: list[str] = []
        cursor = 0
        for step in expected:
            while cursor < len(actual) and actual[cursor] != step:
                cursor += 1
            if cursor == len(actual):
                break
            matched.append(step)
            cursor += 1
        missing = list(expected[len(matched) :])
        unexpected = [name for name in actual if name not in expected]
        return matched, missing, unexpected


class ToolSelection(LedgerEvaluator):
    """Which tools the agent chose, against which it should have chosen.

    Args:
        expected: Tools that should have been selected at least once.
        forbidden: Tools that must not have been selected. A selection of one
            fails the verdict whatever else matched.
        allow_extra: Whether tools outside ``expected`` may be selected. On by
            default: a lookup the agent added is not a wrong selection. When
            off, every selected tool must be an expected one.
        executed_only: Count only calls that ran, as on :class:`TrajectoryMatch`.

    The verdict records precision and recall over the set of selected tools.
    The score is the recall when extras are allowed and the F1 otherwise,
    reduced by the share of selections that were forbidden.
    """

    kind: Literal["tool_selection"] = "tool_selection"
    expected: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)
    allow_extra: bool = True
    executed_only: bool = False

    @property
    def label(self) -> str:
        return "tool_selection"

    def _evaluate(self, ledger: EvidenceLedger) -> EvaluatorVerdict:
        selected: list[str] = []
        for row in self._rows(ledger, executed_only=self.executed_only):
            if row.tool_name not in selected:
                selected.append(row.tool_name)
        expected = set(self.expected)
        chosen = set(selected)
        matched = sorted(expected & chosen)
        missing = sorted(expected - chosen)
        unexpected = sorted(chosen - expected)
        forbidden_hits = sorted(chosen & set(self.forbidden))

        recall = len(matched) / len(expected) if expected else 1.0
        precision = len(matched) / len(chosen) if chosen else (1.0 if not expected else 0.0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        score = recall if self.allow_extra else f1
        if forbidden_hits:
            score *= 1 - len(forbidden_hits) / max(1, len(chosen))
        passed = not missing and not forbidden_hits and (self.allow_extra or not unexpected)

        if passed:
            reason = "the expected tools were selected"
        else:
            parts = []
            if missing:
                parts.append(f"not selected: {missing}")
            if forbidden_hits:
                parts.append(f"forbidden selected: {forbidden_hits}")
            if unexpected and not self.allow_extra:
                parts.append(f"unexpected: {unexpected}")
            reason = "; ".join(parts)
        return EvaluatorVerdict(
            evaluator=self.kind,
            passed=passed,
            score=min(1.0, max(0.0, score)),
            reason=reason,
            matched=matched,
            missing=missing,
            unexpected=unexpected,
            details={
                "selected": selected,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "forbidden_selected": forbidden_hits,
            },
        )


_NUMERIC_TYPES = frozenset({"integer", "number"})


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").strip())
        except ValueError:
            return None
    return None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1"):
            return True
        if lowered in ("false", "no", "0"):
            return False
    return None


class ArgumentCorrectness(LedgerEvaluator):
    """The arguments of one tool's call against the arguments expected.

    Args:
        tool: The tool whose call is checked.
        expected: The arguments that matter and their values. Keys not named
            are not checked unless ``strict_keys`` is on.
        schema: The tool's parameter schema (``{"type": "object",
            "properties": {...}}``), which decides how each value is compared:
            an ``integer`` or ``number`` property compares numerically within
            the tolerance, so ``"42"`` and ``42`` agree; a ``boolean`` accepts
            ``"true"``; an ``array`` compares element by element, or as a
            multiset when ``uniqueItems`` is set; an ``object`` compares its
            ``properties`` recursively. Left ``None``, the runner fills it
            from the tool on the agent under test when it can, and the
            expected value's own type decides otherwise.
        tolerance: Absolute tolerance for every numeric argument.
        tolerances: Absolute tolerance per argument, overriding ``tolerance``.
        relative_tolerance: Relative tolerance for numeric arguments, as a
            share of the expected value; a number agrees when either
            tolerance admits it.
        ignore_case: Compare strings case-insensitively, with whitespace
            folded.
        strict_keys: Also fail on arguments the call carried that ``expected``
            does not name.
        occurrence: Which call of the tool must agree: ``any`` (the best
            match of all calls), ``first``, ``last`` or ``all``.
        executed_only: Count only calls that ran, as on :class:`TrajectoryMatch`.

    The score is the share of expected arguments that agreed on the call
    judged; a tool that was never called scores 0.
    """

    kind: Literal["argument_correctness"] = "argument_correctness"
    tool: str = Field(min_length=1)
    expected: dict[str, Any] = Field(default_factory=dict)
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    tolerance: float = Field(default=0.0, ge=0.0)
    tolerances: dict[str, float] = Field(default_factory=dict)
    relative_tolerance: float = Field(default=0.0, ge=0.0)
    ignore_case: bool = True
    strict_keys: bool = False
    occurrence: Literal["any", "first", "last", "all"] = "any"
    executed_only: bool = False

    model_config = {"populate_by_name": True}

    @property
    def label(self) -> str:
        return f"arguments:{self.tool}"

    def settings(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"kind"}, by_alias=True)

    def with_schema(self, schema: Mapping[str, Any] | None) -> ArgumentCorrectness:
        """A copy carrying ``schema`` when this one has none; this one otherwise."""
        if self.schema_ is not None or schema is None:
            return self
        return self.model_copy(update={"schema_": dict(schema)})

    def _evaluate(self, ledger: EvidenceLedger) -> EvaluatorVerdict:
        calls = [
            row
            for row in self._rows(ledger, executed_only=self.executed_only)
            if row.tool_name == self.tool
        ]
        if not calls:
            return EvaluatorVerdict(
                evaluator=self.kind,
                passed=False,
                score=0.0,
                reason=f"{self.tool} was not called",
                missing=sorted(self.expected),
                details={"calls": 0},
            )
        if self.occurrence == "first":
            judged = [calls[0]]
        elif self.occurrence == "last":
            judged = [calls[-1]]
        else:
            judged = calls

        readings = [self._compare_call(row.arguments) for row in judged]
        if self.occurrence == "all":
            chosen = min(readings, key=lambda r: (r[0], r[1]))
            passed = all(r[0] for r in readings)
        else:
            chosen = max(readings, key=lambda r: (r[0], r[1]))
            passed = chosen[0]
        _, score, matched, mismatches, unexpected = chosen
        if passed:
            reason = f"the arguments of {self.tool} agree"
        else:
            named = [m.argument for m in mismatches]
            reason = f"{self.tool}: {len(mismatches)} argument(s) disagree: {named}"
            if unexpected:
                reason += f"; unexpected: {unexpected}"
        return EvaluatorVerdict(
            evaluator=self.kind,
            passed=passed,
            score=score,
            reason=reason,
            matched=matched,
            missing=[m.argument for m in mismatches if m.reason == "missing"],
            unexpected=unexpected,
            mismatches=mismatches,
            details={"calls": len(calls), "judged": len(judged), "occurrence": self.occurrence},
        )

    def _compare_call(
        self, arguments: Mapping[str, Any]
    ) -> tuple[bool, float, list[str], list[ArgumentMismatch], list[str]]:
        properties = (self.schema_ or {}).get("properties") or {}
        matched: list[str] = []
        mismatches: list[ArgumentMismatch] = []
        for key, wanted in self.expected.items():
            if key not in arguments:
                mismatches.append(ArgumentMismatch(argument=key, expected=wanted, reason="missing"))
                continue
            actual = arguments[key]
            problem = self._compare(
                wanted,
                actual,
                properties.get(key) if isinstance(properties, Mapping) else None,
                self.tolerances.get(key, self.tolerance),
            )
            if problem is None:
                matched.append(key)
            else:
                mismatches.append(
                    ArgumentMismatch(argument=key, expected=wanted, actual=actual, reason=problem)
                )
        unexpected = sorted(k for k in arguments if k not in self.expected)
        if not self.strict_keys:
            unexpected_counted: list[str] = []
        else:
            unexpected_counted = unexpected
        total = len(self.expected)
        score = len(matched) / total if total else 1.0
        passed = not mismatches and not unexpected_counted
        return passed, score, matched, mismatches, unexpected_counted

    def _compare(
        self, expected: Any, actual: Any, schema: Mapping[str, Any] | None, tolerance: float
    ) -> str | None:
        """``None`` when the values agree, else why they do not."""
        declared = schema.get("type") if isinstance(schema, Mapping) else None
        if isinstance(declared, list):
            declared = next((t for t in declared if t != "null"), None)

        if declared in _NUMERIC_TYPES or (
            declared is None
            and isinstance(expected, int | float)
            and not isinstance(expected, bool)
        ):
            want, have = _as_float(expected), _as_float(actual)
            if want is None or have is None:
                return f"expected a number, got {actual!r}"
            if math.isclose(want, have, rel_tol=self.relative_tolerance, abs_tol=tolerance):
                return None
            return f"{have:g} is not within {tolerance:g} of {want:g}"

        if declared == "boolean" or (declared is None and isinstance(expected, bool)):
            want, have = _as_bool(expected), _as_bool(actual)
            if want is None or have is None or want != have:
                return f"expected {expected!r}, got {actual!r}"
            return None

        if declared == "array" or (declared is None and isinstance(expected, list)):
            if not isinstance(expected, list) or not isinstance(actual, list):
                return f"expected a list, got {actual!r}"
            items = schema.get("items") if isinstance(schema, Mapping) else None
            unique = bool(schema.get("uniqueItems")) if isinstance(schema, Mapping) else False
            if len(expected) != len(actual):
                return f"expected {len(expected)} item(s), got {len(actual)}"
            if unique:
                remaining = list(actual)
                for item in expected:
                    found = next(
                        (
                            i
                            for i, candidate in enumerate(remaining)
                            if self._compare(item, candidate, items, tolerance) is None
                        ),
                        None,
                    )
                    if found is None:
                        return f"{item!r} not in {actual!r}"
                    remaining.pop(found)
                return None
            for index, (item, candidate) in enumerate(zip(expected, actual, strict=True)):
                problem = self._compare(item, candidate, items, tolerance)
                if problem is not None:
                    return f"item {index}: {problem}"
            return None

        if declared == "object" or (declared is None and isinstance(expected, Mapping)):
            if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
                return f"expected an object, got {actual!r}"
            nested = schema.get("properties") if isinstance(schema, Mapping) else None
            for key, item in expected.items():
                if key not in actual:
                    return f"{key} missing"
                problem = self._compare(
                    item,
                    actual[key],
                    nested.get(key) if isinstance(nested, Mapping) else None,
                    tolerance,
                )
                if problem is not None:
                    return f"{key}: {problem}"
            return None

        if expected is None:
            return None if actual is None else f"expected null, got {actual!r}"

        want_text, have_text = str(expected), str(actual)
        if self.ignore_case:
            want_text = _WS.sub(" ", want_text).strip().lower()
            have_text = _WS.sub(" ", have_text).strip().lower()
        if want_text == have_text:
            return None
        return f"expected {expected!r}, got {actual!r}"


#: The evaluators a case may carry, told apart by ``kind`` when loaded from a file.
CaseEvaluator = Annotated[
    TrajectoryMatch | ToolSelection | ArgumentCorrectness, Field(discriminator="kind")
]


def evaluator_label(evaluator: LedgerEvaluator, taken: set[str]) -> str:
    """The check name for ``evaluator``, suffixed when the case already uses it."""
    label = evaluator.label
    if label not in taken:
        taken.add(label)
        return label
    index = 2
    while f"{label}#{index}" in taken:
        index += 1
    label = f"{label}#{index}"
    taken.add(label)
    return label
