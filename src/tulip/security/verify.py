# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Verification — the independent challenge that prevents security hallucinations.

A `Evidence` is a *claim*. Before it drives an action or reaches an analyst,
:func:`verify` subjects it to an independent **skeptic** whose job is to *refute*
it — challenge the assumptions, weigh the evidence quality, and flag any
inconsistency between what is claimed and what is supported. The result is a
:class:`VerificationResult`: ``survives`` (with a confidence), or refuted, with the
refutations recorded.

This is the question generic agent frameworks don't answer — *"how do I know the
agent is right?"* — and it works on **any** finding, not just Tulip's: pass a
:class:`~tulip.security.findings.Evidence` or a finding-shaped mapping from an
external agent (LangGraph/CrewAI/anything), so Tulip can sit above the stack as
the verification layer.

Two skeptics ship, both satisfying the :class:`Skeptic` protocol:

- :class:`EvidenceQualitySkeptic` — deterministic and offline; grades the
  evidence a finding *carries* (no model, no network).
- :class:`AdversarialSkeptic` — LLM-backed; actively tries to **refute** the
  claim, scrutinising overreach (a conclusion stronger than its evidence),
  missing evidence, internal contradictions, and alternative explanations the
  evidence doesn't rule out.

Compose a panel via ``verify(finding, skeptics=[...])`` — e.g. the cheap
deterministic gate plus the adversarial LLM challenge. Both work on **any**
finding (Tulip's or a finding-shaped mapping from another framework).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast, runtime_checkable

from pydantic import BaseModel, Field

from tulip.security.findings import Evidence
from tulip.security.taxonomy import Severity, severity_at_least


# A finding to verify: Tulip's typed Evidence, or a finding-shaped mapping from
# any other agent/framework (keys: title, severity, gsar_score, evidence_refs,
# confidence).
FindingLike = Evidence | Mapping[str, Any]

# The GSAR proceed threshold (Appendix B) — a grounded Evidence cleared this.
_PROCEED_THRESHOLD = 0.80


@dataclass(frozen=True)
class Refutation:
    """One objection a skeptic raises. ``weight`` ∈ {weak, concern, fatal}."""

    reason: str
    weight: str = "concern"


@dataclass(frozen=True)
class VerificationResult:
    """The outcome of verifying a finding.

    ``survives`` is False if any refutation is ``fatal`` or confidence falls
    below the threshold. ``alternatives`` is populated by semantic skeptics
    (the deterministic one leaves it empty).
    """

    survives: bool
    confidence: float
    evidence_quality: float
    refutations: list[Refutation] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass(frozen=True)
class _View:
    """Normalised view over a Evidence or a finding-shaped mapping."""

    title: str
    severity: Severity | None
    gsar_score: float
    evidence_refs: list[str]
    confidence: float


def _parse_severity(value: Any) -> Severity | None:
    if isinstance(value, Severity):
        return value
    if isinstance(value, str):
        try:
            return Severity(value.lower())
        except ValueError:
            try:
                return Severity[value.upper()]
            except KeyError:
                return None
    return None


def _coerce(finding: FindingLike) -> _View:
    if isinstance(finding, Evidence):
        return _View(
            title=finding.title,
            severity=finding.severity,
            gsar_score=finding.gsar_score,
            evidence_refs=list(finding.evidence_refs),
            confidence=finding.confidence,
        )
    score = float(finding.get("gsar_score", 0.0) or 0.0)
    return _View(
        title=str(finding.get("title", "<untitled>")),
        severity=_parse_severity(finding.get("severity")),
        gsar_score=score,
        evidence_refs=[str(r) for r in (finding.get("evidence_refs") or [])],
        confidence=float(finding.get("confidence", score) or 0.0),
    )


@runtime_checkable
class Skeptic(Protocol):
    """Something that tries to refute a finding. ``name`` is read-only."""

    @property
    def name(self) -> str: ...

    async def challenge(self, finding: FindingLike) -> list[Refutation]: ...


@dataclass(frozen=True)
class EvidenceQualitySkeptic:
    """Deterministic, offline skeptic — grades the evidence a finding carries.

    No model required. It does not invent contradictory evidence; it checks
    that the claim is actually supported by what it ships with.
    """

    name: str = "evidence-quality"
    proceed_threshold: float = _PROCEED_THRESHOLD

    async def challenge(self, finding: FindingLike) -> list[Refutation]:
        v = _coerce(finding)
        out: list[Refutation] = []
        if not v.evidence_refs:
            out.append(Refutation("No evidence references — the claim is unsupported.", "fatal"))
        if v.gsar_score < self.proceed_threshold:
            out.append(
                Refutation(
                    f"Grounding score {v.gsar_score:.2f} is below the proceed "
                    f"threshold {self.proceed_threshold:.2f}.",
                    "concern",
                )
            )
        if (
            v.severity is not None
            and severity_at_least(v.severity, Severity.HIGH)
            and len(v.evidence_refs) == 1
        ):
            out.append(
                Refutation(
                    f"{v.severity.value} severity asserted on a single evidence reference.",
                    "concern",
                )
            )
        if v.confidence - v.gsar_score > 0.25:
            out.append(
                Refutation(
                    f"Stated confidence {v.confidence:.2f} materially exceeds the "
                    f"grounding score {v.gsar_score:.2f}.",
                    "weak",
                )
            )
        return out


# ---------------------------------------------------------------------------
# Adversarial LLM skeptic — the semantic challenge
# ---------------------------------------------------------------------------

_WEIGHTS = frozenset({"weak", "concern", "fatal"})


class _Objection(BaseModel):
    """One refutation the reviewer raises."""

    reason: str = Field(description="A specific, concrete objection to the finding.")
    weight: str = Field(default="concern", description="One of: weak, concern, fatal.")


class _AdversarialReview(BaseModel):
    """Structured output of the adversarial reviewer."""

    supported: bool = Field(
        description="True ONLY if the cited evidence directly and fully supports "
        "the conclusion, with no overreach.",
    )
    objections: list[_Objection] = Field(
        default_factory=list,
        description="Refutations: overreach, missing evidence, internal contradictions.",
    )
    alternatives: list[str] = Field(
        default_factory=list,
        description="Alternative/benign explanations the cited evidence does not rule out.",
    )


_ADVERSARIAL_SYSTEM_PROMPT = """\
You are an adversarial security reviewer. Your job is to REFUTE the finding
below, not to confirm it. Treat it as a possible false positive until its own
evidence proves otherwise. Plausibility is not proof.

Scrutinise exactly four things:
1. OVERREACH — does the conclusion claim more than the cited evidence supports
   (an identity, a severity, an attribution the evidence cannot carry)?
2. MISSING EVIDENCE — what specific evidence would you need to believe this that
   is NOT cited?
3. INTERNAL CONTRADICTION — does stated confidence exceed the grounding score,
   or does the severity exceed what the cited references can justify?
4. ALTERNATIVE EXPLANATIONS — what benign or different cause could produce the
   same evidence?

Be conservative: if uncertain, raise a concern. Weights:
- "fatal" ONLY when the finding is unsupported (no usable evidence) or
  self-contradictory;
- "concern" for a material gap or overreach;
- "weak" for a minor caveat.
Set supported=true ONLY if the evidence directly and fully supports the
conclusion. Return JSON only.
"""


def _describe(finding: FindingLike, v: _View) -> str:
    """Render a finding as the reviewer's prompt body."""
    lines = [f"TITLE: {v.title}"]
    if isinstance(finding, Evidence) and finding.description:
        lines.append(f"DESCRIPTION: {finding.description}")
    lines.append(f"SEVERITY: {v.severity.value if v.severity else 'unspecified'}")
    lines.append(f"GROUNDING SCORE: {v.gsar_score:.2f}")
    lines.append(f"STATED CONFIDENCE: {v.confidence:.2f}")
    lines.append(f"EVIDENCE REFERENCES ({len(v.evidence_refs)}):")
    lines.extend(f"  - {r}" for r in v.evidence_refs) if v.evidence_refs else lines.append(
        "  (none)"
    )
    lines.append("\nRefute this finding. Return JSON only.")
    return "\n".join(lines)


class AdversarialSkeptic:
    """LLM skeptic that actively tries to refute a finding (the semantic challenge).

    Drives any :class:`tulip.models.base.ModelProtocol` (or a ``"provider:model"``
    string) with an adversarial prompt + constrained decoding, and maps the
    reviewer's objections and unruled-out alternatives into :class:`Refutation`\\ s.
    It **fails safe**: if the model call or its output can't be processed, it
    raises a ``weak`` refutation noting the finding went *unchallenged* rather
    than silently passing it.

    Pair it with :class:`EvidenceQualitySkeptic` in a panel::

        await verify(
            finding,
            skeptics=[
                EvidenceQualitySkeptic(),
                AdversarialSkeptic("anthropic:claude-sonnet-4-6"),
            ],
        )
    """

    name = "adversarial-llm"

    def __init__(self, model: Any, *, strict: bool = True) -> None:
        self._model = model
        self.strict = strict

    def _resolve(self) -> Any:
        if isinstance(self._model, str):
            from tulip.models.registry import get_model

            self._model = get_model(self._model)
        return self._model

    async def challenge(self, finding: FindingLike) -> list[Refutation]:
        from tulip.core.messages import Message
        from tulip.core.structured import build_response_format, parse_structured

        model = self._resolve()
        v = _coerce(finding)
        messages = [Message.system(_ADVERSARIAL_SYSTEM_PROMPT), Message.user(_describe(finding, v))]
        try:
            response = await model.complete(
                messages=messages,
                tools=None,
                response_format=build_response_format(_AdversarialReview, strict=self.strict),
            )
        except Exception as exc:  # noqa: BLE001 — fail safe: an unrun challenge is a caveat
            return [
                Refutation(
                    f"Adversarial verification did not run ({type(exc).__name__}); "
                    "finding was not independently challenged.",
                    "weak",
                )
            ]

        parsed = parse_structured(
            response.message.content or "{}", _AdversarialReview, strict=False
        )
        if not parsed.success or parsed.parsed is None:
            return [
                Refutation(
                    "Adversarial verification output could not be parsed; "
                    "treat the finding as unchallenged.",
                    "weak",
                )
            ]
        review = cast("_AdversarialReview", parsed.parsed)

        out: list[Refutation] = [
            Refutation(o.reason, o.weight if o.weight in _WEIGHTS else "concern")
            for o in review.objections
        ]
        out.extend(
            Refutation(f"Alternative explanation not ruled out: {alt}", "weak")
            for alt in review.alternatives
        )
        if not review.supported and not any(r.weight in ("concern", "fatal") for r in out):
            out.append(
                Refutation(
                    "Reviewer judged the conclusion not fully supported by the cited evidence.",
                    "concern",
                )
            )
        return out


_PENALTY = {"fatal": 1.0, "concern": 0.2, "weak": 0.1}

# Non-fatal objections erode confidence, but capped — so a thorough skeptic that
# lists many concerns/alternatives can't, by sheer volume, refute a well-grounded
# finding. Only a *fatal* objection (or a sub-threshold grounding score) refutes.
# This is what keeps verification from crying wolf in reverse.
_MAX_NONFATAL_PENALTY = 0.3


async def verify(
    finding: FindingLike,
    *,
    skeptics: Sequence[Skeptic] | None = None,
    threshold: float = 0.6,
) -> VerificationResult:
    """Independently challenge a finding; return whether it survives.

    Runs each skeptic (default: a single :class:`EvidenceQualitySkeptic`),
    collects their refutations, and re-grades confidence as the grounding score
    minus the refutation penalties — where non-fatal penalties are **capped**
    (:data:`_MAX_NONFATAL_PENALTY`) so volume of caveats alone can't refute a
    well-grounded finding; a single ``fatal`` refutation zeroes it outright. A
    finding survives only if nothing fatal was raised and confidence clears
    ``threshold``.

    Args:
        finding: A :class:`~tulip.security.findings.Evidence` or finding-shaped
            mapping (framework-agnostic).
        skeptics: The challenge panel; defaults to the deterministic skeptic.
            Plug semantic/LLM skeptics here.
        threshold: Minimum confidence to survive (default 0.6).

    Returns:
        A :class:`VerificationResult`.
    """
    panel: list[Skeptic] = list(skeptics) if skeptics is not None else [EvidenceQualitySkeptic()]
    refutations: list[Refutation] = []
    for skeptic in panel:
        refutations.extend(await skeptic.challenge(finding))

    base = _coerce(finding).gsar_score
    fatal = any(r.weight == "fatal" for r in refutations)
    nonfatal = sum(_PENALTY.get(r.weight, 0.2) for r in refutations if r.weight != "fatal")
    confidence = 0.0 if fatal else max(0.0, min(1.0, base - min(nonfatal, _MAX_NONFATAL_PENALTY)))

    survives = not fatal and confidence >= threshold
    notes = (
        "Survives independent challenge."
        if survives
        else "Refuted by independent challenge — do not act on this finding as-is."
    )
    return VerificationResult(
        survives=survives,
        confidence=confidence,
        evidence_quality=confidence,
        refutations=refutations,
        alternatives=[],
        notes=notes,
    )


__all__ = [
    "AdversarialSkeptic",
    "EvidenceQualitySkeptic",
    "FindingLike",
    "Refutation",
    "Skeptic",
    "VerificationResult",
    "verify",
]
