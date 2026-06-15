# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Verification — the independent challenge that prevents security hallucinations.

A `Finding` is a *claim*. Before it drives an action or reaches an analyst,
:func:`verify` subjects it to an independent **skeptic** whose job is to *refute*
it — challenge the assumptions, weigh the evidence quality, and flag any
inconsistency between what is claimed and what is supported. The result is a
:class:`Verdict`: ``survives`` (with a confidence), or refuted, with the
refutations recorded.

This is the question generic agent frameworks don't answer — *"how do I know the
agent is right?"* — and it works on **any** finding, not just Tulip's: pass a
:class:`~tulip.security.findings.Finding` or a finding-shaped mapping from an
external agent (LangGraph/CrewAI/anything), so Tulip can sit above the stack as
the verification layer.

The bundled :class:`EvidenceQualitySkeptic` is deterministic and offline — it
grades the evidence a finding carries. Richer *semantic* skeptics (search for
contradictory evidence, propose alternative explanations) plug in through the
same :class:`Skeptic` protocol via ``verify(finding, skeptics=[...])``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from tulip.security.findings import Finding
from tulip.security.taxonomy import Severity, severity_at_least


# A finding to verify: Tulip's typed Finding, or a finding-shaped mapping from
# any other agent/framework (keys: title, severity, gsar_score, evidence_refs,
# confidence).
FindingLike = Finding | Mapping[str, Any]

# The GSAR proceed threshold (Appendix B) — a grounded Finding cleared this.
_PROCEED_THRESHOLD = 0.80


@dataclass(frozen=True)
class Refutation:
    """One objection a skeptic raises. ``weight`` ∈ {weak, concern, fatal}."""

    reason: str
    weight: str = "concern"


@dataclass(frozen=True)
class Verdict:
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
    """Normalised view over a Finding or a finding-shaped mapping."""

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
    if isinstance(finding, Finding):
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


_PENALTY = {"fatal": 1.0, "concern": 0.2, "weak": 0.1}


async def verify(
    finding: FindingLike,
    *,
    skeptics: Sequence[Skeptic] | None = None,
    threshold: float = 0.6,
) -> Verdict:
    """Independently challenge a finding; return whether it survives.

    Runs each skeptic (default: a single :class:`EvidenceQualitySkeptic`),
    collects their refutations, and re-grades the finding's confidence from its
    grounding score minus the refutation penalties. A finding survives only if
    nothing fatal was raised and confidence clears ``threshold``.

    Args:
        finding: A :class:`~tulip.security.findings.Finding` or finding-shaped
            mapping (framework-agnostic).
        skeptics: The challenge panel; defaults to the deterministic skeptic.
            Plug semantic/LLM skeptics here.
        threshold: Minimum confidence to survive (default 0.6).

    Returns:
        A :class:`Verdict`.
    """
    panel: list[Skeptic] = list(skeptics) if skeptics is not None else [EvidenceQualitySkeptic()]
    refutations: list[Refutation] = []
    for skeptic in panel:
        refutations.extend(await skeptic.challenge(finding))

    confidence = _coerce(finding).gsar_score
    fatal = False
    for r in refutations:
        penalty = _PENALTY.get(r.weight, 0.2)
        if r.weight == "fatal":
            fatal = True
            confidence = 0.0
        else:
            confidence -= penalty
    confidence = max(0.0, min(1.0, confidence))

    survives = not fatal and confidence >= threshold
    notes = (
        "Survives independent challenge."
        if survives
        else "Refuted by independent challenge — do not act on this finding as-is."
    )
    return Verdict(
        survives=survives,
        confidence=confidence,
        evidence_quality=confidence,
        refutations=refutations,
        alternatives=[],
        notes=notes,
    )


__all__ = [
    "EvidenceQualitySkeptic",
    "FindingLike",
    "Refutation",
    "Skeptic",
    "Verdict",
    "verify",
]
