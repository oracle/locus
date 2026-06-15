# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""The red-team probe contract.

A :class:`Probe` runs one adversarial technique against a :class:`~tulip.security.target.Target`
and returns a :class:`ProbeOutcome`: a *candidate* finding plus the GSAR
:class:`~tulip.reasoning.gsar.Partition` of its evidence. The job runner
(:func:`tulip.security.red_team`) routes that partition through
:func:`tulip.security.ground_finding`, so a probe whose attack actually
landed (strong, tool-backed evidence) yields a :class:`~tulip.security.findings.Finding`,
while an inconclusive attempt (weak / inference-only evidence) **abstains**.
The probe never decides whether to ship — grounding does. That is what
keeps red-team output free of hallucinated vulnerabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from tulip.reasoning.gsar import Partition
from tulip.security.findings import Indicator
from tulip.security.target import Target
from tulip.security.taxonomy import Severity, TaxonomyTag


@dataclass(frozen=True)
class ProbeOutcome:
    """A candidate finding produced by a probe, pending grounding.

    ``partition`` carries the evidence: tool-backed claims when the attack
    landed (so it grounds), inference-only claims when it did not (so it
    abstains). ``transcript`` keeps the raw prompt/response pair for audit.
    """

    title: str
    description: str
    severity: Severity
    asset: str
    remediation: str
    partition: Partition
    taxonomy: list[TaxonomyTag] = field(default_factory=list)
    indicators: list[Indicator] = field(default_factory=list)
    transcript: list[str] = field(default_factory=list)


@runtime_checkable
class Probe(Protocol):
    """One adversarial technique, runnable against a :class:`Target`.

    ``name`` is a stable id (e.g. ``"direct-prompt-injection"``); ``taxonomy``
    is the primary OWASP/ATLAS tag the probe maps to. Both are read-only so a
    ``frozen`` dataclass satisfies the contract.
    """

    @property
    def name(self) -> str: ...

    @property
    def taxonomy(self) -> TaxonomyTag: ...

    async def run(self, target: Target) -> ProbeOutcome: ...


__all__ = ["Probe", "ProbeOutcome"]
