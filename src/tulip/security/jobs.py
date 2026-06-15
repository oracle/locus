# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""The job verbs — what a cybersecurity agent does to a target AI.

Each verb takes a :class:`~tulip.security.target.Target` and returns a list
of :data:`~tulip.security.grounded.GroundedFinding` (each a
:class:`~tulip.security.findings.Finding` or an
:class:`~tulip.security.grounded.Abstention`). Grounding is applied here, in
the runner, not in the probes — so the abstain-by-construction guarantee
holds for every job uniformly.

Stage 1 implements :func:`red_team`. :func:`assure` and :func:`monitor`
arrive in later stages and currently raise :class:`NotImplementedError`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from tulip.reasoning.gsar import GSARThresholds
from tulip.security.grounded import GroundedFinding, ground_finding
from tulip.security.redteam import Probe, suite_probes
from tulip.security.target import Target


async def red_team(
    target: Target,
    *,
    suite: str = "owasp-asi",
    probes: Sequence[Probe] | None = None,
    thresholds: GSARThresholds | None = None,
) -> list[GroundedFinding]:
    """Red-team a target AI: run adversarial probes, return grounded findings.

    Runs each probe in ``probes`` (default: the named ``suite``) against
    ``target`` and grounds its outcome. A probe whose attack landed yields a
    :class:`~tulip.security.findings.Finding`; an inconclusive one yields an
    :class:`~tulip.security.grounded.Abstention`. Both are returned, in probe
    order, so the caller has a complete, auditable record of what was asserted
    *and* what was declined.

    Args:
        target: The AI system under assessment.
        suite: Named probe suite to run when ``probes`` is not given
            (``"owasp-asi"`` / ``"owasp-llm"``).
        probes: Explicit probes to run, overriding ``suite``.
        thresholds: Optional GSAR threshold override for grounding.

    Returns:
        One :data:`~tulip.security.grounded.GroundedFinding` per probe.
    """
    selected = list(probes) if probes is not None else suite_probes(suite)
    results: list[GroundedFinding] = []
    for probe in selected:
        outcome = await probe.run(target)
        results.append(
            ground_finding(
                title=outcome.title,
                description=outcome.description,
                severity=outcome.severity,
                asset=outcome.asset,
                remediation=outcome.remediation,
                partition=outcome.partition,
                indicators=outcome.indicators,
                taxonomy=outcome.taxonomy,
                thresholds=thresholds,
            )
        )
    return results


async def assure(target: Target) -> list[GroundedFinding]:
    """Assess a target AI's posture (AI-BOM, guardrail coverage, fingerprint).

    Not yet implemented — arrives in the assurance stage.
    """
    raise NotImplementedError("assure() is implemented in a later stage")


async def monitor(target: Target) -> AsyncIterator[GroundedFinding]:
    """Watch a live target AI for attacks/anomalies (tamper-evident trail).

    Not yet implemented — a later, optional supporting capability.
    """
    raise NotImplementedError("monitor() is implemented in a later stage")
    yield  # pragma: no cover - makes this an async generator


__all__ = ["assure", "monitor", "red_team"]
