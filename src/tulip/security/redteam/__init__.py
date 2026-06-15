# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Red-team probe library — adversarial techniques run against a target AI.

Probes implement the :class:`~tulip.security.redteam.base.Probe` contract and
are grouped into named *suites* (taxonomy collections). The job runner
:func:`tulip.security.red_team` resolves a suite to its probes, runs each
against the :class:`~tulip.security.target.Target`, and grounds the outcomes.
"""

from __future__ import annotations

from tulip.security.redteam.base import Probe, ProbeOutcome
from tulip.security.redteam.probes import DirectPromptInjection


# Named suites → the probes they run. Stage 1 seeds both suites with the
# reference probe; the catalogue grows here without touching the runner.
_SUITES: dict[str, list[Probe]] = {
    "owasp-asi": [DirectPromptInjection()],
    "owasp-llm": [DirectPromptInjection()],
}


def suite_probes(suite: str) -> list[Probe]:
    """Return a fresh probe list for a named suite, or raise on an unknown name."""
    try:
        return list(_SUITES[suite])
    except KeyError:
        known = ", ".join(sorted(_SUITES))
        raise ValueError(f"unknown red-team suite {suite!r}; known suites: {known}") from None


def all_probes() -> list[Probe]:
    """Every distinct bundled probe, across suites (de-duplicated by name)."""
    seen: dict[str, Probe] = {}
    for probes in _SUITES.values():
        for probe in probes:
            seen.setdefault(probe.name, probe)
    return list(seen.values())


__all__ = [
    "DirectPromptInjection",
    "Probe",
    "ProbeOutcome",
    "all_probes",
    "suite_probes",
]
