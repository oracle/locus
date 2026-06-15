# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Tests for red-team probes + the ``red_team`` job.

The defining behaviour: a vulnerable target yields a grounded Finding; a
hardened target yields an Abstention. Red-team output never fabricates a
vulnerability that the evidence does not support.
"""

from __future__ import annotations

import pytest

from tulip.security import (
    DirectPromptInjection,
    OwaspLLM,
    Severity,
    Target,
    all_probes,
    is_finding,
    red_team,
    suite_probes,
)


def _vulnerable() -> Target:
    # Echoes whatever it is sent — so it parrots back the injection canary.
    return Target.from_callable(lambda p: p, name="vulnerable-bot")


def _hardened() -> Target:
    return Target.from_callable(lambda _p: "I can't help with that request.", name="hardened-bot")


async def test_red_team_grounds_finding_on_vulnerable_target() -> None:
    results = await red_team(_vulnerable(), suite="owasp-asi")
    assert len(results) == 1
    finding = results[0]
    assert is_finding(finding)
    assert finding.severity is Severity.HIGH
    assert OwaspLLM.PROMPT_INJECTION in finding.taxonomy
    assert finding.gsar_score > 0.0
    assert finding.asset == "vulnerable-bot"
    assert finding.evidence_refs  # carries the probe evidence refs


async def test_red_team_abstains_on_hardened_target() -> None:
    results = await red_team(_hardened(), suite="owasp-asi")
    assert len(results) == 1
    assert not is_finding(results[0])  # an Abstention
    assert "withheld" in results[0].reason


async def test_red_team_accepts_explicit_probes() -> None:
    results = await red_team(_vulnerable(), probes=[DirectPromptInjection()])
    assert is_finding(results[0])


async def test_unknown_suite_raises() -> None:
    with pytest.raises(ValueError, match="unknown red-team suite"):
        await red_team(_hardened(), suite="does-not-exist")


def test_suite_probes_and_all_probes() -> None:
    assert any(p.name == "direct-prompt-injection" for p in suite_probes("owasp-asi"))
    assert any(p.name == "direct-prompt-injection" for p in all_probes())


async def test_direct_probe_outcome_shape() -> None:
    outcome = await DirectPromptInjection().run(_vulnerable())
    assert outcome.taxonomy == [OwaspLLM.PROMPT_INJECTION]
    assert len(outcome.transcript) == 2  # payload + response
    assert outcome.partition.grounded  # canary leaked -> tool-backed evidence
