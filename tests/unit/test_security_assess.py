# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Tests for assurance assessments — grounded guardrail-coverage posture."""

from __future__ import annotations

from tulip.security import (
    OwaspLLM,
    Severity,
    Target,
    assure,
    guardrail_coverage,
    is_finding,
)


def _vulnerable() -> Target:
    return Target.from_callable(lambda p: p, name="vulnerable-bot")


def _hardened() -> Target:
    return Target.from_callable(lambda _p: "I can't help with that.", name="hardened-bot")


def _partial() -> Target:
    # Leaks only for the direct-injection payload ("maintenance mode"); resists
    # the rest. -> 4/5 resisted = 80% coverage, one LLM01 gap.
    return Target.from_callable(
        lambda p: p if "maintenance mode" in p else "denied", name="partial-bot"
    )


async def test_coverage_critical_on_vulnerable_target() -> None:
    result = await guardrail_coverage(_vulnerable(), suite="owasp-asi")
    assert is_finding(result)
    assert result.severity is Severity.CRITICAL
    assert "0%" in result.title or "resisted 0/" in result.title
    assert result.taxonomy  # every gap is recorded
    assert result.confidence == 0.0  # zero coverage


async def test_coverage_info_on_hardened_target() -> None:
    result = await guardrail_coverage(_hardened(), suite="owasp-asi")
    assert is_finding(result)
    assert result.severity is Severity.INFO
    assert result.taxonomy == []  # no gaps
    assert result.confidence == 1.0  # full coverage
    assert "No gaps" in result.description


async def test_coverage_partial_is_medium_with_named_gap() -> None:
    result = await guardrail_coverage(_partial(), suite="owasp-asi")
    assert is_finding(result)
    assert result.severity is Severity.MEDIUM
    assert OwaspLLM.PROMPT_INJECTION in result.taxonomy
    assert 0.0 < result.confidence < 1.0


async def test_assure_returns_posture_finding() -> None:
    results = await assure(_hardened())
    assert len(results) == 1
    assert is_finding(results[0])
    assert results[0].severity is Severity.INFO


async def test_coverage_is_grounded_in_observations() -> None:
    result = await guardrail_coverage(_vulnerable(), suite="owasp-asi")
    assert is_finding(result)
    # one evidence ref per probe observed
    assert all("assess:guardrail-coverage" in ref for ref in result.evidence_refs)
    assert len(result.evidence_refs) == 5
