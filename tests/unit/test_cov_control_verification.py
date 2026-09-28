# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Coverage-gap tests for :mod:`tulip.control.verification`.

Edge-case branches of ``verify`` and the bundled skeptics, all offline.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from tulip.control.verification import (
    AdversarialSkeptic,
    EvidenceQualitySkeptic,
    FindingLike,
    verify,
)


# ---------------------------------------------------------------------------
# verify.py — _parse_severity branches (lines 92, 96-101)
# ---------------------------------------------------------------------------


async def test_verify_mapping_with_severity_enum() -> None:
    """Line 92: _parse_severity receives a Severity instance directly."""
    from tulip.control.taxonomy import Severity

    finding: FindingLike = {
        "title": "test",
        "severity": Severity.HIGH,
        "gsar_score": 1.0,
        "evidence_refs": ["r1", "r2"],
        "confidence": 1.0,
    }
    result = await verify(finding)
    # severity is recognised as HIGH → single-ref concern fires when only 1 ref, here 2 so no concern
    assert isinstance(result.confidence, float)


async def test_verify_mapping_with_bad_severity_string() -> None:
    """Lines 96-100: _parse_severity tries Severity("BADVAL") → ValueError,
    then Severity["BADVAL"] → KeyError → returns None."""
    finding: FindingLike = {
        "title": "test",
        "severity": "NOTAVALIDONE",
        "gsar_score": 1.0,
        "evidence_refs": ["r1", "r2"],
        "confidence": 1.0,
    }
    result = await verify(finding)
    assert isinstance(result.confidence, float)


async def test_verify_mapping_with_none_severity() -> None:
    """Line 101: severity is not a str or Severity → _parse_severity returns None."""
    finding: FindingLike = {
        "title": "test",
        "gsar_score": 1.0,
        "evidence_refs": ["r1", "r2"],
        "confidence": 1.0,
    }
    result = await verify(finding)
    assert isinstance(result.confidence, float)


# ---------------------------------------------------------------------------
# verify.py — EvidenceQualitySkeptic confidence-overreach branch (line 169)
# ---------------------------------------------------------------------------


async def test_verify_confidence_much_higher_than_gsar_raises_weak() -> None:
    """Line 169: confidence - gsar_score > 0.25 → weak refutation appended."""
    finding: FindingLike = {
        "title": "Overconfident finding",
        "severity": "medium",
        "gsar_score": 0.5,
        "evidence_refs": ["r1"],
        "confidence": 0.9,  # 0.9 - 0.5 = 0.4 > 0.25
    }
    result = await verify(finding, skeptics=[EvidenceQualitySkeptic()])
    assert any("materially exceeds" in r.reason for r in result.refutations), result.refutations


# ---------------------------------------------------------------------------
# verify.py — AdversarialSkeptic with string model (lines 280-282)
# ---------------------------------------------------------------------------


async def test_adversarial_skeptic_string_model_resolved() -> None:
    """Lines 280-282: _resolve() sees a str → calls get_model → swaps it."""
    review = json.dumps({"supported": True, "objections": [], "alternatives": []})

    class _FakeModel:
        async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001, ANN003
            from tulip.core.messages import Message
            from tulip.models.base import ModelResponse

            return ModelResponse(message=Message.assistant(content=review), usage={})

    with patch("tulip.models.registry.get_model", return_value=_FakeModel()):
        skeptic = AdversarialSkeptic("anthropic:mock")
        finding: FindingLike = {
            "title": "t",
            "severity": "medium",
            "gsar_score": 0.9,
            "evidence_refs": ["r1"],
            "confidence": 0.9,
        }
        refs = await skeptic.challenge(finding)
    assert refs == []


# ---------------------------------------------------------------------------
# verify.py — AdversarialSkeptic parse failure (line 311)
# ---------------------------------------------------------------------------


async def test_adversarial_skeptic_unparseable_output_is_weak() -> None:
    """Line 311: parse_structured fails (missing required 'supported') → weak refutation."""

    class _BadModel:
        async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001, ANN003
            from tulip.core.messages import Message
            from tulip.models.base import ModelResponse

            # {} is valid JSON but fails _AdversarialReview validation (missing 'supported')
            return ModelResponse(message=Message.assistant(content="{}"), usage={})

    from tulip.control.findings import Evidence
    from tulip.control.taxonomy import Severity

    finding = Evidence(
        title="test",
        description="",
        severity=Severity.MEDIUM,
        asset="a",
        remediation="r",
        gsar_score=0.9,
        confidence=0.9,
        evidence_refs=["r1"],
    )
    skeptic = AdversarialSkeptic(model=_BadModel())
    refs = await skeptic.challenge(finding)
    assert any("could not be parsed" in r.reason for r in refs)
    assert all(r.weight == "weak" for r in refs)


# ---------------------------------------------------------------------------
# verify.py — AdversarialSkeptic line 329 (supported=False, no concern/fatal)
# ---------------------------------------------------------------------------


async def test_adversarial_skeptic_unsupported_no_objections_appends_concern() -> None:
    """Line 329: review.supported=False with empty objections/alternatives → concern added."""
    review = json.dumps({"supported": False, "objections": [], "alternatives": []})

    class _UnsupportedModel:
        async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001, ANN003
            from tulip.core.messages import Message
            from tulip.models.base import ModelResponse

            return ModelResponse(message=Message.assistant(content=review), usage={})

    from tulip.control.findings import Evidence
    from tulip.control.taxonomy import Severity

    finding = Evidence(
        title="test",
        description="desc",
        severity=Severity.MEDIUM,
        asset="a",
        remediation="r",
        gsar_score=0.9,
        confidence=0.9,
        evidence_refs=["r1"],
    )
    skeptic = AdversarialSkeptic(model=_UnsupportedModel())
    refs = await skeptic.challenge(finding)
    assert any("not fully supported" in r.reason for r in refs)


# ---------------------------------------------------------------------------
# verify.py — _describe branch for non-Evidence finding (238->240 branch miss)
# ---------------------------------------------------------------------------


async def test_adversarial_skeptic_with_mapping_finding_no_description() -> None:
    """238->240 branch: finding is a dict (not Evidence) → description block skipped."""
    review = json.dumps({"supported": True, "objections": [], "alternatives": []})

    class _Model:
        async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001, ANN003
            from tulip.core.messages import Message
            from tulip.models.base import ModelResponse

            return ModelResponse(message=Message.assistant(content=review), usage={})

    skeptic = AdversarialSkeptic(model=_Model())
    finding: FindingLike = {
        "title": "no-description-finding",
        "severity": "low",
        "gsar_score": 0.85,
        "evidence_refs": ["r1"],
        "confidence": 0.85,
    }
    refs = await skeptic.challenge(finding)
    assert refs == []


# ---------------------------------------------------------------------------
# verify.py — edge-case branches not in extra.py
# ---------------------------------------------------------------------------


async def test_evidence_quality_skeptic_no_refs_fatal() -> None:
    """Line 148: evidence_refs empty → fatal refutation."""
    finding = {
        "title": "unsupported",
        "gsar_score": 0.9,
        "evidence_refs": [],
        "confidence": 0.9,
    }
    refs = await EvidenceQualitySkeptic().challenge(finding)
    assert any(r.weight == "fatal" for r in refs)


async def test_evidence_quality_skeptic_high_single_ref_concern() -> None:
    """Line 162: HIGH severity + 1 ref → concern appended."""
    from tulip.control.findings import Evidence
    from tulip.control.taxonomy import Severity

    finding = Evidence(
        title="cert expired",
        description="cert",
        severity=Severity.HIGH,
        asset="host:443",
        remediation="rotate",
        gsar_score=1.0,
        confidence=1.0,
        evidence_refs=["tool:scan:tls"],  # only one
    )
    refs = await EvidenceQualitySkeptic().challenge(finding)
    assert any("single evidence reference" in r.reason for r in refs)


async def test_adversarial_skeptic_model_error_fails_safe() -> None:
    """Lines 298-299: model.complete raises → fail-safe weak refutation."""

    class _BoomModel:
        async def complete(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
            raise RuntimeError("provider down")

    from tulip.control.findings import Evidence
    from tulip.control.taxonomy import Severity

    finding = Evidence(
        title="t",
        description="d",
        severity=Severity.LOW,
        asset="a",
        remediation="r",
        gsar_score=0.9,
        confidence=0.9,
        evidence_refs=["r1"],
    )
    skeptic = AdversarialSkeptic(model=_BoomModel())
    refs = await skeptic.challenge(finding)
    assert any("not independently challenged" in r.reason for r in refs)
    assert all(r.weight == "weak" for r in refs)


# ---------------------------------------------------------------------------
