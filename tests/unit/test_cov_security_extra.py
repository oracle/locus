# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Coverage-gap tests for security modules.

Targets: fingerprint, scanner, edr, verify, soc, adapter, target, playbooks, context.
All paths tested offline or with httpx mocks (respx / MockTransport).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from tulip.security.adapter import indicator_type
from tulip.security.edr import (
    fetch_host_timeline,
    fetch_host_timeline_tool,
    isolate_host,
    isolate_host_tool,
    list_detections,
    list_detections_tool,
)
from tulip.security.fingerprint import (
    _SAMPLE_FEATURES,
    _has_content,
    _stream_once,
    default_classifier,
    dispatch_timing_probe_reference,
    fingerprint_endpoint_tool,
    measure_endpoint_timing,
)
from tulip.security.playbooks import (
    all_playbooks,
    cloud_posture_audit,
    nist_800_61_ir,
    phishing_triage,
    ransomware_containment,
)
from tulip.security.scanner import scan_dependencies_tool, scan_endpoint, scan_endpoint_tool
from tulip.security.soc import SecurityControls
from tulip.security.target import Target, _extract_text
from tulip.security.taxonomy import IndicatorType
from tulip.security.verify import (
    AdversarialSkeptic,
    EvidenceQualitySkeptic,
    FindingLike,
    verify,
)


# ---------------------------------------------------------------------------
# adapter.py — indicator_type branches (lines 84-90)
# ---------------------------------------------------------------------------


def test_indicator_type_domain() -> None:
    assert indicator_type("domain", "example.com") == IndicatorType.DOMAIN


def test_indicator_type_url() -> None:
    assert indicator_type("url", "https://x.com/p") == IndicatorType.URL


def test_indicator_type_hash_md5() -> None:
    assert indicator_type("hash", "a" * 32) == IndicatorType.MD5


def test_indicator_type_hash_sha256() -> None:
    assert indicator_type("hash", "b" * 64) == IndicatorType.SHA256


def test_indicator_type_host_fallback() -> None:
    assert indicator_type("unknown", "myserver") == IndicatorType.HOST


# ---------------------------------------------------------------------------
# fingerprint.py — _has_content (lines 150-154)
# ---------------------------------------------------------------------------


def test_has_content_valid_token() -> None:
    payload = json.dumps({"choices": [{"delta": {"content": "hello"}}]})
    assert _has_content(payload) is True


def test_has_content_empty_string_delta() -> None:
    payload = json.dumps({"choices": [{"delta": {"content": ""}}]})
    assert _has_content(payload) is False


def test_has_content_no_content_key() -> None:
    payload = json.dumps({"choices": [{"delta": {}}]})
    assert _has_content(payload) is False


def test_has_content_invalid_json() -> None:
    assert _has_content("not-json {{{{") is False


def test_has_content_missing_choices() -> None:
    assert _has_content(json.dumps({"other": "thing"})) is False


# ---------------------------------------------------------------------------
# fingerprint.py — dispatch_timing_probe_reference (line 173)
# ---------------------------------------------------------------------------


def test_dispatch_timing_probe_reference_returns_sample() -> None:
    result = dispatch_timing_probe_reference("http://example.com")
    assert set(result) >= {"ttft_ms_p50", "itl_ms_mean", "itl_cv", "tps_mean"}


def test_dispatch_timing_probe_reference_provider_kwarg() -> None:
    result = dispatch_timing_probe_reference("http://example.com", provider="lambda_cloud")
    assert result["tps_mean"] > 0.0


# ---------------------------------------------------------------------------
# fingerprint.py — default_classifier 70B+ branch (line 197)
# ---------------------------------------------------------------------------


def test_default_classifier_large_model_itl_over_25() -> None:
    features = {"ttft_ms_p50": 120.0, "itl_ms_mean": 30.0, "itl_cv": 0.2, "tps_mean": 15.0}
    verdict = default_classifier(features)
    assert "70B+" in verdict.model


# ---------------------------------------------------------------------------
# fingerprint.py — measure_endpoint_timing live path (lines 87-101)
# ---------------------------------------------------------------------------


def test_measure_timing_live_with_valid_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live path: _stream_once returns timing data → features computed from it."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def _fake(base_url: str, api_key: str, model: str, prompt: str) -> dict:
        return {"ttft_ms": 38.0, "itl_ms": [10.0, 12.0, 11.5], "tps": 90.0}

    monkeypatch.setattr("tulip.security.fingerprint._stream_once", _fake)
    result = measure_endpoint_timing(samples=3)
    assert "ttft_ms_p50" in result
    assert result["tps_mean"] > 0.0
    assert 0.0 <= result["itl_cv"] < 1.0


def test_measure_timing_live_all_failures_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live path: all _stream_once calls return None → falls back to sample."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("tulip.security.fingerprint._stream_once", lambda *a, **k: None)
    result = measure_endpoint_timing(samples=2)
    assert result["itl_ms_mean"] == _SAMPLE_FEATURES["itl_ms_mean"]


def test_measure_timing_live_custom_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """TIMING_BASE_URL is forwarded to _stream_once (line 87)."""
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("TIMING_BASE_URL", "https://custom.example.com/v1")
    captured: list[str] = []

    def _capture(base_url: str, api_key: str, model: str, prompt: str) -> dict:
        captured.append(base_url)
        return {"ttft_ms": 40.0, "itl_ms": [11.0, 13.0], "tps": 80.0}

    monkeypatch.setattr("tulip.security.fingerprint._stream_once", _capture)
    measure_endpoint_timing(samples=1)
    assert captured[0] == "https://custom.example.com/v1"


# ---------------------------------------------------------------------------
# fingerprint.py — _stream_once (lines 111-141)
# ---------------------------------------------------------------------------


@respx.mock
def test_stream_once_success_returns_timing() -> None:
    sse_body = (
        ": keep-alive\n"  # non-data → skipped
        'data: {"choices": [{"delta": {"content": "A"}}]}\n'
        'data: {"choices": [{"delta": {"content": "B"}}]}\n'
        'data: {"choices": [{"delta": {"content": "C"}}]}\n'
        "data: [DONE]"
    )
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=sse_body)
    )
    result = _stream_once("https://api.openai.com/v1", "test-key", "gpt-4o-mini", "hi")
    assert result is not None
    assert "ttft_ms" in result
    assert "itl_ms" in result
    assert result["tps"] >= 0.0


@respx.mock
def test_stream_once_http_error_returns_none() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="internal error")
    )
    result = _stream_once("https://api.openai.com/v1", "test-key", "gpt-4o-mini", "hi")
    assert result is None


@respx.mock
def test_stream_once_too_few_arrivals_returns_none() -> None:
    """Only 1 content token → len(arrivals) < 2 → return None."""
    sse_body = 'data: {"choices": [{"delta": {"content": "A"}}]}\ndata: [DONE]'
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=sse_body)
    )
    result = _stream_once("https://api.openai.com/v1", "test-key", "gpt-4o-mini", "hi")
    assert result is None


@respx.mock
def test_stream_once_no_content_tokens_returns_none() -> None:
    """No content tokens at all → first is None → return None."""
    sse_body = 'data: {"choices": [{"delta": {}}]}\ndata: [DONE]'
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=sse_body)
    )
    result = _stream_once("https://api.openai.com/v1", "test-key", "gpt-4o-mini", "hi")
    assert result is None


# ---------------------------------------------------------------------------
# fingerprint.py — fingerprint_endpoint_tool (lines 252-254)
# ---------------------------------------------------------------------------


async def test_fingerprint_endpoint_tool_returns_json() -> None:
    result = json.loads(await fingerprint_endpoint_tool("https://example.com"))
    assert result["endpoint"] == "https://example.com"
    assert "verdict" in result
    assert "features" in result


# ---------------------------------------------------------------------------
# scanner.py — _scan_live (lines 76-100) via scan_endpoint with SCANNER_LIVE
# ---------------------------------------------------------------------------


def _make_socket_mocks(not_after: str | None) -> tuple[MagicMock, MagicMock]:
    """Return (mock_ctx, mock_sock) for patching ssl/socket in _scan_live."""
    mock_tls = MagicMock()
    mock_tls.__enter__ = MagicMock(return_value=mock_tls)
    mock_tls.__exit__ = MagicMock(return_value=False)
    mock_tls.getpeercert.return_value = {"notAfter": not_after} if not_after else {}

    mock_ctx = MagicMock()
    mock_ctx.wrap_socket.return_value = mock_tls

    mock_sock = MagicMock()
    mock_sock.__enter__ = MagicMock(return_value=mock_sock)
    mock_sock.__exit__ = MagicMock(return_value=False)

    return mock_ctx, mock_sock


def test_scan_live_expired_cert(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCANNER_LIVE", "1")
    mock_ctx, mock_sock = _make_socket_mocks("Jan  1 00:00:00 2020 GMT")

    with (
        patch("socket.create_connection", return_value=mock_sock),
        patch("ssl.create_default_context", return_value=mock_ctx),
        patch("ssl.cert_time_to_seconds", return_value=0.0),
    ):
        result = scan_endpoint("198.51.100.50")

    assert result["source"] == "live-scan"
    assert result["open"] is True
    assert result["tls_not_after"] == "Jan  1 00:00:00 2020 GMT"
    assert result["tls_expired"] is True


def test_scan_live_no_not_after(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cert present but no notAfter field → tls_expired stays False."""
    monkeypatch.setenv("SCANNER_LIVE", "1")
    mock_ctx, mock_sock = _make_socket_mocks(None)

    with (
        patch("socket.create_connection", return_value=mock_sock),
        patch("ssl.create_default_context", return_value=mock_ctx),
    ):
        result = scan_endpoint("198.51.100.50")

    assert result["source"] == "live-scan"
    assert result["open"] is True
    assert result["tls_expired"] is False


def test_scan_live_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Socket refuses → OSError caught → open stays False."""
    monkeypatch.setenv("SCANNER_LIVE", "1")

    with (
        patch("socket.create_connection", side_effect=OSError("refused")),
        patch("ssl.create_default_context", return_value=MagicMock()),
    ):
        result = scan_endpoint("198.51.100.50")

    assert result["source"] == "live-scan"
    assert result["open"] is False
    assert result["tls_expired"] is False


# ---------------------------------------------------------------------------
# scanner.py — async tool wrappers (lines 145, 154)
# ---------------------------------------------------------------------------


async def test_scan_dependencies_tool_returns_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TULIP_MCP_SKIP_OSV", "1")
    result = json.loads(await scan_dependencies_tool("npx", ["lodash@4.17.21"]))
    assert "clean" in result
    assert result["command"] == "npx"


async def test_scan_endpoint_tool_returns_json() -> None:
    result = json.loads(await scan_endpoint_tool("192.0.2.10"))
    assert "tls_expired" in result


# ---------------------------------------------------------------------------
# edr.py — live path via _edr_get / _edr_post (lines 90, 100, 115, 121-142)
# ---------------------------------------------------------------------------


@respx.mock
def test_fetch_host_timeline_live_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDR_URL", "https://edr.example.com")
    monkeypatch.setenv("EDR_TOKEN", "tok")
    respx.get("https://edr.example.com/timeline").mock(
        return_value=httpx.Response(
            200, json={"events": [{"ts": "2026-01-01T00:00:00Z", "kind": "process"}]}
        )
    )
    result = fetch_host_timeline("WS-0142")
    assert result["source"] == "edr"
    assert "events" in result


@respx.mock
def test_list_detections_live_no_host_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDR_URL", "https://edr.example.com")
    monkeypatch.setenv("EDR_TOKEN", "tok")
    respx.get("https://edr.example.com/detections").mock(
        return_value=httpx.Response(200, json={"detections": []})
    )
    result = list_detections(None)
    assert result["source"] == "edr"


@respx.mock
def test_list_detections_live_with_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDR_URL", "https://edr.example.com")
    monkeypatch.setenv("EDR_TOKEN", "tok")
    respx.get("https://edr.example.com/detections").mock(
        return_value=httpx.Response(200, json={"detections": [{"id": "d1"}]})
    )
    result = list_detections("WS-0142")
    assert result["source"] == "edr"


@respx.mock
def test_isolate_host_live_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDR_URL", "https://edr.example.com")
    monkeypatch.setenv("EDR_TOKEN", "tok")
    respx.post("https://edr.example.com/devices/actions/contain").mock(
        return_value=httpx.Response(200, json={"status": "contained"})
    )
    result = isolate_host("HOST-001")
    assert result["source"] == "edr"
    assert result["status"] == "contained"


# ---------------------------------------------------------------------------
# edr.py — async tool wrappers (lines 151, 157, 167)
# ---------------------------------------------------------------------------


async def test_fetch_host_timeline_tool_returns_json() -> None:
    result = json.loads(await fetch_host_timeline_tool("WS-0142"))
    assert "host" in result
    assert result["host"] == "WS-0142"


async def test_list_detections_tool_empty_host_uses_none() -> None:
    result = json.loads(await list_detections_tool())
    assert "count" in result


async def test_isolate_host_tool_offline_simulated() -> None:
    result = json.loads(await isolate_host_tool("HOST-001"))
    assert "status" in result


# ---------------------------------------------------------------------------
# verify.py — _parse_severity branches (lines 92, 96-101)
# ---------------------------------------------------------------------------


async def test_verify_mapping_with_severity_enum() -> None:
    """Line 92: _parse_severity receives a Severity instance directly."""
    from tulip.security.taxonomy import Severity

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

    from tulip.security.findings import Evidence
    from tulip.security.taxonomy import Severity

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

    from tulip.security.findings import Evidence
    from tulip.security.taxonomy import Severity

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
# soc.py — soc_triage() and tools() branches (lines 182, 197-207)
# ---------------------------------------------------------------------------


def test_soc_triage_factory_returns_controls() -> None:
    """Line 182: soc_triage() return statement."""
    controls = SecurityControls.soc_triage()
    assert controls.threat_intel is True
    assert controls.siem is True
    assert controls.edr is True
    assert controls.scanner is True
    assert controls.fingerprint is True
    assert controls.readonly_aws is False
    assert controls.allow_containment is False


def test_soc_triage_tools_include_all_soc_tools() -> None:
    """Lines 197, 199, 201, 205, 207: all non-AWS branches active."""
    names = {t.name for t in SecurityControls.soc_triage().tools()}
    assert "enrich_indicator" in names  # threat_intel
    assert "query_siem" in names  # siem
    assert "fetch_host_timeline" in names  # edr
    assert "list_detections" in names  # edr
    assert "scan_dependencies" in names  # scanner
    assert "scan_endpoint" in names  # scanner
    assert "fingerprint_endpoint" in names  # fingerprint
    assert "isolate_host" not in names  # allow_containment=False


def test_tools_with_containment_enabled() -> None:
    """Lines 202-203: allow_containment=True adds isolate_host."""
    controls = SecurityControls(edr=True, allow_containment=True)
    names = {t.name for t in controls.tools()}
    assert "isolate_host" in names
    assert "fetch_host_timeline" in names


# ---------------------------------------------------------------------------
# target.py — _extract_text missing branches (lines 56, 59, 73-74, 149-150)
# ---------------------------------------------------------------------------


def test_extract_text_path_navigation_fallback() -> None:
    """Line 56: path segment missing in nested dict → as_json(body) fallback."""
    body = {"a": {"b": "val"}}
    # "a.c" → body["a"] exists but body["a"]["c"] doesn't → else branch
    result = _extract_text(body, "a.c")
    assert "a" in result  # returns raw JSON of body


def test_extract_text_body_is_string() -> None:
    """Line 59: body is already a string (no path) → return it directly."""
    assert _extract_text("plain text response", None) == "plain text response"


def test_extract_text_openai_text_field() -> None:
    """Lines 73-74: choices[0] has a 'text' key (completions v1 shape)."""
    body = {"choices": [{"text": "completion-result"}]}
    result = _extract_text(body, None)
    assert result == "completion-result"


def test_extract_text_choices_first_not_mapping() -> None:
    """69->75 branch: choices[0] is not a Mapping → fall to as_json."""
    body = {"choices": ["not-a-dict"]}
    result = _extract_text(body, None)
    assert "choices" in result


def test_extract_text_body_not_mapping_or_string() -> None:
    """60->75 branch: body is an int (not str, not Mapping) → as_json."""
    result = _extract_text(42, None)
    assert result == "42"


async def test_endpoint_target_non_json_response_returns_text() -> None:
    """Lines 149-150: resp.json() raises ValueError → return resp.text."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not valid json <<<")

    target = Target.endpoint(
        "https://bot.example/chat",
        transport=httpx.MockTransport(handler),
    )
    result = await target.send("hello")
    assert result == "not valid json <<<"


# ---------------------------------------------------------------------------
# playbooks.py — factory functions (lines 26, 64, 106, 141, 171)
# ---------------------------------------------------------------------------


def test_phishing_triage_playbook() -> None:
    pb = phishing_triage()
    assert pb.id == "phishing_triage"
    assert any(s.id == "gather" for s in pb.steps)


def test_nist_800_61_ir_playbook() -> None:
    pb = nist_800_61_ir()
    assert pb.id == "nist_800_61_ir"
    assert len(pb.steps) >= 3


def test_ransomware_containment_playbook() -> None:
    pb = ransomware_containment()
    assert pb.id == "ransomware_containment"
    assert any("isolate_host" in s.expected_tools for s in pb.steps)


def test_cloud_posture_audit_playbook() -> None:
    pb = cloud_posture_audit()
    assert pb.id == "cloud_posture_audit"
    assert any("describe_aws" in s.expected_tools for s in pb.steps)


def test_all_playbooks_keyed_by_id() -> None:
    pbs = all_playbooks()
    assert set(pbs) == {
        "phishing_triage",
        "nist_800_61_ir",
        "ransomware_containment",
        "cloud_posture_audit",
    }
    for pb_id, pb in pbs.items():
        assert pb.id == pb_id


# ---------------------------------------------------------------------------
# context.py — missing provider methods (lines 133, 178, 182, 192-201)
# ---------------------------------------------------------------------------


async def test_ref_endpoint_isolate() -> None:
    """Line 133: _RefEndpoint.isolate."""
    from tulip.security import SecurityContext

    ctx = SecurityContext()
    result = await ctx.endpoint.isolate("WS-0142")
    assert "status" in result  # offline-sample returns contained (simulated)


async def test_ref_identity_signins() -> None:
    """Line 178: _RefIdentity.signins."""
    from tulip.security import SecurityContext

    ctx = SecurityContext()
    result = await ctx.identity.signins("jsmith@example.com")
    assert "signins" in result
    assert result["user"] == "jsmith@example.com"


async def test_ref_identity_disable() -> None:
    """Line 182: _RefIdentity.disable (offline simulated receipt)."""
    from tulip.security import SecurityContext

    ctx = SecurityContext()
    result = await ctx.identity.disable("jsmith@example.com")
    assert result["disabled"] is True
    assert result["source"] == "offline-sample"


async def test_ref_cloud_describe_exception_degrades_gracefully() -> None:
    """Lines 192-193: describe_aws raises → returns degraded dict."""
    from tulip.security.context import _RefCloud

    cloud = _RefCloud()
    with patch("tulip.security.context.describe_aws", side_effect=RuntimeError("no boto3")):
        result = await cloud.describe(service="iam")
    assert result["available"] is False
    assert "no boto3" in result["reason"]


async def test_ref_cloud_events_success() -> None:
    """Lines 198-199: use_aws returns a dict → forwarded."""
    from tulip.security.context import _RefCloud

    cloud = _RefCloud()
    fake_response = {"ResponseMetadata": {}, "Items": []}
    with patch("tulip.security.context.use_aws", return_value=fake_response):
        result = await cloud.events("iam", "GetAccountSummary")
    assert result == fake_response


async def test_ref_cloud_events_exception_degrades_gracefully() -> None:
    """Lines 200-201: use_aws raises → returns degraded dict."""
    from tulip.security.context import _RefCloud

    cloud = _RefCloud()
    with patch("tulip.security.context.use_aws", side_effect=RuntimeError("no creds")):
        result = await cloud.events("iam", "GetAccountSummary")
    assert result["available"] is False
    assert "no creds" in result["reason"]
