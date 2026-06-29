# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Second coverage-gap file for security modules.

Covers lines missed when test_cov_security_extra.py runs in isolation:
adapter helpers, fingerprint branches, scanner finding, soc factory,
target shapes, context providers, verify edge cases.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
import respx

# ---------------------------------------------------------------------------
# adapter.py — helpers not tested in extra.py (lines 68, 77, 83, 132)
# ---------------------------------------------------------------------------
from tulip.security.adapter import (
    ToolAdapter,
    indicator_type,
    inference_claim,
    tool_match,
)
from tulip.security.taxonomy import IndicatorType


def test_tool_match_returns_claim() -> None:
    """Line 68: tool_match return statement."""
    claim = tool_match("host has 2 open ports", "tool:scan:host:443")
    assert claim.text == "host has 2 open ports"
    assert len(claim.evidence_refs) == 1


def test_inference_claim_returns_claim() -> None:
    """Line 77: inference_claim return statement."""
    claim = inference_claim("attacker may pivot", "inference:host")
    assert "pivot" in claim.text


def test_indicator_type_ip_branch() -> None:
    """Line 83: indicator_type 'ip' → IndicatorType.IP."""
    assert indicator_type("ip", "198.51.100.1") == IndicatorType.IP


def test_tool_adapter_tools_returns_list() -> None:
    """Line 132: ToolAdapter.tools() return."""
    from tulip.tools.decorator import tool

    @tool
    def dummy_tool(x: str) -> str:
        """A dummy tool."""
        return x

    adapter = ToolAdapter(name="test", vendor="Test Vendor", _tools=[dummy_tool])
    tools = adapter.tools()
    assert len(tools) == 1
    assert tools[0].name == "dummy_tool"


# ---------------------------------------------------------------------------
# fingerprint.py — remaining branches
# ---------------------------------------------------------------------------

from tulip.security.fingerprint import (
    _SAMPLE_FEATURES,
    _stream_once,
    default_classifier,
    fingerprint_to_finding,
)


def test_default_classifier_medium_model_no_itl() -> None:
    """Line 195 (13-34B class): itl=0 (missing key) → elif 0 < 25 → True."""
    features = {"ttft_ms_p50": 50.0}  # no itl_ms_mean → itl=0.0
    verdict = default_classifier(features)
    assert "13-34B" in verdict.model


@respx.mock
def test_stream_once_empty_response_body_returns_none() -> None:
    """Branch 124->137: empty body → for loop never executes → first is None → None."""
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, text="")
    )
    result = _stream_once("https://api.openai.com/v1", "key", "gpt-4o-mini", "hi")
    assert result is None


def test_fingerprint_to_finding_full_coverage_ships() -> None:
    """Lines 225-230: all 4 features present → coverage >= min_coverage → Finding."""
    from tulip.security import is_finding

    result = fingerprint_to_finding(dict(_SAMPLE_FEATURES), asset="203.0.113.10:443")
    assert is_finding(result)


def test_fingerprint_to_finding_thin_abstains() -> None:
    """Lines 231-234: 1/4 features → coverage < min_coverage → Abstention."""
    from tulip.security import is_finding

    result = fingerprint_to_finding({"ttft_ms_p50": 38.2}, asset="203.0.113.10:443")
    assert not is_finding(result)


def test_fingerprint_to_finding_custom_classifier() -> None:
    """Line 225 classifier branch: non-None classifier is used."""
    from tulip.security import is_finding
    from tulip.security.findings import FingerprintVerdict

    def _always_small(features):
        return FingerprintVerdict(
            model="7-8B class",
            engine="test-engine",
            hardware="H100",
            confidence=0.95,
            feature_coverage=1.0,
        )

    result = fingerprint_to_finding(
        dict(_SAMPLE_FEATURES), asset="host:443", classifier=_always_small
    )
    assert is_finding(result)
    assert result.verdict.model == "7-8B class"


# ---------------------------------------------------------------------------
# scanner.py — scan_endpoint_to_finding (lines 110-127)
# ---------------------------------------------------------------------------

from tulip.security.scanner import scan_endpoint_to_finding


def test_scan_endpoint_to_finding_expired_cert_ships() -> None:
    """Lines 110-120: expired cert → grounded partition → Finding."""
    from tulip.security import is_finding
    from tulip.security.taxonomy import Severity

    result = scan_endpoint_to_finding("192.0.2.10")
    assert is_finding(result)
    assert result.severity == Severity.HIGH


def test_scan_endpoint_to_finding_valid_cert_abstains() -> None:
    """Lines 121-127: valid cert → ungrounded partition → Abstention."""
    from tulip.security import is_finding

    result = scan_endpoint_to_finding("198.51.100.5")
    assert not is_finding(result)


# ---------------------------------------------------------------------------
# verify.py — edge-case branches not in extra.py
# ---------------------------------------------------------------------------

from tulip.security.verify import AdversarialSkeptic, EvidenceQualitySkeptic


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
    from tulip.security.findings import Evidence
    from tulip.security.taxonomy import Severity

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

    from tulip.security.findings import Evidence
    from tulip.security.taxonomy import Severity

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
# soc.py — submit_posture, default(), ground_report, create_soc_analyst
# ---------------------------------------------------------------------------

from tulip.security import (
    PostureEvidence,
    PostureFinding,
    PostureReport,
    SecurityControls,
    create_soc_analyst,
    ground_report,
    is_finding,
    submit_posture,
)
from tulip.security.taxonomy import Severity


def test_submit_posture_formats_message() -> None:
    """Line 136: submit_posture return statement."""
    report = PostureReport(summary="clean", confidence=0.83, findings=[])
    msg = submit_posture(report)
    assert "0 finding" in msg
    assert "83%" in msg


def test_security_controls_default_returns_aws_toolset() -> None:
    """Line 177: SecurityControls.default() return cls()."""
    controls = SecurityControls.default()
    assert controls.readonly_aws is True
    names = {t.name for t in controls.tools()}
    assert "describe_aws" in names
    assert "use_aws" in names


def test_ground_report_grounded_finding_ships() -> None:
    """Lines 243-272: ground_report with grounded evidence → Finding."""
    finding = PostureFinding(
        title="Root access keys present",
        description="CIS 1.4 violation.",
        severity=Severity.HIGH,
        asset="aws-account:000:root",
        remediation="Delete root access keys.",
        evidence=[
            PostureEvidence(
                statement="AccountAccessKeysPresent=1",
                ref="aws:iam:GetAccountSummary:AccountAccessKeysPresent",
                grounded=True,
            )
        ],
    )
    report = PostureReport(summary="audit", confidence=0.9, findings=[finding])
    results = ground_report(report)
    assert len(results) == 1
    assert is_finding(results[0])


def test_ground_report_ungrounded_finding_abstains() -> None:
    """Lines 255-272: inference-only evidence → Abstention."""
    from tulip.security.grounded import Abstention

    finding = PostureFinding(
        title="Speculative risk",
        description="Pure speculation.",
        severity=Severity.LOW,
        asset="aws-account:000",
        remediation="n/a",
        evidence=[
            PostureEvidence(
                statement="Attacker could pivot",
                ref="inference:none",
                grounded=False,
            )
        ],
    )
    report = PostureReport(summary="audit", confidence=0.5, findings=[finding])
    results = ground_report(report)
    assert not is_finding(results[0])
    assert isinstance(results[0], Abstention)


def test_ground_report_custom_controls_threshold() -> None:
    """Passes non-default controls → thresholds derived correctly."""
    finding = PostureFinding(
        title="S3 public",
        description="Public bucket.",
        severity=Severity.HIGH,
        asset="s3:my-bucket",
        remediation="Block public access.",
        evidence=[
            PostureEvidence(
                statement="BlockPublicAcls=False",
                ref="aws:s3:GetBucketPublicAccessBlock",
                grounded=True,
            )
        ],
    )
    report = PostureReport(summary="audit", confidence=0.9, findings=[finding])
    controls = SecurityControls(min_gsar=0.5)
    results = ground_report(report, controls=controls)
    assert is_finding(results[0])


def test_create_soc_analyst_wires_tools_and_schema() -> None:
    """Lines 357-370: create_soc_analyst default path."""
    with patch("tulip.deepagent.create_deepagent") as make:
        create_soc_analyst(model="mock:test")
        kwargs = make.call_args.kwargs
    assert kwargs["output_schema"] is PostureReport
    assert kwargs["grounding"] is True
    tool_names = {t.name for t in kwargs["tools"]}
    assert {"describe_aws", "use_aws", "submit_posture"} <= tool_names


def test_create_soc_analyst_with_scope_appended() -> None:
    """Line 364: scope appended to default prompt."""
    with patch("tulip.deepagent.create_deepagent") as make:
        create_soc_analyst(model="mock:test", scope="Focus on IAM only.")
        assert "Focus on IAM only." in make.call_args.kwargs["system_prompt"]


def test_create_soc_analyst_explicit_system_prompt() -> None:
    """Line 362: explicit system_prompt overrides default + scope."""
    with patch("tulip.deepagent.create_deepagent") as make:
        create_soc_analyst(model="mock:test", system_prompt="custom", scope="ignored")
        assert make.call_args.kwargs["system_prompt"] == "custom"


# ---------------------------------------------------------------------------
# target.py — shapes not tested in extra.py
# ---------------------------------------------------------------------------

from tulip.security.target import Target, _extract_text


def test_extract_text_list_indexing_in_path() -> None:
    """Line 54: path includes a numeric index into a list."""
    body = {"items": ["alpha", "beta"]}
    assert _extract_text(body, "items.0") == "alpha"


def test_extract_text_non_str_at_path_end() -> None:
    """Line 57: path navigated but result is not a str → as_json."""
    body = {"a": {"b": 42}}
    result = _extract_text(body, "a.b")
    assert result == "42"


def test_extract_text_common_response_key() -> None:
    """Line 64: body is a Mapping with a well-known response key."""
    assert _extract_text({"response": "ok"}, None) == "ok"
    assert _extract_text({"output": "val"}, None) == "val"


def test_extract_text_openai_chat_shape() -> None:
    """Line 72: choices[0].message.content → returned directly."""
    body = {"choices": [{"message": {"content": "chat-reply"}}]}
    assert _extract_text(body, None) == "chat-reply"


def test_extract_text_choices_not_a_list() -> None:
    """67->75 branch: choices present but is not a list → falls to as_json."""
    body = {"choices": "not-a-list"}
    result = _extract_text(body, None)
    assert "choices" in result


def test_extract_text_choices_mapping_no_content_or_text() -> None:
    """73->75 branch: first is a Mapping, no usable message.content OR text → as_json."""
    # message exists but content is int (not str), and no "text" key
    body = {"choices": [{"message": {"content": 42}}]}
    result = _extract_text(body, None)
    # Falls through to as_json(body)
    assert "choices" in result


async def test_from_callable_sync_fn() -> None:
    """Lines 107-112: from_callable wraps sync function."""
    target = Target.from_callable(lambda p: f"echo:{p}", name="sync-target")
    assert target.kind == "callable"
    assert await target.send("hello") == "echo:hello"


async def test_from_callable_async_fn() -> None:
    """Lines 107-112: from_callable wraps async function (isawaitable path)."""

    async def _fn(p: str) -> str:
        return p.upper()

    target = Target.from_callable(_fn, name="async-target")
    assert await target.send("world") == "WORLD"


async def test_endpoint_target_normal_json_path() -> None:
    """Line 151: successful JSON response → _extract_text called."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": "from-bot"})

    target = Target.endpoint("https://bot.example/chat", transport=httpx.MockTransport(handler))
    result = await target.send("hi")
    assert result == "from-bot"


async def test_agent_target_captures_final_message() -> None:
    """Lines 176-184: Target.agent drives the async run loop."""

    class _FakeAgent:
        async def run(self, prompt: str):  # noqa: ANN201
            class _Ev:
                final_message = None

            class _Fin:
                final_message = f"answer:{prompt}"

            yield _Ev()
            yield _Fin()

    target = Target.agent(_FakeAgent(), name="bot")
    assert target.kind == "agent"
    assert await target.send("ping") == "answer:ping"


async def test_a2a_target_wraps_coroutine() -> None:
    """Lines 201-204: Target.a2a with async sender."""

    async def _peer(prompt: str) -> str:
        return f"peer:{prompt}"

    target = Target.a2a(_peer, name="peer-1")
    assert target.kind == "a2a"
    assert await target.send("x") == "peer:x"


# ---------------------------------------------------------------------------
# context.py — domain providers not yet exercised in extra.py
# ---------------------------------------------------------------------------

from tulip.security import SecurityContext


async def test_ref_logs_search() -> None:
    """Line 121: _RefLogs.search delegates to query_siem."""
    ctx = SecurityContext()
    result = await ctx.logs.search("failed login")
    assert isinstance(result, dict)
    assert "events" in result


async def test_ref_endpoint_get_host() -> None:
    """Line 127: _RefEndpoint.get_host delegates to fetch_host_timeline."""
    ctx = SecurityContext()
    result = await ctx.endpoint.get_host("WS-0142")
    assert isinstance(result, dict)
    assert "events" in result


async def test_ref_endpoint_detections() -> None:
    """Line 130: _RefEndpoint.detections delegates to list_detections."""
    ctx = SecurityContext()
    result = await ctx.endpoint.detections()
    assert isinstance(result, dict)
    assert "count" in result


async def test_ref_identity_get_user() -> None:
    """Line 167: _RefIdentity.get_user."""
    ctx = SecurityContext()
    result = await ctx.identity.get_user("jsmith@example.com")
    assert result["risk"] == "low"
    assert result["source"] == "offline-sample"


async def test_ref_identity_risk() -> None:
    """Lines 170-171: _RefIdentity.risk returns risk + impossible_travel."""
    ctx = SecurityContext()
    result = await ctx.identity.risk("mallory@example.com")
    assert result["risk"] == "high"
    assert result["impossible_travel"] is True


async def test_ref_threat_intel_enrich() -> None:
    """Line 207: _RefThreatIntel.enrich."""
    ctx = SecurityContext()
    result = await ctx.threat_intel.enrich("198.51.100.23")
    assert isinstance(result, dict)
    assert "verdict" in result


async def test_ref_actions_request_approval_allow() -> None:
    """Line 221: _RefActions.request_approval via ctx.actions."""
    from tulip.control import Action, ApprovalOutcome
    from tulip.security.verify import VerificationResult

    ctx = SecurityContext()
    verdict = VerificationResult(survives=True, confidence=0.9, evidence_quality=0.9)
    action = Action(name="scan", asset="host", blast_radius=1, environment="dev")
    decision = ctx.actions.request_approval(action, verdict=verdict)
    assert decision.outcome == ApprovalOutcome.ALLOW


async def test_ref_actions_execute_admitted() -> None:
    """Lines 231-233: _RefActions.execute — action is admitted → perform called."""
    from tulip.control import Action
    from tulip.security.verify import VerificationResult

    ctx = SecurityContext()
    action = Action(name="read-only", asset="host", blast_radius=1, environment="dev")
    # Default policy requires require_verification_score=0.8; supply a passing verdict.
    verdict = VerificationResult(survives=True, confidence=0.95, evidence_quality=0.95)

    async def _perform() -> str:
        return "done"

    result = await ctx.actions.execute(action, _perform, verdict=verdict)
    assert result == "done"


def test_security_context_toolset_delegates() -> None:
    """Lines 265-267: SecurityContext.toolset() → security_toolset()."""
    ctx = SecurityContext()
    tools = ctx.toolset()
    assert isinstance(tools, list)
    assert any(getattr(t, "name", "") == "query_siem" for t in tools)
