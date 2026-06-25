# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Tests for SecurityContext — the domain facade (zero-config offline)."""

from __future__ import annotations

from typing import Any

from tulip.control import Action, ApprovalOutcome
from tulip.security import (
    CloudSource,
    EndpointSource,
    IdentitySource,
    LogSource,
    SecurityContext,
    ThreatIntelSource,
    VerificationResult,
)
from tulip.security.context import _RefCloud, _RefEndpoint, _RefIdentity, _RefLogs


def test_reference_providers_satisfy_their_ports() -> None:
    assert isinstance(_RefLogs(), LogSource)
    assert isinstance(_RefEndpoint(), EndpointSource)
    assert isinstance(_RefIdentity(), IdentitySource)
    assert isinstance(_RefCloud(), CloudSource)


async def test_zero_config_logs_and_endpoint() -> None:
    ctx = SecurityContext()
    logs = await ctx.logs.search("failed login")
    assert isinstance(logs, dict)
    assert "events" in logs
    host = await ctx.endpoint.get_host("WS-0142")
    assert isinstance(host, dict)
    assert isinstance(await ctx.endpoint.detections(), dict)


async def test_identity_surfaces_risk() -> None:
    ctx = SecurityContext()
    high = await ctx.identity.risk("mallory@example.com")
    assert high["risk"] == "high"
    assert high["impossible_travel"] is True
    low = await ctx.identity.get_user("jsmith@example.com")
    assert low["risk"] == "low"
    unknown = await ctx.identity.risk("nobody@example.com")
    assert unknown["risk"] == "unknown"


async def test_threat_intel_and_cloud_return_dicts() -> None:
    ctx = SecurityContext()
    assert isinstance(await ctx.threat_intel.enrich("198.51.100.23"), dict)
    # cloud degrades gracefully whether or not boto3 is installed
    assert isinstance(await ctx.cloud.describe(), dict)


async def test_actions_gates_via_policy() -> None:
    ctx = SecurityContext()
    verdict = VerificationResult(survives=True, confidence=0.95, evidence_quality=0.95)
    decision = ctx.actions.request_approval(
        Action(name="quarantine", asset="WS-0142", blast_radius=1, environment="staging"),
        verdict=verdict,
    )
    assert decision.outcome == ApprovalOutcome.ALLOW


async def test_vendor_provider_injection() -> None:
    # A custom (framework-agnostic) provider plugs in by satisfying the port.
    class _FakeSplunk:
        async def search(self, query: str, *, window: str = "24h") -> dict[str, Any]:
            return {"source": "fake-splunk", "query": query, "events": [{"id": 1}]}

    fake = _FakeSplunk()
    assert isinstance(fake, LogSource)
    ctx = SecurityContext(logs=fake)
    result = await ctx.logs.search("x")
    assert result["source"] == "fake-splunk"
    # other domains keep their offline defaults
    assert isinstance(ctx.threat_intel, ThreatIntelSource)


def test_toolset_returns_agent_tools() -> None:
    tools = SecurityContext().toolset()
    assert isinstance(tools, list)
    assert any(getattr(t, "name", "") == "query_siem" for t in tools)
