# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the first-class tulip.security vendor adapters.

Everything runs offline: the deterministic sample paths need no credentials,
and the live-path tests mock httpx with ``respx`` rather than touching a real
vendor. Grounding flows assert the admit/abstain contract end-to-end.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from tulip.security import (
    classify_indicator,
    default_classifier,
    enrich_indicator,
    enrich_indicator_tool,
    enrich_to_finding,
    fetch_host_timeline,
    fingerprint_to_finding,
    is_finding,
    isolate_host,
    isolate_host_tool,
    list_detections,
    measure_endpoint_timing,
    query_siem,
    scan_dependencies,
    scan_endpoint,
    scan_endpoint_to_finding,
    security_toolset,
)
from tulip.security.taxonomy import Severity


# --------------------------------------------------------------------------- #
# Threat-intel
# --------------------------------------------------------------------------- #


def test_classify_indicator() -> None:
    assert classify_indicator("198.51.100.23") == "ip"
    assert classify_indicator("phish.example.net") == "domain"
    assert classify_indicator("a" * 64) == "hash"


def test_enrich_indicator_offline_shape() -> None:
    out = enrich_indicator("198.51.100.23")
    assert out["source"] == "offline-sample"
    assert out["verdict"] == "malicious"
    assert int(out["malicious"]) >= 3


def test_enrich_to_finding_ships_malicious() -> None:
    result = enrich_to_finding("198.51.100.23")
    assert is_finding(result)
    assert result.severity == Severity.HIGH
    assert result.indicators[0].value == "198.51.100.23"


def test_enrich_to_finding_abstains_on_clean() -> None:
    result = enrich_to_finding("203.0.113.99")  # not in sample feed → no reports
    assert not is_finding(result)


@respx.mock
def test_enrich_indicator_live_path_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VT_API_KEY", "test-key")
    respx.get("https://www.virustotal.com/api/v3/ip_addresses/198.51.100.23").mock(
        return_value=httpx.Response(
            200, json={"data": {"attributes": {"last_analysis_stats": {"malicious": 5}}}}
        )
    )
    out = enrich_indicator("198.51.100.23")
    assert out["source"] == "virustotal"
    assert out["malicious"] == 5
    assert out["verdict"] == "malicious"


async def test_enrich_indicator_tool_returns_json() -> None:
    payload = json.loads(await enrich_indicator_tool("198.51.100.23"))
    assert payload["indicator"] == "198.51.100.23"


# --------------------------------------------------------------------------- #
# SIEM
# --------------------------------------------------------------------------- #


def test_query_siem_offline_filters() -> None:
    out = query_siem("powershell")
    assert out["source"] == "offline-sample"
    assert out["count"] >= 1
    assert all("powershell" in json.dumps(e).lower() for e in out["events"])


@respx.mock
def test_query_siem_live_path_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIEM_URL", "https://siem.example")
    monkeypatch.setenv("SIEM_TOKEN", "tok")
    respx.post("https://siem.example/services/search/jobs/export").mock(
        return_value=httpx.Response(200, json={"results": [{"host": "WS-1", "event": "x"}]})
    )
    out = query_siem("anything")
    assert out["source"] == "siem"
    assert out["count"] == 1


# --------------------------------------------------------------------------- #
# EDR
# --------------------------------------------------------------------------- #


def test_edr_timeline_and_detections_offline() -> None:
    timeline = fetch_host_timeline("WS-0142")
    assert timeline["source"] == "offline-sample"
    assert len(timeline["events"]) == 3
    dets = list_detections("WS-0142")
    assert dets["count"] == 2


def test_isolate_host_offline_simulated() -> None:
    out = isolate_host("WS-0142")
    assert out["status"] == "contained (simulated)"


def test_isolate_host_tool_is_idempotent() -> None:
    # A containment write must dedupe on (name, args) across retries.
    assert isolate_host_tool.idempotent is True


# --------------------------------------------------------------------------- #
# Scanner
# --------------------------------------------------------------------------- #


def test_scan_dependencies_clean_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TULIP_MCP_SKIP_OSV", "1")  # no network
    out = scan_dependencies("npx", ["left-pad@1.0.0"])
    assert out["clean"] is True
    assert out["advisory"] is None


def test_scan_endpoint_offline_expired() -> None:
    out = scan_endpoint("192.0.2.10")
    assert out["source"] == "offline-sample"
    assert out["tls_expired"] is True


def test_scan_endpoint_to_finding_ships_on_expiry() -> None:
    result = scan_endpoint_to_finding("192.0.2.10")
    assert is_finding(result)
    assert result.severity == Severity.HIGH
    assert result.asset == "192.0.2.10:443"


def test_scan_endpoint_to_finding_abstains_when_healthy() -> None:
    result = scan_endpoint_to_finding("198.51.100.5")  # valid cert in sample
    assert not is_finding(result)


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #


def test_measure_endpoint_timing_offline_sample() -> None:
    feats = measure_endpoint_timing()
    assert set(feats) >= {"ttft_ms_p50", "itl_ms_mean", "itl_cv", "tps_mean"}


def test_default_classifier_coverage() -> None:
    verdict = default_classifier(measure_endpoint_timing())
    assert verdict.feature_coverage == 1.0
    assert verdict.model


def test_fingerprint_to_finding_ships_full_coverage() -> None:
    result = fingerprint_to_finding(measure_endpoint_timing(), asset="203.0.113.10:443")
    assert is_finding(result)
    assert result.verdict.model


def test_fingerprint_to_finding_abstains_low_coverage() -> None:
    thin = {"ttft_ms_p50": 38.2}  # 1/4 features → under-observed
    result = fingerprint_to_finding(thin, asset="203.0.113.10:443")
    assert not is_finding(result)


# --------------------------------------------------------------------------- #
# Toolset assembly
# --------------------------------------------------------------------------- #


def test_security_toolset_defaults_are_readonly() -> None:
    names = {t.name for t in security_toolset()}
    assert "enrich_indicator" in names
    assert "query_siem" in names
    assert "fetch_host_timeline" in names
    assert "scan_endpoint" in names
    assert "fingerprint_endpoint" in names
    assert "isolate_host" not in names  # containment off by default
    assert "use_aws" not in names  # aws off by default


def test_security_toolset_containment_and_aws_optin() -> None:
    names = {t.name for t in security_toolset(allow_containment=True, aws=True)}
    assert "isolate_host" in names
    assert {"describe_aws", "use_aws"} <= names
