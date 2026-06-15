# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""In-process integration test for the A2A mesh.

Boots both A2AServers via FastAPI's TestClient (no real network),
exercises ``GET /agent-card`` and ``POST /a2a/invoke``, and asserts the
orchestrator's skill-routing rule picks the right peer.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


# Force MockModel for the test even if the user has env vars set.
os.environ["TULIP_MODEL_PROVIDER"] = "mock"

from a2a_mesh.orchestrator import pick  # noqa: E402
from a2a_mesh.soc_triage import build_server as build_triage  # noqa: E402
from a2a_mesh.threat_intel import build_server as build_intel  # noqa: E402


@pytest.fixture
def intel_client() -> TestClient:
    return TestClient(build_intel().app)


@pytest.fixture
def triage_client() -> TestClient:
    return TestClient(build_triage().app)


PEERS = [
    ("http://127.0.0.1:8001", ["threat_intel", "ioc_enrichment"]),
    ("http://127.0.0.1:8002", ["alert_triage", "severity_scoring"]),
]


def test_intel_card(intel_client: TestClient) -> None:
    card = intel_client.get("/agent-card").json()
    assert card["name"] == "threat-intel-agent"
    assert "threat_intel" in card["skills"]
    assert "ioc_enrichment" in card["skills"]


def test_triage_card(triage_client: TestClient) -> None:
    card = triage_client.get("/agent-card").json()
    assert card["name"] == "soc-triage-agent"
    assert "alert_triage" in card["skills"]
    assert "severity_scoring" in card["skills"]


def test_intel_invoke(intel_client: TestClient) -> None:
    resp = intel_client.post(
        "/a2a/invoke",
        json={"messages": [{"role": "user", "content": "hi", "metadata": {}}], "metadata": {}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["messages"]
    assert body["status"] in {"completed", "ok", "success"} or body["messages"]


def test_triage_invoke(triage_client: TestClient) -> None:
    resp = triage_client.post(
        "/a2a/invoke",
        json={"messages": [{"role": "user", "content": "A-101", "metadata": {}}], "metadata": {}},
    )
    assert resp.status_code == 200


def test_orchestrator_routing_triage() -> None:
    assert pick("Is alert A-101 a true positive?", PEERS, force=None).endswith(":8002")
    assert pick("score the severity of this finding", PEERS, force=None).endswith(":8002")


def test_orchestrator_routing_intel() -> None:
    assert pick("Enrich 198.51.100.7", PEERS, force=None).endswith(":8001")


def test_orchestrator_force_skill() -> None:
    # Even though the query names an alert, --skill ioc_enrichment forces threat-intel.
    assert pick("alert A-101", PEERS, force="ioc_enrichment").endswith(":8001")
