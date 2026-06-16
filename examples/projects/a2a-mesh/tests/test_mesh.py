# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
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
os.environ["LOCUS_MODEL_PROVIDER"] = "mock"

from a2a_mesh.finance import build_server as build_finance  # noqa: E402
from a2a_mesh.orchestrator import pick  # noqa: E402
from a2a_mesh.research import build_server as build_research  # noqa: E402


@pytest.fixture
def research_client() -> TestClient:
    return TestClient(build_research().app)


@pytest.fixture
def finance_client() -> TestClient:
    return TestClient(build_finance().app)


def test_research_card(research_client: TestClient) -> None:
    card = research_client.get("/agent-card").json()
    assert card["name"] == "research-agent"
    assert "research" in card["skills"]
    assert "summarize" in card["skills"]


def test_finance_card(finance_client: TestClient) -> None:
    card = finance_client.get("/agent-card").json()
    assert card["name"] == "finance-agent"
    assert "finance" in card["skills"]
    assert "valuation" in card["skills"]


def test_research_invoke(research_client: TestClient) -> None:
    resp = research_client.post(
        "/a2a/invoke",
        json={"messages": [{"role": "user", "content": "hi", "metadata": {}}], "metadata": {}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["messages"]
    assert body["status"] in {"completed", "ok", "success"} or body["messages"]


def test_finance_invoke(finance_client: TestClient) -> None:
    resp = finance_client.post(
        "/a2a/invoke",
        json={"messages": [{"role": "user", "content": "TSLA", "metadata": {}}], "metadata": {}},
    )
    assert resp.status_code == 200


def test_orchestrator_routing_finance() -> None:
    peers = [
        ("http://127.0.0.1:8001", ["research", "summarize"]),
        ("http://127.0.0.1:8002", ["finance", "valuation"]),
    ]
    assert pick("Should I buy TSLA?", peers, force=None).endswith(":8002")
    assert pick("price of AAPL", peers, force=None).endswith(":8002")


def test_orchestrator_routing_research() -> None:
    peers = [
        ("http://127.0.0.1:8001", ["research", "summarize"]),
        ("http://127.0.0.1:8002", ["finance", "valuation"]),
    ]
    assert pick("Summarise quantum computing", peers, force=None).endswith(":8001")


def test_orchestrator_force_skill() -> None:
    peers = [
        ("http://127.0.0.1:8001", ["research", "summarize"]),
        ("http://127.0.0.1:8002", ["finance", "valuation"]),
    ]
    # Even though the query mentions a ticker, --skill summarize forces research.
    assert pick("TSLA", peers, force="summarize").endswith(":8001")
