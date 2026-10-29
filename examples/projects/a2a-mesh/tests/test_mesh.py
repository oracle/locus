# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""In-process integration test for the A2A mesh.

Boots both A2AServers via FastAPI's TestClient (no real network) and
exercises the A2A v1.0 surface the orchestrator and the web console use
(``GET /.well-known/agent-card.json``, ``SendMessage``,
``SendStreamingMessage``), the legacy ``GET /agent-card`` and
``POST /a2a/invoke`` kept for backwards compatibility, and the
orchestrator's skill-routing rule on the skill ids a v1.0 card carries.
"""

from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient


# Force MockModel for the test even if the user has env vars set.
os.environ["LOCUS_MODEL_PROVIDER"] = "mock"

from a2a_mesh.orchestrator import pick, reply_text, skill_ids  # noqa: E402
from a2a_mesh.soc_triage import build_server as build_triage  # noqa: E402
from a2a_mesh.threat_intel import build_server as build_intel  # noqa: E402

from locus.a2a import AgentCard, Task  # noqa: E402


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


# --- A2A v1.0 ------------------------------------------------------------

V1_HEADERS = {"A2A-Version": "1.0"}


def _rpc(method: str, text: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": "1",
        "method": method,
        "params": {"message": {"role": "ROLE_USER", "parts": [{"text": text}], "messageId": "m-1"}},
    }


def test_v1_card_carries_skill_objects_and_interfaces(intel_client: TestClient) -> None:
    card = intel_client.get("/.well-known/agent-card.json", headers=V1_HEADERS).json()
    assert card["protocolVersion"] == "1.0"
    assert card["supportedInterfaces"][0]["protocolBinding"] == "JSONRPC"
    assert {s["id"] for s in card["skills"]} == {"threat_intel", "ioc_enrichment"}


def test_routing_works_on_the_skill_ids_of_v1_cards(
    intel_client: TestClient, triage_client: TestClient
) -> None:
    """Regression: v1.0 cards list skills as objects, and routing must match their ids."""
    discovered = [
        (
            url,
            skill_ids(AgentCard.model_validate(client.get("/.well-known/agent-card.json").json())),
        )
        for url, client in (
            ("http://127.0.0.1:8001", intel_client),
            ("http://127.0.0.1:8002", triage_client),
        )
    ]
    assert pick("Is alert A-101 a true positive?", discovered, force=None).endswith(":8002")
    assert pick("Enrich 198.51.100.7", discovered, force=None).endswith(":8001")


def test_v1_send_message_returns_the_reply_as_an_artifact(triage_client: TestClient) -> None:
    resp = triage_client.post("/", json=_rpc("SendMessage", "A-101"), headers=V1_HEADERS)
    assert resp.status_code == 200
    task = resp.json()["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert task["artifacts"][-1]["parts"][0]["text"]


def test_v1_streaming_reports_the_task_lifecycle(intel_client: TestClient) -> None:
    frames = []
    with intel_client.stream(
        "POST", "/", json=_rpc("SendStreamingMessage", "Enrich 198.51.100.7"), headers=V1_HEADERS
    ) as resp:
        for line in resp.iter_lines():
            if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
                frames.append(json.loads(line[5:])["result"])
    kinds = [next(iter(frame)) for frame in frames]
    assert kinds[0] == "task"
    assert "artifactUpdate" in kinds
    assert frames[-1]["statusUpdate"]["status"]["state"] == "TASK_STATE_COMPLETED"


def test_reply_text_reads_the_last_artifact() -> None:
    task = Task.model_validate(
        {
            "id": "t1",
            "contextId": "c1",
            "status": {"state": "completed"},
            "artifacts": [{"artifactId": "a1", "parts": [{"kind": "text", "text": "verdict: TP"}]}],
        }
    )
    assert reply_text(task) == "verdict: TP"


def test_reply_text_without_artifacts_reports_the_state() -> None:
    task = Task.model_validate({"id": "t1", "contextId": "c1", "status": {"state": "failed"}})
    assert reply_text(task) == "(no reply; task is failed)"
