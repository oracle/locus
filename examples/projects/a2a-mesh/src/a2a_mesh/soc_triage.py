# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""SOC triage agent — exposed over A2A on :8002 by default.

Skills: ``alert_triage``, ``severity_scoring``.
"""

from __future__ import annotations

import argparse
from typing import Any

from a2a_mesh._model import get_model
from locus.a2a import A2AServer, PushNotificationSettings
from locus.agent import Agent, AgentConfig
from locus.tools import tool


@tool
def fetch_alert(alert_id: str) -> dict:
    """Fetch the details of a SOC alert by id. (Stubbed for the demo.)"""
    canned = {
        "A-101": {"rule": "impossible-travel", "src_ip": "198.51.100.7", "user": "jdoe"},
        "A-204": {"rule": "outbound-beaconing", "src_ip": "192.0.2.14", "user": "svc-batch"},
    }
    return {"alert_id": alert_id.upper(), **canned.get(alert_id.upper(), {"rule": "unknown"})}


@tool
def score_severity(vendor_detections: int, asset_exposed: bool) -> dict:
    """Score an alert's severity from its enrichment signals."""
    score = min(100, vendor_detections * 25 + (40 if asset_exposed else 0))
    tier = "HIGH" if score >= 70 else "MEDIUM" if score >= 40 else "LOW"
    return {"score": score, "severity": tier}


def build_server(push_client: Any | None = None) -> A2AServer:
    """The peer's ``A2AServer``.

    Serves the JSON-RPC and HTTP+JSON bindings and delivers push
    notifications; ``allow_private_urls`` lets the demo post to a receiver on
    this machine (a production peer would leave it off). ``push_client`` is
    an ``httpx.AsyncClient`` for the webhook calls, which the in-process test
    binds to its receiver.
    """
    agent = Agent(
        config=AgentConfig(
            model=get_model(),
            tools=[fetch_alert, score_severity],
            system_prompt=(
                "You are a SOC triage analyst. Pull alert detail, weigh the "
                "enrichment, score severity, and recommend escalate or close."
            ),
            max_iterations=5,
        )
    )
    return A2AServer(
        agent=agent,
        name="soc-triage-agent",
        description="Triages SOC alerts, scores severity, and recommends a verdict.",
        skills=["alert_triage", "severity_scoring"],
        transports=("JSONRPC", "HTTP+JSON"),
        push_notifications=PushNotificationSettings(
            allow_private_urls=True, http_client=push_client
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    build_server().run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
