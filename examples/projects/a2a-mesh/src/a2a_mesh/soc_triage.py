# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""SOC triage agent — exposed over A2A on :8002 by default.

Skills: ``alert_triage``, ``severity_scoring``.
"""

from __future__ import annotations

import argparse

from a2a_mesh._model import get_model
from tulip.a2a import A2AServer
from tulip.agent import Agent, AgentConfig
from tulip.tools import tool


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


def build_server() -> A2AServer:
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
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    build_server().run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
