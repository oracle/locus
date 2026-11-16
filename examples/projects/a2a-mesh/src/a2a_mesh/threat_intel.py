# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Threat-intel agent — exposed over A2A on :8001 by default.

Skills: ``threat_intel``, ``ioc_enrichment``.
"""

from __future__ import annotations

import argparse
from typing import Any

from a2a_mesh._model import get_model
from locus.a2a import A2AServer, PushNotificationSettings
from locus.agent import Agent, AgentConfig
from locus.tools import tool


@tool
def lookup_ioc(indicator: str) -> str:
    """Look up an indicator (IP / domain / hash) against threat intel.

    (Stubbed for the demo — returns a one-line verdict.)
    """
    return f"{indicator}: 3 vendor detections, first seen 2 days ago in a phishing campaign."


@tool
def enrich_domain(domain: str) -> str:
    """Return registrar age, category, and reputation for a domain."""
    return f"{domain}: registered 2 days ago, category 'newly observed', reputation 'suspicious'."


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
            tools=[lookup_ioc, enrich_domain],
            system_prompt=(
                "You are a threat-intel analyst. Enrich indicators, summarise "
                "what the evidence says, and cite the source of each detection."
            ),
            max_iterations=5,
        )
    )
    return A2AServer(
        agent=agent,
        name="threat-intel-agent",
        description="Enriches indicators of compromise and summarises the intel.",
        skills=["threat_intel", "ioc_enrichment"],
        transports=("JSONRPC", "HTTP+JSON"),
        push_notifications=PushNotificationSettings(
            allow_private_urls=True, http_client=push_client
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    build_server().run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
