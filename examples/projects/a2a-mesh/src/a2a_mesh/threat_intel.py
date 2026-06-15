# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Threat-intel agent — exposed over A2A on :8001 by default.

Skills: ``threat_intel``, ``ioc_enrichment``.
"""

from __future__ import annotations

import argparse

from a2a_mesh._model import get_model
from tulip.a2a import A2AServer
from tulip.agent import Agent, AgentConfig
from tulip.tools import tool


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


def build_server() -> A2AServer:
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
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    build_server().run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
