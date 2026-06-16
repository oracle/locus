# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Research agent — exposed over A2A on :8001 by default.

Skills: ``research``, ``summarize``.
"""

from __future__ import annotations

import argparse

from a2a_mesh._model import get_model
from locus.a2a import A2AServer
from locus.agent import Agent, AgentConfig
from locus.tools import tool


@tool
def web_search(query: str) -> str:
    """Look up a topic. (Stubbed for the demo — returns a one-line summary.)"""
    return f"Top result for {query!r}: a curated summary appears here."


@tool
def cite(claim: str, source: str) -> str:
    """Attach a source URL to a claim."""
    return f"{claim} [source: {source}]"


def build_server() -> A2AServer:
    agent = Agent(
        config=AgentConfig(
            model=get_model(),
            tools=[web_search, cite],
            system_prompt=(
                "You are a research assistant. Look things up, summarise "
                "concisely, and cite sources where possible."
            ),
            max_iterations=5,
        )
    )
    return A2AServer(
        agent=agent,
        name="research-agent",
        description="Reads sources and produces summaries with citations.",
        skills=["research", "summarize"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    build_server().run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
