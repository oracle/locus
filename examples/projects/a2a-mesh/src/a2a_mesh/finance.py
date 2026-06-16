# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Finance agent — exposed over A2A on :8002 by default.

Skills: ``finance``, ``valuation``.
"""

from __future__ import annotations

import argparse

from a2a_mesh._model import get_model
from locus.a2a import A2AServer
from locus.agent import Agent, AgentConfig
from locus.tools import tool


@tool
def fetch_price(ticker: str) -> dict:
    """Fetch the latest quote for a ticker. (Stubbed for the demo.)"""
    canned = {"TSLA": 421.0, "AAPL": 211.0, "ORCL": 159.0}
    return {"ticker": ticker.upper(), "price_usd": canned.get(ticker.upper(), 100.0)}


@tool
def compute_pe(price: float, eps: float) -> float:
    """Compute price-to-earnings ratio."""
    return round(price / eps, 2) if eps else float("inf")


def build_server() -> A2AServer:
    agent = Agent(
        config=AgentConfig(
            model=get_model(),
            tools=[fetch_price, compute_pe],
            system_prompt=(
                "You are a financial analyst. Fetch quotes, compute simple "
                "valuation metrics, and explain your reasoning."
            ),
            max_iterations=5,
        )
    )
    return A2AServer(
        agent=agent,
        name="finance-agent",
        description="Quotes, valuations, and back-of-envelope financial analysis.",
        skills=["finance", "valuation"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    build_server().run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
