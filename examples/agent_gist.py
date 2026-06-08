# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL
"""Tulip Agent — Quick Start.

Uses ``get_model`` to pick a provider from a model string. Set
``TULIP_MODEL`` to ``openai:gpt-4o`` / ``anthropic:claude-sonnet-4-6`` /
``anthropic:claude-sonnet-4-6`` and export the matching API key. See
``docs/concepts/models.md``.
"""

import os

from tulip.agent import Agent
from tulip.models import get_model


def main():
    model = get_model(os.environ.get("TULIP_MODEL", "openai:gpt-4o"))

    agent = Agent(
        model=model,
        system_prompt="You are a helpful assistant. Be concise.",
    )

    result = agent.run_sync("What is the capital of France?")
    print(f"Response: {result.message}")
    print(f"Iterations: {result.metrics.iterations}")


if __name__ == "__main__":
    main()
