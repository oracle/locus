# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 44: plugins — composable agent extensions.

Plugins bundle hooks (and optionally tools) into one reusable object.
Drop a plugin onto an agent and every relevant hook method runs
automatically. Three pieces:

- ``Plugin`` base class — subclass it, give it a ``name``, decorate any
  method with ``@hook`` and the agent picks it up.
- ``@hook`` decorator — marks methods like ``on_before_model_call`` and
  ``on_before_tool_call`` for auto-discovery.
- ``callback_handler`` — a plain function that receives every event;
  the lighter-weight alternative when you don't need a class.
- ``Agent.cancel()`` — stop a running agent from another thread; the
  next step returns ``stop_reason="cancelled"``.

Run it:
    # The bundled mock model is the default; set TULIP_MODEL_PROVIDER for a live provider.
    TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_49_plugins.py

    # Offline:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_49_plugins.py

Prerequisites:
- An OpenAI or Anthropic API key, or set ``TULIP_MODEL_PROVIDER`` to
  ``openai`` / ``anthropic`` / ``mock``.
"""

import threading
import time

from config import get_model

from tulip.agent import Agent, AgentConfig
from tulip.hooks.plugin import Plugin, hook
from tulip.tools.decorator import tool


# =============================================================================
# Part 1: A small audit plugin — logs every model and tool call.
# =============================================================================


def example_plugin():
    print("=== Part 1: Plugin System ===\n")

    model = get_model()

    class AuditPlugin(Plugin):
        """Tracks all model and tool calls."""

        name = "audit"

        def __init__(self):
            self.log = []

        @hook
        async def on_before_model_call(self, event):
            self.log.append(f"model: {len(event.messages)} msgs")

        @hook
        async def on_before_tool_call(self, event):
            self.log.append(f"tool: {event.tool_name}")

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results for: {query}"

    plugin = AuditPlugin()
    agent = Agent(
        config=AgentConfig(
            system_prompt="Use the search tool to answer questions.",
            max_iterations=5,
            model=model,
            tools=[search],
            plugins=[plugin],
        )
    )

    result = agent.run_sync("Search for Python best practices")
    print(f"Response: {result.message[:100]}...")
    print(f"Audit log: {plugin.log}")


# =============================================================================
# Part 2: callback_handler — when a plain function is enough.
# =============================================================================


def example_callback():
    print("\n=== Part 2: Callback Handler ===\n")

    model = get_model()
    events = []

    agent = Agent(
        config=AgentConfig(
            system_prompt="Answer concisely.",
            max_iterations=3,
            model=model,
            callback_handler=lambda e: events.append(e.event_type),
        )
    )

    agent.run_sync("What is 2+2?")
    print(f"Events received: {events}")


# =============================================================================
# Part 3: Agent.cancel() — stop a run from another thread.
# =============================================================================


def example_cancel():
    print("\n=== Part 3: Cancel Signal ===\n")

    model = get_model(max_tokens=80)

    # Run one normal call first so this part still exercises the provider.
    live_agent = Agent(
        config=AgentConfig(
            system_prompt="Answer in one sentence.",
            max_iterations=3,
            model=model,
        )
    )
    t0 = time.perf_counter()
    live_result = live_agent.run_sync("In one sentence, why does an agent need a cancel signal?")
    dt = time.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · "
        f"{live_result.metrics.prompt_tokens}→{live_result.metrics.completion_tokens} tokens]"
    )
    print(f"  AI rationale: {live_result.message.strip()}")

    # Cancel a fresh agent before it starts — the run returns immediately.
    agent = Agent(
        config=AgentConfig(
            system_prompt="Answer concisely.",
            max_iterations=3,
            model=model,
        )
    )
    agent.cancel()
    result = agent.run_sync("This should be cancelled")
    print(f"Stop reason: {result.stop_reason}")


if __name__ == "__main__":
    example_plugin()
    example_callback()
    example_cancel()
