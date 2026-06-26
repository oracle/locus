# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 47: plugins — package cloud-cost triage as a reusable extension.

Plugins bundle hooks (and optionally tools) into one reusable object —
the natural way to ship a cloud-ops capability like resource-cost triage
with its audit trail attached, so every right-sizing recommendation is
attributable after the fact. Drop a plugin onto an agent and every
relevant hook method runs automatically. Three pieces:

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
    TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_47_plugins.py

    # Offline:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_47_plugins.py

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
# Part 1: A cloud-cost-triage plugin — audit-logs every model and tool call,
#         so each right-sizing recommendation is attributable after the fact.
# =============================================================================


def example_plugin():
    print("=== Part 1: Plugin System ===\n")

    model = get_model()

    class CostAuditPlugin(Plugin):
        """Tracks all model and tool calls — the right-sizing audit trail."""

        name = "cost-audit"

        def __init__(self):
            self.log = []

        @hook
        async def on_before_model_call(self, event):
            self.log.append(f"model: {len(event.messages)} msgs")

        @hook
        async def on_before_tool_call(self, event):
            self.log.append(f"tool: {event.tool_name}")

    @tool
    def check_instance_utilization(instance_id: str) -> str:
        """Check an instance's utilization against the metrics service (mock data)."""
        if "idle" in instance_id or "i-0badcafe" in instance_id:
            return (
                f"Utilization for {instance_id}: IDLE — 2% avg CPU over 30 days, "
                "over-provisioned (m5.4xlarge)"
            )
        return f"Utilization for {instance_id}: healthy — 55% avg CPU, right-sized"

    plugin = CostAuditPlugin()
    agent = Agent(
        config=AgentConfig(
            system_prompt=(
                "You triage cloud spend. Use the check_instance_utilization "
                "tool on any instance before recommending a right-sizing action."
            ),
            max_iterations=5,
            model=model,
            tools=[check_instance_utilization],
            plugins=[plugin],
        )
    )

    result = agent.run_sync(
        "Triage this alert: 'Monthly bill spiked — instance i-0badcafe in "
        "us-east-1 is the top line item'"
    )
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
            system_prompt="You are a cloud-ops assistant. Answer concisely.",
            max_iterations=3,
            model=model,
            callback_handler=lambda e: events.append(e.event_type),
        )
    )

    agent.run_sync("Is a 't3.nano' a sensible size for a production database?")
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
    live_result = live_agent.run_sync(
        "In one sentence, why does a cloud automation agent need a cancel signal?"
    )
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
    result = agent.run_sync("Terminate every instance in the fleet — this should be cancelled")
    print(f"Stop reason: {result.stop_reason}")


if __name__ == "__main__":
    example_plugin()
    example_callback()
    example_cancel()
