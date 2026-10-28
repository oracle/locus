# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
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
- ``Agent.cancel()`` — stop a running agent from another task or
  thread; the next loop step returns ``stop_reason="cancelled"``.

Run it:
    # Default: OCI GenAI if ~/.oci/config has a usable profile, else the bundled mock model
    LOCUS_MODEL_PROVIDER=openai LOCUS_MODEL_ID=gpt-4o python examples/notebook_47_plugins.py

    # Offline:
    LOCUS_MODEL_PROVIDER=mock python examples/notebook_47_plugins.py

Prerequisites:
- An OpenAI or Anthropic API key, or set ``LOCUS_MODEL_PROVIDER`` to
  ``openai`` / ``anthropic`` / ``mock``.
"""

import asyncio

from config import get_model

from locus.agent import Agent, AgentConfig
from locus.hooks.plugin import Plugin, hook
from locus.tools.decorator import tool


# =============================================================================
# Part 1: A cloud-cost-triage plugin — audit-logs every model and tool call,
#         so each right-sizing recommendation is attributable after the fact.
# =============================================================================


async def example_plugin():
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
        if "idle" in instance_id or "badcafe" in instance_id:
            return (
                f"Utilization for {instance_id}: IDLE — 2% avg CPU over 30 days, "
                "over-provisioned (VM.Standard.E4.Flex, 16 OCPUs)"
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

    result = await agent.arun(
        "Triage this alert: 'Monthly bill spiked — instance "
        "ocid1.instance.oc1.iad.0badcafe in us-ashburn-1 is the top line item'"
    )
    print(f"Response: {result.message[:100]}...")
    print(f"Audit log: {plugin.log}")


# =============================================================================
# Part 2: callback_handler — when a plain function is enough.
# =============================================================================


async def example_callback():
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

    await agent.arun("Is a 1-OCPU VM.Standard.A1.Flex a sensible size for a production database?")
    print(f"Events received: {events}")


# =============================================================================
# Part 3: Agent.cancel() — stop an in-flight run from another task.
# =============================================================================


async def example_cancel():
    print("\n=== Part 3: Cancel Signal ===\n")

    @tool
    async def terminate_instance(instance_id: str) -> str:
        """Terminate a compute instance (mock: slow, so the run is still in flight)."""
        # A real control-plane call takes a while; the pause is what gives
        # the cancel below a running loop to interrupt, under any provider.
        await asyncio.sleep(0.5)
        return f"{instance_id}: termination requested"

    agent = Agent(
        config=AgentConfig(
            system_prompt="You operate the fleet. Use terminate_instance when asked.",
            max_iterations=5,
            model=get_model(max_tokens=80),
            tools=[terminate_instance],
        )
    )

    # The run goes into the background; the cancel comes from this task a
    # moment later — the shape of an operator hitting "stop" mid-run.
    run = asyncio.create_task(
        agent.arun("Terminate every instance in the fleet, starting with web-01.")
    )
    await asyncio.sleep(0.05)
    signalled = agent.cancel()
    print(f"cancel() signalled {signalled} in-flight run(s)")

    result = await run
    print(f"Stop reason: {result.stop_reason}")
    print(f"Tool calls completed before the stop: {len(result.tool_executions)}")


async def main() -> None:
    await example_plugin()
    await example_callback()
    await example_cancel()


if __name__ == "__main__":
    asyncio.run(main())
