# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 14: RELEASE GUARD — guardrails around a deploy-ops agent.

Notebook 12 covered hook basics. Here we build RELEASE GUARD, the
platform team's change-gating layer: the hooks that sit between a
deploy-ops agent and its tools and refuse the actions that would tear
down running infrastructure or violate a change freeze. This notebook
focuses on the safety properties Tulip enforces on the event objects
hooks see, and on the control lever a hook pulls mid-flight —
``event.cancel`` to block a destructive tool call. Write-protected
events are what keep the change log honest: a hook cannot quietly
relabel the action an agent took.

A deploy agent that can delete namespaces or destroy environments on
its own initiative is exactly the kind of unbounded blast radius that
turns a routine rollout into an outage. RELEASE GUARD is the bound on
that agency.

Key ideas:
- Most fields on hook event objects are read-only. Try to overwrite
  ``event.tool_name`` and you get an ``AttributeError`` — that's the
  framework guarding the agent's invariants (and your change log).
- A small set of fields *is* writable: ``event.arguments`` (so a hook
  can rewrite tool input) and ``event.cancel`` (set it to a string to
  block the call and surface that message as the tool's "result").
- Hooks declare a ``priority``; lower runs first. The reverse order
  applies on the "after" callbacks so cleanup unwinds in LIFO order.

Run it:
    .venv/bin/python examples/notebook_14_hooks_advanced.py

The default provider is the mock model; set TULIP_MODEL_PROVIDER for a live one (e.g.
``openai.gpt-4.1`` or ``meta.llama-3.3-70b-instruct``). Set
``TULIP_MODEL_PROVIDER=mock`` for offline runs; OpenAI and Anthropic
also work.

Prerequisite: notebook 12.
"""

import asyncio

from config import get_model

from tulip.agent import Agent, AgentConfig
from tulip.hooks.provider import HookProvider
from tulip.tools.decorator import tool


# =============================================================================
# Part 1: RELEASE GUARD cancels a destructive tool call
# =============================================================================


async def example_cancel_tool():
    """Set event.cancel to short-circuit a tool call and feed back a message."""
    print("=== Part 1: Cancel Tool via Hook ===\n")

    model = get_model()

    class ReleaseGuardHook(HookProvider):
        """RELEASE GUARD: block any tool whose name contains 'delete' — infra is live.

        Deleting a namespace mid-rollout would take down the very
        workloads the deploy is supposed to ship. RELEASE GUARD gates
        it so a destructive op can't fire without a human in the loop.
        """

        @property
        def priority(self):
            return 50  # Lower than the default band so this runs first.

        async def on_before_tool_call(self, event):
            if "delete" in event.tool_name:
                # event.cancel = "<reason>" tells the loop: don't run the
                # tool; surface "<reason>" as the tool's result so the
                # model sees what happened.
                event.cancel = (
                    f"BLOCKED: {event.tool_name} is forbidden during an active change freeze"
                )
                # event.tool_name = "spoofed"  # would raise AttributeError

    @tool
    def delete_namespace(name: str) -> str:
        """Delete a Kubernetes namespace and everything in it."""
        return f"Deleted namespace {name}"

    @tool
    def get_rollout_status(name: str) -> str:
        """Read the rollout status for a deployment."""
        return f"Status for {name}: 3/3 replicas ready, last revision healthy"

    agent = Agent(
        config=AgentConfig(
            system_prompt="You operate the production cluster. If an action is blocked, "
            "tell the on-call engineer why.",
            max_iterations=5,
            model=model,
            tools=[delete_namespace, get_rollout_status],
            hooks=[ReleaseGuardHook()],
        )
    )

    result = await agent.arun("Delete the namespace payments-staging")
    print(f"Response: {result.message[:150]}")
    for te in result.tool_executions:
        print(f"  Tool: {te.tool_name} → {te.result}")


# =============================================================================
# Part 2: which fields are writable and which raise
# =============================================================================


async def example_write_protection():
    """Probe a BeforeToolCallEvent directly — see what mutations are allowed."""
    print("\n=== Part 2: Write Protection ===\n")

    from tulip.hooks.provider import BeforeToolCallEvent

    event = BeforeToolCallEvent(
        tool_name="scale_deployment", tool_call_id="c1", arguments={"replicas": 10}
    )

    # Writable: arguments and cancel.
    event.arguments = {"replicas": 3}
    event.cancel = "blocked pending change-approval ticket"
    print(f"arguments (writable): {event.arguments}")
    print(f"cancel (writable): {event.cancel}")

    # Read-only: tool_name, tool_call_id, ...
    try:
        event.tool_name = "spoofed"
    except AttributeError as e:
        print(f"tool_name (read-only): {e}")

    # Ask the model to explain the design — exercises the configured provider.
    import time as _t

    agent = Agent(model=get_model(max_tokens=80), system_prompt="Reply in one short sentence.")
    t0 = _t.perf_counter()
    res = await agent.arun(
        "In one sentence, why does Tulip mark BeforeToolCallEvent.tool_name as "
        "read-only while letting hooks edit `arguments` and `cancel` — and why "
        "does that matter for an infrastructure change log?"
    )
    dt = _t.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · "
        f"{res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    print(f"  AI rationale: {res.message.strip()}")


async def main():
    """Run all notebook parts."""
    await example_cancel_tool()
    await example_write_protection()


if __name__ == "__main__":
    asyncio.run(main())
