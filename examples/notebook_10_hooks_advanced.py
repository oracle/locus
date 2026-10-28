# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""
Notebook 10: RELEASE GUARD — guardrails around a deploy-ops agent.

Notebook 09 covered hook basics. Here we build RELEASE GUARD, the
platform team's change-gating layer: the hooks that sit between a
deploy-ops agent and its tools and refuse the actions that would tear
down running infrastructure or violate a change freeze. This notebook
focuses on the safety properties Locus enforces on the event objects
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
- Hooks run in the order they are listed (a ``HookRegistry`` sorts them
  by ``priority`` first); the "after" callbacks unwind in reverse so
  cleanup is LIFO.

Run it:
    .venv/bin/python examples/notebook_10_hooks_advanced.py

The default provider is OCI GenAI when ``~/.oci/config`` holds a usable profile,
otherwise the bundled mock model; set ``LOCUS_MODEL_PROVIDER`` (``oci``,
``openai``, ``anthropic``, ``mock``) to choose.

Prerequisite: notebook 09.
"""

import asyncio

from config import get_model, print_config

from locus.agent import Agent, AgentConfig
from locus.hooks.provider import HookProvider
from locus.tools.decorator import tool


@tool
def get_rollout_status(name: str) -> str:
    """Read the rollout status for a deployment."""
    return f"Status for {name}: 3/3 replicas ready, last revision healthy"


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
            return 50  # SECURITY band; list it first so it runs first.

        async def on_before_tool_call(self, event):
            if "delete" in event.tool_name:
                # event.cancel = "<reason>" tells the loop: don't run the
                # tool; surface "<reason>" as the tool's result so the
                # model sees what happened.
                event.cancel = (
                    f"BLOCKED: {event.tool_name} is forbidden during an active change freeze"
                )

    @tool
    def delete_namespace(name: str) -> str:
        """Delete a Kubernetes namespace and everything in it."""
        return f"Deleted namespace {name}"

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
    print()


# =============================================================================
# Part 2: which fields are writable and which raise
# =============================================================================


async def example_write_protection():
    """Probe a BeforeToolCallEvent directly — see what mutations are allowed."""
    print("=== Part 2: Write Protection ===\n")

    from locus.hooks.provider import BeforeToolCallEvent

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
    print()


# =============================================================================
# Part 3: list order on the way in, reverse order on the way out
# =============================================================================


class TraceHook(HookProvider):
    """Append its name to a shared log on the way into and out of each tool call."""

    def __init__(self, name: str, priority: int, log: list[str]):
        self._name = name
        self._priority = priority
        self._log = log

    @property
    def priority(self) -> int:
        return self._priority

    async def on_before_tool_call(self, event):
        self._log.append(f"before:{self._name}")

    async def on_after_tool_call(self, event):
        self._log.append(f"after:{self._name}")


async def example_hook_order():
    """Two hooks — watch the order flip on the 'after' side.

    The agent runs the "before" callbacks in the order the hooks were
    given, so list the security gate first and it sees a call before the
    business ledger does. The "after" callbacks unwind in reverse, like
    nested context managers: whoever was let in last is let out first,
    and the gate gets the final word on the record.
    """
    print("=== Part 3: Hook ordering ===\n")

    order: list[str] = []
    gate = TraceHook("gate", priority=50, log=order)  # security band
    ledger = TraceHook("ledger", priority=250, log=order)  # business band

    agent = Agent(
        model=get_model(),
        tools=[get_rollout_status],
        system_prompt="Use get_rollout_status to answer rollout questions.",
        # Gate first: the agent dispatches before-hooks in list order and
        # unwinds the after-hooks in reverse (LIFO).
        hooks=[gate, ledger],
    )
    await agent.arun("Check the rollout status of payments-api")

    print(f"Call sequence: {' → '.join(order)}")
    print("  gate runs first on the way in; ledger unwinds first on the way out.")
    print()


async def main():
    """Run all notebook parts."""
    print("=" * 60)
    print("Notebook 10: RELEASE GUARD — Guardrails Around a Deploy-Ops Agent")
    print("=" * 60)
    print()

    print_config()
    print()

    await example_cancel_tool()
    await example_write_protection()
    await example_hook_order()

    print("=" * 60)
    print("Next: Notebook 11 — Model providers: one cloud-ops agent, swap models with a string")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
