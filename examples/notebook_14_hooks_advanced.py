# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 14: WARDEN — guardrails around a vuln-triage agent.

Notebook 12 covered hook basics. Here we build WARDEN, the SOC's
risk-gating layer: the hooks that sit between a triage agent and its
tools and refuse the actions that would destroy evidence or overreach.
This notebook focuses on the safety properties Tulip enforces on the
event objects hooks see, and on the control lever a hook pulls
mid-flight — ``event.cancel`` to block a destructive tool call.
Write-protected events are what keep the audit trail honest: a hook
cannot quietly relabel the action an agent took.

A triage agent that can delete quarantined samples or isolate hosts on
its own initiative is a textbook case of excessive agency (OWASP LLM06)
and tool misuse (OWASP ASI02). WARDEN is the bound on that agency.

Key ideas:
- Most fields on hook event objects are read-only. Try to overwrite
  ``event.tool_name`` and you get an ``AttributeError`` — that's the
  framework guarding the agent's invariants (and your audit log).
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

from config import get_model

from tulip.agent import Agent, AgentConfig
from tulip.hooks.provider import HookProvider
from tulip.tools.decorator import tool


# =============================================================================
# Part 1: WARDEN cancels a destructive tool call
# =============================================================================


def example_cancel_tool():
    """Set event.cancel to short-circuit a tool call and feed back a message."""
    print("=== Part 1: Cancel Tool via Hook ===\n")

    model = get_model()

    class WardenEvidenceHook(HookProvider):
        """WARDEN: block any tool whose name contains 'delete' — samples are evidence.

        Deleting a quarantined sample mid-investigation would destroy the
        artifact a finding rests on. WARDEN gates it (OWASP LLM06,
        Excessive Agency).
        """

        @property
        def priority(self):
            return 50  # Lower than the default security band so this runs first.

        async def on_before_tool_call(self, event):
            if "delete" in event.tool_name:
                # event.cancel = "<reason>" tells the loop: don't run the
                # tool; surface "<reason>" as the tool's result so the
                # model sees what happened.
                event.cancel = f"BLOCKED: {event.tool_name} is forbidden while the case is open"
                # event.tool_name = "spoofed"  # would raise AttributeError

    @tool
    def delete_sample(path: str) -> str:
        """Delete a quarantined malware sample."""
        return f"Deleted {path}"

    @tool
    def read_sandbox_report(path: str) -> str:
        """Read the sandbox detonation report for a quarantined sample."""
        return f"Report for {path}: matches EICAR test signature, severity low"

    agent = Agent(
        config=AgentConfig(
            system_prompt="You manage quarantined samples. If an action is blocked, "
            "tell the analyst why.",
            max_iterations=5,
            model=model,
            tools=[delete_sample, read_sandbox_report],
            hooks=[WardenEvidenceHook()],
        )
    )

    result = agent.run_sync("Delete /quarantine/sample_aa11bb22.bin")
    print(f"Response: {result.message[:150]}")
    for te in result.tool_executions:
        print(f"  Tool: {te.tool_name} → {te.result}")


# =============================================================================
# Part 2: which fields are writable and which raise
# =============================================================================


def example_write_protection():
    """Probe a BeforeToolCallEvent directly — see what mutations are allowed."""
    print("\n=== Part 2: Write Protection ===\n")

    from tulip.hooks.provider import BeforeToolCallEvent

    event = BeforeToolCallEvent(
        tool_name="isolate_host", tool_call_id="c1", arguments={"host": "web-01"}
    )

    # Writable: arguments and cancel.
    event.arguments = {"host": "web-02"}
    event.cancel = "blocked pending approval"
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
    res = agent.run_sync(
        "In one sentence, why does Tulip mark BeforeToolCallEvent.tool_name as "
        "read-only while letting hooks edit `arguments` and `cancel` — and why "
        "does that matter for a security audit trail?"
    )
    dt = _t.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · "
        f"{res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    print(f"  AI rationale: {res.message.strip()}")


if __name__ == "__main__":
    example_cancel_tool()
    example_write_protection()
