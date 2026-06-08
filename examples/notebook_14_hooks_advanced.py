# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL
"""
Notebook 15: advanced hooks — cancel, retry, write-protected events.

Notebook 12 covered hook basics. This one focuses on the safety
properties Tulip enforces on the event objects hooks see, and on the
two control levers a hook can pull mid-flight: ``event.cancel`` to skip
a tool call, and ``event.retry`` to re-issue a model call.

Key ideas:
- Most fields on hook event objects are read-only. Try to overwrite
  ``event.tool_name`` and you get an ``AttributeError`` — that's the
  framework guarding the agent's invariants.
- A small set of fields *is* writable: ``event.arguments`` (so a hook
  can rewrite tool input) and ``event.cancel`` (set it to a string to
  block the call and surface that message as the tool's "result").
- Hooks declare a ``priority``; lower runs first. The reverse order
  applies on the "after" callbacks so cleanup unwinds in LIFO order.

Run it:
    .venv/bin/python examples/notebook_20_hooks_advanced.py

The default provider is the mock model; set TULIP_MODEL_PROVIDER for a live one (e.g.
``openai.gpt-4.1`` or ``meta.llama-3.3-70b-instruct``). Set
``TULIP_MODEL_PROVIDER=mock`` for offline runs; OpenAI, Anthropic, and
Anthropic also works.

Prerequisite: notebook 12.
"""

from config import get_model

from tulip.agent import Agent, AgentConfig
from tulip.hooks.provider import HookProvider
from tulip.tools.decorator import tool


# =============================================================================
# Part 1: cancel a dangerous tool call from a hook
# =============================================================================


def example_cancel_tool():
    """Set event.cancel to short-circuit a tool call and feed back a message."""
    print("=== Part 1: Cancel Tool via Hook ===\n")

    model = get_model()

    class SecurityHook(HookProvider):
        """Block any tool whose name contains 'delete'."""

        @property
        def priority(self):
            return 50  # Lower than the default security band so this runs first.

        async def on_before_tool_call(self, event):
            if "delete" in event.tool_name:
                # event.cancel = "<reason>" tells the loop: don't run the
                # tool; surface "<reason>" as the tool's result so the
                # model sees what happened.
                event.cancel = f"BLOCKED: {event.tool_name} is forbidden"
                # event.tool_name = "hacked"  # would raise AttributeError

    @tool
    def delete_file(path: str) -> str:
        """Delete a file."""
        return f"Deleted {path}"

    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        return f"Contents of {path}"

    agent = Agent(
        config=AgentConfig(
            system_prompt="You manage files. If blocked, tell the user.",
            max_iterations=5,
            model=model,
            tools=[delete_file, read_file],
            hooks=[SecurityHook()],
        )
    )

    result = agent.run_sync("Delete /tmp/secret.txt")
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

    event = BeforeToolCallEvent(tool_name="test", tool_call_id="c1", arguments={"x": 1})

    # Writable: arguments and cancel.
    event.arguments = {"x": 2}
    event.cancel = "blocked"
    print(f"arguments (writable): {event.arguments}")
    print(f"cancel (writable): {event.cancel}")

    # Read-only: tool_name, tool_call_id, ...
    try:
        event.tool_name = "hacked"
    except AttributeError as e:
        print(f"tool_name (read-only): {e}")

    # Ask the model to explain the design — exercises the configured provider.
    import time as _t

    agent = Agent(model=get_model(max_tokens=80), system_prompt="Reply in one short sentence.")
    t0 = _t.perf_counter()
    res = agent.run_sync(
        "In one sentence, why does Tulip mark BeforeToolCallEvent.tool_name as "
        "read-only while letting hooks edit `arguments` and `cancel`?"
    )
    dt = _t.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · {res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    print(f"  AI rationale: {res.message.strip()}")


if __name__ == "__main__":
    example_cancel_tool()
    example_write_protection()
