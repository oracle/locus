# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""
Notebook 09: audit hooks — every agent action on the record.

A ``HookProvider`` exposes nine lifecycle callbacks; this notebook uses
the four most-used of them: before/after the whole invocation, and
before/after each tool call. For a cloud-ops
agent that's an audit trail for free — every scaling action it takes
can be logged, timed, validated, or blocked without touching the agent
or the tools themselves.

Key ideas:
- Subclass ``HookProvider`` and override only the callbacks you need.
- Each hook declares a ``priority`` band (``HookPriority`` names the
  common ones: SECURITY, OBSERVABILITY, BUSINESS). A ``HookRegistry``
  sorts on it; an ``Agent`` runs hooks in the order you list them.
- ``on_before_tool_call(event)`` can mutate ``event.arguments`` — handy
  for clamping or sanitising values before the tool actually runs.
- Multiple hooks compose. Locus runs them in list order on the way in
  and in reverse on the way out.

The Part 5 guardrail watches untrusted manifest text for secrets before
they flow onward — the control behind OWASP LLM02 (Sensitive Information
Disclosure). Notebook 10 builds on this with RELEASE GUARD, a hook that
cancels destructive tool calls before they run.

Run it:
    .venv/bin/python examples/notebook_09_agent_hooks.py

The default provider is OCI GenAI when ``~/.oci/config`` holds a usable profile,
otherwise the bundled mock model; set ``LOCUS_MODEL_PROVIDER`` (``oci``,
``openai``, ``anthropic``, ``mock``) to choose.

Prerequisite: notebook 06.
"""

import asyncio
from datetime import datetime

from config import get_model, print_config

from locus.agent import Agent
from locus.hooks import HookPriority, HookProvider
from locus.tools import tool


# =============================================================================
# Part 1: an audit-trail hook
# =============================================================================


class AuditTrailHook(HookProvider):
    """Print an audit line at each of the four lifecycle points this notebook uses."""

    @property
    def priority(self) -> int:
        return HookPriority.OBSERVABILITY_DEFAULT

    async def on_before_invocation(self, prompt, state):
        print(f"  [AUDIT] Run started: '{prompt[:50]}...'")
        return state

    async def on_after_invocation(self, state, success):
        print(f"  [AUDIT] Run finished: success={success}")

    async def on_before_tool_call(self, event):
        print(f"  [AUDIT] Tool call: {event.tool_name}({event.arguments})")

    async def on_after_tool_call(self, event):
        if event.error:
            print(f"  [AUDIT] Tool error: {event.tool_name} -> {event.error}")
        else:
            print(f"  [AUDIT] Tool done: {event.tool_name} -> {str(event.result)[:50]}")


@tool
def adjust_capacity(base: int, delta: int) -> int:
    """Add a delta to a service's current instance count."""
    return base + delta


async def example_simple_hook():
    """Wire an audit hook into an autoscaling agent and run one prompt."""
    print("=== Part 1: Understanding Hooks ===\n")

    model = get_model(max_tokens=100)

    agent = Agent(
        model=model,
        tools=[adjust_capacity],
        system_prompt="Use the adjust_capacity tool to scale service instances.",
        hooks=[AuditTrailHook()],
    )

    print("Running agent with audit-trail hook:\n")
    result = await agent.arun("Scale the service: base 5, delta 3")
    print(f"\nResult: {result.message}")
    print()


# =============================================================================
# Part 2: a timing hook
# =============================================================================


class TimingHook(HookProvider):
    """Time the whole invocation and each tool call — cloud-ops telemetry."""

    def __init__(self):
        self.start_time = None
        self.started: dict[str, float] = {}  # tool_call_id -> start (ms)
        self.tool_times: list[tuple[str, float]] = []  # (tool_name, duration ms)

    @property
    def priority(self) -> int:
        return HookPriority.OBSERVABILITY_MIN

    async def on_before_invocation(self, prompt, state):
        self.start_time = datetime.now()
        self.started = {}
        self.tool_times = []
        return state

    async def on_after_invocation(self, state, success):
        elapsed = (datetime.now() - self.start_time).total_seconds() * 1000
        print("\n  Timing Report:")
        print(f"    Total: {elapsed:.1f}ms")
        for name, ms in self.tool_times:
            print(f"    {name}: {ms:.1f}ms")

    async def on_before_tool_call(self, event):
        # Keyed by tool_call_id, not tool_name: the same tool called twice in
        # one run would otherwise overwrite its own start time and report a
        # duration that belongs to neither call.
        self.started[event.tool_call_id] = datetime.now().timestamp() * 1000

    async def on_after_tool_call(self, event):
        start = self.started.pop(event.tool_call_id, None)
        if start is not None:
            self.tool_times.append((event.tool_name, datetime.now().timestamp() * 1000 - start))


async def example_timing_hook():
    """Use a hook to time the agent and its tool calls."""
    print("=== Part 2: Timing Hook ===\n")

    model = get_model(max_tokens=100)

    agent = Agent(
        model=model,
        tools=[adjust_capacity],
        system_prompt="Use the adjust_capacity tool to scale service instances.",
        hooks=[TimingHook()],
    )

    result = await agent.arun("Scale the service: base 10, delta 20")
    print(f"Result: {result.message}")
    print()


# =============================================================================
# Part 3: a hook that rewrites tool arguments
# =============================================================================


class ValidationHook(HookProvider):
    """Clamp out-of-range capacity changes before the tool runs."""

    def __init__(self, max_value: int = 1000):
        self.max_value = max_value
        self.blocked_count = 0

    @property
    def priority(self) -> int:
        return HookPriority.SECURITY_DEFAULT

    async def on_before_tool_call(self, event):
        if event.tool_name == "adjust_capacity":
            base = event.arguments.get("base", 0)
            delta = event.arguments.get("delta", 0)

            # event.arguments is writable; mutating it changes what the
            # tool actually receives — instance counts stay within the
            # quota that keeps the cloud bill in check.
            if base > self.max_value:
                print(f"  [VALIDATION] Clamping base={base} to {self.max_value}")
                event.arguments["base"] = self.max_value
            if delta > self.max_value:
                print(f"  [VALIDATION] Clamping delta={delta} to {self.max_value}")
                event.arguments["delta"] = self.max_value


async def example_validation_hook():
    """Clamp arguments before the tool sees them."""
    print("=== Part 3: Validation Hook ===\n")

    model = get_model(max_tokens=150)

    agent = Agent(
        model=model,
        tools=[adjust_capacity],
        system_prompt="Use the adjust_capacity tool. Try large numbers if asked.",
        hooks=[ValidationHook(max_value=100)],
    )

    result = await agent.arun("Scale the service: base 5000, delta 3000")
    print(f"Result: {result.message}")
    print()


# =============================================================================
# Part 4: composing multiple hooks
# =============================================================================


class AuditHook(HookProvider):
    """Record every tool call as structured entries for compliance review."""

    def __init__(self):
        self.audit_log = []

    @property
    def priority(self) -> int:
        return HookPriority.BUSINESS_DEFAULT

    async def on_before_tool_call(self, event):
        self.audit_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "tool": event.tool_name,
                "arguments": dict(event.arguments),
                "status": "started",
            }
        )

    async def on_after_tool_call(self, event):
        self.audit_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "tool": event.tool_name,
                "result": str(event.result)[:100] if event.result else None,
                "error": event.error,
                "status": "completed" if not event.error else "failed",
            }
        )

    def get_log(self):
        return self.audit_log


async def example_multiple_hooks():
    """Two hooks on one agent — list order decides who runs first."""
    print("=== Part 4: Multiple Hooks ===\n")

    model = get_model(max_tokens=100)

    timing = TimingHook()
    audit = AuditHook()

    # The Agent runs hooks in the order given: timing first, then audit. The
    # priority bands (OBSERVABILITY_MIN 100, BUSINESS_DEFAULT 250) only order
    # hooks that arrive through a HookRegistry.
    agent = Agent(
        model=model,
        tools=[adjust_capacity],
        system_prompt="Use the adjust_capacity tool.",
        hooks=[timing, audit],
    )

    result = await agent.arun("Scale the service: base 7, delta 8")
    print(f"Result: {result.message}")

    print("\nAudit Log:")
    for entry in audit.get_log():
        print(f"  {entry}")
    print()


# =============================================================================
# Part 5: a guardrail hook
# =============================================================================


class GuardrailsHook(HookProvider):
    """Watch prompts and tool arguments for sensitive-data patterns.

    A first line against OWASP LLM02 (Sensitive Information Disclosure):
    secrets in untrusted input never reach a tool that might log or
    forward them.
    """

    def __init__(self, blocked_patterns: list[str] | None = None):
        self.blocked_patterns = blocked_patterns or []
        self.blocked_calls = []

    @property
    def priority(self) -> int:
        # SECURITY_MIN places this hook ahead of every other band, so it
        # gets first look at the prompt and at every tool call.
        return HookPriority.SECURITY_MIN

    async def on_before_invocation(self, prompt, state):
        prompt_lower = prompt.lower()
        for pattern in self.blocked_patterns:
            if pattern.lower() in prompt_lower:
                print(f"  [GUARDRAIL] Blocked pattern detected: '{pattern}'")
                # Raising here would abort the run; this demo just warns.
        return state

    async def on_before_tool_call(self, event):
        args_str = str(event.arguments).lower()
        for pattern in self.blocked_patterns:
            if pattern.lower() in args_str:
                self.blocked_calls.append(
                    {
                        "tool": event.tool_name,
                        "pattern": pattern,
                        "arguments": dict(event.arguments),
                    }
                )
                print(f"  [GUARDRAIL] Warning: '{pattern}' in {event.tool_name} args")


@tool
def analyze_manifest(text: str) -> str:
    """Word and character counts plus a sha-256 digest of the manifest text."""
    import hashlib
    import re

    words = re.findall(r"\b\w+\b", text)
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    return (
        f"chars={len(text)} words={len(words)} unique_words={len({w.lower() for w in words})} "
        f"sha256={digest}"
    )


async def example_guardrails_hook():
    """A guardrail hook spots a sensitive-data pattern and warns."""
    print("=== Part 5: Guardrails Hook ===\n")

    model = get_model(max_tokens=100)

    guardrails = GuardrailsHook(blocked_patterns=["password", "api key", "access key"])

    agent = Agent(
        model=model,
        tools=[analyze_manifest],
        system_prompt="Analyze any deployment manifest text the operator provides.",
        hooks=[guardrails],
    )

    # The word "password" trips the guardrail — manifest text is untrusted
    # input and may carry secrets that must not flow onward.
    result = await agent.arun("Analyze this manifest: 'env: DB_PASSWORD=hunter2'")
    print(f"Result: {result.message}")

    if guardrails.blocked_calls:
        print(f"\nBlocked calls detected: {len(guardrails.blocked_calls)}")
    print()


async def main():
    """Run all notebook parts."""
    print("=" * 60)
    print("Notebook 09: Audit Hooks & Lifecycle")
    print("=" * 60)
    print()

    print_config()
    print()

    await example_simple_hook()
    await example_timing_hook()
    await example_validation_hook()
    await example_multiple_hooks()
    await example_guardrails_hook()

    print("=" * 60)
    print("Next: Notebook 10 — RELEASE GUARD: guardrails around a deploy-ops agent")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
