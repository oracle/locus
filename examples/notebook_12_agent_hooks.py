# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 12: audit hooks — every agent action on the record.

A ``HookProvider`` plugs into four lifecycle points: before/after the
whole invocation, and before/after each tool call. For a cloud-ops
agent that's an audit trail for free — every scaling action it takes
can be logged, timed, validated, or blocked without touching the agent
or the tools themselves.

Key ideas:
- Subclass ``HookProvider`` and override only the callbacks you need.
- Each hook declares a ``priority``; lower runs earlier. ``HookPriority``
  exposes named constants for the common bands (SECURITY, OBSERVABILITY,
  BUSINESS).
- ``on_before_tool_call(event)`` can mutate ``event.arguments`` — handy
  for clamping or sanitising values before the tool actually runs.
- Multiple hooks compose. Tulip iterates them in priority order.

The Part 5 guardrail watches untrusted manifest text for secrets before
they flow onward — the control behind OWASP LLM02 (Sensitive Information
Disclosure). Notebook 14 builds on this with a deploy-gating agent that
risk-scores each change before it lands.

Run it:
    .venv/bin/python examples/notebook_12_agent_hooks.py

The default provider is the mock model; set TULIP_MODEL_PROVIDER for a live one (e.g.
``openai.gpt-4.1`` or ``meta.llama-3.3-70b-instruct``). Set
``TULIP_MODEL_PROVIDER=mock`` for offline runs; OpenAI and Anthropic
also work.

Prerequisite: notebook 11.
"""

from datetime import datetime

# Import shared config
from config import get_model, print_config

from tulip.agent import Agent
from tulip.hooks import HookPriority, HookProvider
from tulip.tools import tool


# =============================================================================
# Part 1: an audit-trail hook
# =============================================================================


class AuditTrailHook(HookProvider):
    """Print an audit line at each of the four lifecycle points."""

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


def example_simple_hook():
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
    result = agent.run_sync("Scale the service: base 5, delta 3")
    print(f"\nResult: {result.message}")
    print()


# =============================================================================
# Part 2: a timing hook
# =============================================================================


class TimingHook(HookProvider):
    """Time the whole invocation and each tool call — cloud-ops telemetry."""

    def __init__(self):
        self.start_time = None
        self.tool_times = {}

    @property
    def priority(self) -> int:
        return HookPriority.OBSERVABILITY_MIN

    async def on_before_invocation(self, prompt, state):
        self.start_time = datetime.now()
        self.tool_times = {}
        return state

    async def on_after_invocation(self, state, success):
        elapsed = (datetime.now() - self.start_time).total_seconds() * 1000
        print("\n  Timing Report:")
        print(f"    Total: {elapsed:.1f}ms")
        for name, ms in self.tool_times.items():
            print(f"    {name}: {ms:.1f}ms")

    async def on_before_tool_call(self, event):
        self.tool_times[event.tool_name] = datetime.now().timestamp() * 1000

    async def on_after_tool_call(self, event):
        start = self.tool_times.get(event.tool_name, 0)
        self.tool_times[event.tool_name] = (datetime.now().timestamp() * 1000) - start


def example_timing_hook():
    """Use a hook to time the agent and its tool calls."""
    print("=== Part 2: Timing Hook ===\n")

    model = get_model(max_tokens=100)

    agent = Agent(
        model=model,
        tools=[adjust_capacity],
        system_prompt="Use the adjust_capacity tool to scale service instances.",
        hooks=[TimingHook()],
    )

    result = agent.run_sync("Scale the service: base 10, delta 20")
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


def example_validation_hook():
    """Clamp arguments before the tool sees them."""
    print("=== Part 3: Validation Hook ===\n")

    model = get_model(max_tokens=150)

    agent = Agent(
        model=model,
        tools=[adjust_capacity],
        system_prompt="Use the adjust_capacity tool. Try large numbers if asked.",
        hooks=[ValidationHook(max_value=100)],
    )

    result = agent.run_sync("Scale the service: base 5000, delta 3000")
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


def example_multiple_hooks():
    """Two hooks on one agent — priority decides who runs first."""
    print("=== Part 4: Multiple Hooks ===\n")

    model = get_model(max_tokens=100)

    timing = TimingHook()
    audit = AuditHook()

    # Lower priority value runs earlier: timing (100) then audit (200).
    agent = Agent(
        model=model,
        tools=[adjust_capacity],
        system_prompt="Use the adjust_capacity tool.",
        hooks=[timing, audit],
    )

    result = agent.run_sync("Scale the service: base 7, delta 8")
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


def example_guardrails_hook():
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
    result = agent.run_sync("Analyze this manifest: 'env: DB_PASSWORD=hunter2'")
    print(f"Result: {result.message}")

    if guardrails.blocked_calls:
        print(f"\nBlocked calls detected: {len(guardrails.blocked_calls)}")
    print()


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all notebook parts."""
    print("=" * 60)
    print("Notebook 12: Audit Hooks & Lifecycle")
    print("=" * 60)
    print()

    print_config()
    print()

    example_simple_hook()
    example_timing_hook()
    example_validation_hook()
    example_multiple_hooks()
    example_guardrails_hook()

    print("=" * 60)
    print("Next: Notebook 13 — Cloud Ops Dashboard SSE Streaming")
    print("=" * 60)


if __name__ == "__main__":
    main()
