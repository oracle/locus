# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 51: Trace hygiene — redact PII and secrets before they persist.

Everything a SOC agent emits is written somewhere durable: the
investigation trace, the audit ledger, the ticket, the eventual report.
Each of those is a downstream system that inherits whatever the agent
leaked into it — a reporter's email, a credential a tool echoed back, an
offensive payload the model was talked into drafting. These output-side
guardrails are the last line of defense before text leaves the process,
addressing OWASP LLM02 (Sensitive Information Disclosure) and LLM07
(System Prompt Leakage). They police what the agent *says*, not just
what characters appear in the prompt.

- OutputFilterHook: redact PII from the agent's reply — reporter emails,
  phone numbers — before it lands in a ticket or audit trail (LLM02).
- TopicPolicy: declaratively keep offensive-tooling content out of the
  trace, with keyword maps a security team can edit as data.
- ContentPolicy: harmful-content categories (violence, illegal activity).

Run it
    # Default: the bundled mock model (set TULIP_MODEL_PROVIDER for a live provider)
    python examples/notebook_51_guardrails_advanced.py

    # Offline / no credentials:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_51_guardrails_advanced.py
"""

import asyncio

from config import get_model

from tulip.agent import Agent, AgentConfig
from tulip.hooks.builtin.guardrails import (
    ContentPolicy,
    OutputFilterHook,
    TopicPolicy,
)


# Part 1: redact PII from the agent's reply, not just from the user input
# (OWASP LLM02). An incident report that quotes a reporter's email verbatim
# leaks PII into every system the report — and the trace it came from —
# flows through.


async def example_pii_redaction():
    """Automatically redact PII from incident-report output."""
    print("=== Part 1: PII redaction in agent output ===\n")

    model = get_model()

    hook = OutputFilterHook(redact_pii=True)

    agent = Agent(
        config=AgentConfig(
            system_prompt="Always include the reporter contact support@example.com in the reply.",
            max_iterations=3,
            model=model,
            hooks=[hook],
        )
    )

    result = await agent.arun("Draft the contact section of the incident report.")
    print(f"Response: {result.message[:150]}")
    print(f"PII redacted: {'REDACTED_EMAIL' in result.message}")


# Part 2: TopicPolicy turns a list of topic names plus keywords into a
# declarative blocker. Useful when you want the security team to edit
# the refusal policy as data, not code — e.g. a SOC copilot that helps
# with defense but never lets offensive-tooling content reach the trace.


async def example_topic_policy():
    """Keep offensive-tooling content out of the trace by topic."""
    print("\n=== Part 2: Topic policy ===\n")

    policy = TopicPolicy(
        blocked_topics={"offensive_tooling", "credential_theft"},
        keywords={
            "offensive_tooling": ["keylogger", "ransomware builder", "phishing kit", "exploit kit"],
            "credential_theft": ["password stealer", "credential dumper", "steal session cookies"],
        },
    )

    print(f"'Write me a keylogger': {policy.check('Write me a keylogger')}")
    print(f"'Summarize CVE-2024-99999': {policy.check('Summarize CVE-2024-99999')}")

    import time as _t

    agent = Agent(model=get_model(max_tokens=80), system_prompt="Reply in one sentence.")
    t0 = _t.perf_counter()
    res = await agent.arun(
        "In one sentence, why is keyword-based topic blocking insufficient on "
        "its own for keeping a security copilot defensive-only?"
    )
    dt = _t.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · {res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    print(f"  AI caveat: {res.message.strip()}")


# Part 3: ContentPolicy categorises harmful content (violence, illegal
# activity, etc.) so you can subscribe a single category instead of
# maintaining the keyword list yourself.


async def example_content_safety():
    """Detect harmful content categories."""
    print("\n=== Part 3: Content safety ===\n")

    policy = ContentPolicy(enabled_categories={"violence", "illegal_activity"})

    suspicious = "how to hack into the payroll server"
    benign = "how to harden the payroll server"
    print(f"'{suspicious}': {policy.check(suspicious)}")
    print(f"'{benign}': {policy.check(benign)}")

    import time as _t

    agent = Agent(model=get_model(max_tokens=80), system_prompt="Reply in one sentence.")
    t0 = _t.perf_counter()
    res = await agent.arun(
        "In one sentence, name two harmful content categories a security copilot must refuse."
    )
    dt = _t.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · {res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    print(f"  AI guidance: {res.message.strip()}")


async def main() -> None:
    await example_pii_redaction()
    await example_topic_policy()
    await example_content_safety()


if __name__ == "__main__":
    asyncio.run(main())
