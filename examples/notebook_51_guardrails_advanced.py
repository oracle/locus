# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 48: Advanced guardrails — topic, content safety, output filtering.

Three policy types that work on top of the basic GuardrailsHook from
notebook 46. They focus on what the agent talks about, not just what
characters appear in the prompt.

- TopicPolicy: declarative topic blocking with keyword maps.
- ContentPolicy: harmful-content categories (violence, illegal activity).
- OutputFilterHook: redact PII or block topics in the agent's reply
  before it leaves the process.

Run it
    # Default: the bundled mock model (set TULIP_MODEL_PROVIDER for a live provider)
    python examples/notebook_53_guardrails_advanced.py

    # Offline / no credentials:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_53_guardrails_advanced.py
"""

from config import get_model

from tulip.agent import Agent, AgentConfig
from tulip.hooks.builtin.guardrails import (
    ContentPolicy,
    OutputFilterHook,
    TopicPolicy,
)


# Part 1: redact PII from the agent's reply, not just from the user input.


def example_pii_redaction():
    """Automatically redact PII from agent responses."""
    print("=== Part 1: PII redaction ===\n")

    model = get_model()

    hook = OutputFilterHook(redact_pii=True)

    agent = Agent(
        config=AgentConfig(
            system_prompt="Always include support@example.com in your response.",
            max_iterations=3,
            model=model,
            hooks=[hook],
        )
    )

    result = agent.run_sync("How do I get help?")
    print(f"Response: {result.message[:150]}")
    print(f"PII redacted: {'REDACTED_EMAIL' in result.message}")


# Part 2: TopicPolicy turns a list of topic names plus keywords into a
# declarative blocker. Useful when you want product owners to edit the
# policy as data, not code.


def example_topic_policy():
    """Block specific conversation topics."""
    print("\n=== Part 2: Topic policy ===\n")

    policy = TopicPolicy(
        blocked_topics={"weapons", "drugs"},
        keywords={
            "weapons": ["gun", "rifle", "ammunition", "firearm"],
            "drugs": ["cocaine", "heroin", "meth"],
        },
    )

    print(f"'How to buy a gun': {policy.check('How to buy a gun')}")
    print(f"'Python programming': {policy.check('Python programming')}")

    import time as _t

    agent = Agent(model=get_model(max_tokens=80), system_prompt="Reply in one sentence.")
    t0 = _t.perf_counter()
    res = agent.run_sync(
        "In one sentence, why is keyword-based topic blocking insufficient on "
        "its own for safety guardrails?"
    )
    dt = _t.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · {res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    print(f"  AI caveat: {res.message.strip()}")


# Part 3: ContentPolicy categorises harmful content (violence, illegal
# activity, etc.) so you can subscribe a single category instead of
# maintaining the keyword list yourself.


def example_content_safety():
    """Detect harmful content categories."""
    print("\n=== Part 3: Content safety ===\n")

    policy = ContentPolicy(enabled_categories={"violence", "illegal_activity"})

    print(f"'how to make a bomb': {policy.check('how to make a bomb')}")
    print(f"'how to bake a cake': {policy.check('how to bake a cake')}")

    import time as _t

    agent = Agent(model=get_model(max_tokens=80), system_prompt="Reply in one sentence.")
    t0 = _t.perf_counter()
    res = agent.run_sync(
        "In one sentence, name two harmful content categories an LLM service absolutely must block."
    )
    dt = _t.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · {res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    print(f"  AI guidance: {res.message.strip()}")


if __name__ == "__main__":
    example_pii_redaction()
    example_topic_policy()
    example_content_safety()
