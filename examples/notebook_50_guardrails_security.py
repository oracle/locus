# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 47: Guardrails and security — block dangerous calls before the model sees them.

Each part wires a guardrail into a real agent run and prints the model
round-trip cost, so the safety policy is exercised live, not described
in the abstract.

- GuardrailsHook with a typed GuardrailConfig (block list, length caps,
  default action).
- PII detection and redaction on user input.
- Content pattern blocking (SQL injection, path traversal, shell escapes).
- Tool allowlist vs denylist.
- Stacked hooks via HookRegistry plus a separate ContentFilterHook.

Run it
    # Default: the bundled mock model (set TULIP_MODEL_PROVIDER for a live provider)
    python examples/notebook_52_guardrails_security.py

    # Offline / no credentials:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_52_guardrails_security.py

    # Pin a specific model:
    TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_52_guardrails_security.py
"""

import asyncio
import time

from config import get_model, print_config

from tulip.agent import Agent
from tulip.core.events import BeforeToolCallEvent
from tulip.core.state import AgentState
from tulip.hooks import HookRegistry
from tulip.hooks.builtin.guardrails import (
    ContentFilterHook,
    GuardrailAction,
    GuardrailConfig,
    GuardrailsHook,
    GuardrailViolation,
)


# Helper used by every Part: one model call with a timing/token banner so
# you can see the guardrail running against a real round-trip.


def _llm_call(
    prompt: str,
    *,
    system: str = "Reply in one short sentence.",
    max_tokens: int = 100,
    hooks: list | None = None,
) -> str:
    agent = Agent(
        model=get_model(max_tokens=max_tokens),
        system_prompt=system,
        hooks=hooks,
    )
    t0 = time.perf_counter()
    result = agent.run_sync(prompt)
    dt = time.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · "
        f"{result.metrics.prompt_tokens}→{result.metrics.completion_tokens} tokens]"
    )
    return result.message.strip()


async def main():
    print("=" * 60)
    print("Notebook 47: Guardrails and security")
    print("=" * 60)
    print()
    print_config()

    # Part 1: declare a GuardrailConfig, then ask the model to summarise it.
    print("\n=== Part 1: Basic guardrail configuration ===\n")
    config = GuardrailConfig(
        block_dangerous_tools=frozenset(
            {"eval", "exec", "system", "shell", "rm", "delete", "drop", "truncate"}
        ),
        max_prompt_length=100000,
        max_tool_result_length=50000,
        default_action=GuardrailAction.BLOCK,
    )
    print(f"  block_dangerous_tools: {sorted(config.block_dangerous_tools)[:5]}…")
    print(f"  max_prompt_length: {config.max_prompt_length:,}")
    print(f"  default_action: {config.default_action.value}")
    summary = _llm_call(
        "In one sentence, summarise what a security policy that blocks "
        "{eval, exec, system, shell, rm, delete, drop, truncate} protects "
        "an LLM agent against.",
        max_tokens=80,
    )
    print(f"AI policy summary: {summary}")

    # Part 2: wire the config into a GuardrailsHook and run an agent through it.
    print("\n=== Part 2: GuardrailsHook on a live agent ===\n")
    violations_log: list[GuardrailViolation] = []

    def on_violation(v: GuardrailViolation):
        violations_log.append(v)
        print(f"  VIOLATION: {v.rule_name} - {v.description}")

    guardrails = GuardrailsHook(config=config, on_violation=on_violation)
    print(f"  Hook: {guardrails.name}, priority={guardrails.priority}")
    answer = _llm_call(
        "What's a sensible default password policy length?",
        system="Reply in one short sentence.",
        hooks=[guardrails],
    )
    print(f"Guarded answer: {answer}")

    # Part 3: PII detection. The hook scans each input against the
    # configured patterns before the model ever sees it.
    print("\n=== Part 3: PII detection ===\n")
    print("Built-in PII patterns:")
    for name in list(config.pii_patterns)[:5]:
        print(f"  - {name}")

    test_inputs = [
        "Contact me at john@example.com for details",
        "Call 555-123-4567 for support",
        "SSN: 123-45-6789",
        "No sensitive data here",
    ]
    state = AgentState(agent_id="test")
    print("\nSDK-side PII detection:")
    for text in test_inputs:
        guardrails.clear_violations()
        try:
            await guardrails.on_before_invocation(text, state)
            seen = guardrails.violations
            label = ", ".join(v.rule_name for v in seen) if seen else "Clean"
            print(f"  '{text[:40]}…' -> {label}")
        except ValueError as e:
            print(f"  '{text[:40]}…' -> BLOCKED: {e}")

    pii_advice = _llm_call(
        "Give one concrete piece of advice for an SRE on what to do when an "
        "LLM application logs PII like emails or SSNs.",
        max_tokens=80,
    )
    print(f"AI advice: {pii_advice}")

    # Part 4: block known-malicious input shapes (SQL injection, path
    # traversal, shell escapes) by pattern.
    print("\n=== Part 4: Content pattern blocking ===\n")
    dangerous_inputs = [
        "DROP TABLE users;",
        "../../etc/passwd",
        "ls -la; rm -rf /",
        "Normal query SELECT * FROM users",
    ]
    for text in dangerous_inputs:
        guardrails.clear_violations()
        try:
            await guardrails.on_before_invocation(text, state)
            print(f"  '{text[:40]}…' -> Allowed")
        except ValueError:
            print(f"  '{text[:40]}…' -> BLOCKED")
    risk_summary = _llm_call(
        "List the top three classes of malicious input an LLM service should "
        "filter at the gateway. Three short bullets.",
        max_tokens=120,
    )
    print(f"AI risk summary:\n{risk_summary}")

    # Part 5: tool denylist. block_dangerous_tools rejects calls before
    # they reach the tool runner.
    print("\n=== Part 5: Tool restrictions ===\n")
    tool_tests = [
        ("read_file", {"path": "/app/data.txt"}),
        ("exec", {"code": "print('hello')"}),
        ("shell", {"command": "ls"}),
        ("search", {"query": "test"}),
    ]
    for name, args in tool_tests:
        guardrails.clear_violations()
        try:
            await guardrails.on_before_tool_call(
                BeforeToolCallEvent(tool_name=name, arguments=args)
            )
            print(f"  {name} -> Allowed")
        except ValueError:
            print(f"  {name} -> BLOCKED")
    rationale = _llm_call(
        "Why is it dangerous to expose `exec` or `shell` tools to an LLM agent?",
        max_tokens=80,
    )
    print(f"AI rationale: {rationale}")

    # Part 6: allowlist mode — safer default for production because new
    # tools added later need explicit listing.
    print("\n=== Part 6: Tool allowlist mode ===\n")
    allowlist_config = GuardrailConfig(
        allow_only_tools=frozenset({"read_file", "search", "analyze"})
    )
    allowlist_guardrails = GuardrailsHook(config=allowlist_config)
    for name in ["read_file", "write_file", "search", "delete"]:
        try:
            await allowlist_guardrails.on_before_tool_call(
                BeforeToolCallEvent(tool_name=name, arguments={})
            )
            print(f"  {name} -> Allowed")
        except ValueError:
            print(f"  {name} -> BLOCKED")
    contrast = _llm_call(
        "In one sentence, compare allowlist vs denylist for tool access in an "
        "LLM agent — which is safer and why?",
        max_tokens=80,
    )
    print(f"AI contrast: {contrast}")

    # Part 7: per-rule actions. REDACT replaces the match in-place,
    # WARN logs but allows, BLOCK rejects the call.
    print("\n=== Part 7: Action types ===\n")
    for action in GuardrailAction:
        print(f"  {action.value}")
    custom_config = GuardrailConfig(
        default_action=GuardrailAction.BLOCK,
        action_overrides={
            "pii_email": GuardrailAction.REDACT,
            "pii_phone_us": GuardrailAction.WARN,
            "blocked_sql_injection": GuardrailAction.BLOCK,
        },
    )
    print("\naction_overrides:")
    for rule, act in custom_config.action_overrides.items():
        print(f"  {rule} -> {act.value}")
    explainer = _llm_call(
        "Briefly explain when an LLM service should REDACT vs BLOCK vs WARN "
        "on policy violations. One sentence per action.",
        max_tokens=140,
    )
    print(f"AI explainer:\n{explainer}")

    # Part 8: a second hook type — ContentFilterHook scans plain text
    # for blocked words and credential patterns.
    print("\n=== Part 8: ContentFilterHook on a live agent ===\n")
    content_filter = ContentFilterHook(
        blocked_words=["password", "secret", "api_key"],
        blocked_patterns=[r"sk-[a-zA-Z0-9]+", r"ghp_[a-zA-Z0-9]+"],
        max_input_length=10000,
        case_sensitive=False,
    )
    benign = _llm_call(
        "Suggest one good practice for handling developer credentials in CI.",
        hooks=[content_filter],
    )
    print(f"Filtered answer: {benign}")
    try:
        _llm_call("What's my password?", hooks=[content_filter])
    except Exception as e:  # noqa: BLE001
        print(f"  (filter blocked the input as expected: {type(e).__name__})")

    # Part 9: stack multiple hooks. HookRegistry runs them in priority
    # order; the first BLOCK wins.
    print("\n=== Part 9: Stacking guardrail hooks ===\n")
    registry = HookRegistry()
    registry.add_provider(
        GuardrailsHook(config=GuardrailConfig(block_dangerous_tools=frozenset({"exec", "eval"})))
    )
    registry.add_provider(ContentFilterHook(blocked_words=["forbidden"]))
    print("Registered hook providers:")
    for prov in registry.providers:
        print(f"  - {prov.name} (priority={prov.priority})")
    stacked = _llm_call(
        "Name two security risks of giving an LLM agent unrestricted shell "
        "access. One bullet each.",
        hooks=[
            GuardrailsHook(
                config=GuardrailConfig(block_dangerous_tools=frozenset({"exec", "eval"}))
            ),
            ContentFilterHook(blocked_words=["forbidden"]),
        ],
    )
    print(f"Stacked-hooks answer: {stacked}")

    # Part 10: prod vs dev policy presets. Dev is permissive (WARN);
    # prod blocks irreversible operations and redacts PII.
    print("\n=== Part 10: Custom security policies ===\n")

    def production_config() -> GuardrailConfig:
        return GuardrailConfig(
            block_dangerous_tools=frozenset(
                {"exec", "eval", "system", "shell", "delete", "drop", "truncate", "rm", "sudo"}
            ),
            max_prompt_length=50000,
            max_tool_result_length=25000,
            default_action=GuardrailAction.BLOCK,
            action_overrides={
                "pii_email": GuardrailAction.REDACT,
                "pii_ssn": GuardrailAction.BLOCK,
                "pii_credit_card": GuardrailAction.BLOCK,
            },
        )

    def development_config() -> GuardrailConfig:
        return GuardrailConfig(
            block_dangerous_tools=frozenset({"exec", "eval"}),
            max_prompt_length=200000,
            max_tool_result_length=100000,
            default_action=GuardrailAction.WARN,
        )

    prod = production_config()
    dev = development_config()
    print(
        f"prod blocks {len(prod.block_dangerous_tools)} tools, "
        f"dev blocks {len(dev.block_dangerous_tools)}; "
        f"prod default={prod.default_action.value}, dev default={dev.default_action.value}"
    )
    suggestion = _llm_call(
        "List one extra guardrail rule a fintech company should add on top of "
        "blocking shell tools. One short sentence.",
        max_tokens=80,
    )
    print(f"AI suggestion: {suggestion}")

    # Part 11: ask the model to write a guardrail cheat sheet.
    print("\n=== Part 11: Best practices ===\n")
    best = _llm_call(
        "Write a six-line cheat sheet of best practices for guarding LLM "
        "agents in production. Six bullets, terse.",
        max_tokens=240,
    )
    print(best)

    # Part 12: an end-to-end Agent run with a guardrail attached.
    print("\n=== Part 12: Live Agent + Guardrails ===\n")
    safe_guardrails = GuardrailsHook(
        config=GuardrailConfig(
            block_dangerous_tools=frozenset({"exec", "eval", "shell"}),
            default_action=GuardrailAction.WARN,
        ),
    )
    safe_agent = Agent(
        model=get_model(max_tokens=200),
        system_prompt=(
            "You are a friendly assistant. Refuse to share secrets or "
            "anything the guardrails would block."
        ),
        hooks=[safe_guardrails],
    )
    t0 = time.perf_counter()
    safe_result = safe_agent.run_sync("How can I improve the security posture of a small SaaS app?")
    dt = time.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · "
        f"{safe_result.metrics.prompt_tokens}→{safe_result.metrics.completion_tokens} tokens]"
    )
    print(f"Guarded answer: {safe_result.message[:300]}")

    print(f"\nTotal violations logged in this notebook: {len(violations_log)}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
