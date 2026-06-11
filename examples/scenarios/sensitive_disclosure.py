# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Sensitive-information disclosure and system-prompt leakage.

Threat: an agent passes PII (emails, SSNs) into a tool call that logs or
ships it, or leaks a secret/credential out through an outbound action. The
data crosses a trust boundary it shouldn't.

Defense (built-in SDK primitive): a GuardrailsHook redacts PII in tool
arguments before the tool runs, and blocks arguments that carry a
secret/credential signature.

Taxonomy: OWASP LLM02 (Sensitive Information Disclosure) · OWASP LLM07
(System Prompt Leakage).
"""

from __future__ import annotations

import asyncio

from tulip.hooks.builtin.guardrails import GuardrailAction, GuardrailConfig, GuardrailsHook
from tulip.hooks.provider import BeforeToolCallEvent


def main() -> None:
    print("Scenario: sensitive disclosure / secret leakage  [LLM02 · LLM07]\n")

    # Part A — redact PII in tool arguments before the tool sees them.
    redactor = GuardrailsHook(
        config=GuardrailConfig(
            action_overrides={
                "pii_email": GuardrailAction.REDACT,
                "pii_ssn": GuardrailAction.REDACT,
            },
        )
    )
    args = {"note": "escalate for john.doe@example.com, SSN 123-45-6789"}
    event = BeforeToolCallEvent(tool_name="open_ticket", tool_call_id="t", arguments=dict(args))
    asyncio.run(redactor.on_before_tool_call(event))
    print("PII redaction in tool args:")
    print(f"  before: {args['note']}")
    print(f"  after:  {event.arguments['note']}")

    # Part B — block an outbound call carrying a secret/credential.
    secret_guard = GuardrailsHook(
        config=GuardrailConfig(
            blocked_content_patterns={
                "secret": r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*\S+",
            },
        )
    )
    print("\nSecret-egress block:")
    for tool_name, payload in (
        ("post_status", {"text": "triage complete, closing case"}),  # clean
        ("post_status", {"text": "debug: api_key=sk-live-9f8a7b6c5d"}),  # leaks a secret
    ):
        ev = BeforeToolCallEvent(tool_name=tool_name, tool_call_id="t", arguments=payload)
        try:
            asyncio.run(secret_guard.on_before_tool_call(ev))
            print(f"  [ALLOW] {tool_name}")
        except ValueError as exc:
            print(f"  [BLOCK] {tool_name} -> {exc}")

    print("\nSensitive data is redacted or stopped at the tool boundary, before it leaves.")


if __name__ == "__main__":
    main()
