# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Excessive agency — an agent reaching for power it was never granted.

Threat: a triage agent that should only *read* is steered into a
high-impact action — disabling an account, wiring a payment, deleting data.
The blast radius of a compromised or confused agent is whatever its tools
allow, so the fix is to grant the least authority that does the job.

Defense (built-in SDK primitive): a GuardrailsHook with an allow_only_tools
allowlist — deny-by-default. Any tool outside the read-only set is blocked
before it runs. (High-impact actions that genuinely need to happen go
through an approval interrupt — see notebook_19.)

Taxonomy: OWASP LLM06 (Excessive Agency) · OWASP ASI03 (Identity &
Privilege Abuse).
"""

from __future__ import annotations

import asyncio

from tulip.hooks.builtin.guardrails import GuardrailConfig, GuardrailsHook
from tulip.hooks.provider import BeforeToolCallEvent


def main() -> None:
    print("Scenario: excessive agency  [LLM06 · ASI03]\n")

    # The triage agent is granted read-only authority — nothing else.
    guardrails = GuardrailsHook(
        config=GuardrailConfig(
            allow_only_tools=frozenset({"enrich_indicator", "query_siem", "get_alert"}),
        )
    )

    attempts = [
        ("query_siem", {"q": "powershell"}),  # in-scope read
        ("enrich_indicator", {"indicator": "198.51.100.23"}),  # in-scope read
        ("disable_account", {"user": "svc-backup"}),  # out of scope
        ("wire_transfer", {"amount": 50000}),  # wildly out of scope
    ]
    for tool_name, args in attempts:
        event = BeforeToolCallEvent(tool_name=tool_name, tool_call_id="t", arguments=args)
        try:
            asyncio.run(guardrails.on_before_tool_call(event))
            print(f"  [ALLOW] {tool_name}")
        except ValueError as exc:
            print(f"  [BLOCK] {tool_name} -> {exc}")

    print("\nLeast authority by default; a high-impact action needs an explicit grant or approval.")


if __name__ == "__main__":
    main()
