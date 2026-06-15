#!/usr/bin/env python3
# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""Tulip "hello world" deepagent demo.

Two `@tool` functions + `create_deepagent(...)`, asking the agent to
call both tools in one turn.

Run:
    TULIP_MODEL=anthropic:claude-sonnet-4-6 \\
    ANTHROPIC_API_KEY=sk-ant-... \\
    python examples/projects/deep-research/demo_hello_world.py
"""

from __future__ import annotations

import os
import sys

from tulip.deepagent import create_deepagent
from tulip.models import get_model
from tulip.tools import tool


@tool
def lookup_cve(cve_id: str) -> str:
    """Return the CVSS severity and a one-line summary for a CVE id."""
    return f"{cve_id}: CVSS 10.0 (Critical) — remote code execution via JNDI lookup in Log4j."


@tool
def fetch_advisory(vendor: str) -> str:
    """Fetch the latest security advisory headline for a vendor."""
    return f"{vendor} advisory: patched release available — upgrade immediately and rotate exposed secrets."


def main() -> int:
    model_id = os.environ.get("TULIP_MODEL", "anthropic:claude-sonnet-4-6")

    print("== tulip hello-world deepagent ==")
    print(f"   model      : {model_id}")
    print(f"   tools      : lookup_cve, fetch_advisory")
    print()

    chat = get_model(model_id)

    agent = create_deepagent(
        model=chat,
        tools=[lookup_cve, fetch_advisory],
        system_prompt=(
            "You are a vulnerability-research assistant. When asked to call "
            "tools, use them before responding. After all tools have returned, "
            "summarize the findings in one short sentence."
        ),
        reflexion=False,
        grounding=False,
        max_iterations=6,
    )

    result = agent.run_sync("Look up CVE-2021-44228 and fetch the Apache advisory, then summarize.")

    text = getattr(result, "text", "") or ""
    tool_execs = list(result.tool_executions or ())  # type: ignore[arg-type]
    metrics = getattr(result, "metrics", None)

    print(f"Tool calls   : {len(tool_execs)}")
    for t in tool_execs:
        print(f"  - {t.tool_name}({t.arguments}) -> {t.result!r}")
    if metrics:
        print(f"Iterations   : {metrics.iterations}")
        print(
            f"Tokens       : prompt={metrics.prompt_tokens} "
            f"completion={metrics.completion_tokens} total={metrics.total_tokens}"
        )
    print()
    print("Agent response:")
    print(text)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
