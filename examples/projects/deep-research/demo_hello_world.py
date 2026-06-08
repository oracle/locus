#!/usr/bin/env python3
# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

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
def hello_world() -> str:
    """Return a hello string."""
    return "Hello from tulip deepagent!"


@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


def main() -> int:
    model_id = os.environ.get("TULIP_MODEL", "anthropic:claude-sonnet-4-6")

    print("== tulip hello-world deepagent ==")
    print(f"   model      : {model_id}")
    print(f"   tools      : hello_world, add")
    print()

    chat = get_model(model_id)

    agent = create_deepagent(
        model=chat,
        tools=[hello_world, add],
        system_prompt=(
            "You are a helpful assistant. When asked to call tools, use them "
            "before responding. After all tools have returned, summarize the "
            "results in one short sentence."
        ),
        reflexion=False,
        grounding=False,
        max_iterations=6,
    )

    result = agent.run_sync("Call hello_world and add 17 + 25, then summarize.")

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
