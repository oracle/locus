#!/usr/bin/env python3
# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Locus port of the deepagents "hello world" gist.

Mirrors `gist 2453c40e/04_deep_agents_hello_world.py`:
two `@tool` functions + `create_deep_agent(...)` against OCI GenAI,
asking the agent to call both tools in one turn.

Locus equivalents:
- `from deepagents import create_deep_agent`           -> `from locus.deepagent import create_deepagent`
- `from langchain_core.tools import tool`               -> `from locus.tools import tool`
- `from langchain_oci import ChatOCIGenAI`              -> `from locus.models import get_model`
- `agent.invoke({"messages": [...]})`                   -> `agent.run_sync("...")`

Run:
    OCI_PROFILE=DEFAULT \\
    OCI_COMPARTMENT=ocid1.tenancy.oc1..xxx \\
    .venv/bin/python examples/projects/deep-research/demo_hello_world.py
"""

from __future__ import annotations

import os
import sys

from locus.deepagent import create_deepagent
from locus.models import get_model
from locus.tools import tool


@tool
def hello_world() -> str:
    """Return a hello string."""
    return "Hello from locus deepagent with OCI!"


@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


def main() -> int:
    profile = os.environ.get("OCI_PROFILE", "DEFAULT")
    auth_type = os.environ.get("OCI_AUTH_TYPE", "api_key")
    compartment = os.environ.get(
        "OCI_COMPARTMENT",
        "ocid1.tenancy.oc1..<your-tenancy>",
    )
    region = os.environ.get("OCI_REGION", "us-chicago-1")
    model_id = os.environ.get("OCI_MODEL_ID", "oci:openai.gpt-4o-mini")

    print("== locus hello-world deepagent ==")
    print(f"   model      : {model_id}")
    print(f"   profile    : {profile} ({auth_type})")
    print(f"   tools      : hello_world, add")
    print()

    chat = get_model(
        model_id,
        profile=profile,
        compartment_id=compartment,
        region=region,
    )

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
