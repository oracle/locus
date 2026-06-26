# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 07: Deployment readiness with tools.

A model without tools can only guess about a release from what's already
in its context — and guessed go/no-go calls are how bad deploys ship.
Tools let the agent reach out — look up an image digest, pull a service's
health record — and bring real evidence back into the conversation. Tulip
runs this as a small ReAct loop: the model decides whether to call a tool,
Tulip runs the tool, the result is fed back into the next model call.

Key ideas:
- ``@tool`` turns a plain Python function into something the model can
  call. The docstring is the description the model sees.
- Pass tools to ``Agent(tools=[...])`` and the agent picks when to use
  them.
- Each tool call shows up as a ``ToolStartEvent`` / ``ToolCompleteEvent``
  pair in the event stream — an auditable record of every lookup.
- Tools can take typed arguments (including optional ones) and return
  anything JSON-serialisable — strings, dicts, lists.

The inventory here is fictional by design: example.com hostnames,
placeholder image digests, and made-up service names. The degraded
``payment-svc`` stands in for a release that should hold for review.

Run it:
    .venv/bin/python examples/notebook_07_agent_with_tools.py

The default provider is the mock model; set TULIP_MODEL_PROVIDER for a live one (e.g.
``openai.gpt-4.1`` or ``meta.llama-3.3-70b-instruct``). Drop in
``TULIP_MODEL_PROVIDER=mock`` for an offline run. Tool-calling also
works against OpenAI, Anthropic.

Prerequisite: notebook 06.
"""

import asyncio
from datetime import datetime

# Import shared config
from config import get_model, print_config

from tulip.agent import Agent
from tulip.tools import tool


# =============================================================================
# Part 1: define a lookup tool
# =============================================================================

# A tool is a plain Python function decorated with @tool. The docstring
# is what the model reads to decide when to call it. All inventory data
# below is invented — placeholder digests and example.com hostnames.


@tool
def lookup_image(digest: str) -> str:
    """Look up a container image digest in the build registry."""
    known = {
        "sha256:aa11bb22": "api-gateway:v2.3.1 — built 2h ago, scan clean (0 critical CVEs)",
        "sha256:dd44ee55": "payment-svc:v1.9.0 — built 6d ago, scan flagged 3 critical CVEs",
    }
    return known.get(digest.lower(), f"Digest {digest} not present in the build registry")


@tool
def dns_record(hostname: str) -> str:
    """Look up the DNS / deployment record for a service hostname."""
    records = {
        "canary.example.com": "points at canary pool, 5% traffic, deployed 12 minutes ago",
        "payments.example.net": "points at blue pool, 100% traffic, last change 3 days ago",
    }
    return records.get(hostname.lower(), f"{hostname}: stable CNAME, last changed 2024")


def example_simple_tools():
    """Show the tool metadata Tulip generates from a decorated function."""
    print("=== Part 1: Simple Tools ===\n")

    result = lookup_image("sha256:aa11bb22")
    print(f"Direct call: lookup_image('sha256:aa11bb22') = {result}")

    print(f"\nTool name: {lookup_image.name}")
    print(f"Tool description: {lookup_image.description}")
    print(f"Tool parameters: {lookup_image.parameters}")

    import time as _t

    agent = Agent(
        model=get_model(max_tokens=80),
        system_prompt="Reply in one short sentence.",
    )
    t0 = _t.perf_counter()
    desc = agent.run_sync(
        f"In one sentence, when would an SRE agent use a tool called '{lookup_image.name}' "
        f"that {lookup_image.description}?"
    )
    dt = _t.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · "
        f"{desc.metrics.prompt_tokens}→{desc.metrics.completion_tokens} tokens]"
    )
    print(f"  AI commentary: {desc.message.strip()}")
    print()


# =============================================================================
# Part 2: hand tools to a release agent
# =============================================================================


def example_agent_with_tools():
    """Wire tools into an Agent and let the model decide when to call them."""
    print("=== Part 2: Agent Using Tools ===\n")

    model = get_model(max_tokens=200)

    agent = Agent(
        model=model,
        tools=[lookup_image, dns_record],
        system_prompt="You are a release-readiness assistant. Use the provided tools to look up "
        "images and hosts before giving a go/no-go.",
    )

    print(f"Agent has {len(agent.tools)} tools registered")

    result = agent.run_sync("Is the image sha256:aa11bb22 safe to deploy?")
    print("\nQ: Is the image sha256:aa11bb22 safe to deploy?")
    print(f"A: {result.message}")
    print(f"Tool calls made: {result.metrics.tool_calls}")
    print()


# =============================================================================
# Part 3: tools with optional and typed arguments
# =============================================================================


@tool
def get_current_time() -> str:
    """Get the current date and time, for deploy-log timestamps."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def image_age(built_year: int) -> str:
    """Calculate how stale a container image is given the year it was built."""
    current_year = datetime.now().year
    age = current_year - built_year
    return f"An image built in {built_year} is {age} years old."


@tool
def format_change_title(service: str, urgent: bool = False) -> str:
    """Create a change-request title for a service being deployed.

    Args:
        service: The service the change is about
        urgent: Whether to flag the change as urgent (default: False)
    """
    if urgent:
        return f"[URGENT] Deploy: {service} — expedited rollout requested."
    return f"Deploy: {service} — standard rollout."


def example_complex_tools():
    """Tools with default arguments and varied return types."""
    print("=== Part 3: Complex Tools ===\n")

    model = get_model(max_tokens=200)

    agent = Agent(
        model=model,
        tools=[get_current_time, image_age, format_change_title],
        system_prompt="You are an SRE assistant with access to time and change-management tools.",
    )

    prompts = [
        "What time is it right now? I need it for the deploy log.",
        "How old is a container image built in 2019?",
        "Give me an urgent change title for the service payment-svc",
    ]

    for prompt in prompts:
        result = agent.run_sync(prompt)
        print(f"Q: {prompt}")
        print(f"A: {result.message}")
        print()


# =============================================================================
# Part 4: watch lookups happen in the event stream
# =============================================================================


async def example_tool_events():
    """Stream events to see the model plan, call a tool, and use its evidence."""
    print("=== Part 4: Tool Execution Events ===\n")

    model = get_model(max_tokens=200)

    agent = Agent(
        model=model,
        tools=[lookup_image, dns_record],
        system_prompt="Use tools to check a release. Always look up images and hosts "
        "before answering.",
    )

    print("Q: Check image sha256:aa11bb22 and host canary.example.com, then give a go/no-go.\n")
    print("Events:")

    async for event in agent.run(
        "Check image sha256:aa11bb22 and host canary.example.com, then give a go/no-go."
    ):
        event_type = event.event_type

        if event_type == "tool_start":
            print(f"  TOOL_START: {event.tool_name}({event.arguments})")
        elif event_type == "tool_complete":
            print(f"  TOOL_COMPLETE: {event.tool_name} -> {event.result}")
        elif event_type == "think":
            if event.tool_calls:
                print(f"  THINK: Planning to call {len(event.tool_calls)} tool(s)")
        elif event_type == "terminate":
            print(f"  TERMINATE: {event.reason}")
            if event.final_message:
                print(f"\nFinal Answer: {event.final_message}")

    print()


# =============================================================================
# Part 5: tools that return structured data
# =============================================================================


@tool
def search_services(query: str, max_results: int = 3) -> list[dict]:
    """Search for services in the deployment inventory.

    Args:
        query: Search query (matches service name or type)
        max_results: Maximum number of results to return
    """
    # In-memory inventory stands in for a service catalogue / CMDB. The
    # search logic below is the part worth reading. All entries are fake.
    services = [
        {"id": 1, "name": "api-gateway", "type": "deployment", "status": "healthy"},
        {
            "id": 2,
            "name": "payment-svc",
            "type": "deployment",
            "status": "degraded",
        },
        {"id": 3, "name": "nightly-backup", "type": "cronjob", "status": "healthy"},
        {"id": 4, "name": "auth-svc", "type": "deployment", "status": "degraded"},
        {"id": 5, "name": "redis-cache", "type": "statefulset", "status": "healthy"},
        {"id": 6, "name": "image-resizer", "type": "deployment", "status": "down"},
        {
            "id": 7,
            "name": "metrics-agent",
            "type": "daemonset",
            "status": "healthy",
        },
        {
            "id": 8,
            "name": "report-export",
            "type": "cronjob",
            "status": "degraded",
        },
    ]

    # Case-insensitive match on name OR type.
    q = query.lower()
    matches = [s for s in services if q in s["name"].lower() or q in s["type"].lower()]
    return matches[:max_results]


@tool
def get_service_details(service_id: int) -> dict:
    """Get detailed status about a specific service in the inventory."""
    details = {
        1: {
            "id": 1,
            "name": "api-gateway",
            "status": "healthy",
            "notes": "3 replicas ready, p99 latency 80ms, last deploy 2h ago",
        },
        2: {
            "id": 2,
            "name": "payment-svc",
            "status": "degraded",
            "notes": "1/3 replicas crash-looping, error rate 8%, deployed 6d ago",
        },
        3: {
            "id": 3,
            "name": "nightly-backup",
            "status": "healthy",
            "notes": "last run succeeded, 4m12s, retention 30 days",
        },
        4: {
            "id": 4,
            "name": "auth-svc",
            "status": "degraded",
            "notes": "elevated 401s after token-cache change",
        },
        5: {"id": 5, "name": "redis-cache", "status": "healthy", "notes": "98% hit rate"},
        6: {
            "id": 6,
            "name": "image-resizer",
            "status": "down",
            "notes": "OOMKilled repeatedly, memory limit too low",
        },
        7: {
            "id": 7,
            "name": "metrics-agent",
            "status": "healthy",
            "notes": "running on all nodes, no drops",
        },
        8: {
            "id": 8,
            "name": "report-export",
            "status": "degraded",
            "notes": "last run timed out at 30m",
        },
    }
    return details.get(service_id, {"error": f"Service {service_id} not found"})


def example_structured_tools():
    """Tools can return dicts and lists — the model parses them on the next turn."""
    print("=== Part 5: Structured Data Tools ===\n")

    model = get_model(max_tokens=300)

    agent = Agent(
        model=model,
        tools=[search_services, get_service_details],
        system_prompt="You are an SRE assistant. Help engineers look up services.",
    )

    result = agent.run_sync("Find deployment services, then tell me more about payment-svc")
    print("Q: Find deployment services, then tell me more about payment-svc")
    print(f"A: {result.message}")
    print(f"\nTool calls made: {result.metrics.tool_calls}")
    print()


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all notebook parts."""
    print("=" * 60)
    print("Notebook 07: Deployment Readiness with Tools")
    print("=" * 60)
    print()

    print_config()
    print()

    example_simple_tools()
    example_agent_with_tools()
    example_complex_tools()
    asyncio.run(example_tool_events())
    example_structured_tools()

    print("=" * 60)
    print("Next: Notebook 08 — Investigation Memory")
    print("=" * 60)


if __name__ == "__main__":
    main()
