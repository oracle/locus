# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 07: giving an agent tools.

A model on its own can only answer from what is already in its context. It
cannot tell you which image digest is actually running in production, or
whether a service is passing its health checks right now. Tools let the
agent reach out, get a real answer, and bring it back. Tulip runs this as
a small ReAct loop: the model decides whether to call a tool, Tulip runs
it, the result goes into the next model call.

The scenario is a deployment-readiness check. The agent looks up a
container image digest in the build registry, pulls a service's DNS and
health record, and only then calls go or no-go. It is a good first
tool-using example because the answer is *not* in the model — every fact
it needs has to be fetched — and because the go/no-go at the end is the
kind of judgement people actually want an agent to make from fetched
data.

Key ideas:
- ``@tool`` turns a plain Python function into something the model can
  call. The docstring is the description the model reads.
- Pass tools to ``Agent(tools=[...])`` and the agent picks when to use
  them.
- Each call shows up as a ``ToolStartEvent`` / ``ToolCompleteEvent`` pair
  in the event stream — a record of every lookup the agent made.
- Tools take typed arguments (including optional ones with defaults) and
  return anything JSON-serialisable: strings, numbers, dicts, lists.

The data is fictional throughout — ``example.com`` hostnames, placeholder
digests, and made-up service names — so the notebook runs the same way
every time and reaches nothing on the network.

Run it:
    .venv/bin/python examples/notebook_07_agent_with_tools.py

The default provider is the bundled deterministic mock model, so this
runs offline with no credentials. Set ``TULIP_MODEL_PROVIDER=openai`` (or
``anthropic``) and the matching key for a live model.

Prerequisite: notebook 06.
"""

import asyncio
from datetime import UTC, datetime

# Import shared config
from config import get_model, print_config

from tulip.agent import Agent
from tulip.tools import tool


# --- The fictional estate -------------------------------------------------
# Fixed sample data, so the notebook is deterministic and offline. Every
# hostname is under example.com and every digest is a placeholder.

_REGISTRY = {
    "checkout": {
        "tag": "v4.2.1",
        "digest": "sha256:9f2c1a7e4b3d0c8a6f5e2d1b0a9c8f7e6d5c4b3a2f1e0d9c8b7a6f5e4d3c2b1a",
        "built": "2026-08-11T09:14:00Z",
        "signed": True,
    },
    "search": {
        "tag": "v1.9.0",
        "digest": "sha256:1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f809",
        "built": "2026-06-02T17:40:00Z",
        "signed": False,
    },
}

_SERVICES = {
    "checkout": {
        "hostname": "checkout.svc.example.com",
        "dns_ttl_seconds": 60,
        "healthy_replicas": 6,
        "total_replicas": 6,
        "last_health_check": "passing",
    },
    "search": {
        "hostname": "search.svc.example.com",
        "dns_ttl_seconds": 3600,
        "healthy_replicas": 2,
        "total_replicas": 5,
        "last_health_check": "failing",
    },
}


# =============================================================================
# Part 1: define the tools
# =============================================================================

# A tool is a plain Python function decorated with @tool. The docstring is
# what the model reads to decide when to call it, so it is written for the
# model as much as for the next human.


@tool
def lookup_image_digest(service: str) -> str:
    """Look up the container image tag and digest the build registry has for a service."""
    entry = _REGISTRY.get(service.lower())
    if entry is None:
        return f"No image on file for {service!r}."
    signed = "signed" if entry["signed"] else "UNSIGNED"
    return f"{service}: {entry['tag']} ({signed}), built {entry['built']}, digest {entry['digest']}"


@tool
def lookup_service_health(service: str) -> str:
    """Look up a service's DNS record and current health-check status."""
    entry = _SERVICES.get(service.lower())
    if entry is None:
        return f"No service record for {service!r}."
    return (
        f"{entry['hostname']} · DNS TTL {entry['dns_ttl_seconds']}s · "
        f"{entry['healthy_replicas']}/{entry['total_replicas']} replicas healthy · "
        f"health check {entry['last_health_check']}"
    )


async def example_simple_tools():
    """Call the tools directly, before an agent is anywhere near them.

    Worth doing once: a tool is an ordinary function, and ``@tool`` did not
    take that away. If it misbehaves here, the problem is not the model.
    """
    print("=== Part 1: Tools Are Just Functions ===\n")

    print(
        f"lookup_image_digest('checkout') → {await lookup_image_digest.execute(service='checkout')}"
    )
    print(
        f"lookup_service_health('search') → {await lookup_service_health.execute(service='search')}"
    )
    print()


# =============================================================================
# Part 2: hand the tools to an agent
# =============================================================================


async def example_agent_with_tools():
    """The agent decides when to call what. You do not script the sequence."""
    print("=== Part 2: Agent with Tools ===\n")

    agent = Agent(
        model=get_model(max_tokens=150),
        tools=[lookup_image_digest, lookup_service_health],
        system_prompt=(
            "You are a release engineer's assistant. Before answering any "
            "deployment question, look up the facts with your tools — never "
            "answer from memory. Then give a clear go or no-go with the "
            "reason."
        ),
    )

    prompt = "Are we clear to deploy Checkout? Check the image and the service health."
    result = await agent.arun(prompt)

    print(f"Prompt: {prompt}")
    print(f"Response: {result.message}")
    print(f"Tool calls made: {result.metrics.tool_calls}")
    print()


# =============================================================================
# Part 3: optional arguments and defaults
# =============================================================================


@tool
def deployment_window_open(environment: str = "production") -> str:
    """Say whether the change window for an environment is currently open."""
    # Fixed rather than clock-dependent, so the notebook reads the same at
    # 3pm and at 3am. A real one would consult the change calendar.
    windows = {
        "production": "closed until 22:00 UTC (business hours freeze)",
        "staging": "open",
    }
    return f"{environment}: change window {windows.get(environment.lower(), 'unknown')}"


@tool
def rollback_target(service: str, versions_back: int = 1) -> str:
    """Name the version this service would roll back to, N releases back."""
    history = {
        "checkout": ["v4.2.1", "v4.2.0", "v4.1.7"],
        "search": ["v1.9.0", "v1.8.3", "v1.8.2"],
    }
    releases = history.get(service.lower())
    if releases is None:
        return f"No release history for {service!r}."
    if versions_back >= len(releases):
        return f"{service}: only {len(releases) - 1} release(s) of history retained."
    return f"{service}: rolling back {versions_back} would land on {releases[versions_back]}"


@tool
def utc_now() -> str:
    """Get the current UTC time, for stamping a go/no-go decision."""
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


async def example_optional_arguments():
    """Defaults mean the model can call a tool without knowing every argument."""
    print("=== Part 3: Optional Arguments and Defaults ===\n")

    print(f"deployment_window_open() → {await deployment_window_open.execute()}")
    print(
        "deployment_window_open('staging') → "
        f"{await deployment_window_open.execute(environment='staging')}"
    )
    print(f"rollback_target('checkout') → {await rollback_target.execute(service='checkout')}")
    print(
        "rollback_target('checkout', 2) → "
        f"{await rollback_target.execute(service='checkout', versions_back=2)}"
    )
    print()


# =============================================================================
# Part 4: watching the tool calls happen
# =============================================================================


async def example_tool_events():
    """Every lookup the agent made, in order, as it made it.

    This is the part worth internalising. An agent that fetched the wrong
    thing and an agent that fetched nothing look identical in the final
    message; they look completely different here.
    """
    print("=== Part 4: Tool Execution Events ===\n")

    agent = Agent(
        model=get_model(max_tokens=150),
        tools=[lookup_image_digest, lookup_service_health, deployment_window_open],
        system_prompt=(
            "You are a release engineer's assistant. Look up the facts before "
            "answering. Give a go or no-go with the reason."
        ),
    )

    print("Prompt: Run the pre-deploy checks on Search.")
    print("Events:")

    async for event in agent.run("Run the pre-deploy checks on Search."):
        if event.event_type == "tool_start":
            print(f"  → calling {event.tool_name}({event.arguments})")
        elif event.event_type == "tool_complete":
            print(f"  ← {event.tool_name}: {str(event.result)[:80]}")
        elif event.event_type == "terminate":
            print(f"  ✓ {str(getattr(event, 'final_message', '') or '').strip()[:100]}")

    print()


# =============================================================================
# Part 5: tools that return structure, not prose
# =============================================================================


@tool
def readiness_report(service: str) -> dict:
    """Return the full readiness record for a service as structured data."""
    image = _REGISTRY.get(service.lower())
    health = _SERVICES.get(service.lower())
    if image is None or health is None:
        return {"service": service, "known": False}
    return {
        "service": service,
        "known": True,
        "image": {"tag": image["tag"], "signed": image["signed"], "digest": image["digest"][:19]},
        "health": {
            "hostname": health["hostname"],
            "replicas": f"{health['healthy_replicas']}/{health['total_replicas']}",
            "check": health["last_health_check"],
        },
        # The blockers the agent should not have to re-derive. A tool that
        # returns a judgement alongside the facts is doing half the work the
        # model would otherwise do less reliably.
        "blockers": [
            *([] if image["signed"] else ["image is unsigned"]),
            *(
                []
                if health["healthy_replicas"] == health["total_replicas"]
                else [
                    f"{health['total_replicas'] - health['healthy_replicas']} replica(s) unhealthy"
                ]
            ),
            *([] if health["last_health_check"] == "passing" else ["health check failing"]),
        ],
    }


async def example_structured_tools():
    """A dict comes back as a dict, not as a string the model has to parse."""
    print("=== Part 5: Structured Return Types ===\n")

    # ``.func`` is the undecorated function, so this is the real dict. The
    # agent sees ``.execute()``'s output instead — the same data as JSON,
    # because that is what crosses the wire to a model.
    for service in ("checkout", "search"):
        report = readiness_report.func(service=service)
        blockers = report["blockers"] or ["none"]
        print(f"{service}: {report['image']['tag']} · {report['health']['replicas']} replicas")
        print(f"  blockers: {', '.join(blockers)}")
    print()

    agent = Agent(
        model=get_model(max_tokens=150),
        tools=[readiness_report, utc_now],
        system_prompt=(
            "You are a release engineer's assistant. Pull the readiness report "
            "and give a go or no-go. If there are blockers, say no-go and list "
            "them."
        ),
    )

    result = await agent.arun("Give me a go/no-go on Search.")
    print(f"Agent: {result.message}")
    print()


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all notebook parts."""
    print("=" * 60)
    print("Notebook 07: Deployment Readiness — Giving an Agent Tools")
    print("=" * 60)
    print()

    print_config()
    print()

    await example_simple_tools()
    await example_agent_with_tools()
    await example_optional_arguments()
    await example_tool_events()
    await example_structured_tools()

    print("=" * 60)
    print("Next: Notebook 08 — Giving an Agent Memory")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
