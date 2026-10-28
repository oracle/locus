# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Notebook 68: steering — a policy LLM gates a live production incident.

``SteeringHook`` runs a second LLM ("the steering model") in front of
every tool call, which is how you steer an on-call agent while it is
running — keep it read-only, redirect it ("check the canary first"),
or stop a mutating action that nobody approved. The steering model
reads a natural-language policy plus the agent's activity so far, then
returns one of three actions:

- ``PROCEED`` — let the tool call go through.
- ``GUIDE`` — let it through but inject a note for the agent to read
  (e.g. "check the canary metrics before scaling").
- ``INTERRUPT`` — block the tool call and return a refusal message.

The result is a real-time guardrail you can author in plain English —
no rules engine, no policy DSL — and every decision is recorded for
the post-incident review. Holding a diagnostics session read-only is a
direct control against blast radius: the agent can query Prometheus and
read logs all day, but it cannot reach a mutating operation — restarting
a service, scaling a deployment, terminating an instance — that no one
approved.

- ``SteeringHook(model=..., policy="...")`` — attach it to any agent
  via the ``hooks=`` parameter.
- ``steering.decisions`` — every action with its reason, for audit.

The configured provider drives both the agent and the steering model.

Run it:
    # Default: OCI GenAI if ~/.oci/config has a usable profile, else the bundled mock model
    LOCUS_MODEL_PROVIDER=openai LOCUS_MODEL_ID=gpt-4o python examples/notebook_68_steering.py

    # Offline:
    LOCUS_MODEL_PROVIDER=mock python examples/notebook_68_steering.py

Prerequisites:
- An OpenAI or Anthropic API key, or set ``LOCUS_MODEL_PROVIDER`` to
  ``openai`` / ``anthropic`` / ``mock``.
"""

import asyncio

from config import get_model

from locus.agent import Agent, AgentConfig
from locus.hooks.builtin.steering import SteeringHook
from locus.tools.decorator import tool


_POLICY = (
    "This is a read-only diagnostics session. Only allow metric and log queries. "
    "Never allow mutating or destructive operations such as restarting services, "
    "scaling deployments, or terminating instances."
)


@tool
def query_metrics(query: str) -> str:
    """Run a read-only PromQL query against the metrics backend."""
    return f"Metrics results: {query}"


@tool
def restart_service(service: str) -> str:
    """Restart a production service (mutating action)."""
    return f"Restarted {service}"


def _make_agent(model, hook: SteeringHook) -> Agent:
    """One diagnostics agent per steering hook, so each run's decisions stay separate."""
    return Agent(
        config=AgentConfig(
            system_prompt="You are an SRE incident diagnostics assistant.",
            max_iterations=5,
            model=model,
            tools=[query_metrics, restart_service],
            hooks=[hook],
        )
    )


def _print_decisions(steering: SteeringHook) -> None:
    print("\nSteering decisions:")
    for d in steering.decisions:
        print(f"  {d.action}: {d.reason[:60]}")


# =============================================================================
# Part 1: A read-only diagnostics policy. Mutating ops are blocked,
#         metric queries are allowed.
# =============================================================================


async def example_steering():
    print("=== Steering: LLM-Powered Tool Approval ===\n")

    model = get_model()

    # Should be INTERRUPTed — the policy forbids mutating actions.
    print("Attempt: Restart service checkout-api")
    steering = SteeringHook(model=model, policy=_POLICY)
    result = await _make_agent(model, steering).arun("Restart service checkout-api")
    print(f"Response: {result.message[:150]}")
    _print_decisions(steering)

    # Should PROCEED — read-only metric queries are allowed. A fresh hook
    # keeps this run's decision log separate from the first.
    print("\nAttempt: Query p99 latency for checkout-api")
    steering2 = SteeringHook(model=model, policy=_POLICY)
    result2 = await _make_agent(model, steering2).arun(
        "Query the metrics backend for p99 latency on checkout-api"
    )
    print(f"Response: {result2.message[:150]}")
    _print_decisions(steering2)


async def main() -> None:
    await example_steering()


if __name__ == "__main__":
    asyncio.run(main())
