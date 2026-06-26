# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 49: steering — a policy LLM gates a live production incident.

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
    # The bundled mock model is the default; set TULIP_MODEL_PROVIDER for a live provider.
    TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_49_steering.py

    # Offline:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_49_steering.py

Prerequisites:
- An OpenAI or Anthropic API key, or set ``TULIP_MODEL_PROVIDER`` to
  ``openai`` / ``anthropic`` / ``mock``.
"""

from config import get_model

from tulip.agent import Agent, AgentConfig
from tulip.hooks.builtin.steering import SteeringHook
from tulip.tools.decorator import tool


# =============================================================================
# Part 1: A read-only diagnostics policy. Mutating ops are blocked,
#         metric queries are allowed.
# =============================================================================


def example_steering():
    print("=== Steering: LLM-Powered Tool Approval ===\n")

    model = get_model()

    @tool
    def query_metrics(query: str) -> str:
        """Run a read-only PromQL query against the metrics backend."""
        return f"Metrics results: {query}"

    @tool
    def restart_service(service: str) -> str:
        """Restart a production service (mutating action)."""
        return f"Restarted {service}"

    steering = SteeringHook(
        model=model,
        policy=(
            "This is a read-only diagnostics session. Only allow metric and log queries. "
            "Never allow mutating or destructive operations such as restarting services, "
            "scaling deployments, or terminating instances."
        ),
    )

    agent = Agent(
        config=AgentConfig(
            system_prompt="You are an SRE incident diagnostics assistant.",
            max_iterations=5,
            model=model,
            tools=[query_metrics, restart_service],
            hooks=[steering],
        )
    )

    # Should be INTERRUPTed — the policy forbids mutating actions.
    print("Attempt: Restart service checkout-api")
    result = agent.run_sync("Restart service checkout-api")
    print(f"Response: {result.message[:150]}")
    print(f"\nSteering decisions:")
    for d in steering.decisions:
        print(f"  {d.action}: {d.reason[:60]}")

    # Should PROCEED — read-only metric queries are allowed.
    print("\nAttempt: Query p99 latency for checkout-api")
    steering2 = SteeringHook(
        model=model,
        policy=(
            "This is a read-only diagnostics session. Only allow metric and log queries. "
            "Never allow mutating or destructive operations such as restarting services, "
            "scaling deployments, or terminating instances."
        ),
    )
    agent2 = Agent(
        config=AgentConfig(
            system_prompt="You are an SRE incident diagnostics assistant.",
            max_iterations=5,
            model=model,
            tools=[query_metrics, restart_service],
            hooks=[steering2],
        )
    )
    result2 = agent2.run_sync("Query the metrics backend for p99 latency on checkout-api")
    print(f"Response: {result2.message[:150]}")


if __name__ == "__main__":
    example_steering()
