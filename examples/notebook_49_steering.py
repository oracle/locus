# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""Notebook 49: steering — a policy LLM gates a live investigation.

``SteeringHook`` runs a second LLM ("the steering model") in front of
every tool call, which is how you steer an investigation while it is
running — keep it read-only, redirect it ("pivot to lateral movement"),
or stop a containment action that nobody approved. The steering model
reads a natural-language policy plus the agent's activity so far, then
returns one of three actions:

- ``PROCEED`` — let the tool call go through.
- ``GUIDE`` — let it through but inject a note for the agent to read
  (e.g. "pivot to lateral movement next").
- ``INTERRUPT`` — block the tool call and return a refusal message.

The result is a real-time guardrail you can author in plain English —
no rules engine, no policy DSL — and every decision is recorded for
the post-incident audit. Holding an investigation read-only is a direct
control against excessive agency (OWASP LLM06) and tool misuse
(OWASP ASI02): the agent can read the SIEM all day but cannot reach a
containment action no one approved.

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
# Part 1: A read-only investigation policy. Containment is blocked,
#         SIEM queries are allowed.
# =============================================================================


def example_steering():
    print("=== Steering: LLM-Powered Tool Approval ===\n")

    model = get_model()

    @tool
    def query_siem(query: str) -> str:
        """Run a read-only query against the SIEM."""
        return f"SIEM results: {query}"

    @tool
    def isolate_host(hostname: str) -> str:
        """Isolate a host from the network (containment action)."""
        return f"Isolated {hostname}"

    steering = SteeringHook(
        model=model,
        policy=(
            "This is a read-only investigation. Only allow queries and log reads. "
            "Never allow containment or destructive actions such as host isolation."
        ),
    )

    agent = Agent(
        config=AgentConfig(
            system_prompt="You are a SOC investigation assistant.",
            max_iterations=5,
            model=model,
            tools=[query_siem, isolate_host],
            hooks=[steering],
        )
    )

    # Should be INTERRUPTed — the policy forbids containment actions.
    print("Attempt: Isolate host web-prod-03")
    result = agent.run_sync("Isolate host web-prod-03")
    print(f"Response: {result.message[:150]}")
    print(f"\nSteering decisions:")
    for d in steering.decisions:
        print(f"  {d.action}: {d.reason[:60]}")

    # Should PROCEED — read-only SIEM queries are allowed.
    print("\nAttempt: Query failed logins from 198.51.100.7")
    steering2 = SteeringHook(
        model=model,
        policy=(
            "This is a read-only investigation. Only allow queries and log reads. "
            "Never allow containment or destructive actions such as host isolation."
        ),
    )
    agent2 = Agent(
        config=AgentConfig(
            system_prompt="You are a SOC investigation assistant.",
            max_iterations=5,
            model=model,
            tools=[query_siem, isolate_host],
            hooks=[steering2],
        )
    )
    result2 = agent2.run_sync("Query the SIEM for failed logins from 198.51.100.7")
    print(f"Response: {result2.message[:150]}")


if __name__ == "__main__":
    example_steering()
