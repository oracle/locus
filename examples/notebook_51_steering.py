# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""Notebook 46: steering — runtime tool approval driven by a policy LLM.

``SteeringHook`` runs a second LLM ("the steering model") in front of
every tool call. The steering model reads a natural-language policy
plus the agent's activity so far, then returns one of three actions:

- ``PROCEED`` — let the tool call go through.
- ``GUIDE`` — let it through but inject a note for the agent to read.
- ``INTERRUPT`` — block the tool call and return a refusal message.

The result is a real-time guardrail you can author in plain English —
no rules engine, no policy DSL.

- ``SteeringHook(model=..., policy="...")`` — attach it to any agent
  via the ``hooks=`` parameter.
- ``steering.decisions`` — every action with its reason, for audit.

OCI GenAI drives both the agent and the steering model by default.

Run it:
    # OCI GenAI is the default — auto-detected from ~/.oci/config.
    LOCUS_MODEL_ID=openai.gpt-4.1 python examples/notebook_51_steering.py

    # Offline:
    LOCUS_MODEL_PROVIDER=mock python examples/notebook_51_steering.py

Prerequisites:
- An OCI profile with GenAI access, or set ``LOCUS_MODEL_PROVIDER`` to
  ``openai`` / ``anthropic`` / ``mock``.
"""

from config import get_model

from locus.agent import Agent, AgentConfig
from locus.hooks.builtin.steering import SteeringHook
from locus.tools.decorator import tool


# =============================================================================
# Part 1: A read-only policy. Delete is blocked, read is allowed.
# =============================================================================


def example_steering():
    print("=== Steering: LLM-Powered Tool Approval ===\n")

    model = get_model()

    @tool
    def read_data(query: str) -> str:
        """Read data from the database."""
        return f"Data: {query}"

    @tool
    def delete_data(table: str) -> str:
        """Delete a database table."""
        return f"Deleted {table}"

    steering = SteeringHook(
        model=model,
        policy="Only allow read operations. Never allow delete or write operations.",
    )

    agent = Agent(
        config=AgentConfig(
            system_prompt="You are a database assistant.",
            max_iterations=5,
            model=model,
            tools=[read_data, delete_data],
            hooks=[steering],
        )
    )

    # Should be INTERRUPTed — the policy forbids deletes.
    print("Attempt: Delete the users table")
    result = agent.run_sync("Delete the users table")
    print(f"Response: {result.message[:150]}")
    print(f"\nSteering decisions:")
    for d in steering.decisions:
        print(f"  {d.action}: {d.reason[:60]}")

    # Should PROCEED — reads are allowed.
    print("\nAttempt: Read all users")
    steering2 = SteeringHook(
        model=model,
        policy="Only allow read operations. Never allow delete or write operations.",
    )
    agent2 = Agent(
        config=AgentConfig(
            system_prompt="You are a database assistant.",
            max_iterations=5,
            model=model,
            tools=[read_data, delete_data],
            hooks=[steering2],
        )
    )
    result2 = agent2.run_sync("Read all users from the database")
    print(f"Response: {result2.message[:150]}")


if __name__ == "__main__":
    example_steering()
