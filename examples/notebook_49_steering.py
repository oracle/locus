# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL
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

The configured provider drives both the agent and the steering model.

Run it:
    # The bundled mock model is the default; set TULIP_MODEL_PROVIDER for a live provider.
    TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_51_steering.py

    # Offline:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_51_steering.py

Prerequisites:
- An OpenAI or Anthropic API key, or set ``TULIP_MODEL_PROVIDER`` to
  ``openai`` / ``anthropic`` / ``mock``.
"""

from config import get_model

from tulip.agent import Agent, AgentConfig
from tulip.hooks.builtin.steering import SteeringHook
from tulip.tools.decorator import tool


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
