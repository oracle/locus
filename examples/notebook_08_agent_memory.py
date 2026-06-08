# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL
"""
Notebook 08: agent memory backed by Redis.

Give an agent a checkpointer and every turn is written to a real
store. Restart the process, attach a fresh agent to the same
connection, hand it the same ``thread_id``, and the conversation picks
up where it left off. This is the production memory story for Tulip.

Key ideas:
- ``RedisBackend(...)`` opens a connection to Redis and stores agent
  state under keys you scope.
- ``thread_id`` keys conversations — one Redis can host many independent
  threads side by side.
- ``checkpoint_every_n_iterations=1`` writes after every loop iteration,
  so a crash mid-tool-call still resumes cleanly.
- The saved state is rich: messages, tool history, iteration count,
  Reflexion confidence — all loadable for inspection.

Run it:
    export REDIS_URL=redis://localhost:6379/0
    python examples/notebook_08_agent_memory.py

The agent itself goes through whichever provider you configure via
``TULIP_MODEL_PROVIDER`` (``openai`` / ``anthropic``).
Set ``TULIP_MODEL_PROVIDER=mock`` to bypass the model for offline runs.

If REDIS_URL is missing the script prints a skip banner and
exits cleanly — it never falls back to an in-memory checkpointer.
"""

import asyncio
import os
import sys

from config import get_model, print_config

from tulip.agent import Agent
from tulip.memory.backends import RedisBackend
from tulip.tools import tool


_REQUIRED_ENV = ("REDIS_URL",)


def _missing_env() -> list[str]:
    return [name for name in _REQUIRED_ENV if not os.environ.get(name)]


def _build_checkpointer(table_suffix: str = "default"):
    """Build a Redis-backed checkpointer described by env vars."""
    backend = RedisBackend(
        url=os.environ["REDIS_URL"],
        namespace=f"tulip_notebook_08_{table_suffix}",
    )
    return backend.as_checkpointer()


# =============================================================================
# Part 1: a two-turn conversation that survives in Redis
# =============================================================================


def example_conversation_memory():
    """Same thread_id across two run_sync calls — the agent recalls turn one."""
    print("=== Part 1: Conversation memory (Redis) ===\n")

    model = get_model(max_tokens=100)
    checkpointer = _build_checkpointer("convo")

    agent = Agent(
        model=model,
        system_prompt="You are a helpful assistant. Remember what the user tells you.",
        checkpointer=checkpointer,
    )

    thread_id = "conversation_001"

    result1 = agent.run_sync("My name is Alice.", thread_id=thread_id)
    print("User: My name is Alice.")
    print(f"Agent: {result1.message}")

    # Same thread_id — the agent loads prior state from Redis before
    # the next model call.
    result2 = agent.run_sync("What's my name?", thread_id=thread_id)
    print("\nUser: What's my name?")
    print(f"Agent: {result2.message}")
    print()


# =============================================================================
# Part 2: write a checkpoint after every iteration
# =============================================================================


_NOTES: list[str] = []


@tool
def save_note(content: str) -> str:
    """Save a note for later reference."""
    _NOTES.append(content)
    return f"Note saved: {content}"


@tool
def get_notes() -> str:
    """Get all saved notes."""
    if not _NOTES:
        return "No notes saved yet."
    lines = "\n".join(f"- {n}" for n in _NOTES)
    return f"You have {len(_NOTES)} note(s):\n{lines}"


def example_checkpointing_with_tools():
    """checkpoint_every_n_iterations=1 means a mid-loop crash still recovers."""
    print("=== Part 2: Checkpoint after each iteration ===\n")

    model = get_model(max_tokens=150)
    checkpointer = _build_checkpointer("notes")

    agent = Agent(
        model=model,
        tools=[save_note, get_notes],
        system_prompt="You are a note-taking assistant.",
        checkpointer=checkpointer,
        checkpoint_every_n_iterations=1,
    )

    thread_id = "notes_session"

    result1 = agent.run_sync("Save a note: Buy groceries", thread_id=thread_id)
    print("User: Save a note: Buy groceries")
    print(f"Agent: {result1.message}")
    print(f"Tool calls: {result1.metrics.tool_calls}")

    result2 = agent.run_sync("What notes do I have?", thread_id=thread_id)
    print("\nUser: What notes do I have?")
    print(f"Agent: {result2.message}")
    print()


# =============================================================================
# Part 3: a fresh agent picks up where the old one stopped
# =============================================================================


def example_persistence_across_processes():
    """Two Agent objects, one Redis. The second loads the first's state."""
    print("=== Part 3: Cross-process persistence ===\n")

    model = get_model(max_tokens=100)

    agent1 = Agent(
        model=model,
        system_prompt="You are a helpful assistant.",
        checkpointer=_build_checkpointer("persist"),
    )

    thread_id = "persistent_chat"

    result1 = agent1.run_sync("Remember: The secret code is 42.", thread_id=thread_id)
    print("User: Remember: The secret code is 42.")
    print(f"Agent: {result1.message}")

    # Pretend the process restarted. A *new* Agent + checkpointer object
    # connects to the same Redis and reads back the state.
    agent2 = Agent(
        model=model,
        system_prompt="You are a helpful assistant.",
        checkpointer=_build_checkpointer("persist"),
    )

    result2 = agent2.run_sync("What was the secret code?", thread_id=thread_id)
    print("\n[New process — same Redis]")
    print("User: What was the secret code?")
    print(f"Agent: {result2.message}")
    print()


# =============================================================================
# Part 4: many threads sharing one database
# =============================================================================


def example_multiple_threads():
    """Two users, two thread_ids, one Redis — no cross-talk."""
    print("=== Part 4: Multiple threads ===\n")

    model = get_model(max_tokens=100)
    checkpointer = _build_checkpointer("multi")

    agent = Agent(
        model=model,
        system_prompt="You are a helpful assistant.",
        checkpointer=checkpointer,
    )

    thread_alice = "thread_alice"
    thread_bob = "thread_bob"

    agent.run_sync("I'm Alice and I like pizza.", thread_id=thread_alice)
    agent.run_sync("I'm Bob and I like sushi.", thread_id=thread_bob)

    result_alice = agent.run_sync("What's my favorite food?", thread_id=thread_alice)
    print("Thread 'alice': What's my favorite food?")
    print(f"Agent: {result_alice.message}")

    result_bob = agent.run_sync("What's my favorite food?", thread_id=thread_bob)
    print("\nThread 'bob': What's my favorite food?")
    print(f"Agent: {result_bob.message}")
    print()


# =============================================================================
# Part 5: load and walk a saved checkpoint
# =============================================================================


async def example_inspect_checkpoint():
    """Load the persisted AgentState directly and look at every field.

    Reflexion's confidence score only moves when tools succeed, so the
    inspector gets a ``record_fact`` tool. After two turns that each
    fire a tool call, ``state.confidence`` should be > 0.
    """
    print("=== Part 5: Inspecting a Redis-backed checkpoint ===\n")

    model = get_model(max_tokens=200)
    checkpointer = _build_checkpointer("inspect")

    _facts: list[str] = []

    @tool
    def record_fact(fact: str) -> str:
        """Persist a fact the user has shared so the agent can remember it."""
        _facts.append(fact)
        return f"Recorded fact #{len(_facts)}: {fact}"

    agent = Agent(
        agent_id="inspector",
        model=model,
        system_prompt=(
            "You are a helpful assistant. Whenever the user shares a fact "
            "about themselves (their name, job, hobbies, etc.), call "
            "record_fact exactly once with a short summary of that fact, "
            "then reply naturally."
        ),
        tools=[record_fact],
        checkpointer=checkpointer,
        reflexion=True,
    )

    thread_id = "inspect_thread"

    agent.run_sync("Hello, my name is Charlie.", thread_id=thread_id)
    agent.run_sync("I work as a data scientist.", thread_id=thread_id)

    state = await checkpointer.load(thread_id)

    if state:
        print(f"Thread ID: {thread_id}")
        print(f"Agent ID: {state.agent_id}")
        print(f"Iteration: {state.iteration}")
        print(f"Message count: {len(state.messages)}")
        print(f"Tool calls so far: {len(state.tool_history)}")
        print(f"Confidence: {state.confidence:.2f}")
        print(f"Confidence history: {[round(c, 2) for c in state.confidence_history]}")

        print("\nMessages:")
        for i, msg in enumerate(state.messages):
            content = (
                msg.content[:50] + "..." if msg.content and len(msg.content) > 50 else msg.content
            )
            print(f"  {i}. [{msg.role.value}] {content}")

    print(f"\nFacts recorded by record_fact: {_facts}")
    print()


# =============================================================================
# Main
# =============================================================================


def _print_skip_banner(missing: list[str]) -> None:
    print("\n--- Notebook 10: Agent Memory & Checkpointing ---")
    print(
        "Required environment variables not set; skipping the live demo "
        "so this file still runs cleanly in CI.\n"
    )
    for name in missing:
        print(f"  - {name}")
    print("\nStart a Redis instance, set REDIS_URL, and re-run.")


def main():
    """Run all notebook parts."""
    print("=" * 60)
    print("Notebook 10: Agent Memory & Checkpointing on Redis")
    print("=" * 60)
    print()

    missing = _missing_env()
    if missing:
        _print_skip_banner(missing)
        return

    print_config()
    print()

    example_conversation_memory()
    example_checkpointing_with_tools()
    example_persistence_across_processes()
    example_multiple_threads()
    asyncio.run(example_inspect_checkpoint())

    print("=" * 60)
    print("Next: Notebook 11 — Agent Streaming")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
