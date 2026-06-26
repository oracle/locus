# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 08: conversation memory — the Desk, backed by Redis.

Give a support copilot a checkpointer and every turn of the conversation
is written to the Desk: the help desk's durable store of checkpointed
ticket state. Restart the process, attach a fresh agent to the same
connection, hand it the same ``thread_id``, and the conversation picks
up where it left off — earlier messages, order numbers, and notes
intact. This is the production memory story for Tulip.

Key ideas:
- ``RedisBackend(...)`` opens a connection to Redis and stores agent
  state under keys you scope.
- ``thread_id`` keys conversations — one Redis can host many independent
  customer tickets side by side without cross-contamination.
- ``checkpoint_every_n_iterations=1`` writes after every loop iteration,
  so a crash mid-tool-call still resumes cleanly.
- The saved state is rich: messages, tool history, iteration count,
  Reflexion confidence — all loadable for ticket review.

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
    """Same thread_id across two run_sync calls — the copilot recalls turn one."""
    print("=== Part 1: Conversation memory (Redis) ===\n")

    model = get_model(max_tokens=100)
    checkpointer = _build_checkpointer("ticket")

    agent = Agent(
        model=model,
        system_prompt="You are a customer-support copilot. Remember details the customer shares.",
        checkpointer=checkpointer,
    )

    thread_id = "ticket_4042"

    result1 = agent.run_sync(
        "Hi, my order #A-4042 hasn't arrived and tracking shows it stuck at the depot.",
        thread_id=thread_id,
    )
    print("Customer: Hi, my order #A-4042 hasn't arrived and tracking shows it stuck at the depot.")
    print(f"Copilot: {result1.message}")

    # Same thread_id — the agent loads prior conversation state from Redis
    # before the next model call.
    result2 = agent.run_sync("Which order number are we talking about?", thread_id=thread_id)
    print("\nCustomer: Which order number are we talking about?")
    print(f"Copilot: {result2.message}")
    print()


# =============================================================================
# Part 2: write a checkpoint after every iteration
# =============================================================================


_TICKET_NOTES: list[str] = []


@tool
def save_ticket_note(content: str) -> str:
    """Save a support note for later reference."""
    _TICKET_NOTES.append(content)
    return f"Ticket note saved: {content}"


@tool
def get_ticket_notes() -> str:
    """Get all saved support notes."""
    if not _TICKET_NOTES:
        return "No ticket notes saved yet."
    lines = "\n".join(f"- {n}" for n in _TICKET_NOTES)
    return f"You have {len(_TICKET_NOTES)} ticket note(s):\n{lines}"


def example_checkpointing_with_tools():
    """checkpoint_every_n_iterations=1 means a mid-loop crash still recovers."""
    print("=== Part 2: Checkpoint after each iteration ===\n")

    model = get_model(max_tokens=150)
    checkpointer = _build_checkpointer("notes")

    agent = Agent(
        model=model,
        tools=[save_ticket_note, get_ticket_notes],
        system_prompt="You are a support note-taking assistant.",
        checkpointer=checkpointer,
        checkpoint_every_n_iterations=1,
    )

    thread_id = "ticket_notes_session"

    result1 = agent.run_sync(
        "Save a ticket note: customer confirmed shipping address 12 Elm St is correct",
        thread_id=thread_id,
    )
    print("Customer: Save a ticket note: customer confirmed shipping address 12 Elm St is correct")
    print(f"Copilot: {result1.message}")
    print(f"Tool calls: {result1.metrics.tool_calls}")

    result2 = agent.run_sync("What ticket notes do we have so far?", thread_id=thread_id)
    print("\nCustomer: What ticket notes do we have so far?")
    print(f"Copilot: {result2.message}")
    print()


# =============================================================================
# Part 3: a fresh agent picks up where the old one stopped
# =============================================================================


def example_persistence_across_processes():
    """Two Agent objects, one Redis. The second loads the first's ticket state."""
    print("=== Part 3: Cross-process persistence ===\n")

    model = get_model(max_tokens=100)

    agent1 = Agent(
        model=model,
        system_prompt="You are a customer-support copilot.",
        checkpointer=_build_checkpointer("persist"),
    )

    thread_id = "persistent_ticket"

    result1 = agent1.run_sync(
        "Remember: the support ticket for this conversation is CS-2026-042.",
        thread_id=thread_id,
    )
    print("Customer: Remember: the support ticket for this conversation is CS-2026-042.")
    print(f"Copilot: {result1.message}")

    # Pretend the agent's shift ended and the process restarted. A *new*
    # Agent + checkpointer object connects to the same Redis and reads
    # back the ticket state.
    agent2 = Agent(
        model=model,
        system_prompt="You are a customer-support copilot.",
        checkpointer=_build_checkpointer("persist"),
    )

    result2 = agent2.run_sync("What was the support ticket number?", thread_id=thread_id)
    print("\n[New process — same Redis]")
    print("Customer: What was the support ticket number?")
    print(f"Copilot: {result2.message}")
    print()


# =============================================================================
# Part 4: many conversations sharing one database
# =============================================================================


def example_multiple_threads():
    """Two tickets, two thread_ids, one Redis — no cross-talk between them."""
    print("=== Part 4: Multiple conversations ===\n")

    model = get_model(max_tokens=100)
    checkpointer = _build_checkpointer("multi")

    agent = Agent(
        model=model,
        system_prompt="You are a customer-support copilot.",
        checkpointer=checkpointer,
    )

    thread_billing = "ticket_billing"
    thread_shipping = "ticket_shipping"

    agent.run_sync(
        "This ticket is about a double charge on invoice INV-7781.",
        thread_id=thread_billing,
    )
    agent.run_sync(
        "This ticket is about a package marked delivered but never received.",
        thread_id=thread_shipping,
    )

    result_billing = agent.run_sync("What is this ticket about?", thread_id=thread_billing)
    print("Thread 'ticket_billing': What is this ticket about?")
    print(f"Copilot: {result_billing.message}")

    result_shipping = agent.run_sync("What is this ticket about?", thread_id=thread_shipping)
    print("\nThread 'ticket_shipping': What is this ticket about?")
    print(f"Copilot: {result_shipping.message}")
    print()


# =============================================================================
# Part 5: load and walk a saved checkpoint
# =============================================================================


async def example_inspect_checkpoint():
    """Load the persisted AgentState directly and look at every field.

    Reflexion's confidence score only moves when tools succeed, so the
    inspector gets a ``record_detail`` tool. After two turns that each
    fire a tool call, ``state.confidence`` should be > 0.
    """
    print("=== Part 5: Inspecting a Redis-backed checkpoint ===\n")

    model = get_model(max_tokens=200)
    checkpointer = _build_checkpointer("inspect")

    _details: list[str] = []

    @tool
    def record_detail(detail: str, category: str = "") -> str:
        """Persist a detail from the conversation so the ticket record keeps it.

        Args:
            detail: Short summary of what the customer reported.
            category: An optional tag (e.g. order, billing, shipping, account).
        """
        label = f"[{category}] {detail}" if category else detail
        _details.append(label)
        return f"Recorded detail #{len(_details)}: {label}"

    agent = Agent(
        agent_id="ticket_inspector",
        model=model,
        system_prompt=(
            "You are a customer-support copilot. Whenever the customer reports "
            "a fact about their issue (an order number, a charge, a delivery "
            "status, etc.), call record_detail exactly once with a short "
            "summary of that fact — and a category tag when one fits — then "
            "reply naturally."
        ),
        tools=[record_detail],
        checkpointer=checkpointer,
        reflexion=True,
    )

    thread_id = "inspect_ticket"

    agent.run_sync(
        "Reported: order #A-4042 was charged twice on the same card.",
        thread_id=thread_id,
    )
    agent.run_sync(
        "Reported: the customer is on the premium support plan.",
        thread_id=thread_id,
    )

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

    print(f"\nDetails recorded by record_detail: {_details}")
    print()


# =============================================================================
# Main
# =============================================================================


def _print_skip_banner(missing: list[str]) -> None:
    print("\n--- Notebook 08: The Desk — Conversation Memory ---")
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
    print("Notebook 08: The Desk — Conversation Memory on Redis")
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
    print("Next: Notebook 11 — Streaming a Support Reply")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
