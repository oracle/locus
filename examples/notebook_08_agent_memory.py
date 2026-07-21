# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 08: conversation memory — remembering what you told it.

By default each call to an agent is independent — it forgets everything
the moment it answers. Give an agent a checkpointer and that changes:
every turn of the conversation is saved, so the next turn can pick up
where the last one left off. Earlier messages, names, and notes stay
intact.

This notebook uses ``MemoryCheckpointer`` — a simple in-memory store — so
it runs end-to-end with no setup. The exact same code works with a
durable backend (Redis, Postgres, and others) when you want the memory
to survive a restart; you just swap the checkpointer.

Key ideas:
- A ``checkpointer`` saves and restores agent state under keys you scope.
- ``thread_id`` keys conversations — one store can hold many independent
  conversations side by side without mixing them up.
- ``checkpoint_every_n_iterations=1`` writes after every loop iteration,
  so even a mid-tool-call interruption resumes cleanly.
- The saved state is rich: messages, tool history, iteration count,
  Reflexion confidence — all loadable for review.

Run it:
    .venv/bin/python examples/notebook_08_agent_memory.py

The agent itself goes through whichever provider you configure via
``TULIP_MODEL_PROVIDER`` (``openai`` / ``anthropic``).
Set ``TULIP_MODEL_PROVIDER=mock`` to bypass the model for offline runs.
"""

import asyncio
import sys

from config import get_model, print_config

from tulip.agent import Agent
from tulip.memory.backends import MemoryCheckpointer
from tulip.tools import tool


# One in-memory store per "suffix", cached so that two agents asking for
# the same suffix share the same state (see Part 3). A durable backend
# would share state through the external database instead.
_STORES: dict[str, MemoryCheckpointer] = {}


def _build_checkpointer(suffix: str = "default") -> MemoryCheckpointer:
    """Return a checkpointer for the given scope, reusing it if it exists."""
    return _STORES.setdefault(suffix, MemoryCheckpointer())


# =============================================================================
# Part 1: a two-turn conversation that remembers turn one
# =============================================================================


async def example_conversation_memory():
    """Same thread_id across two arun calls — the agent recalls turn one."""
    print("=== Part 1: Conversation memory ===\n")

    model = get_model(max_tokens=100)
    checkpointer = _build_checkpointer("chat")

    agent = Agent(
        model=model,
        system_prompt="You are a friendly assistant. Remember details the person shares.",
        checkpointer=checkpointer,
    )

    thread_id = "chat_1"

    result1 = await agent.arun(
        "Hi! My name is Sam and I'm planning a trip to Japan next spring.",
        thread_id=thread_id,
    )
    print("You: Hi! My name is Sam and I'm planning a trip to Japan next spring.")
    print(f"Assistant: {result1.message}")

    # Same thread_id — the agent loads the prior conversation state before
    # the next model call.
    result2 = await agent.arun("What's my name, and where am I going?", thread_id=thread_id)
    print("\nYou: What's my name, and where am I going?")
    print(f"Assistant: {result2.message}")
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


async def example_checkpointing_with_tools():
    """checkpoint_every_n_iterations=1 means a mid-loop interruption still recovers."""
    print("=== Part 2: Checkpoint after each iteration ===\n")

    model = get_model(max_tokens=150)
    checkpointer = _build_checkpointer("notes")

    agent = Agent(
        model=model,
        tools=[save_note, get_notes],
        system_prompt="You are a helpful note-taking assistant.",
        checkpointer=checkpointer,
        checkpoint_every_n_iterations=1,
    )

    thread_id = "notes_session"

    result1 = await agent.arun(
        "Save a note: buy oat milk and coffee filters on the way home.",
        thread_id=thread_id,
    )
    print("You: Save a note: buy oat milk and coffee filters on the way home.")
    print(f"Assistant: {result1.message}")
    print(f"Tool calls: {result1.metrics.tool_calls}")

    result2 = await agent.arun("What notes do we have so far?", thread_id=thread_id)
    print("\nYou: What notes do we have so far?")
    print(f"Assistant: {result2.message}")
    print()


# =============================================================================
# Part 3: a fresh agent picks up where the old one stopped
# =============================================================================


async def example_persistence_across_agents():
    """Two Agent objects, one store. The second loads the first's state."""
    print("=== Part 3: Reattaching to the same store ===\n")

    model = get_model(max_tokens=100)

    agent1 = Agent(
        model=model,
        system_prompt="You are a friendly assistant.",
        checkpointer=_build_checkpointer("persist"),
    )

    thread_id = "persistent_chat"

    result1 = await agent1.arun(
        "Remember: my favorite color is teal.",
        thread_id=thread_id,
    )
    print("You: Remember: my favorite color is teal.")
    print(f"Assistant: {result1.message}")

    # A *new* Agent object attaches to the same store and reads back the
    # conversation. With a durable backend this could even be a different
    # process after a restart — here they share one in-memory store.
    agent2 = Agent(
        model=model,
        system_prompt="You are a friendly assistant.",
        checkpointer=_build_checkpointer("persist"),
    )

    result2 = await agent2.arun("What's my favorite color?", thread_id=thread_id)
    print("\n[New agent — same store]")
    print("You: What's my favorite color?")
    print(f"Assistant: {result2.message}")
    print()


# =============================================================================
# Part 4: many conversations sharing one store
# =============================================================================


async def example_multiple_threads():
    """Two conversations, two thread_ids, one store — no cross-talk between them."""
    print("=== Part 4: Multiple conversations ===\n")

    model = get_model(max_tokens=100)
    checkpointer = _build_checkpointer("multi")

    agent = Agent(
        model=model,
        system_prompt="You are a friendly assistant.",
        checkpointer=checkpointer,
    )

    thread_travel = "chat_travel"
    thread_music = "chat_music"

    await agent.arun(
        "Let's talk about my upcoming hiking trip in the Alps.",
        thread_id=thread_travel,
    )
    await agent.arun(
        "I just started learning to play the guitar.",
        thread_id=thread_music,
    )

    result_travel = await agent.arun("What are we talking about?", thread_id=thread_travel)
    print("Thread 'chat_travel': What are we talking about?")
    print(f"Assistant: {result_travel.message}")

    result_music = await agent.arun("What are we talking about?", thread_id=thread_music)
    print("\nThread 'chat_music': What are we talking about?")
    print(f"Assistant: {result_music.message}")
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
    print("=== Part 5: Inspecting a saved checkpoint ===\n")

    model = get_model(max_tokens=200)
    checkpointer = _build_checkpointer("inspect")

    _details: list[str] = []

    @tool
    def record_detail(detail: str, category: str = "") -> str:
        """Persist a detail from the conversation so the record keeps it.

        Args:
            detail: Short summary of what the person shared.
            category: An optional tag (e.g. name, location, hobby, plan).
        """
        label = f"[{category}] {detail}" if category else detail
        _details.append(label)
        return f"Recorded detail #{len(_details)}: {label}"

    agent = Agent(
        agent_id="chat_inspector",
        model=model,
        system_prompt=(
            "You are a friendly assistant. Whenever the person shares a fact "
            "about themselves (a name, a place, a hobby, a plan, etc.), call "
            "record_detail exactly once with a short summary of that fact — "
            "and a category tag when one fits — then reply naturally."
        ),
        tools=[record_detail],
        checkpointer=checkpointer,
        reflexion=True,
    )

    thread_id = "inspect_chat"

    await agent.arun(
        "My name is Sam and I live in Portland.",
        thread_id=thread_id,
    )
    await agent.arun(
        "On weekends I like to go rock climbing.",
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


async def main():
    """Run all notebook parts."""
    print("=" * 60)
    print("Notebook 08: Conversation Memory")
    print("=" * 60)
    print()

    print_config()
    print()

    await example_conversation_memory()
    await example_checkpointing_with_tools()
    await example_persistence_across_agents()
    await example_multiple_threads()
    await example_inspect_checkpoint()

    print("=" * 60)
    print("Next: Notebook 11 — Streaming a Reply")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
