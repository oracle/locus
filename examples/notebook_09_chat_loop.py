# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 09: a chat loop — actually talking to your agent.

Every other example in this suite is a batch script: it asks one thing,
prints the answer, and exits. That is the right shape for teaching a
primitive and the wrong shape for the first thing most people want to do,
which is have a conversation.

This is the missing REPL. It is deliberately small — a loop, a checkpointer,
and one ``thread_id`` — because the interesting part is not the loop, it is
that continuity comes from the checkpointer rather than from anything you
have to hold in your own code. Every turn is saved; the next turn resumes
from the saved state. Swap ``MemoryCheckpointer`` for Redis or Postgres and
the same conversation survives a restart.

Key ideas:
- A ``thread_id`` names the conversation; the checkpointer stores it.
- The loop keeps no history of its own — the agent's state is the history.
- Tool calls are printed as they happen, so you can see the agent act
  rather than infer it from the answer.
- ``/reset`` starts a fresh thread, which shows the boundary a
  ``thread_id`` draws: same agent, no shared memory.

Run it:
    .venv/bin/python examples/notebook_09_chat_loop.py

Non-interactive (CI, or just to see the shape) — the script detects a
non-tty and replays a scripted conversation instead of prompting:
    echo "" | .venv/bin/python examples/notebook_09_chat_loop.py

Live model:
    TULIP_MODEL_PROVIDER=openai TULIP_MODEL_ID=gpt-4o
    (or TULIP_MODEL_PROVIDER=anthropic, or any OpenAI-compatible
    provider: ollama, vllm, groq, together, openrouter, ...).
Set TULIP_MODEL_PROVIDER=mock for an offline run.
"""

import asyncio
import os
import sys

from config import get_model, print_config

from tulip.agent import Agent
from tulip.memory.backends import MemoryCheckpointer
from tulip.tools import tool


BANNER = """
Type a message and press enter. The agent remembers the conversation.

  /reset   start a new thread (drops the memory)
  /thread  show the current thread id
  /exit    quit
"""

# A scripted conversation for non-interactive runs. The second turn only
# makes sense if the first was remembered, which is the point being shown.
SCRIPTED = [
    "My name is Federico and I'm looking at flights to Lisbon.",
    "What's the weather there?",
    "What did I say my name was?",
]


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    known = {
        "lisbon": "18°C, light rain",
        "berlin": "9°C, overcast",
        "tokyo": "24°C, clear",
    }
    return f"{city}: {known.get(city.lower(), 'no data on file')}"


def build_agent() -> Agent:
    """One agent, one checkpointer. Continuity lives in the checkpointer."""
    return Agent(
        model=get_model(),
        tools=[get_weather],
        checkpointer=MemoryCheckpointer(),
        # Write after every iteration so an interrupted turn resumes cleanly
        # rather than losing the tool call that was in flight.
        checkpoint_every_n_iterations=1,
        system_prompt=(
            "You are a concise travel assistant. Use the weather tool when "
            "asked about weather. Keep answers to one or two sentences."
        ),
    )


async def say(agent: Agent, thread_id: str, text: str) -> None:
    """Send one turn and print the answer plus any tools the agent used."""
    result = await agent.arun(text, thread_id=thread_id)
    for execution in result.tool_executions:
        print(f"    · {execution.tool_name}({execution.arguments})")
    print(f"  agent: {(result.text or '').strip()}\n")


async def scripted_run(agent: Agent) -> None:
    """Replay a fixed conversation — used when there is no terminal."""
    thread_id = "chat-scripted"
    print("(no terminal detected — replaying a scripted conversation)\n")
    for line in SCRIPTED:
        print(f"  you  : {line}")
        await say(agent, thread_id, line)

    if os.environ.get("TULIP_MODEL_PROVIDER", "mock").lower() == "mock":
        # Be straight about it: the mock answers the same way whatever it is
        # asked, so this run shows the loop and the tool calls but proves
        # nothing about memory. Against a real model the third answer names
        # you, which is the part worth seeing.
        print(
            "The mock answers identically every turn, so memory is not "
            "visible here.\nRe-run with a live provider and the third answer "
            "will repeat your name back."
        )
    else:
        print("The third answer is only possible because the first turn was saved.")


async def interactive_run(agent: Agent) -> None:
    """Read from stdin until the user leaves."""
    thread_id = "chat-1"
    turn = 1
    print(BANNER)
    while True:
        try:
            # Off the event loop: a bare input() blocks it, which would stall
            # any streaming or background work the agent has in flight.
            line = (await asyncio.to_thread(input, "  you  : ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line == "/exit":
            break
        if line == "/thread":
            print(f"  thread: {thread_id}\n")
            continue
        if line == "/reset":
            turn += 1
            thread_id = f"chat-{turn}"
            print(f"  (new thread: {thread_id} — previous memory is not visible)\n")
            continue

        await say(agent, thread_id, line)

    print("Bye.")


async def main() -> None:
    print("=" * 60)
    print("Notebook 09: A chat loop")
    print("=" * 60)
    print()
    print_config()

    agent = build_agent()
    if sys.stdin.isatty():
        await interactive_run(agent)
    else:
        await scripted_run(agent)

    print()
    print("=" * 60)
    print("Next: Notebook 11 — Streaming")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
