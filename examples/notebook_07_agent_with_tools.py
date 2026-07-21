# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 07: giving an agent tools.

A model on its own can only answer from what's already in its context —
it can't check today's weather or count the words in your draft. Tools
let the agent reach out — look something up, run a small calculation —
and bring a real answer back into the conversation. Tulip runs this as a
small ReAct loop: the model decides whether to call a tool, Tulip runs
the tool, the result is fed back into the next model call.

Key ideas:
- ``@tool`` turns a plain Python function into something the model can
  call. The docstring is the description the model sees.
- Pass tools to ``Agent(tools=[...])`` and the agent picks when to use
  them.
- Each tool call shows up as a ``ToolStartEvent`` / ``ToolCompleteEvent``
  pair in the event stream — a clear record of every lookup.
- Tools can take typed arguments (including optional ones) and return
  anything JSON-serialisable — strings, numbers, dicts, lists.

The data here is made up on purpose: the weather table and the little
library catalog are fixed sample values, so the notebook runs the same
way every time.

Run it:
    .venv/bin/python examples/notebook_07_agent_with_tools.py

The default provider is the mock model; set TULIP_MODEL_PROVIDER for a live one (e.g.
``openai.gpt-4.1`` or ``meta.llama-3.3-70b-instruct``). Drop in
``TULIP_MODEL_PROVIDER=mock`` for an offline run. Tool-calling also
works against OpenAI, Anthropic.

Prerequisite: notebook 06.
"""

import asyncio
from datetime import datetime

# Import shared config
from config import get_model, print_config

from tulip.agent import Agent
from tulip.tools import tool


# =============================================================================
# Part 1: define a lookup tool
# =============================================================================

# A tool is a plain Python function decorated with @tool. The docstring
# is what the model reads to decide when to call it. The weather data
# below is invented — a small fixed table of sample values.


@tool
def get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    known = {
        "paris": "Paris: 18°C, cloudy",
        "tokyo": "Tokyo: 24°C, sunny",
        "cairo": "Cairo: 33°C, clear and dry",
    }
    return known.get(city.lower(), f"No weather data on file for {city}.")


@tool
def word_count(text: str) -> int:
    """Count the number of words in a piece of text."""
    return len(text.split())


async def example_simple_tools():
    """Show the tool metadata Tulip generates from a decorated function."""
    print("=== Part 1: Simple Tools ===\n")

    result = get_weather("Tokyo")
    print(f"Direct call: get_weather('Tokyo') = {result}")

    print(f"\nTool name: {get_weather.name}")
    print(f"Tool description: {get_weather.description}")
    print(f"Tool parameters: {get_weather.parameters}")

    import time as _t

    agent = Agent(
        model=get_model(max_tokens=80),
        system_prompt="Reply in one short sentence.",
    )
    t0 = _t.perf_counter()
    desc = await agent.arun(
        f"In one sentence, when would an assistant use a tool called '{get_weather.name}' "
        f"that {get_weather.description}?"
    )
    dt = _t.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · "
        f"{desc.metrics.prompt_tokens}→{desc.metrics.completion_tokens} tokens]"
    )
    print(f"  AI commentary: {desc.message.strip()}")
    print()


# =============================================================================
# Part 2: hand tools to an agent
# =============================================================================


async def example_agent_with_tools():
    """Wire tools into an Agent and let the model decide when to call them."""
    print("=== Part 2: Agent Using Tools ===\n")

    model = get_model(max_tokens=200)

    agent = Agent(
        model=model,
        tools=[get_weather, word_count],
        system_prompt="You are a helpful assistant. Use the provided tools to look up "
        "the weather or count words when the question calls for it.",
    )

    print(f"Agent has {len(agent.tools)} tools registered")

    result = await agent.arun("What's the weather in Tokyo right now?")
    print("\nQ: What's the weather in Tokyo right now?")
    print(f"A: {result.message}")
    print(f"Tool calls made: {result.metrics.tool_calls}")
    print()


# =============================================================================
# Part 3: tools with optional and typed arguments
# =============================================================================


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def add(a: int, b: int) -> int:
    """Add two whole numbers together."""
    return a + b


@tool
def greet(name: str, formal: bool = False) -> str:
    """Write a short greeting for a person.

    Args:
        name: The person to greet
        formal: Whether to use a formal tone (default: False)
    """
    if formal:
        return f"Good day, {name}. It is a pleasure to meet you."
    return f"Hey {name}! Nice to meet you."


async def example_complex_tools():
    """Tools with default arguments and varied return types."""
    print("=== Part 3: Complex Tools ===\n")

    model = get_model(max_tokens=200)

    agent = Agent(
        model=model,
        tools=[get_current_time, add, greet],
        system_prompt="You are a helpful assistant with access to a clock, a calculator, "
        "and a greeting tool.",
    )

    prompts = [
        "What time is it right now?",
        "What is 23 plus 19?",
        "Write a formal greeting for Dr. Chen.",
    ]

    for prompt in prompts:
        result = await agent.arun(prompt)
        print(f"Q: {prompt}")
        print(f"A: {result.message}")
        print()


# =============================================================================
# Part 4: watch lookups happen in the event stream
# =============================================================================


async def example_tool_events():
    """Stream events to see the model plan, call a tool, and use its result."""
    print("=== Part 4: Tool Execution Events ===\n")

    model = get_model(max_tokens=200)

    agent = Agent(
        model=model,
        tools=[get_weather, word_count],
        system_prompt="You are a helpful assistant. Use the tools to answer questions "
        "about the weather or about how long a piece of text is.",
    )

    print("Q: What's the weather in Paris, and how many words are in 'the quick brown fox'?\n")
    print("Events:")

    async for event in agent.run(
        "What's the weather in Paris, and how many words are in 'the quick brown fox'?"
    ):
        event_type = event.event_type

        if event_type == "tool_start":
            print(f"  TOOL_START: {event.tool_name}({event.arguments})")
        elif event_type == "tool_complete":
            print(f"  TOOL_COMPLETE: {event.tool_name} -> {event.result}")
        elif event_type == "think":
            if event.tool_calls:
                print(f"  THINK: Planning to call {len(event.tool_calls)} tool(s)")
        elif event_type == "terminate":
            print(f"  TERMINATE: {event.reason}")
            if event.final_message:
                print(f"\nFinal Answer: {event.final_message}")

    print()


# =============================================================================
# Part 5: tools that return structured data
# =============================================================================


@tool
def search_books(query: str, max_results: int = 3) -> list[dict]:
    """Search a small library catalog by title or genre.

    Args:
        query: Search query (matches book title or genre)
        max_results: Maximum number of results to return
    """
    # In-memory catalog stands in for a real library database. The search
    # logic below is the part worth reading. All entries are made up.
    books = [
        {"id": 1, "title": "The Blue Kite", "genre": "fiction", "available": True},
        {"id": 2, "title": "Cooking at Home", "genre": "cookbook", "available": True},
        {"id": 3, "title": "A Short History of Maps", "genre": "history", "available": False},
        {"id": 4, "title": "The Quiet Garden", "genre": "fiction", "available": True},
        {"id": 5, "title": "Stars and Seasons", "genre": "science", "available": True},
        {"id": 6, "title": "Bread Every Day", "genre": "cookbook", "available": False},
    ]

    # Case-insensitive match on title OR genre.
    q = query.lower()
    matches = [b for b in books if q in b["title"].lower() or q in b["genre"].lower()]
    return matches[:max_results]


@tool
def get_book_details(book_id: int) -> dict:
    """Get detailed information about a specific book in the catalog."""
    details = {
        1: {
            "id": 1,
            "title": "The Blue Kite",
            "author": "M. Okafor",
            "notes": "A quiet novel about a family and a summer that changes them.",
        },
        2: {
            "id": 2,
            "title": "Cooking at Home",
            "author": "L. Romano",
            "notes": "120 weeknight recipes, most under 30 minutes.",
        },
        3: {
            "id": 3,
            "title": "A Short History of Maps",
            "author": "P. Anand",
            "notes": "How people have drawn the world, from clay tablets to satellites.",
        },
        4: {
            "id": 4,
            "title": "The Quiet Garden",
            "author": "S. Fields",
            "notes": "Short stories set in one small town over forty years.",
        },
        5: {
            "id": 5,
            "title": "Stars and Seasons",
            "author": "R. Vega",
            "notes": "A friendly introduction to why the night sky changes.",
        },
        6: {
            "id": 6,
            "title": "Bread Every Day",
            "author": "T. Ito",
            "notes": "A gentle guide to baking your own loaves.",
        },
    }
    return details.get(book_id, {"error": f"Book {book_id} not found"})


async def example_structured_tools():
    """Tools can return dicts and lists — the model parses them on the next turn."""
    print("=== Part 5: Structured Data Tools ===\n")

    model = get_model(max_tokens=300)

    agent = Agent(
        model=model,
        tools=[search_books, get_book_details],
        system_prompt="You are a helpful library assistant. Help people find books.",
    )

    result = await agent.arun("Find some fiction books, then tell me more about 'The Blue Kite'.")
    print("Q: Find some fiction books, then tell me more about 'The Blue Kite'.")
    print(f"A: {result.message}")
    print(f"\nTool calls made: {result.metrics.tool_calls}")
    print()


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all notebook parts."""
    print("=" * 60)
    print("Notebook 07: Giving an Agent Tools")
    print("=" * 60)
    print()

    print_config()
    print()

    await example_simple_tools()
    await example_agent_with_tools()
    await example_complex_tools()
    await example_tool_events()
    await example_structured_tools()

    print("=" * 60)
    print("Next: Notebook 08 — Conversation Memory")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
