# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL
"""
Notebook 09: give an agent tools.

A model without tools can only answer from what's already in its
context. Tools let the agent reach out — do math, look up data, call
APIs — and bring the result back into the conversation. Tulip runs this
as a small loop: the model decides whether to call a tool, Tulip runs
the tool, the result is fed back into the next model call.

Key ideas:
- ``@tool`` turns a plain Python function into something the model can
  call. The docstring is the description the model sees.
- Pass tools to ``Agent(tools=[...])`` and the agent picks when to use
  them.
- Each tool call shows up as a ``ToolStartEvent`` / ``ToolCompleteEvent``
  pair in the event stream.
- Tools can take typed arguments (including optional ones) and return
  anything JSON-serialisable — strings, dicts, lists.

Run it:
    .venv/bin/python examples/notebook_15_agent_with_tools.py

The default provider is the mock model; set TULIP_MODEL_PROVIDER for a live one (e.g.
``openai.gpt-4.1`` or ``meta.llama-3.3-70b-instruct``). Drop in
``TULIP_MODEL_PROVIDER=mock`` for an offline run. Tool-calling also
works against OpenAI, Anthropic.

Prerequisite: notebook 08.
"""

import asyncio
from datetime import datetime

# Import shared config
from config import get_model, print_config

from tulip.agent import Agent
from tulip.tools import tool


# =============================================================================
# Part 1: define a tool
# =============================================================================

# A tool is a plain Python function decorated with @tool. The docstring
# is what the model reads to decide when to call it.


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


def example_simple_tools():
    """Show the tool metadata Tulip generates from a decorated function."""
    print("=== Part 1: Simple Tools ===\n")

    result = add_numbers(5, 3)
    print(f"Direct call: add_numbers(5, 3) = {result}")

    print(f"\nTool name: {add_numbers.name}")
    print(f"Tool description: {add_numbers.description}")
    print(f"Tool parameters: {add_numbers.parameters}")

    import time as _t

    agent = Agent(
        model=get_model(max_tokens=80),
        system_prompt="Reply in one short sentence.",
    )
    t0 = _t.perf_counter()
    desc = agent.run_sync(
        f"In one sentence, when would an LLM agent use a tool called '{add_numbers.name}' "
        f"that {add_numbers.description}?"
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


def example_agent_with_tools():
    """Wire tools into an Agent and let the model decide when to call them."""
    print("=== Part 2: Agent Using Tools ===\n")

    model = get_model(max_tokens=200)

    agent = Agent(
        model=model,
        tools=[add_numbers, multiply_numbers],
        system_prompt="You are a calculator assistant. Use the provided tools to perform calculations.",
    )

    print(f"Agent has {len(agent.tools)} tools registered")

    result = agent.run_sync("What is 15 + 27?")
    print("\nQ: What is 15 + 27?")
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
def calculate_age(birth_year: int) -> str:
    """Calculate someone's age given their birth year."""
    current_year = datetime.now().year
    age = current_year - birth_year
    return f"A person born in {birth_year} is {age} years old."


@tool
def format_greeting(name: str, formal: bool = False) -> str:
    """Create a greeting for someone.

    Args:
        name: The person's name
        formal: Whether to use formal greeting (default: False)
    """
    if formal:
        return f"Good day, {name}. It is a pleasure to meet you."
    return f"Hey {name}! Nice to meet you!"


def example_complex_tools():
    """Tools with default arguments and varied return types."""
    print("=== Part 3: Complex Tools ===\n")

    model = get_model(max_tokens=200)

    agent = Agent(
        model=model,
        tools=[get_current_time, calculate_age, format_greeting],
        system_prompt="You are a helpful assistant with access to time and greeting tools.",
    )

    prompts = [
        "What time is it right now?",
        "How old would someone born in 1990 be?",
        "Give me a formal greeting for Dr. Smith",
    ]

    for prompt in prompts:
        result = agent.run_sync(prompt)
        print(f"Q: {prompt}")
        print(f"A: {result.message}")
        print()


# =============================================================================
# Part 4: watch tool calls happen in the event stream
# =============================================================================


async def example_tool_events():
    """Stream events to see the model plan, call a tool, and use its result."""
    print("=== Part 4: Tool Execution Events ===\n")

    model = get_model(max_tokens=200)

    agent = Agent(
        model=model,
        tools=[add_numbers, multiply_numbers],
        system_prompt="Use tools to calculate. Always use tools for math.",
    )

    print("Q: What is (5 + 3) * 2?\n")
    print("Events:")

    async for event in agent.run("What is (5 + 3) * 2?"):
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
def search_products(query: str, max_results: int = 3) -> list[dict]:
    """Search for products in the catalog.

    Args:
        query: Search query
        max_results: Maximum number of results to return
    """
    # In-memory catalogue stands in for a database. The search logic
    # below is the part worth reading.
    products = [
        {"id": 1, "name": "Laptop", "price": 999.99, "category": "electronics", "in_stock": True},
        {
            "id": 2,
            "name": "Headphones",
            "price": 149.99,
            "category": "electronics",
            "in_stock": True,
        },
        {"id": 3, "name": "Mouse", "price": 49.99, "category": "electronics", "in_stock": True},
        {"id": 4, "name": "Keyboard", "price": 79.99, "category": "electronics", "in_stock": False},
        {"id": 5, "name": "Monitor", "price": 299.99, "category": "electronics", "in_stock": True},
        {"id": 6, "name": "Webcam", "price": 89.99, "category": "electronics", "in_stock": True},
        {
            "id": 7,
            "name": "Standing Desk",
            "price": 449.99,
            "category": "furniture",
            "in_stock": True,
        },
        {
            "id": 8,
            "name": "Office Chair",
            "price": 329.99,
            "category": "furniture",
            "in_stock": False,
        },
    ]

    # Case-insensitive match on name OR category.
    q = query.lower()
    matches = [p for p in products if q in p["name"].lower() or q in p["category"].lower()]
    return matches[:max_results]


@tool
def get_product_details(product_id: int) -> dict:
    """Get detailed information about a specific product."""
    details = {
        1: {
            "id": 1,
            "name": "Laptop",
            "price": 999.99,
            "specs": '16GB RAM, 512GB SSD, 14" 2.8K display',
        },
        2: {
            "id": 2,
            "name": "Headphones",
            "price": 149.99,
            "specs": "Noise-canceling, 40h battery, USB-C",
        },
        3: {
            "id": 3,
            "name": "Mouse",
            "price": 49.99,
            "specs": "Wireless, 16k DPI, programmable buttons",
        },
        4: {"id": 4, "name": "Keyboard", "price": 79.99, "specs": "Mechanical, hot-swappable, RGB"},
        5: {"id": 5, "name": "Monitor", "price": 299.99, "specs": '27" 4K IPS, 144Hz, USB-C 90W'},
        6: {"id": 6, "name": "Webcam", "price": 89.99, "specs": "1080p60, dual mic, auto-framing"},
        7: {
            "id": 7,
            "name": "Standing Desk",
            "price": 449.99,
            "specs": "Sit-stand, 60×30, programmable presets",
        },
        8: {
            "id": 8,
            "name": "Office Chair",
            "price": 329.99,
            "specs": "Lumbar support, adjustable arms",
        },
    }
    return details.get(product_id, {"error": f"Product {product_id} not found"})


def example_structured_tools():
    """Tools can return dicts and lists — the model parses them on the next turn."""
    print("=== Part 5: Structured Data Tools ===\n")

    model = get_model(max_tokens=300)

    agent = Agent(
        model=model,
        tools=[search_products, get_product_details],
        system_prompt="You are a shopping assistant. Help users find products.",
    )

    result = agent.run_sync("Find me some electronics, then tell me about the laptop")
    print("Q: Find me some electronics, then tell me about the laptop")
    print(f"A: {result.message}")
    print(f"\nTool calls made: {result.metrics.tool_calls}")
    print()


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all notebook parts."""
    print("=" * 60)
    print("Notebook 09: Agent with Tools")
    print("=" * 60)
    print()

    print_config()
    print()

    example_simple_tools()
    example_agent_with_tools()
    example_complex_tools()
    asyncio.run(example_tool_events())
    example_structured_tools()

    print("=" * 60)
    print("Next: Notebook 10 — Agent Memory")
    print("=" * 60)


if __name__ == "__main__":
    main()
