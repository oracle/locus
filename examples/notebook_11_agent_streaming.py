# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL
"""
Notebook 11: streaming events from an agent.

Every Tulip agent run is a stream of events: thinking, tool start, tool
complete, terminate. ``agent.run(prompt)`` returns an async iterator
that yields them in order. This notebook walks through the full set,
then shows useful patterns built on top — a live console UI, event
filtering, metric collection, and a progress bar.

Key ideas:
- ``ThinkEvent``: the model produced a plan (and possibly tool calls).
- ``ToolStartEvent`` / ``ToolCompleteEvent``: a tool ran; the complete
  event carries the result and duration.
- ``TerminateEvent``: the loop ended; carries the final message, stop
  reason, and iteration count.
- You filter the stream by ``isinstance(event, EventType)``.
- ``StructuredStream`` (mentioned at the end) wraps the stream and
  yields partial Pydantic instances as JSON arrives.

Run it:
    .venv/bin/python examples/notebook_17_agent_streaming.py

The default provider is the mock model; set TULIP_MODEL_PROVIDER for a live one (e.g.
``openai.gpt-4.1`` or ``meta.llama-3.3-70b-instruct``). For an offline
run set ``TULIP_MODEL_PROVIDER=mock``; OpenAI, Anthropic
work too.

Prerequisite: notebook 09.
"""

import ast
import asyncio
import operator as _op
from datetime import datetime

# Import shared config
from config import get_model, print_config

from tulip.agent import Agent
from tulip.core.events import (
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)
from tulip.tools import tool


# =============================================================================
# Part 1: every event type the agent emits
# =============================================================================


_SAFE_MATH_BIN_OPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
    ast.Pow: _op.pow,
}
_SAFE_MATH_UNARY_OPS = {ast.USub: _op.neg, ast.UAdd: _op.pos}


def _safe_math_eval(expression: str) -> float:
    """Evaluate arithmetic via AST — no eval, no attribute access, no names."""
    tree = ast.parse(expression, mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_MATH_BIN_OPS:
            return _SAFE_MATH_BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_MATH_UNARY_OPS:
            return _SAFE_MATH_UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported expression")

    return _eval(tree)


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        return str(_safe_math_eval(expression))
    except (ValueError, SyntaxError, ZeroDivisionError):
        return "Error: Invalid expression"


async def example_all_events():
    """Print every event with its distinguishing fields."""
    print("=== Part 1: Understanding Events ===\n")

    model = get_model(max_tokens=200)

    agent = Agent(
        model=model,
        tools=[calculate],
        system_prompt="You are a calculator. Always use the calculate tool for math.",
    )

    print("Running: 'What is 25 * 4?'\n")
    print("Events received:")

    async for event in agent.run("What is 25 * 4?"):
        print(f"\n  Event Type: {event.event_type}")
        print(f"  Timestamp:  {event.timestamp}")

        if isinstance(event, ThinkEvent):
            print(f"  Iteration:  {event.iteration}")
            print(f"  Tool Calls: {len(event.tool_calls)}")
            if event.reasoning:
                preview = (
                    event.reasoning[:80] + "..." if len(event.reasoning) > 80 else event.reasoning
                )
                print(f"  Reasoning:  {preview}")

        elif isinstance(event, ToolStartEvent):
            print(f"  Tool Name:  {event.tool_name}")
            print(f"  Arguments:  {event.arguments}")

        elif isinstance(event, ToolCompleteEvent):
            print(f"  Tool Name:  {event.tool_name}")
            print(f"  Result:     {event.result}")
            print(f"  Duration:   {event.duration_ms:.1f}ms")

        elif isinstance(event, TerminateEvent):
            print(f"  Reason:     {event.reason}")
            print(f"  Iterations: {event.iterations_used}")
            if event.final_message:
                print(f"  Answer:     {event.final_message}")

    print()


# =============================================================================
# Part 2: a live console UI driven by the event stream
# =============================================================================


async def example_console_ui():
    """Render a tiny progress UI as the agent thinks and calls tools."""
    print("=== Part 2: Console UI ===\n")

    model = get_model(max_tokens=200)

    @tool
    def search_database(query: str) -> str:
        """Search the internal database."""
        # Sleep so the UI has visible work to show.
        import time

        time.sleep(0.5)
        return f"Found 3 results for '{query}'"

    @tool
    def analyze_results(data: str) -> str:
        """Analyze search results."""
        return f"Analysis complete: {data} contains useful information"

    agent = Agent(
        model=model,
        tools=[search_database, analyze_results],
        system_prompt="You are a research assistant. Search and analyze data.",
    )

    print("Query: Find information about Python and analyze it\n")

    async for event in agent.run("Find information about Python and analyze it"):
        if isinstance(event, ThinkEvent):
            print("[Thinking...]")
            if event.tool_calls:
                for tc in event.tool_calls:
                    print(f"  Planning to call: {tc.name}")

        elif isinstance(event, ToolStartEvent):
            print(f"[Running] {event.tool_name}...", end=" ", flush=True)

        elif isinstance(event, ToolCompleteEvent):
            if event.error:
                print(f"ERROR: {event.error}")
            else:
                print(f"Done ({event.duration_ms:.0f}ms)")

        elif isinstance(event, TerminateEvent):
            print(f"\n[Complete] {event.reason}")
            if event.final_message:
                print(f"\nAnswer: {event.final_message}")

    print()


# =============================================================================
# Part 3: pick out just the events you care about
# =============================================================================


async def example_event_filtering():
    """Build an audit log by keeping only ToolStart/ToolComplete events."""
    print("=== Part 3: Event Filtering ===\n")

    model = get_model(max_tokens=200)

    agent = Agent(
        model=model,
        tools=[calculate],
        system_prompt="Use the calculate tool for math.",
    )

    tool_log = []

    async for event in agent.run("Calculate 10 + 20 + 30"):
        if isinstance(event, ToolStartEvent):
            tool_log.append(
                {
                    "action": "start",
                    "tool": event.tool_name,
                    "args": event.arguments,
                    "time": datetime.now().isoformat(),
                }
            )

        elif isinstance(event, ToolCompleteEvent):
            tool_log.append(
                {
                    "action": "complete",
                    "tool": event.tool_name,
                    "result": event.result,
                    "duration_ms": event.duration_ms,
                }
            )

    print("Tool execution log:")
    for entry in tool_log:
        print(f"  {entry}")
    print()


# =============================================================================
# Part 4: roll up metrics from the event stream
# =============================================================================


async def example_collect_metrics():
    """Count events and sum tool durations to get per-run telemetry."""
    print("=== Part 4: Collecting Metrics ===\n")

    model = get_model(max_tokens=200)

    @tool
    def step_one(data: str) -> str:
        """First processing step."""
        return f"Step 1 processed: {data}"

    @tool
    def step_two(data: str) -> str:
        """Second processing step."""
        return f"Step 2 processed: {data}"

    agent = Agent(
        model=model,
        tools=[step_one, step_two],
        system_prompt="Process data through step_one then step_two.",
    )

    metrics = {
        "think_events": 0,
        "tool_starts": 0,
        "tool_completes": 0,
        "total_tool_time_ms": 0,
        "iterations": 0,
    }

    start_time = datetime.now()

    async for event in agent.run("Process 'hello world' through both steps"):
        if isinstance(event, ThinkEvent):
            metrics["think_events"] += 1
        elif isinstance(event, ToolStartEvent):
            metrics["tool_starts"] += 1
        elif isinstance(event, ToolCompleteEvent):
            metrics["tool_completes"] += 1
            metrics["total_tool_time_ms"] += event.duration_ms or 0
        elif isinstance(event, TerminateEvent):
            metrics["iterations"] = event.iterations_used

    elapsed = (datetime.now() - start_time).total_seconds() * 1000

    print("Execution Metrics:")
    print(f"  Think events:    {metrics['think_events']}")
    print(f"  Tool starts:     {metrics['tool_starts']}")
    print(f"  Tool completes:  {metrics['tool_completes']}")
    print(f"  Tool time:       {metrics['total_tool_time_ms']:.1f}ms")
    print(f"  Iterations:      {metrics['iterations']}")
    print(f"  Total time:      {elapsed:.1f}ms")
    print()


# =============================================================================
# Part 5: a progress bar from ToolCompleteEvent
# =============================================================================


async def example_progress_tracking():
    """Draw a textual progress bar as each tool completes."""
    print("=== Part 5: Progress Tracking ===\n")

    model = get_model(max_tokens=300)

    # Deterministic tool bodies: fetch_data returns a real summary of an
    # in-memory source, process_data hashes its input. No canned strings.
    sources = {
        "users": [
            {"id": 1, "name": "Alice", "role": "engineer"},
            {"id": 2, "name": "Bob", "role": "designer"},
            {"id": 3, "name": "Charlie", "role": "researcher"},
        ],
        "orders": [
            {"id": 101, "user_id": 1, "total": 49.99},
            {"id": 102, "user_id": 2, "total": 19.99},
        ],
    }

    @tool
    def fetch_data(source: str) -> str:
        """Fetch records from a named in-memory source (users, orders, ...)."""
        import json
        import time

        time.sleep(0.3)
        rows = sources.get(source.lower())
        if rows is None:
            return f"unknown source {source!r}; try one of: {sorted(sources)}"
        return f"{len(rows)} record(s) from {source}: {json.dumps(rows)}"

    @tool
    def process_data(data: str) -> str:
        """Process fetched data — counts records and emits a checksum."""
        import hashlib
        import time

        time.sleep(0.2)
        digest = hashlib.sha256(data.encode()).hexdigest()[:12]
        return f"processed {len(data)} bytes, sha256:{digest}"

    @tool
    def store_results(results: str) -> str:
        """Store processed results."""
        import time

        time.sleep(0.1)
        return "Results stored successfully"

    agent = Agent(
        model=model,
        tools=[fetch_data, process_data, store_results],
        system_prompt="Fetch data from 'api', process it, and store the results.",
    )

    steps_done = 0
    total_steps = 3  # we expect three tool calls: fetch, process, store

    print("Processing...")

    async for event in agent.run("Fetch, process, and store data from the API"):
        if isinstance(event, ToolCompleteEvent):
            steps_done += 1
            progress = (steps_done / total_steps) * 100
            bar = "#" * int(progress / 10) + "-" * (10 - int(progress / 10))
            print(f"  [{bar}] {progress:.0f}% - {event.tool_name} complete")

        elif isinstance(event, TerminateEvent):
            print(f"\nDone! Final message: {event.final_message}")

    print()


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all notebook parts."""
    print("=" * 60)
    print("Notebook 11: Agent Streaming & Events")
    print("=" * 60)
    print()

    print_config()
    print()

    asyncio.run(example_all_events())
    asyncio.run(example_console_ui())
    asyncio.run(example_event_filtering())
    asyncio.run(example_collect_metrics())
    asyncio.run(example_progress_tracking())

    # =========================================================================
    # See also: structured streaming
    # =========================================================================
    print("=== See also: StructuredStream ===\n")
    print(
        "Want incremental Pydantic instances instead of raw chunks?\n"
        "Wrap any agent stream with StructuredStream:\n"
    )
    print(
        """    from tulip.streaming import StructuredStream

    stream = StructuredStream(agent.run("Top 3 vendors."), schema=VendorList)
    async for partial in stream:
        ui.render(partial)               # may have 0, 1, 2, then 3 vendors
    final: VendorList | None = stream.final
"""
    )
    print(
        "Tulip buffers each ModelChunkEvent, closes any unbalanced JSON\n"
        "braces / brackets / strings, runs the result through\n"
        "schema.model_validate, and yields the parsed instance if it\n"
        "validates.\n"
    )

    print("=" * 60)
    print("Next: Notebook 12 — Agent Hooks")
    print("=" * 60)


if __name__ == "__main__":
    main()
