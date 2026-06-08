# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL
"""
Build a workflow as a graph of nodes that pass state to each other.

A StateGraph is a directed graph where each node is an async function
that takes the current state in and returns updates to merge back. Use
it when one Agent isn't enough — multi-step pipelines, branching
logic, fan-out / fan-in, human approval gates.

- Nodes and edges: nodes do work, edges describe order.
- START and END: the two sentinel node ids that frame execution.
- Sequential, parallel, and conditional flow on the same primitives.
- GraphResult: final state plus per-node status, timing, and ordering.
- Streaming: emit progress events from inside a node with emit_custom.

Run it:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_22_basic_graph.py

The default provider is the bundled mock model; set TULIP_MODEL_PROVIDER for a live provider.
Set TULIP_MODEL_PROVIDER=mock for offline runs. Pick a live provider with
TULIP_MODEL_ID=openai.gpt-4.1 (or meta.llama-3.3-70b-instruct, etc.).
"""

import asyncio
import time

from config import get_model

from tulip.agent import Agent
from tulip.multiagent import END, START, StateGraph


def _llm_call(
    prompt: str, *, system: str = "Reply in one short sentence.", max_tokens: int = 80
) -> str:
    """Run a one-shot Agent and print a timing/token banner. Used by every part."""
    agent = Agent(model=get_model(max_tokens=max_tokens), system_prompt=system)
    t0 = time.perf_counter()
    res = agent.run_sync(prompt)
    dt = time.perf_counter() - t0
    print(
        f"  [model call: {dt:.2f}s · {res.metrics.prompt_tokens}→{res.metrics.completion_tokens} tokens]"
    )
    return res.message.strip()


# =============================================================================
# Part 1: A one-node graph
# =============================================================================


async def example_first_graph():
    """The smallest possible graph: START -> greet -> END."""
    print("=== Part 1: A one-node graph ===\n")

    graph = StateGraph()

    async def greet(inputs):
        name = inputs.get("name", "World")
        ai_line = _llm_call(
            f"Greet {name} warmly in one sentence.",
            system="You are a friendly assistant.",
        )
        return {"greeting": ai_line}

    graph.add_node("greet", greet)
    graph.add_edge(START, "greet")
    graph.add_edge("greet", END)

    result = await graph.execute({"name": "Alice"})

    print("Input:   name = 'Alice'")
    print(f"Output:  greeting = '{result.final_state.get('greeting')}'")
    print(f"Success: {result.success}")
    print()


# =============================================================================
# Part 2: A sequence of nodes
# =============================================================================


async def example_sequence():
    """Chain three nodes; each node's return value merges into shared state."""
    print("=== Part 2: A sequence of nodes ===\n")

    graph = StateGraph()

    async def validate(inputs):
        text = inputs.get("text", "")
        return {
            "text": text.strip(),
            "is_valid": len(text.strip()) > 0,
        }

    async def transform(inputs):
        text = inputs.get("text", "")
        return {
            "uppercase": text.upper(),
            "word_count": len(text.split()),
        }

    async def summarize(inputs):
        wc = inputs.get("word_count")
        ai = _llm_call(
            f"In one short sentence, comment on a text with {wc} words "
            f"that was {'valid' if inputs.get('is_valid') else 'invalid'}.",
        )
        return {
            "summary": f"{wc} words, valid={inputs.get('is_valid')} — {ai}",
        }

    graph.add_node("validate", validate)
    graph.add_node("transform", transform)
    graph.add_node("summarize", summarize)

    graph.add_edge(START, "validate")
    graph.add_edge("validate", "transform")
    graph.add_edge("transform", "summarize")
    graph.add_edge("summarize", END)

    result = await graph.execute({"text": "  hello world  "})

    print("Input:     text = '  hello world  '")
    print(f"Validated: is_valid = {result.final_state.get('is_valid')}")
    print(f"Uppercase: {result.final_state.get('uppercase')}")
    print(f"Summary:   {result.final_state.get('summary')}")
    print()


# =============================================================================
# Part 3: How state accumulates
# =============================================================================


async def example_state_flow():
    """Print what each node receives so you can watch state grow."""
    print("=== Part 3: How state accumulates ===\n")

    graph = StateGraph()

    async def step_a(inputs):
        print(f"  Step A receives: {list(inputs.keys())}")
        return {"a_output": "from A", "value": 10}

    async def step_b(inputs):
        print(f"  Step B receives: {list(inputs.keys())}")
        value = inputs.get("value", 0)
        return {"b_output": "from B", "doubled": value * 2}

    async def step_c(inputs):
        print(f"  Step C receives: {list(inputs.keys())}")
        doubled = inputs.get("doubled", 0)
        ai = _llm_call(
            f"Briefly comment on a graph that doubled the value to {doubled}.",
        )
        return {"c_output": "from C", "final": doubled + 5, "ai_comment": ai}

    graph.add_node("step_a", step_a)
    graph.add_node("step_b", step_b)
    graph.add_node("step_c", step_c)

    graph.add_edge(START, "step_a")
    graph.add_edge("step_a", "step_b")
    graph.add_edge("step_b", "step_c")
    graph.add_edge("step_c", END)

    print("Executing graph...")
    result = await graph.execute({"initial_data": True})

    print("\nFinal state:")
    for key, value in result.final_state.items():
        if not key.startswith("_"):
            print(f"  {key}: {value}")
    print()


# =============================================================================
# Part 4: Fan-out and fan-in
# =============================================================================


async def example_parallel():
    """Three independent nodes run concurrently, then converge in one."""
    print("=== Part 4: Fan-out and fan-in ===\n")

    graph = StateGraph()
    graph.config.parallel = True

    async def analyze_sentiment(inputs):
        text = inputs.get("text", "")
        label = _llm_call(
            f"Classify the sentiment of '{text}' as positive, negative, or "
            "neutral. Reply with one word.",
            system="Output one of: positive | negative | neutral. Nothing else.",
            max_tokens=10,
        )
        return {"sentiment": label.lower()}

    async def count_words(inputs):
        text = inputs.get("text", "")
        await asyncio.sleep(0.1)
        return {"word_count": len(text.split())}

    async def detect_language(inputs):
        await asyncio.sleep(0.1)
        return {"language": "en"}

    async def combine_results(inputs):
        return {
            "analysis": {
                "sentiment": inputs.get("sentiment"),
                "words": inputs.get("word_count"),
                "lang": inputs.get("language"),
            }
        }

    graph.add_node("sentiment", analyze_sentiment)
    graph.add_node("words", count_words)
    graph.add_node("language", detect_language)
    graph.add_node("combine", combine_results)

    graph.add_edge(START, "sentiment")
    graph.add_edge(START, "words")
    graph.add_edge(START, "language")

    graph.add_edge("sentiment", "combine")
    graph.add_edge("words", "combine")
    graph.add_edge("language", "combine")

    graph.add_edge("combine", END)

    import time

    start = time.time()
    result = await graph.execute({"text": "This is a great example!"})
    elapsed = (time.time() - start) * 1000

    print("Input: 'This is a great example!'")
    print(f"Analysis: {result.final_state.get('analysis')}")
    print(f"Time: {elapsed:.0f}ms (parallel nodes run concurrently)")
    print()


# =============================================================================
# Part 5: Reading GraphResult
# =============================================================================


async def example_results():
    """GraphResult gives you final state plus per-node status and timing."""
    print("=== Part 5: Reading GraphResult ===\n")

    graph = StateGraph()

    async def process(inputs):
        v = inputs.get("value", 0)
        comment = _llm_call(f"In one sentence, comment on doubling {v}.")
        return {"processed": True, "result": v * 2, "comment": comment}

    graph.add_node("process", process)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)

    result = await graph.execute({"value": 21})

    print("GraphResult fields:")
    print(f"  .success         = {result.success}")
    print(f"  .graph_id        = {result.graph_id}")
    print(f"  .duration_ms     = {result.duration_ms:.1f}")
    print(f"  .iterations      = {result.iterations}")
    print(f"  .execution_order = {result.execution_order}")

    print("\n  .final_state:")
    for k, v in result.final_state.items():
        if not k.startswith("_"):
            print(f"    {k}: {v}")

    print("\n  .node_results:")
    for node_id, node_result in result.node_results.items():
        print(f"    {node_id}: status={node_result.status.value}")
    print()


async def example_streaming():
    """Stream node updates and push custom progress events from inside a node."""
    print("=== Part 6: Streaming with emit_custom ===\n")
    from tulip.multiagent import StreamMode, emit_custom

    graph = StateGraph()

    async def step1(inputs):
        # emit_custom — push an arbitrary payload onto the event stream
        # while the node is still running. Useful for long-running steps.
        await emit_custom({"phase": "starting", "value": inputs.get("x", 0)})
        ai = _llm_call(
            f"In one sentence, congratulate someone who reached {inputs.get('x', 0) * 2}.",
        )
        await emit_custom({"phase": "halfway"})
        return {"y": inputs.get("x", 0) * 2, "ai": ai}

    async def step2(inputs):
        ai = _llm_call(
            f"In one short sentence, narrate adding 10 to {inputs.get('y', 0)}.",
        )
        return {"z": inputs.get("y", 0) + 10, "ai": ai}

    graph.add_node("step1", step1)
    graph.add_node("step2", step2)
    graph.add_edge(START, "step1")
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", END)

    print("Streaming UPDATES + CUSTOM events as they arrive:")
    async for event in graph.stream({"x": 21}, mode=StreamMode.UPDATES):
        if event.mode == StreamMode.CUSTOM:
            print(f"  [CUSTOM]  {event.node_id}: {event.data}")
        else:
            print(f"  [UPDATE]  {event.node_id}: {event.data}")
    print()


# =============================================================================
# Part 7: An Agent inside a node
# =============================================================================


async def example_graph_with_llm():
    """A node can hold a full Agent — graphs and agents compose freely."""
    print("=== Part 7: An Agent inside a node ===\n")

    graph = StateGraph()

    async def ai_summarize(inputs):
        import time as _t

        topic = inputs.get("topic", "")
        agent = Agent(
            model=get_model(max_tokens=80),
            system_prompt="You write one-sentence factual summaries.",
        )
        t0 = _t.perf_counter()
        result = agent.run_sync(f"Summarize the topic '{topic}' in one sentence.")
        dt = _t.perf_counter() - t0
        print(
            f"  [model call: {dt:.2f}s · {result.metrics.prompt_tokens}→{result.metrics.completion_tokens} tokens]"
        )
        return {"summary": result.message}

    graph.add_node("summarize", ai_summarize)
    graph.add_edge(START, "summarize")
    graph.add_edge("summarize", END)

    result = await graph.execute({"topic": "large language models"})
    print(f"AI summary: {result.final_state.get('summary')}")
    print()


# =============================================================================
# Main
# =============================================================================


async def main():
    print("=" * 60)
    print("Notebook 17: Basic graph")
    print("=" * 60)
    print()

    await example_first_graph()
    await example_sequence()
    await example_state_flow()
    await example_parallel()
    await example_results()
    await example_streaming()
    await example_graph_with_llm()

    print("=" * 60)
    print("Next: Notebook 18 — Conditional routing")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
