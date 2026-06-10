# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 22: Resilient sandbox-detonation workflows.

Sandbox infrastructure is flaky by nature — detonation nodes get busy,
reputation feeds rate-limit, reports take time. The graph executor lets
you attach policies to individual nodes: the sandbox submission retries
with backoff without touching the rest of the workflow, and a repeated
hash-reputation lookup gets cached without changing how it's called.
The visualisation helpers and streaming hooks give you the operational
story to go with it. Scenario: malware detonation using EICAR as the
benign test sample (MITRE ATT&CK T1204, User Execution).

- RetryPolicy — exponential backoff with optional jitter, per node.
- CachePolicy — TTL-based result caching, per node, keyed on inputs.
- draw_mermaid / draw_ascii — print the workflow as a diagram.
- graph.stream(...) + emit_custom — push detonation progress from inside a node.

Run it:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_22_graph_advanced.py

The default provider is the bundled mock model; set TULIP_MODEL_PROVIDER for a live provider.
Set TULIP_MODEL_PROVIDER=mock for offline runs. Pick a live provider with
TULIP_MODEL_ID=openai.gpt-4.1 (or meta.llama-3.3-70b-instruct, etc.).
"""

import asyncio
import time

from config import get_model

from tulip.agent import Agent
from tulip.multiagent.graph import (
    END,
    START,
    CachePolicy,
    GraphConfig,
    RetryPolicy,
    StateGraph,
)
from tulip.multiagent.visualize import draw_ascii, draw_mermaid


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
# Part 1: RetryPolicy
# =============================================================================


async def example_retry():
    """A busy sandbox fails twice, accepts the sample on the third attempt."""
    print("=== Part 1: RetryPolicy ===\n")
    print(
        f"AI rationale: {_llm_call('In one sentence, why is exponential backoff with jitter the right retry default?')}"
    )

    attempt = 0

    async def submit_sample(inputs):
        nonlocal attempt
        attempt += 1
        if attempt < 3:
            raise ConnectionError(f"Attempt {attempt}: sandbox node busy")
        # Mock detonation verdict — EICAR is the standard harmless test file.
        return {"verdict": "malicious (EICAR test signature)"}

    graph = StateGraph(config=GraphConfig(parallel=False))
    graph.add_node(
        "submit",
        submit_sample,
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.1, jitter=False),
    )
    graph.add_edge(START, "submit")
    graph.add_edge("submit", END)

    result = await graph.execute({})
    print(f"Success: {result.success}")
    print(f"Attempts needed: {attempt}")
    print(f"Verdict: {result.final_state.get('verdict')}")


# =============================================================================
# Part 2: CachePolicy
# =============================================================================


async def example_cache():
    """Identical hashes to the same node return the cached verdict for ttl_seconds."""
    print("\n=== Part 2: CachePolicy ===\n")
    print(
        f"AI rationale: {_llm_call('In one sentence, when does CachePolicy on a node beat memoising the function yourself?')}"
    )

    call_count = 0

    async def reputation_lookup(inputs):
        nonlocal call_count
        call_count += 1
        return {"reputation": f"lookup_{call_count}"}

    graph = StateGraph(config=GraphConfig(parallel=False))
    graph.add_node(
        "lookup",
        reputation_lookup,
        cache_policy=CachePolicy(ttl_seconds=60),
    )
    graph.add_edge(START, "lookup")
    graph.add_edge("lookup", END)

    r1 = await graph.execute({"sha256": "aa11bb22cc33dd44"})
    r2 = await graph.execute({"sha256": "aa11bb22cc33dd44"})

    print(f"Call count: {call_count}")  # 1 — second lookup was a cache hit
    print(
        f"Both same result: {r1.final_state.get('reputation') == r2.final_state.get('reputation')}"
    )


# =============================================================================
# Part 3: Diagrams
# =============================================================================


async def example_visualization():
    """draw_mermaid and draw_ascii print the detonation workflow as a diagram."""
    print("\n=== Part 3: Diagrams ===\n")
    print(
        f"AI rationale: {_llm_call('In one sentence, why are Mermaid diagrams useful when reviewing a Tulip StateGraph?')}"
    )

    graph = StateGraph(config=GraphConfig(parallel=False))

    async def submit(i):
        return {"submitted": True}

    async def detonate(i):
        return {"detonated": True}

    async def report(i):
        return {"done": True}

    graph.add_node("submit", submit)
    graph.add_node("detonate", detonate)
    graph.add_node("report", report)
    graph.add_edge(START, "submit")
    graph.add_edge("submit", "detonate")
    graph.add_conditional_edges(
        "detonate",
        lambda s: "report" if s.get("submitted") else "__END__",
        {
            "report": "report",
            "__END__": "__END__",
        },
    )
    graph.add_edge("report", END)

    print("Mermaid (paste into https://mermaid.live):")
    print(draw_mermaid(graph))
    print("\nASCII:")
    print(draw_ascii(graph))


async def example_realtime_streaming():
    """Stream node updates while also pushing detonation progress events."""
    print("\n=== Part 4: Live streaming with emit_custom ===\n")
    print(
        f"AI rationale: {_llm_call('In one sentence, why is streaming progress events better than polling for sandbox status?')}"
    )
    from tulip.multiagent import StreamMode, emit_custom

    graph = StateGraph(config=GraphConfig(parallel=False))

    async def detonate(inputs):
        for i in range(3):
            await emit_custom({"stage": i + 1, "of": 3}, node_id="detonate")
            await asyncio.sleep(0.05)
        return {"done": True}

    graph.add_node("detonate", detonate)
    graph.add_edge(START, "detonate")
    graph.add_edge("detonate", END)

    seen_custom = 0
    seen_updates = 0
    async for event in graph.stream({}, mode=StreamMode.UPDATES):
        if event.mode == StreamMode.CUSTOM:
            seen_custom += 1
            print(f"  [CUSTOM]  {event.data}")
        else:
            seen_updates += 1
            print(f"  [UPDATE]  {event.node_id}: {event.data}")
    print(f"\nDelivered {seen_custom} custom events + {seen_updates} updates.")


async def example_retry_with_llm() -> None:
    """RetryPolicy applies to any node — including ones that call an LLM."""
    print("\n=== Part 5: RetryPolicy around a real LLM call ===\n")

    async def summarize_report(inputs):
        import time as _t

        agent = Agent(
            model=get_model(max_tokens=60),
            system_prompt="Answer in one sentence for a SOC analyst.",
        )
        t0 = _t.perf_counter()
        result = agent.run_sync(inputs["question"])
        dt = _t.perf_counter() - t0
        print(
            f"  [model call: {dt:.2f}s · {result.metrics.prompt_tokens}→{result.metrics.completion_tokens} tokens]"
        )
        return {"summary": result.message.strip()}

    graph = StateGraph(config=GraphConfig(parallel=False))
    graph.add_node(
        "summarize",
        summarize_report,
        retry_policy=RetryPolicy(max_attempts=2, initial_interval=0.2, jitter=False),
    )
    graph.add_edge(START, "summarize")
    graph.add_edge("summarize", END)

    result = await graph.execute(
        {
            "question": (
                "Summarize this sandbox report: the sample copied itself to the "
                "startup folder and beaconed to evil.example."
            )
        }
    )
    print(f"Summary: {result.final_state.get('summary')}")


if __name__ == "__main__":
    asyncio.run(example_retry())
    asyncio.run(example_cache())
    asyncio.run(example_visualization())
    asyncio.run(example_realtime_streaming())
    asyncio.run(example_retry_with_llm())
