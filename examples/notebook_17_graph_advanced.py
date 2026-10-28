# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""
Notebook 17: Resilient cloud-provisioning workflows.

Cloud control planes are flaky by nature — provisioning APIs get
throttled, capacity pools fill up, instances take time to boot. The
graph executor lets you attach policies to individual nodes: the
instance-launch call retries with backoff without touching the rest of
the workflow, and a repeated machine-image lookup gets cached without
changing how it's called. The visualisation helpers and streaming hooks
give you the operational story to go with it. Scenario: provisioning a
small web tier on OCI (launch a VM, attach it, run a health check).

Key ideas:
- RetryPolicy — exponential backoff with optional jitter, per node.
- CachePolicy — TTL-based result caching, per node, keyed on inputs.
- draw_mermaid / draw_ascii — print the workflow as a diagram.
- graph.stream(...) + emit_custom — push provisioning progress from inside a node.

Run it:
    LOCUS_MODEL_PROVIDER=mock python examples/notebook_17_graph_advanced.py

The default provider is OCI GenAI when ``~/.oci/config`` holds a usable profile, otherwise the bundled mock model; set ``LOCUS_MODEL_PROVIDER`` (``oci``, ``openai``, ``anthropic``, ``mock``) to choose.
Set LOCUS_MODEL_PROVIDER=mock for offline runs. Pick a live provider with
LOCUS_MODEL_PROVIDER=openai LOCUS_MODEL_ID=gpt-4o
(or LOCUS_MODEL_PROVIDER=anthropic, or any OpenAI-compatible
provider: ollama, vllm, groq, together, openrouter, ...).
"""

import asyncio
import time

from config import get_model, print_config

from locus.agent import Agent
from locus.multiagent import StreamMode, emit_custom
from locus.multiagent.graph import (
    END,
    START,
    CachePolicy,
    GraphConfig,
    RetryPolicy,
    StateGraph,
)
from locus.multiagent.visualize import draw_ascii, draw_mermaid


# Every graph in this notebook is a straight line, so the parallel
# scheduler (the default) buys nothing and would interleave the attempt
# counters and stream output; run nodes one at a time so they read in order.
_SERIAL = GraphConfig(parallel=False)


# =============================================================================
# Part 1: RetryPolicy
# =============================================================================


async def example_retry():
    """A throttled control plane fails twice, launches the instance on the third attempt."""
    print("=== Part 1: RetryPolicy ===\n")

    attempt = 0

    async def launch_instance(inputs):
        nonlocal attempt
        attempt += 1
        if attempt < 3:
            raise ConnectionError(f"Attempt {attempt}: control plane throttled (429)")
        # Mock launch result — a freshly provisioned VM.
        return {"instance": "web-01 RUNNING (VM.Standard.E4.Flex, us-ashburn-1 AD-1)"}

    graph = StateGraph(config=_SERIAL)
    graph.add_node(
        "launch",
        launch_instance,
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.1, jitter=False),
    )
    graph.add_edge(START, "launch")
    graph.add_edge("launch", END)

    result = await graph.execute({})
    print(f"Success: {result.success}")
    print(f"Attempts needed: {attempt}")
    print(f"Instance: {result.final_state.get('instance')}")


# =============================================================================
# Part 2: CachePolicy
# =============================================================================


async def example_cache():
    """Identical image lookups to the same node return the cached result for ttl_seconds."""
    print("\n=== Part 2: CachePolicy ===\n")

    call_count = 0

    async def image_lookup(inputs):
        nonlocal call_count
        call_count += 1
        return {"image_ocid": f"ocid1.image.oc1.iad.resolved_{call_count}"}

    graph = StateGraph(config=_SERIAL)
    graph.add_node(
        "lookup",
        image_lookup,
        cache_policy=CachePolicy(ttl_seconds=60),
    )
    graph.add_edge(START, "lookup")
    graph.add_edge("lookup", END)

    r1 = await graph.execute({"image": "Oracle-Linux-9-2026.01"})
    r2 = await graph.execute({"image": "Oracle-Linux-9-2026.01"})

    print(f"Call count: {call_count}")  # 1 — second lookup was a cache hit
    same = r1.final_state.get("image_ocid") == r2.final_state.get("image_ocid")
    print(f"Both same result: {same}")


# =============================================================================
# Part 3: Diagrams
# =============================================================================


async def example_visualization():
    """draw_mermaid and draw_ascii print the provisioning workflow as a diagram."""
    print("\n=== Part 3: Diagrams ===\n")

    graph = StateGraph(config=_SERIAL)

    async def launch(i):
        return {"launched": True}

    async def attach(i):
        return {"attached": True}

    async def healthcheck(i):
        return {"done": True}

    graph.add_node("launch", launch)
    graph.add_node("attach", attach)
    graph.add_node("healthcheck", healthcheck)
    graph.add_edge(START, "launch")
    graph.add_edge("launch", "attach")
    graph.add_conditional_edges(
        "attach",
        lambda s: "healthcheck" if s.get("launched") else END,
        {
            "healthcheck": "healthcheck",
            END: END,
        },
    )
    graph.add_edge("healthcheck", END)

    print("Mermaid (paste into https://mermaid.live):")
    print(draw_mermaid(graph))
    print("\nASCII:")
    print(draw_ascii(graph))


# =============================================================================
# Part 4: Live streaming with emit_custom
# =============================================================================


async def example_realtime_streaming():
    """Stream node updates while also pushing provisioning progress events."""
    print("\n=== Part 4: Live streaming with emit_custom ===\n")

    graph = StateGraph(config=_SERIAL)

    async def provision(inputs):
        for i in range(3):
            await emit_custom({"stage": i + 1, "of": 3}, node_id="provision")
            await asyncio.sleep(0.05)
        return {"done": True}

    graph.add_node("provision", provision)
    graph.add_edge(START, "provision")
    graph.add_edge("provision", END)

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


# =============================================================================
# Part 5: RetryPolicy around a real LLM call
# =============================================================================


async def example_retry_with_llm() -> None:
    """RetryPolicy applies to any node — including ones that call an LLM."""
    print("\n=== Part 5: RetryPolicy around a real LLM call ===\n")

    # One agent for the whole part; the node closes over it so a retry
    # reuses the same model client rather than rebuilding it.
    summarizer = Agent(
        model=get_model(max_tokens=60),
        system_prompt="Answer in one sentence for an on-call cloud engineer.",
    )
    attempt = 0

    async def summarize_event(inputs):
        nonlocal attempt
        attempt += 1
        # The first attempt simulates a transient gateway failure in front
        # of the model so the RetryPolicy on this node has something to do.
        if attempt == 1:
            raise ConnectionError("Attempt 1: model gateway returned 503")
        t0 = time.perf_counter()
        result = await summarizer.arun(inputs["question"])
        dt = time.perf_counter() - t0
        print(
            f"  [model call: {dt:.2f}s · {result.metrics.prompt_tokens}→{result.metrics.completion_tokens} tokens]"
        )
        return {"summary": result.message.strip()}

    graph = StateGraph(config=_SERIAL)
    graph.add_node(
        "summarize",
        summarize_event,
        retry_policy=RetryPolicy(max_attempts=2, initial_interval=0.2, jitter=False),
    )
    graph.add_edge(START, "summarize")
    graph.add_edge("summarize", END)

    result = await graph.execute(
        {
            "question": (
                "Summarize this autoscaling event: the instance pool scaled from 2 to 6 "
                "instances after CPU stayed above 80% for ten minutes."
            )
        }
    )
    print(f"Attempts needed: {attempt} (first failed, RetryPolicy re-ran the node)")
    print(f"Summary: {result.final_state.get('summary')}")


async def main():
    """Run all notebook parts."""
    print("=" * 60)
    print("Notebook 17: Resilient cloud-provisioning workflows")
    print("=" * 60)
    print()

    print_config()
    print()

    await example_retry()
    await example_cache()
    await example_visualization()
    await example_realtime_streaming()
    await example_retry_with_llm()

    print("\n" + "=" * 60)
    print("Next: Notebook 18 — Payment dispute triage with the functional API")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
