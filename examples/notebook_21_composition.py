# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Chain agents, run them in parallel, or loop one until it's satisfied.

When the work decomposes cleanly into agent-shaped pieces, you don't
need a full StateGraph. The three pipeline classes here are batteries-
included composition primitives that take a list of Agent instances
and orchestrate them for you.

- SequentialPipeline — each agent's output becomes the next agent's input.
- ParallelPipeline — run agents concurrently, then merge their results.
- LoopAgent — run one agent repeatedly until a stop condition fires.
- Helpers sequential() / parallel() / loop() exist for one-liners.

Run it:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_27_composition.py

The default provider is the bundled mock model; set TULIP_MODEL_PROVIDER for a live provider.
Set TULIP_MODEL_PROVIDER=mock for offline runs. Pick a live provider with
TULIP_MODEL_ID=openai.gpt-4.1 (or meta.llama-3.3-70b-instruct, etc.).
"""

import asyncio

from config import get_model

from tulip.agent import (
    Agent,
    AgentConfig,
    LoopAgent,
    ParallelPipeline,
    SequentialPipeline,
)


# =============================================================================
# Part 1: Sequential — researcher then writer
# =============================================================================


async def example_sequential():
    """Researcher gathers facts; writer turns them into prose."""
    print("=== Part 1: Sequential — researcher then writer ===\n")

    model = get_model()

    researcher = Agent(
        config=AgentConfig(
            system_prompt="You are a researcher. Provide 3 key facts about the topic.",
            max_iterations=3,
            model=model,
        )
    )
    writer = Agent(
        config=AgentConfig(
            system_prompt="You are a writer. Take the research and write a short paragraph.",
            max_iterations=3,
            model=model,
        )
    )

    pipeline = SequentialPipeline(agents=[researcher, writer])
    result = await pipeline.run("Benefits of regular exercise")

    print(f"Stage 1 (Researcher): {result.outputs[0][:100]}...")
    print(f"Stage 2 (Writer): {result.outputs[1][:100]}...")
    print(f"Duration: {result.duration_ms:.0f}ms")


# =============================================================================
# Part 2: Parallel — pros vs cons in one call
# =============================================================================


async def example_parallel():
    """Two agents form independent perspectives; the pipeline merges them."""
    print("\n=== Part 2: Parallel — pros vs cons in one call ===\n")

    model = get_model()

    pros = Agent(
        config=AgentConfig(
            system_prompt="List 2 pros of the topic. Be concise.",
            max_iterations=3,
            model=model,
        )
    )
    cons = Agent(
        config=AgentConfig(
            system_prompt="List 2 cons of the topic. Be concise.",
            max_iterations=3,
            model=model,
        )
    )

    pipeline = ParallelPipeline(agents=[pros, cons])
    result = await pipeline.run("Remote work for engineers")

    print(f"Pros: {result.outputs[0][:100]}...")
    print(f"Cons: {result.outputs[1][:100]}...")
    print(f"Merged: {result.final_output[:150]}...")


# =============================================================================
# Part 3: Loop — iterate until APPROVED or max_loops
# =============================================================================


async def example_loop():
    """LoopAgent re-runs the same agent, feeding back the previous output."""
    print("\n=== Part 3: Loop — iterate until APPROVED or max_loops ===\n")

    model = get_model()

    improver = Agent(
        config=AgentConfig(
            system_prompt=(
                "You improve text quality. When the text is good enough, "
                "include the word APPROVED at the end."
            ),
            max_iterations=3,
            model=model,
        )
    )

    loop = LoopAgent(
        agent=improver,
        condition=lambda output: "APPROVED" in output.upper(),
        max_loops=3,
        loop_prompt="Improve this text. Say APPROVED when done:\n{previous_output}",
    )

    result = await loop.run("The quick brown fox jumps over the lazy dog.")
    print(f"Iterations: {len(result.outputs)}")
    print(f"Final: {result.final_output[:100]}...")


if __name__ == "__main__":
    asyncio.run(example_sequential())
    asyncio.run(example_parallel())
    asyncio.run(example_loop())
