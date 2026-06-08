# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL
"""
Express a workflow as decorated async functions instead of a graph.

If `StateGraph` feels like overkill for a straight-line pipeline, the
functional API lets you write the same workflow as ordinary Python:
decorate the units of work with `@task`, decorate the orchestrator
with `@entrypoint`, and Tulip tracks timing, retries, and caching for
you behind the scenes.

- @task — a unit of work; can declare retry_attempts and cache.
- @entrypoint — the top-level coroutine; tracks every task it awaits.
- pipeline.get_result() returns an EntrypointResult with per-task metadata.
- Same execution semantics as StateGraph, written imperatively.

Run it:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_29_functional_api.py

The default provider is the bundled mock model; set TULIP_MODEL_PROVIDER for a live provider.
Set TULIP_MODEL_PROVIDER=mock for offline runs. Pick a live provider with
TULIP_MODEL_ID=openai.gpt-4.1 (or meta.llama-3.3-70b-instruct, etc.).
"""

import asyncio
import time

from config import get_model

from tulip.agent import Agent
from tulip.multiagent.functional import entrypoint, task


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
# Part 1: A two-task pipeline
# =============================================================================


async def example_basic():
    """Two @task functions wired together inside an @entrypoint."""
    print("=== Part 1: A two-task pipeline ===\n")
    print(
        f"AI rationale: {_llm_call('In one sentence, when is the Tulip functional API a better choice than StateGraph?')}"
    )

    @task
    async def fetch(url: str) -> dict:
        return {"data": f"fetched from {url}", "status": 200}

    @task
    async def process(data: dict) -> str:
        return f"processed: {data['data']}"

    @entrypoint
    async def pipeline(url: str) -> str:
        data = await fetch(url)
        result = await process(data)
        return result

    result = await pipeline("https://api.example.com/data")
    print(f"Result: {result}")

    ep = pipeline.get_result()
    print(f"Tasks executed: {len(ep.tasks)}")
    for t in ep.tasks:
        print(f"  {t.task_name}: {t.duration_ms:.1f}ms")
    print(f"Total: {ep.duration_ms:.1f}ms")


# =============================================================================
# Part 2: @task(retry_attempts=3)
# =============================================================================


async def example_retry():
    """retry_attempts on the decorator handles transient failures."""
    print("\n=== Part 2: @task(retry_attempts=3) ===\n")
    print(
        f"AI rationale: {_llm_call('In one sentence, why does @task(retry_attempts=3) belong on the task and not in caller code?')}"
    )

    attempt = 0

    @task(retry_attempts=3)
    async def unreliable_api(query: str) -> str:
        nonlocal attempt
        attempt += 1
        if attempt < 3:
            raise ConnectionError("API timeout")
        return f"result for: {query}"

    @entrypoint
    async def retry_pipeline() -> str:
        return await unreliable_api("test")

    result = await retry_pipeline()
    print(f"Result: {result}")
    print(f"Attempts needed: {attempt}")


# =============================================================================
# Part 3: @task(cache=True)
# =============================================================================


async def example_cache():
    """Same arguments return the cached result without re-running the task."""
    print("\n=== Part 3: @task(cache=True) ===\n")
    print(
        f"AI rationale: {_llm_call('In one sentence, when should you turn @task(cache=True) ON for an LLM-heavy pipeline?')}"
    )

    call_count = 0

    @task(cache=True)
    async def expensive_compute(key: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"computed_{call_count}"

    @entrypoint
    async def cache_pipeline() -> tuple:
        r1 = await expensive_compute("same_key")
        r2 = await expensive_compute("same_key")  # cache hit
        r3 = await expensive_compute("diff_key")
        return (r1, r2, r3)

    r1, r2, r3 = await cache_pipeline()
    print(f"r1={r1}, r2={r2}, r3={r3}")
    print(f"Actual calls: {call_count}")  # 2, not 3


async def example_with_llm():
    """A @task can wrap a full Agent call just like a node would."""
    print("\n=== Part 4: @task wrapping an LLM call ===\n")

    @task
    async def fetch_topic(seed: str) -> str:
        return f"Tell me about {seed}."

    @task
    async def think(prompt: str) -> str:
        import time as _t

        agent = Agent(
            model=get_model(max_tokens=80),
            system_prompt="Answer in one factual sentence.",
        )
        t0 = _t.perf_counter()
        result = agent.run_sync(prompt)
        dt = _t.perf_counter() - t0
        print(
            f"  [model call: {dt:.2f}s · {result.metrics.prompt_tokens}→{result.metrics.completion_tokens} tokens]"
        )
        return result.message.strip()

    @entrypoint
    async def pipeline(seed: str) -> str:
        question = await fetch_topic(seed)
        return await think(question)

    answer = await pipeline("retrieval-augmented generation")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    asyncio.run(example_basic())
    asyncio.run(example_retry())
    asyncio.run(example_cache())
    asyncio.run(example_with_llm())
