# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/
"""
Notebook 18: Payment dispute triage with the functional API.

If `StateGraph` feels like overkill for a straight-line dispute
workflow, the functional API lets you write the same payment dispute
triage as ordinary Python: decorate the units of work with
`@task`, decorate the orchestrator with `@entrypoint`, and Locus
tracks timing, retries, and caching for you behind the scenes. Scenario:
payments operations — fetch transaction, look up risk score, assess dispute.

Key ideas:
- @task — a unit of work; can declare retry_attempts and cache.
- @entrypoint — the top-level coroutine; tracks every task it awaits.
- pipeline.get_result() returns an EntrypointResult with per-task metadata.
- Same execution semantics as StateGraph, written imperatively.

Run it:
    LOCUS_MODEL_PROVIDER=mock python examples/notebook_18_functional_api.py

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
from locus.multiagent.functional import entrypoint, task


# =============================================================================
# Part 1: A two-task pipeline
# =============================================================================


async def example_basic():
    """Two @task functions wired together inside an @entrypoint."""
    print("=== Part 1: A two-task pipeline ===\n")

    @task
    async def fetch_transaction(txn_id: str) -> dict:
        # Mock transaction fetch — invented data, clearly fake.
        return {"transaction": f"transaction record for {txn_id}", "status": 200}

    @task
    async def assess(transaction: dict) -> str:
        return f"dispute assessed: {transaction['transaction']}"

    @entrypoint
    async def pipeline(txn_id: str) -> str:
        transaction = await fetch_transaction(txn_id)
        result = await assess(transaction)
        return result

    result = await pipeline("TXN-2024-99999")
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
    """retry_attempts on the decorator handles a flaky payment processor feed."""
    print("\n=== Part 2: @task(retry_attempts=3) ===\n")

    attempt = 0

    @task(retry_attempts=3)
    async def query_processor_feed(txn_id: str) -> str:
        nonlocal attempt
        attempt += 1
        if attempt < 3:
            raise ConnectionError("payment processor feed timeout")
        return f"processor entry for: {txn_id}"

    @entrypoint
    async def retry_pipeline() -> str:
        return await query_processor_feed("TXN-2024-99999")

    result = await retry_pipeline()
    print(f"Result: {result}")
    print(f"Attempts needed: {attempt}")


# =============================================================================
# Part 3: @task(cache=True)
# =============================================================================


async def example_cache():
    """Same transaction id returns the cached score without re-running the lookup."""
    print("\n=== Part 3: @task(cache=True) ===\n")

    call_count = 0

    @task(cache=True)
    async def lookup_risk_score(txn_id: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"score_lookup_{call_count}"

    @entrypoint
    async def cache_pipeline() -> tuple:
        r1 = await lookup_risk_score("TXN-2024-99999")
        r2 = await lookup_risk_score("TXN-2024-99999")  # cache hit
        r3 = await lookup_risk_score("TXN-2024-88888")
        return (r1, r2, r3)

    r1, r2, r3 = await cache_pipeline()
    print(f"r1={r1}, r2={r2}, r3={r3}")
    print(f"Actual calls: {call_count}")  # 2, not 3


# =============================================================================
# Part 4: @task wrapping an LLM call
# =============================================================================


async def example_with_llm():
    """A @task can wrap a full Agent call just like a node would."""
    print("\n=== Part 4: @task wrapping an LLM call ===\n")

    # Built once, outside the task: the task body is what the functional
    # API tracks, so keep the model client out of it.
    analyst = Agent(
        model=get_model(max_tokens=80),
        system_prompt="Answer in one factual sentence for a payments fraud analyst.",
    )

    @task
    async def build_question(txn_id: str) -> str:
        return f"Assess the likely fraud risk of {txn_id} for a typical card-not-present charge."

    @task
    async def assess_risk(prompt: str) -> str:
        t0 = time.perf_counter()
        result = await analyst.arun(prompt)
        dt = time.perf_counter() - t0
        print(
            f"  [model call: {dt:.2f}s · {result.metrics.prompt_tokens}→{result.metrics.completion_tokens} tokens]"
        )
        return result.message.strip()

    @entrypoint
    async def pipeline(txn_id: str) -> str:
        question = await build_question(txn_id)
        return await assess_risk(question)

    answer = await pipeline("TXN-2024-99999")
    print(f"Assessment: {answer}")


async def main():
    """Run all notebook parts."""
    print("=" * 60)
    print("Notebook 18: Payment dispute triage with the functional API")
    print("=" * 60)
    print()

    print_config()
    print()

    await example_basic()
    await example_retry()
    await example_cache()
    await example_with_llm()

    print("\n" + "=" * 60)
    print("Next: Notebook 19 — Composing data-privacy agents into pipelines")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
