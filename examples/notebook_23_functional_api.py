# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 23: Payment dispute triage with the functional API.

If `StateGraph` feels like overkill for a straight-line dispute
workflow, the functional API lets you write the same payment dispute
triage as ordinary Python: decorate the units of work with
`@task`, decorate the orchestrator with `@entrypoint`, and Tulip
tracks timing, retries, and caching for you behind the scenes. Scenario:
payments operations — fetch transaction, look up risk score, assess dispute.

- @task — a unit of work; can declare retry_attempts and cache.
- @entrypoint — the top-level coroutine; tracks every task it awaits.
- pipeline.get_result() returns an EntrypointResult with per-task metadata.
- Same execution semantics as StateGraph, written imperatively.

Run it:
    TULIP_MODEL_PROVIDER=mock python examples/notebook_23_functional_api.py

The default provider is the bundled mock model; set TULIP_MODEL_PROVIDER for a live provider.
Set TULIP_MODEL_PROVIDER=mock for offline runs. Pick a live provider with
TULIP_MODEL_ID=openai.gpt-4.1 (or meta.llama-3.3-70b-instruct, etc.).
"""

import asyncio
import time

from config import get_model

from tulip.agent import Agent
from tulip.multiagent.functional import entrypoint, task


async def _llm_call(
    prompt: str, *, system: str = "Reply in one short sentence.", max_tokens: int = 80
) -> str:
    """Run a one-shot Agent and print a timing/token banner. Used by every part."""
    agent = Agent(model=get_model(max_tokens=max_tokens), system_prompt=system)
    t0 = time.perf_counter()
    res = await agent.arun(prompt)
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
    _ai_note = await _llm_call("In one sentence, when is the Tulip functional API a better choice than StateGraph?")
    print(f"AI rationale: {_ai_note}")

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
    _ai_note = await _llm_call("In one sentence, why does @task(retry_attempts=3) belong on the task and not in caller code?")
    print(f"AI rationale: {_ai_note}")

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
    _ai_note = await _llm_call("In one sentence, when should you turn @task(cache=True) ON for a transaction-enrichment pipeline?")
    print(f"AI rationale: {_ai_note}")

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


async def example_with_llm():
    """A @task can wrap a full Agent call just like a node would."""
    print("\n=== Part 4: @task wrapping an LLM call ===\n")

    @task
    async def build_question(txn_id: str) -> str:
        return f"Assess the likely fraud risk of {txn_id} for a typical card-not-present charge."

    @task
    async def assess_risk(prompt: str) -> str:
        import time as _t

        agent = Agent(
            model=get_model(max_tokens=80),
            system_prompt="Answer in one factual sentence for a payments fraud analyst.",
        )
        t0 = _t.perf_counter()
        result = await agent.arun(prompt)
        dt = _t.perf_counter() - t0
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
    await example_basic()
    await example_retry()
    await example_cache()
    await example_with_llm()


if __name__ == "__main__":
    asyncio.run(main())
