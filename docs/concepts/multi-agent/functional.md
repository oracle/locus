# Functional API

The functional API is the Locus's "agent as a task" shape — `@task` and `@entrypoint` decorators
that bring agent runs into the regular asyncio universe.

{{ locus_diagram functional }}

## What it is

Two decorators:

| Decorator | What it does |
|---|---|
| **`@task`** | Wraps a coroutine that calls an `Agent`. Calling it returns a plain coroutine you can `await`, `gather`, retry, time-out — anything asyncio gives you. |
| **`@entrypoint`** | Marks the top-level coroutine of a workflow. You `await` it like any coroutine and get the body's return value unchanged; it also keeps an `EntrypointResult` for the last run (the value, a `TaskResult` for each task that ran, the duration and any error), which `.get_result()` returns. |

These are **not a new orchestration runtime**. They're a thin shim
that lets agents participate in plain asyncio. The point is to
compose with `asyncio.gather`, `asyncio.wait_for`, `asyncio.Queue`,
or anything else you already use.

## When to use it

- ✅ You think in **`async def` and `asyncio.gather`** already.
- ✅ The flow is **map/reduce** over agents (summarize N documents in
  parallel).
- ✅ You want to **mix agents with non-agent code** — DB writes,
  HTTP calls, file I/O — in the same coroutine.
- ✅ Tooling like **`tenacity` retries**, **`asyncio.timeout`**, or a
  **`asyncio.Queue` scheduler** already gives you the orchestration
  you need.

## When NOT to use it

- ❌ You want **inspectable, named control-flow** with cycles or
  conditional branches → use [StateGraph](graph.md).
- ❌ You need **per-node retry / cache policies as data** →
  [StateGraph](graph.md).
- ❌ Different agents should **decide who runs** → use
  [Orchestrator](orchestrator.md).

## Code

Inside a task, call the agent with `await agent.arun(...)`. `run_sync`
blocks the event loop until its run finishes, so tasks that use it run
one after another even under `asyncio.gather`; with `arun` their model
calls overlap.

```python
import asyncio
from locus.multiagent.functional import task, entrypoint

@task
async def summarize_doc(doc: dict) -> str:
    """Run the review agent against one document."""
    return (await review_agent.arun(f"Summarize {doc['title']}.")).message

@entrypoint
async def summarize_all(docs: list[dict]) -> list[str]:
    """Summarize every document in parallel; gather the results."""
    return await asyncio.gather(*[summarize_doc(d) for d in docs])

summaries = await summarize_all(documents)   # or: asyncio.run(summarize_all(documents))
```

## Map/reduce with retries and timeouts

Because tasks are plain coroutines, you compose with whatever the
asyncio ecosystem provides:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@task
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5))
async def summarize_doc(doc: dict) -> str:
    return (await review_agent.arun(f"Summarize {doc['title']}.")).message

@entrypoint
async def summarize_all_with_deadline(docs: list[dict]) -> list[str]:
    async with asyncio.timeout(60):                # 60s wall-clock cap
        return await asyncio.gather(*[summarize_doc(d) for d in docs])
```

## Tasks calling tasks

Tasks compose. An `@entrypoint` workflow can call other `@task`s
including parallel batches inside sequential phases:

```python
@task
async def shortlist_docs(queue: list[dict]) -> list[dict]:
    titles = "\n".join(d["title"] for d in queue)
    reply = await review_agent.arun(f"Pick the top 5 of these titles, one per line:\n{titles}")
    picked = set(reply.message.splitlines())
    return [d for d in queue if d["title"] in picked]

@task
async def summarize(doc: dict) -> str:
    return (await review_agent.arun(f"Summarize {doc['title']}.")).message

@entrypoint
async def end_to_end(queue: list[dict]) -> str:
    shortlisted = await shortlist_docs(queue)               # phase 1
    summaries = await asyncio.gather(*[summarize(d) for d in shortlisted])  # phase 2 (parallel)
    final = (await editor_agent.arun(f"Draft the review digest from: {summaries}")).message  # phase 3
    return final
```

## Notebooks

- [`notebook_18_functional_api.py`](https://github.com/oracle/locus/blob/main/examples/notebook_18_functional_api.py)
  — `@task` and `@entrypoint` end-to-end.
- [`notebook_20_map_reduce_support_triage.py`](https://github.com/oracle/locus/blob/main/examples/notebook_20_map_reduce_support_triage.py)
  — same map/reduce shape, written as a graph with `Send` instead.
  Useful as the "graph version" comparison when you're choosing
  between functional and StateGraph for a fan-out workload.

## Source

[`multiagent/functional.py`](https://github.com/oracle/locus/blob/main/src/locus/multiagent/functional.py)
— `task`, `entrypoint`, `TaskResult`, `EntrypointResult`.

## See also

- [Multi-agent overview](../multi-agent.md) — pick a shape.
- [StateGraph](graph.md) — for the same fan-out *as data* with
  inspectable retry/cache policies.
- [Composition](composition.md) — for the same shapes via
  `ParallelPipeline`.
