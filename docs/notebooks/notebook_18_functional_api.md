# Functional API

Express a payment dispute triage workflow as decorated async functions
instead of a graph. When `StateGraph` is more than a straight-line
dispute pipeline needs, the functional API lets you write the same
workflow as ordinary Python: decorate the units of work with `@task`,
decorate the orchestrator with `@entrypoint`, and the SDK records timing
and applies retries and caching without a graph object. The scenario
here is payments operations: fetch the transaction, look up a risk
score, and assess the dispute.

The notebook walks through four parts: a two-task fetch-and-assess
pipeline, `@task(retry_attempts=3)` absorbing two timeouts from a flaky
payment processor feed, `@task(cache=True)` skipping a repeated
risk-score lookup, and a `@task` wrapping a full `Agent` call for
fraud-risk assessment.

What this notebook covers:

- `@task` — a unit of work; can declare `retry_attempts` and `cache`.
- `@entrypoint` — the top-level coroutine; records a `TaskResult` for
  each task that runs inside it.
- `pipeline.get_result()` returns an `EntrypointResult` with per-task
  metadata.
- Plain `async`/`await` control flow — no graph object, so no state
  reducers, conditional edges, checkpointing or interrupt/resume. Use
  [`StateGraph`](../concepts/multi-agent/graph.md) when you need any of
  those.

The diagram shows the fan-out form of the functional API, in which one
`@task` is called once per document and the calls are collected with
`asyncio.gather`, whereas this pipeline awaits its tasks one after
another. In both cases awaiting the `@entrypoint` runs its body as
ordinary asyncio code, each task that runs inside it appends a
`TaskResult` to the run record, and `get_result()` returns the record of
the last run:

{{ locus_diagram functional }}

## Key lines

Two `@task` functions, `fetch_transaction` and `assess`, run in sequence
inside the `@entrypoint` pipeline; once the await returns,
`pipeline.get_result()` returns the `EntrypointResult` with one entry per
task and its duration:

{{ locus_source examples/notebook_18_functional_api.py::example_basic }}

`@task(retry_attempts=3)` re-runs `query_processor_feed` after each
failed attempt (here a `ConnectionError`), pausing briefly before every
retry; the feed times out twice before succeeding, and the entrypoint
returns the result of the third attempt. Because `retry_attempts` counts
attempts in total, the default of 1 means no retry:

{{ locus_source examples/notebook_18_functional_api.py::example_retry }}

`@task(cache=True)` reuses the result of `lookup_risk_score` for a
repeated transaction id, so three awaits run the lookup only twice; a
cache hit returns before any `TaskResult` is recorded, so the run record
lists two tasks:

{{ locus_source examples/notebook_18_functional_api.py::example_cache }}

## Run it

```bash
python examples/notebook_18_functional_api.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

Only the fourth part calls the model. To run offline regardless of what
`~/.oci/config` holds:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_18_functional_api.py
```

## See also

- [Functional API](../concepts/multi-agent/functional.md) — the fan-out
  form in the diagram above, composed with `asyncio.gather`, and when to
  choose a graph instead.
- [StateGraph: per-node policies](../concepts/multi-agent/graph.md#per-node-policies)
  — `RetryPolicy` and `CachePolicy` declared as data on a graph node.
- [Notebook 17 — Resilient graphs](notebook_17_graph_advanced.md) — the
  same retries and caching attached to individual `StateGraph` nodes.
- [Notebook 20 — Scatter-gather with Send](notebook_20_map_reduce_support_triage.md)
  — the fan-out shape of the diagram written as a graph.

## Source

??? note "Full source — `examples/notebook_18_functional_api.py`"

    ````python
    --8<-- "examples/notebook_18_functional_api.py"
    ````
