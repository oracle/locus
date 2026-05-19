# Functional API

Express a workflow as decorated async functions instead of a graph. If
`StateGraph` feels like overkill for a straight-line pipeline, the
functional API lets you write the same workflow as ordinary Python:
decorate the units of work with `@task`, decorate the orchestrator
with `@entrypoint`, and Locus tracks timing, retries, and caching
behind the scenes.

What you'll see:

- `@task` — a unit of work; can declare `retry_attempts` and `cache`.
- `@entrypoint` — the top-level coroutine; tracks every task it awaits.
- `pipeline.get_result()` returns an `EntrypointResult` with per-task
  metadata.
- Same execution semantics as `StateGraph`, written imperatively.

Runs on the same OCI GenAI default as the rest of the notebooks:

```bash
LOCUS_MODEL_ID=openai.gpt-4.1 python examples/notebook_29_functional_api.py
# or, fully offline:
LOCUS_MODEL_PROVIDER=mock python examples/notebook_29_functional_api.py
```

## Source

```python
--8<-- "examples/notebook_29_functional_api.py"
```
