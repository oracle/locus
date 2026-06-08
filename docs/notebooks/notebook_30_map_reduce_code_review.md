# Map-Reduce Code Review

Three source files, three reviewer roles (security, performance,
style) = nine reviewer agents running in parallel, then one
synthesizer collapses everything into a single Markdown report.

This notebook covers:

- `Send(node, payload, metadata)` — first-class graph primitive. The
  splitter returns a list of Sends; the executor fans them out
  concurrently. No queues, no manual `asyncio.gather`.
- Each reviewer is a distinct `Agent` with a role-specific system
  prompt. The graph orchestrates them, not a hand-rolled loop.
- The synthesizer reads each Send's output back from merged state and
  renders the final Markdown report.
- The whole pipeline is one `StateGraph.execute` call — streaming,
  cancellation, checkpointing, and GSAR judgment all attach for free.

```text
Diff splitter ──> N reviewers (parallel via Send) ──> Synthesizer
```

## Prerequisites

- Notebook 17 (basic graph).
- Notebook 25 (Swarm) for the dynamic-claim counterpoint.

## Run

```bash
python examples/notebook_30_map_reduce_code_review.py
```

The default provider is the bundled mock model. Set
`TULIP_MODEL_PROVIDER` (openai / anthropic) and credentials to use a
live model. Set
`TULIP_MODEL_PROVIDER=mock` for offline runs.

## Source

```python
--8<-- "examples/notebook_30_map_reduce_code_review.py"
```
