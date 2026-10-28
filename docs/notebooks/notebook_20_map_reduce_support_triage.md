# Map-Reduce Support Triage

Three inbound tickets, three triage lenses (sentiment, routing,
resolution) = nine analyst agents running in parallel, then one
synthesizer collapses everything into a single Markdown triage summary.

This notebook covers:

- `Send(node, payload, metadata)` — first-class graph primitive. The
  splitter returns a list of Sends; the executor fans them out
  concurrently. No queues, no manual `asyncio.gather`.
- Each analyst is a distinct `Agent` with a lens-specific system
  prompt. The graph orchestrates them, not a hand-rolled loop.
- The synthesizer reads each Send's output back from merged state and
  renders the final Markdown summary.
- The whole pipeline is one `StateGraph.execute` call — streaming,
  cancellation, checkpointing, and GSAR judgment all attach for free.

```text
Ticket splitter ──> N analysts (parallel via Send) ──> Synthesizer
```

## Prerequisites

- Basic graph.
- Swarm, for the dynamic-claim counterpoint.

## Run

```bash
python examples/notebook_20_map_reduce_support_triage.py
```

The default provider is the bundled mock model. Set
`LOCUS_MODEL_PROVIDER` (openai / anthropic) and credentials to use a
live model. Set
`LOCUS_MODEL_PROVIDER=mock` for offline runs.

## Source

````python
--8<-- "examples/notebook_20_map_reduce_support_triage.py"
````
