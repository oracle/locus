# Map-Reduce Support Triage

Three inbound tickets and three triage lenses (sentiment, routing,
resolution) give nine analyst agents running in parallel, then one
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
  cancellation, checkpointing, and GSAR judgment attach to it without
  extra code.

{{ locus_diagram scatter-gather }}

## Prerequisites

- Basic graph.
- Swarm, for the dynamic-claim counterpoint.

## Run

```bash
python examples/notebook_20_map_reduce_support_triage.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

## Source

````python
--8<-- "examples/notebook_20_map_reduce_support_triage.py"
````
