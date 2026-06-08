# Advanced Graphs

Per-node retry and caching, plus graph diagrams and live streaming.
The executor lets you attach policies to individual nodes — so a flaky
API call retries with backoff without touching the rest of the graph,
and an expensive lookup gets cached without changing how it's called.
The visualisation helpers and streaming hooks give you the operational
story to go with it.

What you'll see:

- `RetryPolicy` — exponential backoff with optional jitter, per node.
- `CachePolicy` — TTL-based result caching, per node, keyed on inputs.
- `draw_mermaid` / `draw_ascii` — print the graph as a diagram.
- `graph.stream(...)` + `emit_custom` — push progress events from
  inside a node.

Runs on the same default (mock) as the rest of the notebooks:

```bash
TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_22_graph_advanced.py
# or, fully offline:
TULIP_MODEL_PROVIDER=mock python examples/notebook_22_graph_advanced.py
```

## Source

```python
--8<-- "examples/notebook_22_graph_advanced.py"
```
