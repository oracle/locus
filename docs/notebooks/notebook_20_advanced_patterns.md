# Advanced Patterns

Five primitives that turn a `StateGraph` into a general-purpose
runtime. Reach for these once basic graphs stop being enough: dynamic
routing from inside a node, fan-out to many workers, reusable
subgraphs, cross-conversation key/value storage, and combining them
in one workflow.

What you'll see:

- `Command(update=..., goto=...)` — write state and pick the next node
  in one return value.
- `goto()` and `end()` — short helpers for common `Command` shapes.
- `scatter("worker", items, key=...)` — fan a list of items out to
  copies of a worker node.
- `broadcast(nodes, payload)` — fan one payload out to several different
  nodes.
- Subgraph-as-node — call one `StateGraph` from inside another.
- `InMemoryStore` — durable key/value space that outlives a single run.

Runs on the same default (mock) as the rest of the notebooks:

```bash
TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_20_advanced_patterns.py
# or, fully offline:
TULIP_MODEL_PROVIDER=mock python examples/notebook_20_advanced_patterns.py
```

## Source

```python
--8<-- "examples/notebook_20_advanced_patterns.py"
```
