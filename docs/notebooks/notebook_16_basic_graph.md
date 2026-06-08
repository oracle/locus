# Basic Graph

Build a workflow as a graph of nodes that pass state to each other. A
`StateGraph` is a directed graph where each node is an async function
that takes the current state in and returns updates to merge back. Use
it when one Agent isn't enough — multi-step pipelines, branching
logic, fan-out / fan-in, human approval gates.

What you'll see:

- Nodes and edges; `START` and `END` sentinels.
- Sequential, parallel, and conditional flow on the same primitives.
- `GraphResult` — final state plus per-node status, timing, and order.
- Streaming node updates and pushing custom progress events from inside a node.
- An Agent embedded inside a graph node.

Runs on the same default (mock) as the rest of the notebooks:

```bash
TULIP_MODEL_ID=openai.gpt-4.1 python examples/notebook_16_basic_graph.py
# or, fully offline:
TULIP_MODEL_PROVIDER=mock python examples/notebook_16_basic_graph.py
```

## Source

```python
--8<-- "examples/notebook_16_basic_graph.py"
```
