# Graph streaming

`StateGraph.stream(...)` yields events as nodes complete — not buffered
until the graph finishes — so a UI can render progress in real time.

## Modes

```python
from locus.multiagent import StateGraph, StreamMode

async for event in graph.stream(inputs, mode=StreamMode.UPDATES):
    print(event.node_id, event.data)
```

| Mode | Yields per node | Plus terminal event |
|---|---|---|
| `StreamMode.VALUES` *(default)* | A snapshot of full state after the node completes | `StreamEvent(mode=VALUES, data=final_state)` |
| `StreamMode.UPDATES` | Just the node's own output dict | — |
| `StreamMode.NODES` | The full `NodeResult` with status / duration / error | — |
| `StreamMode.DEBUG` | `{"result": NodeResult, "state": dict}` | — |
| `StreamMode.CUSTOM` | Whatever `emit_custom(...)` pushes from inside a node body | — |

## Custom events from inside a node

Long-running nodes can push intermediate progress events with
`emit_custom`. Outside a `stream()` context the call is a silent no-op,
so the same node code runs unchanged under `execute()` too.

```python
from locus.multiagent import emit_custom

async def long_running_node(state: dict) -> dict:
    for i in range(10):
        await emit_custom({"progress": i / 10, "phase": "indexing"})
        await asyncio.sleep(0.1)
    return {"done": True}

graph.add_node("worker", long_running_node)

async for event in graph.stream(inputs, mode=StreamMode.UPDATES):
    if event.mode == StreamMode.CUSTOM:
        ui.set_progress(event.data["progress"])
    elif event.mode == StreamMode.UPDATES:
        ui.mark_node_complete(event.node_id)
```

`emit_custom` is exported from `locus.multiagent` and accepts an
optional `node_id=` kwarg if you want the event tagged with the
emitting node's identity.

## Real-time delivery

Events arrive as nodes complete, not at the end. A fast-then-slow graph
proves it:

```python
async def fast(state):
    return {"x": 1}                          # ~ms

async def slow(state):
    await asyncio.sleep(2)                   # 2 seconds
    return {"y": 2}

graph.add_node("a", fast)
graph.add_node("b", slow)
graph.add_edge(START, "a"); graph.add_edge("a", "b"); graph.add_edge("b", END)

start = time.perf_counter()
async for ev in graph.stream({}, mode=StreamMode.UPDATES):
    print(f"{time.perf_counter() - start:.2f}s  {ev.node_id}")
# 0.05s  a
# 2.05s  b
```

If `stream()` were buffering, both events would arrive at 2.05s. The
unit test
[`test_stategraph_streaming.py`](https://orahub.oci.oraclecorp.com/saas-observ-eng/locus/-/blob/main/tests/unit/test_stategraph_streaming.py)
guards this property — fails the build if the first event lands at
≥ end / 2.

## Error and cancellation

A node that raises has its `NodeResult.success` set to `False` with the
error message; the stream still yields an event for it (no consumer
deadlock). Breaking out of the iterator early cancels the background
driver task so no work continues in the background.

## Source

`src/locus/multiagent/graph.py:emit_custom`,
`StateGraph.stream`, `StreamMode`.
