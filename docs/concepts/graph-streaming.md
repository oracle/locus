# Graph streaming

`StateGraph.stream(...)` yields events as nodes complete — not buffered
until the graph finishes. For an order-fulfillment graph that means a
live replay: you watch *which specialist decided what, when* —
validation checking the order, pricing computing the totals, fulfillment
dispatching the shipment — instead of seeing only the result at the end.

## Modes

```python
from locus.multiagent import StateGraph, StreamMode

# graph: validate -> price -> fulfill
async for event in graph.stream(order, mode=StreamMode.UPDATES):
    print(event.node_id, event.data)   # who decided, what they wrote to state
```

| Mode | Yields per node | Plus terminal event |
|---|---|---|
| `StreamMode.VALUES` *(default)* | A snapshot of full state after the node completes | `StreamEvent(mode=VALUES, data=final_state)` |
| `StreamMode.UPDATES` | Just the node's own output dict | — |
| `StreamMode.NODES` | The full `NodeResult` with status / duration / error | — |
| `StreamMode.DEBUG` | `{"result": <the NodeResult as a JSON dict, via model_dump(mode="json")>, "state": dict}` | — |
| `StreamMode.CUSTOM` | Whatever `emit_custom(...)` pushes from inside a node body | — |

## Custom events from inside a node

A pricing node sweeping a batch of invoices can push intermediate
progress with `emit_custom` so the replay shows each invoice as it
clears. Outside a `stream()` context the call is a silent no-op, so the
same node code runs unchanged under `execute()` too.

```python
from locus.multiagent import emit_custom

async def pricing_node(state: dict) -> dict:
    invoices = state["batch"]
    for i, invoice in enumerate(invoices):
        await emit_custom({"progress": i / len(invoices), "phase": f"pricing:{invoice}"})
        await price_invoice(invoice)
    return {"priced": ["inv-2041"]}

graph.add_node("pricing", pricing_node)

async for event in graph.stream(order, mode=StreamMode.UPDATES):
    if event.mode == StreamMode.CUSTOM:
        ui.set_progress(event.data["progress"])     # batch advancing, live
    elif event.mode == StreamMode.UPDATES:
        ui.mark_node_complete(event.node_id)         # specialist done deciding
```

`emit_custom` is exported from `locus.multiagent` and accepts an
optional `node_id=` kwarg if you want the event tagged with the
emitting specialist's identity.

## Real-time delivery

Decisions arrive as each specialist finishes, not at the end. A
fast-validate / slow-fulfill graph proves it:

```python
async def validate(state):
    return {"status": "valid"}              # ~ms

async def fulfill(state):
    await asyncio.sleep(2)                  # 2 seconds: warehouse dispatch
    return {"shipped": ["ord-4821"]}

graph.add_node("validate", validate)
graph.add_node("fulfill", fulfill)
graph.add_edge(START, "validate"); graph.add_edge("validate", "fulfill"); graph.add_edge("fulfill", END)

start = time.perf_counter()
async for ev in graph.stream(order, mode=StreamMode.UPDATES):
    print(f"{time.perf_counter() - start:.2f}s  {ev.node_id}")
# 0.05s  validate
# 2.05s  fulfill
```

If `stream()` were buffering, both events would arrive at 2.05s. The
unit test
[`test_stategraph_streaming.py`](https://github.com/oracle/locus/blob/main/tests/unit/test_stategraph_streaming.py)
guards this property — fails the build if the first event lands at
≥ end / 2.

## Error and cancellation

A specialist that raises has its `NodeResult.success` set to `False`
with the error message; the stream still yields an event for it (no
consumer deadlock), so the replay records the failed step instead of
hanging. Breaking out of the iterator early — say the operator already
saw the fulfillment decision and closed the panel — cancels the
background driver task so no work continues in the background.

## Source

`src/locus/multiagent/graph.py:emit_custom`,
`StateGraph.stream`, `StreamMode`.
