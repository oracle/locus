# EventBus Subscribers

The bus has three subscribe shapes, each suited to a different consumer:

- `bus.subscribe(run_id)` — events for one dispatch, with history replay
  on connect then live events, terminated by a sentinel on stream close.
- `bus.subscribe_global()` — every event from every run, no history
  replay. Good fit for a monitoring dashboard that spans concurrent runs.
- `bus._history.get(run_id, ())` — direct read of the per-run history
  deque (test helper; capped at 500 events × 200 runs LRU).

Capacity model::

    publisher
       │
       ├──► subscriber queue A  (max_queue_size=1024)
       │       └─ slow? ─► drop event, increment bus._dropped_events
       │
       ├──► subscriber queue B
       │
       └──► global subscriber queues (capped count)

- Per-run subscriber alongside a global subscriber on two concurrent
  dispatches.
- `bus.stats()` snapshot — queue sizes, history depth, drop counter,
  retained-run count.

Run it (defaults to the bundled mock model; set `TULIP_MODEL_PROVIDER` to `openai` / `anthropic` for a live model):

    python examples/notebook_61_eventbus_subscribers.py

Offline:

    TULIP_MODEL_PROVIDER=mock python examples/notebook_61_eventbus_subscribers.py

## Source

```python
--8<-- "examples/notebook_61_eventbus_subscribers.py"
```
