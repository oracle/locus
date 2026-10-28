# EventBus Subscribers

The bus has three subscribe shapes, each suited to a different consumer:

- `bus.subscribe(run_id)` — events for one deployment, with history replay
  on connect then live events, terminated by a sentinel on stream close.
- `bus.subscribe_global()` — every event from every run, no history
  replay. Good fit for a telemetry forwarder that spans concurrent
  rollouts.
- `bus._history.get(run_id, ())` — direct read of the per-run history
  deque (test helper; capped by default at 500 events per run across
  200 runs, oldest run evicted first).

The capacity model, as the bus implements it:

{{ locus_diagram eventbus-queues }}

The notebook demonstrates:

- Per-deployment subscriber alongside a global telemetry subscriber on
  two concurrent rollouts.
- `bus.stats()` snapshot — counts of active runs, global subscribers and
  retained runs, the drop counter, and per-run queue depths.

Run it (OCI Generative AI with an OCI profile, otherwise the bundled mock model; set `LOCUS_MODEL_PROVIDER` to force a provider):

    python examples/notebook_62_eventbus_subscribers.py

Offline:

    LOCUS_MODEL_PROVIDER=mock python examples/notebook_62_eventbus_subscribers.py

## Source

````python
--8<-- "examples/notebook_62_eventbus_subscribers.py"
````
