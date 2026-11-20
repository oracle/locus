# EventBus subscribers

The in-process `EventBus` has two subscribe shapes, each suited to a
different consumer:

- `bus.subscribe(run_id)` — events for one deployment, with history replay
  on connect then live events, ending when `bus.close_stream(run_id)` is
  called for that run. The shape a rollout console uses for a single
  release.
- `bus.subscribe_global()` — every event from every run, no history
  replay. The shape a telemetry forwarder uses: one tap covering all
  concurrent rollouts.

Both shapes draw on the same bounded buffers. Each subscriber owns one
`asyncio.Queue` holding at most 1024 events. The bus keeps a per-run
history of the last 500 events for at most 200 runs, evicting the oldest
run first, and `bus.close_stream(run_id)` keeps that history, so a
subscriber that connects after the close still replays the run and then
ends, until the run is evicted or `bus.drop_history(run_id)` forgets it.

A publish that cannot place an event on a subscriber's queue within one
second drops the event for that subscriber only and increments the
`dropped_events_total` counter that `bus.stats()` reports; every other
subscriber still receives it. Global subscribers are capped at 50
concurrent; one over the cap ends immediately.

What this notebook covers:

- Per-deployment subscriber alongside a global telemetry subscriber on
  two concurrent rollouts.
- `bus.stats()` snapshot — counts of active runs, global subscribers and
  retained runs, the drop counter, and per-run queue depths.

The drawing below is that capacity model as the bus implements it: the
run's history replayed into each `bus.subscribe(run_id)` queue on
connect, global queues receiving live events from every run with no
replay, and an event still blocked on a full queue after the publish
timeout dropped for that subscriber only:

{{ locus_diagram eventbus-queues }}

## Key lines

`part1_global_vs_per_run` starts a global telemetry forwarder and a
per-deployment subscriber, yields once so both attach before the first
event is published (the global stream has no history replay to fall back
on), then runs two rollouts under `run_context` and closes each stream.
The forwarder records every event from both deployments and returns
after the second `agent.terminate`; the per-deployment subscriber
records only its own rollout. The forwarder tags each kind with the
first six characters of the run id, which for `deploy-A` and `deploy-B`
is the shared prefix `deploy`, so its list shows how many events crossed
the global tap rather than which rollout each came from:

{{ locus_source examples/notebook_62_eventbus_subscribers.py::part1_global_vs_per_run }}

`part2_stats` reads the `bus.stats()` snapshot once both rollouts have
finished: active runs, global subscribers, retained run histories, the
`dropped_events_total` counter and per-run queue depths, the health
metrics of the event pipeline. `close_stream` has already removed the
per-deployment subscriber's queue (`deploy-B` never had one) and kept both
runs' histories, so `active_runs` is 0, `queue_depths` is empty and
`history_runs` is 2. `global_subscribers` still reads 1: the forwarder
returned from inside `async for`, which leaves the `subscribe_global()`
generator suspended at its `yield`, and the bus removes that
subscriber's queue only in the generator's `finally` block. That block
runs when the consumer calls `aclose()` on the generator, or on a later
loop iteration when the event loop runs the `aclose()` it schedules for
a garbage-collected generator; it does not run at the moment the
consuming task returns:

{{ locus_source examples/notebook_62_eventbus_subscribers.py::part2_stats }}

## Run it

```bash
python examples/notebook_62_eventbus_subscribers.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

To stay offline regardless of the profiles on the machine:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_62_eventbus_subscribers.py
```

## Output

Running it offline — no credentials, bundled mock model — prints the event kinds the global telemetry forwarder and the per-deployment subscriber each received across two concurrent rollouts, then the `bus.stats()` snapshot (`active_runs`, `global_subscribers`, `history_runs`, `dropped_events_total`, `queue_depths`):

```text

--- Part 1: telemetry forwarder vs per-deployment subscriber ---
telemetry forwarder saw 4 events across both deployments:
  - deploy/agent.think
  - deploy/agent.terminate
  - deploy/agent.think
  - deploy/agent.terminate
deploy-A subscriber saw 2 events (only its own rollout):
  - agent.think
  - agent.terminate

--- Part 2: bus.stats() ---
  active_runs: 0
  global_subscribers: 1
  history_runs: 2
  dropped_events_total: 0
  pending_publishes: 0
  queue_depths: {}
```
<!-- notebook-output:end -->

## See also

- [Observability](../concepts/observability.md#subscribe-shapes) — the
  subscribe-shape table, and
  [capacity and drop accounting](../concepts/observability.md#capacity-and-drop-accounting)
  for the bounded buffers behind `bus.stats()`.
- [Observability basics](notebook_60_observability_basics.md) —
  `run_context` and the history replay a late `bus.subscribe(run_id)`
  receives.

## Source

??? note "Full source — `examples/notebook_62_eventbus_subscribers.py`"

    ````python
    --8<-- "examples/notebook_62_eventbus_subscribers.py"
    ````
