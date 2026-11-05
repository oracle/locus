# Observability basics

The SDK ships an in-process publish/subscribe (pub/sub) `EventBus` that
publishes typed `StreamEvent` objects for the meaningful steps of
execution: agent thinking, tool calls, multi-agent fan-outs,
checkpoints — each under a canonical `event_type` whose prefix names
the component (`agent.*`, `multiagent.*`, `composition.*`, `memory.*`).
For a customer-support bot, that stream is the ticket *timeline*: an
ordered, replayable record of every action the agent took on a case,
ready for a quality-assurance (QA) review or a customer dispute. The
[Observability](../concepts/observability.md#in-process-sse-eventbus)
concept page describes the bus in full.

Telemetry is opt-in. Code that never enters a `run_context` pays one
`ContextVar.get()` per emission site — no bus, no events, no
allocations.

What this notebook covers:

- Run an `Agent` with no telemetry (the SDK-default path) and confirm
  that the bus singleton was never built.
- Reply on the same ticket inside `run_context()`, this time with a
  lookup tool attached, and subscribe to the bus.
- Watch the canonical events on the agent path land on the timeline:
  `agent.think`, the `agent.tool.started`/`agent.tool.completed` pair
  and `agent.terminate`. `agent.tokens.used` (drawn dashed below) is
  mapped from a [`ModelCompleteEvent`](../concepts/events.md#modelcompleteevent)
  that carries usage, but `Agent.run` has no yield site for that event,
  so the agent path never records it; token totals are available from
  `result.metrics`, which [notebook 61](notebook_61_agent_yield_bridge.md)
  meters.
- Read the per-run history buffer after the fact (replay semantics for
  late subscribers — reconstruct the ticket timeline after the run).

Inside `run_context()`, the think, tool and terminate steps of
`agent.arun(...)` reach the `EventBus` as canonical `agent.*` events, and
`bus.subscribe(rid)` replays the run's history before streaming live:

{{ locus_diagram run-context-events }}

## Key lines

With no `run_context` entered, `agent.arun(...)` runs on the SDK-default
path and leaves the bus singleton unbuilt; the check reads the module
attribute directly because `get_event_bus()` would construct it:

{{ locus_source examples/notebook_60_observability_basics.py::part1_no_telemetry }}

Inside `run_context()`, a consumer task iterates `bus.subscribe(rid)` while
`agent.arun(...)` runs in the same block; the `@_bus_bridge` decorator on
`Agent.run` republishes each yielded event the bridge maps, so
`agent.think`, the `agent.tool.started`/`agent.tool.completed` pair and
`agent.terminate` all land on the timeline, and `close_stream(rid)` ends
the iterator for any other subscriber:

{{ locus_source examples/notebook_60_observability_basics.py::part2_subscribe }}

A subscriber that arrives after the run has finished and the stream is
closed still receives the run from the per-run history deque (the bus
keeps up to 500 events for each of the last 200 runs), which is what an
after-the-fact QA review of the ticket relies on:

{{ locus_source examples/notebook_60_observability_basics.py::part3_history_replay }}

## Run it

```bash
python examples/notebook_60_observability_basics.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

To run offline regardless of the local OCI configuration:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_60_observability_basics.py
```

## Output

Running it offline — no credentials, bundled mock model — prints the bus singleton still `None` after a reply made outside `run_context`, the canonical event trail a tool-calling ticket reply leaves on the bus (`agent.think`, `agent.tool.started`, `agent.tool.completed`, `agent.think`, `agent.terminate`), then the two events a late subscriber replays from history after the run has finished:

```text

--- Part 1: no run_context — bus stays uninstantiated ---
agent reply: This is a mock response for testing purposes.
bus singleton is None — zero allocations spent on telemetry

--- Part 2: run_context active — subscribe to the ticket timeline ---
agent reply: This is a mock response for testing purposes.
timeline events recorded (5):
  - agent.think
  - agent.tool.started
  - agent.tool.completed
  - agent.think
  - agent.terminate

--- Part 3: late subscriber — after-the-fact QA replay ---
replayed 2 events from history (after the run finished)
  - agent.think
  - agent.terminate
```
<!-- notebook-output:end -->

## Source

??? note "Full source — `examples/notebook_60_observability_basics.py`"

    ````python
    --8<-- "examples/notebook_60_observability_basics.py"
    ````
