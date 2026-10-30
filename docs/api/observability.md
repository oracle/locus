# Observability

Centralized in-process event bus + SSE telemetry. The `EventBus` is a
singleton that fans `StreamEvent` instances scoped to a *run id* out
to multiple consumers (Web SSE, CLI tail, JSON logs) from a single
emission point.

Publishers (router, agent loop hooks, custom user code) call `emit()`
or `bus.publish()`; subscribers consume via `bus.subscribe(run_id)`.
An SSE endpoint such as `/api/events/{run_id}` is the natural public
HTTP wrapper around `EventBus.subscribe`.

For the agent's typed `LocusEvent` stream (the thing
`async for event in agent.run(...)` yields) see [Events](events.md);
this page covers the **lower-level** observability bus that
`LocusEvent`s are mirrored onto for cross-process consumption.

For the concepts, start with
[Observability](../concepts/observability.md#in-process-sse-eventbus)
and [the SSE event catalogue](../concepts/sse-events.md).

## Event bus

::: locus.observability.event_bus.EventBus
::: locus.observability.event_bus.StreamEvent
::: locus.observability.event_bus.get_event_bus
::: locus.observability.event_bus.reset_event_bus

## Publishing

::: locus.observability.emit.emit
::: locus.observability.emit.emit_sync

## Run context

Scope events to a single run (e.g. one router dispatch) via `run_context()` — a
context manager that sets and unsets the run id for the current task /
coroutine.

::: locus.observability.context.run_context
::: locus.observability.context.current_run_id
::: locus.observability.context.set_run_id
::: locus.observability.context.reset_run_id

## Hook bridge

`EventBusHook` publishes the agent's hook lifecycle — invocation, iteration,
model-call and tool-call phases — onto the bus as `agent.<phase>` events
without changing existing hook code. The `LocusEvent` stream itself is
mirrored by the runtime whenever a `run_context()` is active.

::: locus.observability.bus_hook.EventBusHook
