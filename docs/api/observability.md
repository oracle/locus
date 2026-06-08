# Observability

Centralised in-process event bus + SSE telemetry. The `EventBus` is a
singleton that fans `StreamEvent` instances scoped to a *run id* out
to multiple consumers (Web SSE, CLI tail, JSON logs) from a single
emission point.

Publishers (router, agent loop hooks, custom user code) call `emit()`
or `bus.publish()`; subscribers consume via `bus.subscribe(run_id)`.
The workbench's SSE endpoint at `/api/events/{run_id}` is the public
HTTP wrapper around `EventBus.subscribe`.

For the agent's typed `TulipEvent` stream (the thing
`async for event in agent.run(...)` yields) see [Events](events.md);
this page covers the **lower-level** observability bus that
`TulipEvent`s are mirrored onto for cross-process consumption.

## Event bus

::: tulip.observability.event_bus.EventBus
::: tulip.observability.event_bus.StreamEvent
::: tulip.observability.event_bus.get_event_bus
::: tulip.observability.event_bus.reset_event_bus

## Publishing

::: tulip.observability.emit.emit
::: tulip.observability.emit.emit_sync

## Run context

Scope events to a single cognitive dispatch via `run_context()` — a
context manager that sets and unsets the run id for the current task /
coroutine.

::: tulip.observability.context.run_context
::: tulip.observability.context.current_run_id
::: tulip.observability.context.set_run_id
::: tulip.observability.context.reset_run_id

## Hook bridge

`EventBusHook` mirrors the agent's `TulipEvent` stream onto the bus
without changing existing hook code.

::: tulip.observability.bus_hook.EventBusHook
