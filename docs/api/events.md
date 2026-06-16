# Events

## Agent event stream

Typed, frozen Pydantic events yielded by `agent.run(...)`. Consume
with `async for event in agent.run(...)` or pipe into a hook.

::: locus.core.events.LocusEvent
::: locus.core.events.ThinkEvent
::: locus.core.events.ToolStartEvent
::: locus.core.events.ToolCompleteEvent
::: locus.core.events.ReflectEvent
::: locus.core.events.GroundingEvent
::: locus.core.events.TerminateEvent
::: locus.core.events.ModelChunkEvent
::: locus.core.events.ModelCompleteEvent
::: locus.core.events.InterruptEvent

## In-process SSE bus

The `EventBus` publishes `StreamEvent`s for every meaningful step from
every framework layer. Opt-in via `run_context()` — zero cost when
unused.

See [Observability](../concepts/observability.md) for usage patterns and
[SSE event catalogue](../concepts/sse-events.md) for the full wire-format
reference (40+ `event_type` strings across 9 prefixes).

::: locus.observability.event_bus.EventBus
::: locus.observability.event_bus.StreamEvent
::: locus.observability.context.run_context
::: locus.observability.context.current_run_id
::: locus.observability.bus_hook.EventBusHook
