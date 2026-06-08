# Events

## Agent event stream

Typed, frozen Pydantic events yielded by `agent.run(...)`. Consume
with `async for event in agent.run(...)` or pipe into a hook.

::: tulip.core.events.TulipEvent
::: tulip.core.events.ThinkEvent
::: tulip.core.events.ToolStartEvent
::: tulip.core.events.ToolCompleteEvent
::: tulip.core.events.ReflectEvent
::: tulip.core.events.GroundingEvent
::: tulip.core.events.TerminateEvent
::: tulip.core.events.ModelChunkEvent
::: tulip.core.events.ModelCompleteEvent
::: tulip.core.events.InterruptEvent

## In-process SSE bus

The `EventBus` publishes `StreamEvent`s for every meaningful step from
every framework layer. Opt-in via `run_context()` — zero cost when
unused.

See [Observability](../concepts/observability.md) for usage patterns and
[SSE event catalogue](../concepts/sse-events.md) for the full wire-format
reference (40+ `event_type` strings across 9 prefixes).

::: tulip.observability.event_bus.EventBus
::: tulip.observability.event_bus.StreamEvent
::: tulip.observability.context.run_context
::: tulip.observability.context.current_run_id
::: tulip.observability.bus_hook.EventBusHook
