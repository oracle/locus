# Hooks

## Contract

`HookProvider` is the base class every hook subclasses; `HookRegistry`
collects providers and dispatches events in priority order.
`HookPriority` is the integer-backed enum that determines dispatch
order (lower = earlier; security hooks run before observability hooks).
`ProtectedEvent` marks events whose payloads can be mutated by hooks
(model calls, tool calls) versus the read-only lifecycle events.

::: tulip.hooks.provider.HookProvider
::: tulip.hooks.provider.HookPriority
::: tulip.hooks.provider.ProtectedEvent
::: tulip.hooks.registry.HookRegistry
::: tulip.hooks.registry.create_registry

## Hook events

### Write-protected events

Hooks observing these events can mutate the payload before it reaches
the model / tool / next stage.

::: tulip.hooks.provider.BeforeModelCallEvent
::: tulip.hooks.provider.AfterModelCallEvent
::: tulip.hooks.provider.BeforeToolCallEvent
::: tulip.hooks.provider.AfterToolCallEvent

### Lifecycle events

Read-only notifications fired at agent / iteration boundaries.

::: tulip.hooks.events.HookEvent
::: tulip.hooks.events.HookResult
::: tulip.hooks.events.BeforeInvocationEvent
::: tulip.hooks.events.AfterInvocationEvent
::: tulip.hooks.events.IterationStartEvent
::: tulip.hooks.events.IterationEndEvent

## Built-in hooks

::: tulip.hooks.builtin.logging.LoggingHook
::: tulip.hooks.builtin.logging.StructuredLoggingHook
::: tulip.hooks.builtin.telemetry.TelemetryHook
::: tulip.hooks.builtin.telemetry.NoOpTelemetryHook
::: tulip.hooks.builtin.retry.ModelRetryHook
::: tulip.hooks.builtin.guardrails.GuardrailsHook
::: tulip.hooks.builtin.steering.SteeringHook
