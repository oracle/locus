# Hooks

## Contract

`HookProvider` is the base class every hook subclasses; `HookRegistry`
collects providers and dispatches events in priority order.
`HookPriority` is the integer-backed enum that determines dispatch
order (lower = earlier; security hooks run before observability hooks).
`ProtectedEvent` marks events whose payloads can be mutated by hooks
(model calls, tool calls) versus the read-only lifecycle events.

::: locus.hooks.provider.HookProvider
::: locus.hooks.provider.HookPriority
::: locus.hooks.provider.ProtectedEvent
::: locus.hooks.registry.HookRegistry
::: locus.hooks.registry.create_registry

## Hook events

### Write-protected events

Hooks observing these events can mutate the payload before it reaches
the model / tool / next stage.

::: locus.hooks.provider.BeforeModelCallEvent
::: locus.hooks.provider.AfterModelCallEvent
::: locus.hooks.provider.BeforeToolCallEvent
::: locus.hooks.provider.AfterToolCallEvent

### Lifecycle events

Read-only notifications fired at agent / iteration boundaries.

::: locus.hooks.events.HookEvent
::: locus.hooks.events.HookResult
::: locus.hooks.events.BeforeInvocationEvent
::: locus.hooks.events.AfterInvocationEvent
::: locus.hooks.events.IterationStartEvent
::: locus.hooks.events.IterationEndEvent

## Built-in hooks

::: locus.hooks.builtin.logging.LoggingHook
::: locus.hooks.builtin.logging.StructuredLoggingHook
::: locus.hooks.builtin.telemetry.TelemetryHook
::: locus.hooks.builtin.telemetry.NoOpTelemetryHook
::: locus.hooks.builtin.retry.ModelRetryHook
::: locus.hooks.builtin.guardrails.GuardrailsHook
::: locus.hooks.builtin.steering.SteeringHook
