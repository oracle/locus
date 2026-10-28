# Hooks

A hook is a callback the agent calls at eight fixed moments in a run:
before / after the run starts, before / after each loop iteration,
before / after each model call, before / after each tool call.
Everything that should happen *around* the
agent's primary task — logging, OpenTelemetry traces, retry, guardrails,
PII redaction, an LLM-as-judge approval gate on tool calls — lives in
a hook.

You can use the ones Locus
ships (covers most production needs out of the box) or write your own
— a hook is a small subclass with the methods it cares about.

## When to write a hook

| You want… | Write a hook |
|---|---|
| Log every tool call to your aggregator | ✓ |
| Add OpenTelemetry spans / metrics | ✓ — use the built-in `TelemetryHook` |
| Retry model calls with backoff | ✓ — `ModelRetryHook` |
| Reject tool calls that look dangerous | ✓ — `GuardrailsHook`, `ContentFilterHook`, `SteeringHook` |
| Add a tool to the registry | use [`tools=[...]` on Agent](tools.md) |
| Change the system prompt mid-run | hooks can read state but not mutate the prompt; use a [skill](skills.md) instead |

## The eight lifecycle phases

A hook can subscribe to any of these. Each method receives a typed,
write-protected event object.

| Phase | Fires | Useful for |
|---|---|---|
| `on_before_invocation` | once, when `agent.run()` starts | initialise per-run state, open spans |
| `on_after_invocation` | once, after the agent finishes | flush metrics, close spans |
| `on_iteration_start` | at the top of each loop iteration | per-iteration budgets, step counters |
| `on_iteration_end` | at the bottom of each loop iteration | per-iteration accounting, early termination |
| `on_before_model_call` | before each request to the model | redact PII, count tokens |
| `on_after_model_call` | after each response from the model | log usage, retry on empty |
| `on_before_tool_call` | before each tool body runs | guardrails, audit, approval gates |
| `on_after_tool_call` | after each tool body completes | log result, update metrics, mirror calls into a host-side queue |

## Getting started

### 1. Subclass `HookProvider`

```python
from locus.hooks.provider import HookPriority, HookProvider

class AuditHook(HookProvider):
    name = "audit"
    priority = HookPriority.OBSERVABILITY_MIN

    async def on_before_tool_call(self, event):
        print(f"→ {event.tool_name}({event.arguments})")

    async def on_after_tool_call(self, event):
        print(f"← {event.tool_name} = {event.result}")
```

Override only the phases you care about. Unimplemented phases inherit
no-op defaults from the base class.

### `on_after_tool_call` — what the event carries

| Field | Type | Mutable? | Meaning |
|---|---|---|---|
| `tool_name` | `str` | read-only | Name of the tool that ran. |
| `tool_call_id` | `str` | read-only | The same id as the matching `BeforeToolCallEvent.tool_call_id` — use it to correlate before/after for parallel tool calls. |
| `arguments` | `dict[str, Any]` | read-only | The arguments the tool was invoked with, *post* any mutation by a `before` hook. |
| `result` | `Any` | writable via `event.result = ...` | Replace the tool result before downstream hooks / the agent see it. |
| `error` | `str \| None` | read-only | Set when the tool raised; mutually exclusive with a useful `result`. |
| `retry` | `bool` | writable via `event.retry = True` | Re-execute the tool with the same arguments. |

**Common pattern — mirror every tool call into a host-side queue** (e.g. an
HTTP response payload that drives an out-of-process side effect):

```python
class ActionQueueHook(HookProvider):
    priority = HookPriority.BUSINESS_DEFAULT

    def __init__(self, queue: list[dict]) -> None:
        self._queue = queue

    async def on_after_tool_call(self, event):
        if event.error is None:
            self._queue.append({
                "id": event.tool_call_id,        # correlate with the model's tool_calls[]
                "tool": event.tool_name,
                "args": event.arguments,         # exact args the tool ran with
                "result": event.result,
            })
```

This is the standard pattern for [MCP integrations](mcp.md) where
the *real* side effect lives in the host process, not in the tool body.

### 2. Pass to the agent

```python
agent = Agent(
    model="{{ locus_example_model }}",
    tools=[lookup_order, issue_refund],
    hooks=[AuditHook()],
)
```

### 3. Run

The hook fires on every agent run — no further wiring.

## What you get out of the box

The SDK ships these hooks. Composed in this order, they cover most
production needs without writing custom code.

```python
from locus.hooks.builtin import (
    LoggingHook, StructuredLoggingHook,
    TelemetryHook,
    ModelRetryHook,
    GuardrailsHook, ContentFilterHook,
    SteeringHook,
)

agent = Agent(
    model="{{ locus_example_model }}",
    tools=[...],
    hooks=[
        StructuredLoggingHook(),       # JSON logs at every phase
        TelemetryHook(),               # OTel spans + metrics + histograms
        ModelRetryHook(max_retries=3), # backoff on empty model responses
        GuardrailsHook(),              # PII / SQL / path-traversal / command-injection
        SteeringHook(model=judge_model),  # LLM-as-judge tool approval
    ],
)
```

### `LoggingHook` / `StructuredLoggingHook`

Plain-text or JSON-structured logs at every lifecycle phase. Drop in
when you want a paper trail without writing your own logger.

### `TelemetryHook`

OpenTelemetry spans for every model + tool call, counters for tool
invocations, histograms for latency. Use `NoOpTelemetryHook` when
you want the API surface but no actual export (useful for tests).

### `ModelRetryHook`

Backoff retries on **empty model responses** — a reply with no content
and no tool calls. Configurable `max_retries`, `initial_delay`,
`max_delay`, and `backoff_factor`. Doesn't intercept your tool calls —
only the model layer.

!!! warning "It does not retry errors"
    `on_after_model_call` has no exception handling, so rate limits,
    5xx responses, and connection failures are **not** retried here.
    Those are handled inside the provider client instead — the OpenAI
    and Anthropic models take `max_retries` and `request_timeout` of
    their own. See [Retry strategies](retry.md).

### `GuardrailsHook` / `ContentFilterHook`

Regex-based policies on tool inputs (`GuardrailsHook`) and model
outputs (`ContentFilterHook`). Catches PII, SQL injection patterns,
shell-command injection, and credit-card-shaped strings. Reject or
redact at the boundary.

### `SteeringHook` — LLM-as-judge tool approval

A *second model* evaluates each tool call before it runs and emits
one of `PROCEED / GUIDE / INTERRUPT` (allow, cancel with feedback, or
pause for a human). Use this when the cost of a wrong tool call is
higher than the cost of a second model round-trip. The `model=` is a
model instance, not a provider string:

```python
from locus.models import AnthropicModel

agent = Agent(
    ...,
    hooks=[SteeringHook(model=AnthropicModel(model="claude-sonnet-5-5"))],
)
```

## Priorities — the ordering rules

An `Agent` runs its hooks in the order you list them on `before_*`
phases and reverses that order for `after_*`, so teardown pairs with
setup. `priority` is what a `HookRegistry` sorts on before handing a
provider list over (lower numbers first); list security hooks first
when you pass them to the `Agent` directly.

Reach for the named constants in `HookPriority` —
`HookPriority.SECURITY_MAX`, `HookPriority.OBSERVABILITY_MIN`,
`HookPriority.BUSINESS_MIN`, etc. — so the intent is obvious in
code review. The underlying number bands are:

| Range | Intended use |
|---|---|
| `0`–`99` | **Security** — guardrails, PII redaction (must run first to short-circuit unsafe calls) |
| `100`–`199` | **Observability** — logging, telemetry |
| `200`–`299` | **Business logic** — domain-specific hooks |
| `300+` | **Cosmetic** — pretty-printing, console UI |

## Write-protected events — by design

Event objects are `ProtectedEvent` instances with a custom
`__setattr__`: each event declares a small `_writable` set, and writing
any other field raises `AttributeError`. The fields that *do* let hooks
steer the agent are plain attribute assignments — on
`BeforeToolCallEvent` you can set `event.arguments` (rewrite the inputs)
and `event.cancel` (skip the call); on `AfterToolCallEvent` you can set
`event.result` and `event.retry`. Setting `cancel` to a string records
the reason:

```python
async def on_before_tool_call(self, event):
    if "DROP TABLE" in str(event.arguments):
        event.cancel = "SQL injection blocked by GuardrailsHook"
```

Compare to a callback-based system where any code can monkey-patch
any field; this is intentionally tight.

## Common gotchas

| Symptom | Likely cause |
|---|---|
| Hook never fires | Forgot to pass it on `Agent(hooks=[...])`. The `HookRegistry` only sees what you register. |
| Hook fires in the wrong order | Reorder the list you pass to `Agent(hooks=[...])`; when the hooks come through a `HookRegistry`, set `priority` explicitly — its default is mid-range so security hooks sort first. |
| `AttributeError: Cannot set 'foo' … read-only` | You tried to write a non-writable field. Only the documented steering fields are assignable (`arguments`/`cancel` on before-tool, `result`/`retry` on after-tool); everything else is observe-only. |
| `on_after_tool_call` doesn't see the result | The tool raised. Check `event.error` instead of `event.result`. |
| Telemetry spans aren't exported | `TelemetryHook` needs an OTel exporter configured upstream — see [Observability](observability.md). |

## Source and examples

- [`HookProvider` and `HookOrchestrator`](https://github.com/oracle/locus/blob/main/src/locus/hooks/provider.py)
- [Built-in hooks](https://github.com/oracle/locus/tree/main/src/locus/hooks/builtin)
- [`notebook_09_agent_hooks.py`](https://github.com/oracle/locus/blob/main/examples/notebook_09_agent_hooks.py) — write your first hook.
- [`notebook_10_hooks_advanced.py`](https://github.com/oracle/locus/blob/main/examples/notebook_10_hooks_advanced.py) — guardrails + steering, end to end.

## See also

- [Tools](tools.md) — the things hooks observe.
- [Events](events.md) — the typed event objects hooks receive.
- [Safety & guardrails](safety.md) — production policies built on `GuardrailsHook`.
- [Observability](observability.md) — wiring `TelemetryHook` to your OTel collector.
- [Retry strategies](retry.md) — how `ModelRetryHook` works under the hood.
