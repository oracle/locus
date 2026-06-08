# Hooks

A hook is a callback the agent calls at six fixed moments in a run:
before / after the run starts, before / after each model call, before /
after each tool call. Everything that should happen *around* the
agent's primary task — logging, OpenTelemetry traces, retry, guardrails,
PII redaction, an LLM-as-judge approval gate on tool calls — lives in
a hook.

You can use the ones Tulip
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

## The six lifecycle phases

A hook can subscribe to any of these. Each method receives a typed,
write-protected event object.

| Phase | Fires | Useful for |
|---|---|---|
| `on_before_invocation` | once, when `agent.run()` starts | initialise per-run state, open spans |
| `on_after_invocation` | once, after the agent finishes | flush metrics, close spans |
| `on_before_model_call` | before each request to the model | redact PII, count tokens |
| `on_after_model_call` | after each response from the model | log usage, retry on empty |
| `on_before_tool_call` | before each tool body runs | guardrails, audit, approval gates |
| `on_after_tool_call` | after each tool body completes | log result, update metrics, mirror calls into a host-side queue |

## Getting started

### 1. Subclass `HookProvider`

```python
from tulip.hooks.provider import HookPriority, HookProvider

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
    model="anthropic:claude-sonnet-4-6",
    tools=[search, book_flight],
    hooks=[AuditHook()],
)
```

### 3. Run

The hook fires on every agent run — no further wiring.

## What you get out of the box

The SDK ships these hooks. Composed in this order, they cover most
production needs without writing custom code.

```python
from tulip.hooks.builtin import (
    LoggingHook, StructuredLoggingHook,
    TelemetryHook,
    ModelRetryHook,
    GuardrailsHook, ContentFilterHook,
    SteeringHook,
)

agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[...],
    hooks=[
        StructuredLoggingHook(),       # JSON logs at every phase
        TelemetryHook(),               # OTel spans + metrics + histograms
        ModelRetryHook(max_retries=3), # backoff on empty / rate-limited responses
        GuardrailsHook(),              # PII / SQL / XSS / command-injection
        SteeringHook(approver=second_model),  # LLM-as-judge tool approval
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

Backoff retries on empty model responses, rate-limit errors, and
transient connection failures. Configurable `max_retries` and
`backoff_seconds`. Doesn't intercept your tool calls — only the
model layer.

### `GuardrailsHook` / `ContentFilterHook`

Regex-based policies on tool inputs (`GuardrailsHook`) and model
outputs (`ContentFilterHook`). Catches PII, SQL injection patterns,
shell-command injection, and credit-card-shaped strings. Reject or
redact at the boundary.

### `SteeringHook` — LLM-as-judge tool approval

A *second model* sees each tool call before it runs and votes
"approve / reject / rewrite". Use this when the cost of a wrong tool
call is higher than the cost of a second model round-trip.

```python
agent = Agent(
    ...,
    hooks=[SteeringHook(approver="anthropic:claude-sonnet-4-6")],
)
```

## Priorities — the ordering rules

Hooks run in priority order. Lower numbers run first on `before_*`
phases; the order reverses for `after_*` so teardown pairs with
setup.

Reach for the named constants in `HookPriority` —
`HookPriority.SECURITY_MAX`, `HookPriority.OBSERVABILITY_MIN`,
`HookPriority.BUSINESS_LOGIC_MIN`, etc. — so the intent is obvious in
code review. The underlying number bands are:

| Range | Intended use |
|---|---|
| `0`–`99` | **Security** — guardrails, PII redaction (must run first to short-circuit unsafe calls) |
| `100`–`199` | **Observability** — logging, telemetry |
| `200`–`299` | **Business logic** — domain-specific hooks |
| `300+` | **Cosmetic** — pretty-printing, console UI |

## Write-protected events — by design

Event objects are frozen Pydantic models. You **cannot** accidentally
mutate them from a hook — try and you get a `ValidationError`. The
methods that *do* let hooks steer the agent (`event.cancel()`,
`event.retry()`, `event.replace_arguments(...)`) are explicit and
named for what they do, so the intent is unambiguous in a review:

```python
async def on_before_tool_call(self, event):
    if "DROP TABLE" in str(event.arguments):
        event.cancel(reason="SQL injection blocked by GuardrailsHook")
```

Compare to a callback-based system where any code can monkey-patch
any field; this is intentionally tight.

## Common gotchas

| Symptom | Likely cause |
|---|---|
| Hook never fires | Forgot to pass it on `Agent(hooks=[...])`. The `HookRegistry` only sees what you register. |
| Hook fires in the wrong order | Set `priority` explicitly. The default priority is intentionally mid-range so security hooks always come before yours. |
| `ValidationError: cannot mutate frozen instance` | You tried to write `event.foo = bar`. Hooks observe, not mutate; use the explicit steering methods. |
| `on_after_tool_call` doesn't see the result | The tool raised. Check `event.error` instead of `event.result`. |
| `on_after_tool_call` doesn't see the arguments / call id | Pre-`0.2.0b4` event payload. Upgrade — `event.arguments` and `event.tool_call_id` were added so hooks can build host-side action queues without a separate `before`-hook stash. |
| Telemetry spans aren't exported | `TelemetryHook` needs an OTel exporter configured upstream — see [Observability](observability.md). |

## Source and examples

- [`HookProvider` and `HookOrchestrator`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/hooks/provider.py)
- [Built-in hooks](https://github.com/tuliplabs-ai/sdk-python/tree/main/src/tulip/hooks/builtin)
- [`notebook_12_agent_hooks.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_12_agent_hooks.py) — write your first hook.
- [`notebook_14_hooks_advanced.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_14_hooks_advanced.py) — guardrails + steering, end to end.

## See also

- [Tools](tools.md) — the things hooks observe.
- [Events](events.md) — the typed event objects hooks receive.
- [Safety & guardrails](safety.md) — production policies built on `GuardrailsHook`.
- [Observability](observability.md) — wiring `TelemetryHook` to your OTel collector.
- [Retry strategies](retry.md) — how `ModelRetryHook` works under the hood.
