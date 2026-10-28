---
title: Observability
---

# Observability

What the agent did, how long each step took, and what it cost — two
built-in hooks plus the standard OpenTelemetry stack cover every piece
you need. No vendor lock-in: Locus emits OTLP (the OpenTelemetry wire
protocol), you point it at whatever backend you run.

## When to wire what

| Need | Add |
|---|---|
| Structured per-event lines for log aggregators (Loki, Splunk) | `StructuredLoggingHook` |
| OTLP traces and metrics for dashboards (Grafana, Honeycomb) | `TelemetryHook` |
| Token and cost totals on every result (a resumed `thread_id` reports the thread's running total) | nothing — `AgentResult.metrics` already has it |
| Per-run trace ID surfaced to the user (for support tickets) | telemetry hook + log the active span's trace ID |

## Getting started

### Structured logs

```python
import logging
from locus.agent import Agent
from locus.hooks.builtin import StructuredLoggingHook

agent = Agent(
    model="{{ locus_example_model }}",
    tools=[lookup_order, check_refund_policy, issue_refund],
    hooks=[StructuredLoggingHook(level=logging.INFO)],
)
```

Each lifecycle point is emitted as a structured JSON record you can
forward to a log aggregator (Loki, Splunk) or a SIEM — a security
team's log platform. The after-tool-call record looks
like this out of the box:

```json
{
  "message": "Tool call completed",
  "event": "tool_call_completed",
  "tool_name": "issue_refund",
  "success": true,
  "timestamp": "2026-05-02T01:31:02.481923+00:00"
}
```

The keys are exactly those the hook sets: `message`, `event` (the
message lowercased with spaces as underscores — e.g.
`tool_call_completed`), `tool_name`, `success`, and `timestamp`, with
an `error` key when the call failed. `StructuredLoggingHook` hard-wires
`log_arguments=False` and `log_results=False` and exposes no
constructor option to turn them on, so the matching `tool_call_starting`
record carries `tool_name` plus `argument_keys` (names only, no values),
and neither record carries argument values or tool results. If you need
`result_preview` + `result_length`, use the parent
`LoggingHook(log_results=True)` instead — it attaches them as attributes
on the standard `logging` record (no `event` or `timestamp` key) — and
note that results and arguments may contain PII.
Anything you pass as `extra={...}` to the hook
constructor is merged into **every** record — that's how you stamp a
stable `agent_id` / `thread_id` for correlation.

Pipe stdout to your log aggregator. The SDK doesn't own the transport —
you choose between stdlib `logging`, `structlog`, or
`opentelemetry-logs`.

### Replaying an incident for auditors

After a disputed action — a refund the customer says never arrived, a
rollback that fired at 2 a.m. — an auditor asks: *what did the agent
do, in what order?* Stamp the hook with a stable `thread_id` so every
record carries it —
`StructuredLoggingHook(extra={"thread_id": "ord-4821"})` — then
filter by `thread_id`, sort by `timestamp`, and pick out the
`tool_call_completed` records to reconstruct the tool timeline:

```python
import json

events = [json.loads(line) for line in open("support-agent.jsonl")]
case = sorted(
    (e for e in events if e.get("thread_id") == "ord-4821"),
    key=lambda e: e["timestamp"],
)
for e in case:
    if e["event"] == "tool_call_completed":
        print(f'{e["timestamp"]}  {e["tool_name"]:<20} success={e["success"]}')
# 2026-05-02T01:30:58.117402+00:00  lookup_order         success=True
# 2026-05-02T01:31:01.390115+00:00  check_refund_policy  success=True
# 2026-05-02T01:31:02.481923+00:00  issue_refund         success=True
```

Because the stamped `thread_id` is stable across a multi-turn
case, the same filter replays the *whole* thread in
chronological order. For per-tool `span_id` correlation (pairing a start
with its completion), consume the `EventBus` stream instead — its
`agent.tool.*` events carry a `span_id` (see below).

The JSONL event log is a faithful trace, not a tamper-evident record —
for decisions an auditor must trust, route the action through `admit()`
so it lands on the hash-chained
[`AuditTrail`](agentic-ai-security.md) — each entry commits to the one
before it, so `verify()` fails on any edit.

### Traces and metrics over OTLP

```python
from locus.hooks.builtin import TelemetryHook

agent = Agent(
    model="{{ locus_example_model }}",
    tools=[lookup_order, issue_refund],
    hooks=[
        TelemetryHook(
            service_name="support-agent",
            record_arguments=False,    # set True to attach tool args to spans
            record_results=False,      # set True for results (watch PII)
        ),
    ],
)
```

Spans are emitted for every agent invocation, every ReAct iteration,
every tool call, and every model call. Metrics include:

| Counter | What it counts |
|---|---|
| `locus.invocations` | Calls to `agent.run(...)` |
| `locus.iterations` | ReAct iterations across all runs |
| `locus.tool_calls` | Tool invocations |
| `locus.tool_errors` | Tool calls that raised |

| Histogram | What it measures |
|---|---|
| `locus.invocation.duration` | Wall-clock per `agent.run(...)` |
| `locus.tool_call.duration` | Wall-clock per tool body |

Configure the exporter the standard OpenTelemetry way — set
`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_RESOURCE_ATTRIBUTES`, etc.
before constructing the agent. Anything OTLP works: Honeycomb, Tempo,
Grafana Cloud.

Install the optional extra:

```bash
pip install "locus-sdk[telemetry]"
```

### Token cost — already on every result

```python
result = agent.run_sync("Resolve the duplicate charge on ord-4821.")
print(f"prompt:     {result.metrics.prompt_tokens}")
print(f"completion: {result.metrics.completion_tokens}")
print(f"total:      {result.metrics.total_tokens}")
print(f"iterations: {result.metrics.iterations}")
print(f"cost_usd:   {result.metrics.cost_usd}")
```

`cost_usd` is the spend in USD, computed from the model's
per-million-token prices in Locus's model metadata. Like the token
counts, it adds up across turns when you resume a checkpointed
`thread_id`, so on a resumed thread it is the thread's total so far,
not just the latest turn's. It is `None` —
unpriced, never free — when those prices are unknown. The built-in
table prices only some models, and lookup is an exact match once the
`openai:` / `anthropic:` prefix is stripped, so
`anthropic:claude-sonnet-4-6` does not inherit the `claude-sonnet-4`
price. Register prices for your model with
`register_metadata(ModelMetadata(..., input_price_per_mtok=..., output_price_per_mtok=...))`
from `locus.models.metadata` before you construct the agent — prices
are read once, when the `Agent` is built. The same prices drive `max_cost_usd`: set
a budget on an unpriced model and the `Agent` constructor raises
`ValueError` rather than run a budget that could never stop anything.
For dashboards, key on `agent_id` plus the same metrics the
`TelemetryHook` already emits — no glue code needed.

## PII and tool arguments

`record_arguments=True` and `record_results=True` are off by default
because tool args and results often contain sensitive input — order
records, account identifiers, untrusted user input. Turn them on
selectively, and only after you've verified your tracing backend has
appropriate retention and access controls. For PII redaction *inside*
the agent before anything leaves, see [Safety](safety.md).

## Common gotchas

| Symptom | Likely cause |
|---|---|
| `TelemetryHook` raises `ImportError` | `pip install "locus-sdk[telemetry]"` to get the OpenTelemetry SDK. |
| No spans show up in your backend | Exporter not configured. Set `OTEL_EXPORTER_OTLP_ENDPOINT` (and `OTEL_EXPORTER_OTLP_HEADERS` if your backend needs auth) *before* creating the agent. |
| Spans land but metrics don't | Some OTLP receivers reject metrics on the trace endpoint. Set `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` separately if needed. |
| Token totals are zero | The provider isn't returning usage in the response (some self-hosted endpoints). The SDK's loop can't make up the numbers. |
| Tool args land in your logs unintentionally | Either `record_arguments=True` or your structured logger is dumping the full event dict. Configure either explicitly. |

## Source and notebooks

- [`notebook_09_agent_hooks.py`](https://github.com/oracle/locus/blob/main/examples/notebook_09_agent_hooks.py) — first hook, including logging.
- [`notebook_10_hooks_advanced.py`](https://github.com/oracle/locus/blob/main/examples/notebook_10_hooks_advanced.py) — telemetry pipelines.
- [`locus.hooks.builtin.logging`](https://github.com/oracle/locus/blob/main/src/locus/hooks/builtin/logging.py) — `LoggingHook`, `StructuredLoggingHook`.
- [`locus.hooks.builtin.telemetry`](https://github.com/oracle/locus/blob/main/src/locus/hooks/builtin/telemetry.py) — `TelemetryHook`, `NoOpTelemetryHook`.

---

## In-process SSE (EventBus)

For browser streaming, real-time dashboards, or any use case where you need
to watch the full inner cognition of a run without standing up an OTLP stack,
the SDK ships a zero-dependency in-process pub/sub bus.

### How it works

Every emission site in the SDK reads `current_run_id()` from a `ContextVar`.
When no `run_context()` is active the emit returns immediately — no bus, no
allocation, one `ContextVar.get()` per call site. The singleton is never
instantiated.

```python
import asyncio
from locus.observability import run_context, get_event_bus


async def main():
    async with run_context() as rid:
        bus = get_event_bus()

        async def consumer():
            # Subscribe before or during a run — history replay delivers the last
            # 500 events on connect, then switches to live mode.
            async for event in bus.subscribe(rid):
                print(event.event_type, event.data)

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0)  # let the subscriber register
        await agent.arun("Resolve the duplicate charge on ord-4821.")
        await bus.close_stream(rid)  # delivers the end-of-stream sentinel
        await task                   # the iterator ends only after close_stream


asyncio.run(main())
```

Nothing publishes until something runs inside the context, and the
iterator ends only when you call `close_stream(rid)` — `run_context`
deliberately does not close it for you.

### The agent yield bridge

`Agent.run` is decorated with `@_bus_bridge`. When a `run_context` is
active, every `LocusEvent` the agent yields is silently republished on the
bus as a canonical `agent.*` event — no hook registration, no config flag:

| Inner event | Bus `event_type` |
|---|---|
| `ThinkEvent` | `agent.think` |
| `ToolStartEvent` | `agent.tool.started` (with `span_id`) |
| `ToolCompleteEvent` | `agent.tool.completed` (matching `span_id`) |
| `ReflectEvent` | `agent.reflect` |
| `GroundingEvent` | `agent.grounding` |
| `ModelChunkEvent` | `agent.model.chunk` (only with `run(..., stream_tokens=True)`) |
| `ModelCompleteEvent` | `agent.model.completed` + `agent.tokens.used` — mapped, but never fires: the loop does not yield `ModelCompleteEvent` (see [Events](events.md#modelcompleteevent)) |
| `InterruptEvent` | `agent.interrupt` (human-in-the-loop pause) |
| `TerminateEvent` | `agent.terminate` |

`span_id` on started/completed pairs lets consumers compute durations and
survive interleaved events from concurrent runs without subtracting timestamps.

### EventBusHook — when you can't use run_context

For non-async host code or agents you don't control at construction time:

```python
from locus.observability import EventBusHook, get_event_bus
from locus.agent import Agent
run_id = "my-run-1"
agent = Agent(
    model="{{ locus_example_model }}",
    tools=[query_metrics, get_deploy_status],
    hooks=[EventBusHook(run_id=run_id)],
)
result = agent.run_sync("Investigate the spike in checkout latency after deploy-77.")

# Read the history after the fact.
bus = get_event_bus()
for event in bus._history.get(run_id, []):
    print(event.event_type, event.data)
```

`EventBusHook` bridges every agent-lifecycle hook (`on_before_invocation`,
`on_iteration_start/end`, `on_before/after_tool_call`, `on_before/after_model_call`)
onto the bus. It never mutates events or breaks execution.

### Subscribe shapes

| Method | Scope | History replay |
|---|---|---|
| `bus.subscribe(run_id)` | One run | Yes — last 500 events, then live |
| `bus.subscribe_global()` | All runs | No — live only |
| `bus._history.get(run_id, ())` | One run | Yes — direct deque read (tests) |

### Capacity and drop accounting

The bus is bounded. When a subscriber's queue fills, the bus drops the event
for that subscriber (1 s timeout) rather than blocking the publisher. Fast
subscribers are unaffected.

```python
stats = bus.stats()
print(stats["dropped_events_total"])  # cumulative drops across all subscribers
print(stats["history_runs"])          # number of runs with retained history
print(stats["global_subscribers"])    # current global subscribers
print(stats["active_runs"])           # runs with at least one live queue
```

Default limits (all configurable at `EventBus(...)` construction time):

| Parameter | Default |
|---|---|
| `max_queue_size` | 1024 events per subscriber |
| `history_per_run` | 500 events |
| `max_runs_retained` | 200 runs (FIFO eviction — oldest run by insertion order dropped first) |

### Full event catalogue

See [SSE event catalogue](sse-events.md) for the complete wire-format
reference — all `EV_*` constants, payload fields, span discipline, and
the cost table for every subscription scenario.

---

## See also

- [Hooks](hooks.md) — both observability hooks plug into the same lifecycle as guardrails / steering / retry.
- [Events](events.md) — what gets emitted before any hook runs.
- [SSE event catalogue](sse-events.md) — full wire-format reference for every `event_type`.
- [Safety](safety.md) — PII redaction *before* logs leave the box.
