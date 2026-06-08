# Observability

What the agent did, how long each step took, and what it cost — two
built-in hooks plus the standard OpenTelemetry stack cover every piece
you need. No vendor lock-in: Tulip emits OTLP, you point it at whatever backend you run.

## When to wire what

| Need | Add |
|---|---|
| Structured per-event lines for log aggregators (Loki, Splunk) | `StructuredLoggingHook` |
| OTLP traces and metrics for dashboards (Grafana, Honeycomb) | `TelemetryHook` |
| Per-run token totals on every result | nothing — `AgentResult.metrics` already has it |
| Per-run trace ID surfaced to the user (for support tickets) | telemetry hook + log the active span's trace ID |

## Getting started

### Structured logs

```python
import logging
from tulip.agent import Agent
from tulip.hooks.builtin import StructuredLoggingHook

agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search, summarise],
    hooks=[StructuredLoggingHook(level=logging.INFO)],
)
```

Every event in the run is emitted as a structured JSON line.
Sample (`ToolCompleteEvent`):

```json
{
  "ts": "2026-05-02T01:31:02Z",
  "thread_id": "th-001",
  "run_id": "run-9c14b1",
  "agent_id": "procurement",
  "event": "tool_complete",
  "tool": "search_vendors",
  "duration_ms": 412,
  "result_size": 2148
}
```

Pipe stdout to your log aggregator. The SDK doesn't own the transport —
you choose between stdlib `logging`, `structlog`, or
`opentelemetry-logs`.

### Traces and metrics over OTLP

```python
from tulip.hooks.builtin import TelemetryHook

agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search, summarise],
    hooks=[
        TelemetryHook(
            service_name="procurement-agent",
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
| `tulip.invocations` | Calls to `agent.run(...)` |
| `tulip.iterations` | ReAct iterations across all runs |
| `tulip.tool_calls` | Tool invocations |
| `tulip.tool_errors` | Tool calls that raised |

| Histogram | What it measures |
|---|---|
| `tulip.invocation.duration` | Wall-clock per `agent.run(...)` |
| `tulip.tool_call.duration` | Wall-clock per tool body |

Configure the exporter the standard OpenTelemetry way — set
`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_RESOURCE_ATTRIBUTES`, etc.
before constructing the agent. Anything OTLP works: Honeycomb, Tempo,
Grafana Cloud.

Install the optional extra:

```bash
pip install "tulip-agents[telemetry]"
```

### Token cost — already on every result

```python
result = agent.run_sync("Plan Q3 launch.")
print(f"prompt:     {result.metrics.prompt_tokens}")
print(f"completion: {result.metrics.completion_tokens}")
print(f"total:      {result.metrics.total_tokens}")
print(f"iterations: {result.metrics.iterations}")
```

Multiply by your provider's per-token rate to get a per-run cost.
For dashboards, key on `agent_id` plus the same metrics the
`TelemetryHook` already emits — no glue code needed.

## PII and tool arguments

`record_arguments=True` and `record_results=True` are off by default
because tool args and results often contain user input — emails,
account numbers, free-text. Turn them on selectively, and only after
you've verified your tracing backend has appropriate retention and
access controls. For PII redaction *inside* the agent before
anything leaves, see [Safety](safety.md).

## Common gotchas

| Symptom | Likely cause |
|---|---|
| `TelemetryHook` raises `ImportError` | `pip install "tulip-agents[telemetry]"` to get the OpenTelemetry SDK. |
| No spans show up in your backend | Exporter not configured. Set `OTEL_EXPORTER_OTLP_ENDPOINT` (and `OTEL_EXPORTER_OTLP_HEADERS` if your backend needs auth) *before* creating the agent. |
| Spans land but metrics don't | Some OTLP receivers reject metrics on the trace endpoint. Set `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` separately if needed. |
| Token totals are zero | The provider isn't returning usage in the response (some self-hosted endpoints). The SDK's loop can't make up the numbers. |
| Tool args land in your logs unintentionally | Either `record_arguments=True` or your structured logger is dumping the full event dict. Configure either explicitly. |

## Source and notebooks

- [`notebook_12_agent_hooks.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_12_agent_hooks.py) — first hook, including logging.
- [`notebook_14_hooks_advanced.py`](https://github.com/tuliplabs-ai/sdk-python/blob/main/examples/notebook_14_hooks_advanced.py) — telemetry pipelines.
- [`tulip.hooks.builtin.logging`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/hooks/builtin/logging.py) — `LoggingHook`, `StructuredLoggingHook`.
- [`tulip.hooks.builtin.telemetry`](https://github.com/tuliplabs-ai/sdk-python/blob/main/src/tulip/hooks/builtin/telemetry.py) — `TelemetryHook`, `NoOpTelemetryHook`.

---

## In-process SSE (EventBus)

For workbench streaming, real-time dashboards, or any use case where you need
to watch the full inner cognition of a run without standing up an OTLP stack,
the SDK ships a zero-dependency in-process pub/sub bus.

### How it works

Every emission site in the SDK reads `current_run_id()` from a `ContextVar`.
When no `run_context()` is active the emit returns immediately — no bus, no
allocation, one `ContextVar.get()` per call site. The singleton is never
instantiated.

```python
from tulip.observability import run_context, get_event_bus

async with run_context() as rid:
    # Subscribe before or during a run — history replay delivers the last
    # 500 events on connect, then switches to live mode.
    async for event in get_event_bus().subscribe(rid):
        print(event.event_type, event.data)
```

### The agent yield bridge

`Agent.run` is decorated with `@_bus_bridge`. When a `run_context` is
active, every `TulipEvent` the agent yields is silently republished on the
bus as a canonical `agent.*` event — no hook registration, no config flag:

| Inner event | Bus `event_type` |
|---|---|
| `ThinkEvent` | `agent.think` |
| `ToolStartEvent` | `agent.tool.started` (with `span_id`) |
| `ToolCompleteEvent` | `agent.tool.completed` (matching `span_id`) |
| `ReflectEvent` | `agent.reflect` |
| `GroundingEvent` | `agent.grounding` |
| `ModelCompleteEvent` | `agent.model.completed` + `agent.tokens.used` |
| `TerminateEvent` | `agent.terminate` |

`span_id` on started/completed pairs lets consumers compute durations and
survive interleaved events from concurrent runs without subtracting timestamps.

### EventBusHook — when you can't use run_context

For non-async host code or agents you don't control at construction time:

```python
from tulip.observability import EventBusHook, get_event_bus
from tulip.agent import Agent
run_id = "my-run-1"
agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search, summarise],
    hooks=[EventBusHook(run_id=run_id)],
)
result = agent.run_sync("Diagnose the checkout slowdown.")

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
print(stats["dropped_events"])    # cumulative drops across all subscribers
print(stats["retained_runs"])     # number of runs with live history
print(stats["subscriber_count"])  # current active subscribers
```

Default limits (all configurable at `EventBus(...)` construction time):

| Parameter | Default |
|---|---|
| `max_queue_size` | 1024 events per subscriber |
| `history_per_run` | 500 events |
| `max_runs_retained` | 200 runs (LRU eviction) |

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
