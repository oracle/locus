---
title: Observability
---

# Observability

What the agent did, how long each step took, and what it cost — two
built-in hooks plus the standard OpenTelemetry stack cover logs, traces
and metrics, and every result carries its token and cost totals. The SDK
emits OTLP (the OpenTelemetry wire protocol) under the OpenTelemetry GenAI
semantic conventions, so any OpenTelemetry backend works and its GenAI
views find the agent, model and tool spans where they expect them; the
[Export](#export) section has the recipe for OCI Application Performance
Monitoring and three open-source backends.

## When to wire what

| Need | Add |
|---|---|
| Structured per-event lines for log aggregators (Loki, Splunk) | `StructuredLoggingHook` |
| OTLP traces and metrics for dashboards (OCI APM, Grafana, Honeycomb), named per the GenAI semantic conventions | `TelemetryHook` |
| Live per-run event stream for a browser or dashboard, without an OTLP stack | `run_context()` + `bus.subscribe(run_id)`, or `EventBusHook` — see [In-process SSE](#in-process-sse-eventbus) |
| Token and cost totals on every result (a resumed `thread_id` reports the thread's running total) | nothing — `AgentResult.metrics` already has it |
| Per-run trace ID surfaced to the user (for support tickets) | telemetry hook + `current_traceparent()` from `locus.observability.propagation` |
| One trace across agents calling agents over A2A or MCP | nothing — the A2A client and server and the MCP client carry W3C `traceparent` |

## Getting started

### Structured logs

{{ locus_model_tabs }}

```python
import logging
from locus.agent import Agent
from locus.hooks.builtin import StructuredLoggingHook

agent = Agent(
    model="{{ locus_model }}",
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
[`AuditTrail`](control-layer.md) — each entry commits to the one
before it, so `verify()` fails on any edit.

### Where the audit trail lives

A hash chain proves its records agree with each other. It cannot stop
whoever holds the file from rewriting the whole chain, and it cannot see
records dropped off its own end. Two backends put the record beyond the
reach of the process that wrote it:

| Backend | What it adds | Use it when |
|---|---|---|
| `AuditTrail(path=...)` | Every record fsynced to a JSONL file before `record()` returns | Development, single-host runners, shipping to a SIEM |
| `OracleAuditTrail` | Rows in an Oracle AI Database **immutable table** (`NO DROP UNTIL n DAYS IDLE NO DELETE UNTIL n DAYS AFTER INSERT`); with `table_kind="blockchain"` the database chains each row with its own hash and `verify()` asks it to re-check them (`DBMS_BLOCKCHAIN_TABLE.VERIFY_ROWS`) | Decisions a regulator or a dispute may turn on, several processes writing one trail, a DBA-run retention policy |
| `anchor_trail(trail, store)` | The chain head (record count, hash, timestamp) as a small object in OCI Object Storage or an S3-compatible bucket; a bucket **retention rule** keeps it from being deleted or overwritten | Any trail, including the file-backed one, when truncation must be detectable; cheap enough to run every few records |

```python
from locus.control import OracleAuditTrail, OCIBucketAnchorStore, anchor_trail, verify_anchor

trail = OracleAuditTrail(
    dsn="audit_low", user="AUDIT", password=pw, wallet_location="~/wallets/audit",
    table_kind="blockchain",
)
store = OCIBucketAnchorStore(bucket_name="audit-anchors", namespace="...")


async def refund(action, perform, policy):
    result = await admit(action, perform, policy=policy, trail=trail)   # one row, committed
    anchor_trail(trail, store, every=100, require_retention_rule=True)
    return result


trail.verify()                 # re-reads the rows, recomputes the chain, asks the database
verify_anchor(trail, store)    # the chain still reaches the last anchored head
```

`OracleAuditTrail` has the same API as `AuditTrail`, so `admit(trail=...)`,
`gate_tool(trail=...)` and `governed_agent(audit_trail=...)` take it
unchanged; `record()` commits before it returns, and `arecord()` /
`averify()` run the same work on a worker thread for async code.
`require_retention_rule=True` refuses to anchor into a bucket that has no
retention rule (OCI) or Object Lock (S3), so a bucket that can be emptied
is never mistaken for an anchor; the rule itself is the bucket's
configuration. `anchor_periodically()` is the loop a server or runner
calls. The table and the rule outlive the agent on purpose: an immutable
table cannot be dropped inside its idle window, and a locked retention
rule cannot be shortened.

### Traces and metrics over OTLP

```python
from locus.hooks.builtin import TelemetryHook

agent = Agent(
    model="{{ locus_example_model }}",
    name="support-agent",
    tools=[lookup_order, issue_refund],
    hooks=[
        TelemetryHook(
            record_arguments=False,    # set True to attach tool args to spans
            record_results=False,      # set True for results (watch PII)
        ),
    ],
)
```

One run is one trace. The hook names spans and attributes per the
OpenTelemetry GenAI semantic conventions — the names published alongside
semantic-conventions 1.45, still marked *Development* upstream — so a
backend's GenAI view shows the agent, its model calls and its tool calls
without a mapping layer:

| Span | Kind | One per | Attributes |
|---|---|---|---|
| `invoke_agent <name>` (`invoke_agent` when the agent has no name) | INTERNAL | `agent.run(...)`, `arun`, `run_sync` | `gen_ai.operation.name=invoke_agent`, `gen_ai.agent.name`, `gen_ai.agent.id`, `gen_ai.request.model`, `gen_ai.conversation.id` (the `thread_id`), `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens` (run totals), `error.type` when the run failed; `locus.run_id`, `locus.iterations`, `locus.inference_calls`, `locus.tool_calls`, `locus.confidence` |
| `locus.iteration` | INTERNAL | ReAct iteration (the SDK's own span; the conventions have no iteration) | `locus.iteration` (the index), `locus.confidence`, `locus.messages` |
| `chat <model>` | CLIENT | model call | `gen_ai.operation.name=chat`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.conversation.id`, `gen_ai.usage.input_tokens` / `output_tokens` (and `gen_ai.usage.cache_read.input_tokens` / `cache_write.input_tokens` when the provider reports them), `gen_ai.response.finish_reasons`, `gen_ai.response.id` (server-stateful transports), `server.address` / `server.port` (the endpoint host), `error.type` when the call never returned |
| `execute_tool <tool>` | INTERNAL | tool call | `gen_ai.operation.name=execute_tool`, `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.type=function`, `gen_ai.tool.description`, `gen_ai.agent.name`, `gen_ai.conversation.id`, `error.type` on failure; `gen_ai.tool.call.arguments` and `gen_ai.tool.call.result` only with `record_arguments` / `record_results` |

Iterations nest under the run; model and tool spans nest under their
iteration. The root span is a child of whatever OpenTelemetry context is
current when the run starts, so a run served behind an HTTP handler, an
A2A request or an MCP call files under the caller's trace (see
[Trace context in and out](#trace-context-in-and-out)).

The `Agent` lends the hook its model and tools when it is constructed, so
`gen_ai.request.model` is the model id (`openai.gpt-4.1` for
`oci:openai.gpt-4.1`), `gen_ai.provider.name` the provider
(`oci.generative_ai`, `openai`, `anthropic`; for an OpenAI-compatible
endpoint the SDK's provider table knows by host and port — Groq, DeepSeek,
Mistral, xAI, Perplexity, or a local vLLM, Ollama or LM Studio on its
default port — that provider's value, for any other `openai`, with
`server.address` naming the host) and `gen_ai.tool.description` each
tool's docstring. `gen_ai.agent.name` is the agent's `name` (or
`agent_id`). `TelemetryHook(model=..., provider_name=..., agent_name=...)`
sets any of them by hand — a `FallbackChain`, say, has no single model
id. The `telemetry` extra is the OpenTelemetry API, SDK and OTLP
exporters:

```bash
pip install "locus-sdk[telemetry]"
```

Metrics follow the same conventions. Durations are in seconds and the
histograms carry the conventions' bucket boundaries as an advisory, which
the Python SDK honours:

| Instrument | Type | Unit | Attributes |
|---|---|---|---|
| `gen_ai.client.inference.usage.input_tokens` | counter | `{token}` | `gen_ai.operation.name=chat`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.token.modality=text` |
| `gen_ai.client.inference.usage.output_tokens` | counter | `{token}` | same |
| `gen_ai.client.inference.duration` | histogram | `s` | `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model` |
| `gen_ai.invoke_agent.duration` | histogram | `s` | `gen_ai.operation.name=invoke_agent`, `gen_ai.agent.name`, `gen_ai.request.model`, `error.type` |
| `gen_ai.invoke_agent.inference_calls` | histogram | `{inference_call}` | `gen_ai.operation.name`, `gen_ai.agent.name` |
| `gen_ai.invoke_agent.tool_calls` | histogram | `{tool_call}` | same |
| `gen_ai.execute_tool.duration` | histogram | `s` | `gen_ai.tool.name`, `gen_ai.tool.type`, `gen_ai.agent.name`, `error.type` |

Token counters record only what the provider reported, so a self-hosted
endpoint that returns no usage adds nothing rather than zeros.

#### The previous names

`TelemetryHook(semconv="locus-legacy")` emits what the SDK emitted before
the conventions: `agent.invocation`, `agent.iteration.<n>` and
`tool.<name>` spans with `locus.*` attributes, the `locus.invocations`,
`locus.iterations`, `locus.tool_calls` and `locus.tool_errors` counters,
the `locus.invocation.duration` and `locus.tool_call.duration` histograms
in milliseconds, and no model-call spans. Keep it while dashboards and
alerts built on those names move over. `semconv` takes `"genai/1.45"`
(alias `"genai"`) or `"locus-legacy"`; when the conventions change
upstream, a new `genai/<version>` value will carry the new names and the
earlier one stays selectable.

#### Checking a trace

`locus.observability.semconv.validate_spans(spans)` checks finished spans —
from the SDK's in-memory exporter, or a recorded trace — against the
conventions: one trace, one `invoke_agent` root, the kind and name each
operation requires, the required attributes present and typed, `error.type`
on every failed span, and no misspelled `gen_ai.*` key. It returns the
problems, one line each; `assert_valid_spans` raises with them. Notebook
60 runs it on the trace of its ticket reply:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from locus.observability.semconv import assert_valid_spans

exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)

result = agent.run_sync("Resolve the duplicate charge on ord-4821.")
assert_valid_spans(exporter.get_finished_spans())
```

#### Trace context in and out

Two helpers in `locus.observability.propagation` carry the W3C `traceparent`
across a process boundary; both are no-ops without OpenTelemetry installed.

- **In.** `trace_scope(headers)` runs a block under the trace a request's
  headers carry, so the `invoke_agent` span the block starts is a child of
  the caller's span. `A2AServer` does this for every request it serves and
  `LocusMCPServer` reads the pair off MCP `_meta`; a FastAPI or Flask
  handler of your own wraps the agent call the same way. When the web
  framework has already picked the headers up (FastAPI does when
  OpenTelemetry is installed), the scope leaves the framework's span in
  place and the run files under it.
- **Out.** `inject_trace_headers(headers)` adds `traceparent` for the tool
  call in flight — the hook's `execute_tool` span — else for the current
  span. `A2AClient` adds it to every request and `MCPClient` writes it into
  `_meta`; a tool that calls another service over HTTP adds it to its own
  headers.

`current_traceparent()` returns the header value for the current span, the
string to put in a support ticket or a log line so the run can be found in
the backend.

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
per-million-token prices in the SDK's model metadata. Like the token
counts, it adds up across turns when you resume a checkpointed
`thread_id`, so on a resumed thread it is the thread's total so far,
not just the latest turn's. It is `None` —
unpriced, never free — when those prices are unknown. The built-in
table prices only some models. Lookup strips the SDK's own provider
prefixes (`openai:`, `anthropic:` and the OpenAI-compatible routing
prefixes such as `vllm:` or `openrouter:`) and then needs an entry for
the slug that remains: a dated snapshot or `-latest` suffix resolves to
its base slug, but a longer id never inherits a shorter entry's price.
`oci:` is not one of those prefixes, so an OCI id is looked up whole,
and the table has no OCI entry — for the `{{ locus_example_model }}`
agent above, `cost_usd` is `None` until you register the model's list
price under its full id. Register prices for your model with
`register_metadata(ModelMetadata(model_id="{{ locus_example_model }}", ..., input_price_per_mtok=..., output_price_per_mtok=...))`
from `locus.models.metadata` before you construct the agent — prices
are read once, when the `Agent` is built. The same prices drive `max_cost_usd`: set
a budget on an unpriced model and the `Agent` constructor raises
`ValueError` rather than run a budget that could never stop anything.
For dashboards, key on `agent_id` plus the same metrics the
`TelemetryHook` already emits — no glue code needed.

## Export

The hook emits standard OpenTelemetry spans and metrics through the
global providers, so exporting is the OpenTelemetry SDK's business: set a
`TracerProvider` and a `MeterProvider` with the exporter for your backend
**before** constructing the agent. The OTLP/HTTP exporters read
`OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_EXPORTER_OTLP_HEADERS` when not
given an endpoint, and `OTEL_SERVICE_NAME` sets the Resource's
`service.name`, which is what every backend groups by:

```python
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

resource = Resource.create({"service.name": "support-agent"})
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(tracer_provider)
metrics.set_meter_provider(
    MeterProvider(
        resource=resource,
        metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())],
    )
)
```

Call `tracer_provider.shutdown()` (and the meter provider's) before the
process exits so the last batch is flushed. Nothing below adds a
dependency beyond the `telemetry` extra.

### OCI Application Performance Monitoring

An APM domain receives OTLP over HTTP on its *data upload endpoint* with a
*private data key* as the `dataKey` authorization; both are on the
domain's page in the console (Administration → Data keys). `oci_apm`
builds the two exporters from them and `install()` sets the providers:

```python
from locus.observability.exporters import oci_apm

apm = oci_apm(
    data_upload_endpoint="https://<domain>.apm-agt.<region>.oci.oraclecloud.com",
    private_data_key="<private data key>",
)
tracer_provider, meter_provider = apm.install(service_name="support-agent")

agent = Agent(model="{{ locus_example_model }}", tools=[lookup_order], hooks=[TelemetryHook()])
result = agent.run_sync("Resolve the duplicate charge on ord-4821.")

tracer_provider.shutdown()
meter_provider.shutdown()
```

Both arguments default to the `OCI_APM_DATA_UPLOAD_ENDPOINT` and
`OCI_APM_PRIVATE_DATA_KEY` environment variables, so `oci_apm().install()`
is the deployment form; `apm.span_processor()` and `apm.metric_reader()`
plug into providers you already own. The exporters post to
`<endpoint>/20200101/opentelemetry/private/v1/traces` and `/v1/metrics`
with `Authorization: dataKey <key>`. In Trace Explorer the run is the
`invoke_agent <name>` span under the service you named, with its `chat`
and `execute_tool` children; the `gen_ai.*` attributes are span
dimensions, so `show (spans) gen_ai.request.model,
sum(gen_ai.usage.input_tokens) where OperationName like 'chat%' group by
gen_ai.request.model` is the token bill by model.

[`examples/observability/oci_apm_dashboard.json`](https://github.com/oracle/locus/blob/main/examples/observability/oci_apm_dashboard.json)
is a Management Dashboard definition — agent runs and latency, tool
latency and errors, token spend by model and over time — built on those
queries. Set `compartmentId` in the file to the compartment that holds
the APM domain and import it:

```bash
oci management-dashboard dashboard import \
    --from-json file://examples/observability/oci_apm_dashboard.json
```

The same directory's README walks through the setup. A public data key
(for browser and mobile agents) is not what a server-side SDK should
carry; `oci_apm` takes private keys only.

### Open-source backends

Each of these speaks OTLP natively; the generic snippet above with the
endpoint set is the whole recipe.

**Jaeger** (traces). Jaeger v2 listens for OTLP on 4317 (gRPC) and 4318
(HTTP):

```bash
docker run --rm -p 16686:16686 -p 4318:4318 jaegertracing/jaeger:2
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OTEL_SERVICE_NAME=support-agent
```

Jaeger stores traces only; leave the `MeterProvider` out, or point the
metric exporter elsewhere.

**Grafana Tempo and Prometheus** (traces and metrics). Tempo takes OTLP on
4318; Prometheus takes OTLP metrics on its own endpoint once started with
`--web.enable-otlp-receiver`:

```bash
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://tempo:4318/v1/traces
export OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=http://prometheus:9090/api/v1/otlp/v1/metrics
export OTEL_SERVICE_NAME=support-agent
```

In Grafana, `gen_ai_client_inference_usage_input_tokens_total` (the
counter, in Prometheus naming) summed by `gen_ai_request_model` is the
token bill, and the trace view shows the `invoke_agent` tree from Tempo.

**SigNoz** (traces and metrics). Its collector takes both signals on one
OTLP endpoint:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OTEL_SERVICE_NAME=support-agent
```

Any other OTLP backend — Honeycomb, Grafana Cloud, an OpenTelemetry
Collector in front of anything — is the same two variables, plus
`OTEL_EXPORTER_OTLP_HEADERS` for its credentials.

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
| A dashboard expects `agent.invocation` / `tool.<name>` spans or `locus.*` metrics | The hook now emits the GenAI semantic conventions. `TelemetryHook(semconv="locus-legacy")` restores the previous names. |
| `chat` spans have no `gen_ai.provider.name` or `gen_ai.request.model` | The model class is one the SDK cannot place (a `FallbackChain`, a custom model). Pass `TelemetryHook(model="oci:openai.gpt-4.1")` or `provider_name=...`. |
| A run served over A2A or MCP is its own trace, not part of the caller's | The caller sent no `traceparent`. `A2AClient` and `MCPClient` send it; another client must add it to the request headers or MCP `_meta`. |
| The `invoke_agent` span sits under a `fastapi.endpoint` span | FastAPI emits its own spans when OpenTelemetry is installed; the run files under the request's span, which carries the caller's trace. Nothing to fix. |

## Source and notebooks

- [`notebook_09_agent_hooks.py`](https://github.com/oracle/locus/blob/main/examples/notebook_09_agent_hooks.py) — first hook, including logging.
- [`notebook_10_hooks_advanced.py`](https://github.com/oracle/locus/blob/main/examples/notebook_10_hooks_advanced.py) — telemetry pipelines.
- [`locus.hooks.builtin.logging`](https://github.com/oracle/locus/blob/main/src/locus/hooks/builtin/logging.py) — `LoggingHook`, `StructuredLoggingHook`.
- [`locus.hooks.builtin.telemetry`](https://github.com/oracle/locus/blob/main/src/locus/hooks/builtin/telemetry.py) — `TelemetryHook`, `NoOpTelemetryHook`.
- [`notebook_60_observability_basics.py`](https://github.com/oracle/locus/blob/main/examples/notebook_60_observability_basics.py) — the bus timeline, then the same reply as GenAI-convention spans with the in-memory exporter.
- [`locus.observability.semconv`](https://github.com/oracle/locus/blob/main/src/locus/observability/semconv.py) — the convention names, `validate_spans`.
- [`locus.observability.propagation`](https://github.com/oracle/locus/blob/main/src/locus/observability/propagation.py) — `inject_trace_headers`, `trace_scope`, `current_traceparent`.
- [`locus.observability.exporters`](https://github.com/oracle/locus/blob/main/src/locus/observability/exporters.py) — `oci_apm`.
- [`examples/observability/`](https://github.com/oracle/locus/tree/main/examples/observability) — the OCI APM dashboard and its README.

---

## In-process SSE (EventBus)

For browser streaming, real-time dashboards, or any use case where you need
to watch every event of a run without standing up an OTLP stack,
the SDK ships a zero-dependency in-process pub/sub bus.

### How it works

The diagram follows an event from the code that emits it, through the
`run_context()` gate, onto the bus and out to the subscribers that read
it.

{{ locus_diagram observability-pipeline }}

Every `emit()` call site in the SDK reads `current_run_id()` from a `ContextVar`.
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
| `CompactionEvent` | `agent.context.compacted` (stage, tokens before/after) |

`span_id` on started/completed pairs lets consumers compute durations and
survive interleaved events from concurrent runs without subtracting timestamps.

### EventBusHook — when you can't use run_context

For non-async host code or agents you don't control at construction time:

```python
import asyncio
from locus.observability import EventBusHook, get_event_bus
from locus.agent import Agent
run_id = "my-run-1"
agent = Agent(
    model="{{ locus_example_model }}",
    tools=[query_metrics, get_deploy_status],
    hooks=[EventBusHook(run_id=run_id)],
)
result = agent.run_sync("Investigate the spike in checkout latency after deploy-77.")


# Read the history after the fact: close the stream, and a late subscriber
# is replayed the run's retained events and then ends.
async def replay():
    bus = get_event_bus()
    await bus.close_stream(run_id)
    async for event in bus.subscribe(run_id):
        print(event.event_type, event.data)


asyncio.run(replay())
```

`EventBusHook` bridges every agent-lifecycle hook (`on_before_invocation`,
`on_iteration_start/end`, `on_before/after_tool_call`, `on_before/after_model_call`)
onto the bus. It never mutates events or breaks execution. `close_stream(run_id)`
marks the run finished but keeps its history, so a subscriber that connects
afterwards receives the retained events followed by the end-of-stream
sentinel; `drop_history(run_id)` forgets them.

### Subscribe shapes

| Method | Scope | History replay |
|---|---|---|
| `bus.subscribe(run_id)` | One run | Yes — last 500 events, then live |
| `bus.subscribe_global()` | All runs | No — live only |
| `bus.subscribe(run_id)` after `close_stream(run_id)` | One finished run | Yes — history, then the end-of-stream sentinel |

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
the cost table for every subscription scenario. Its
[Events and spans](sse-events.md#events-and-spans) section says, event by
event, which `TelemetryHook` span each one falls in.

---

## See also

- [Hooks](hooks.md) — both observability hooks plug into the same lifecycle as guardrails / steering / retry.
- [Events](events.md) — what gets emitted before any hook runs.
- [SSE event catalogue](sse-events.md) — full wire-format reference for every `event_type`.
- [Safety](safety.md) — PII redaction *before* logs leave the box.
