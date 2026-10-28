# SSE event catalogue

Locus publishes a single canonical stream of events on its in-process
`EventBus` — an observability/telemetry stream, one event per
component-visible action (see [Observability](observability.md) for the
hooks that ship it). Every event carries a stable `event_type` string
keyed by the component that produced it (`agent.*`, `multiagent.*`,
`composition.*`, `router.*`, `rag.*`, `memory.*`, `a2a.*`, `skills.*`,
`deepagent.*`, `research.*`, `tool.*`).

Each action — every `lookup_order`, `issue_refund`, `isolate_host` —
surfaces as a timestamped, span-tied event you can render live or replay
from the in-memory history buffer.

!!! warning "Not a durable audit log"
    The `EventBus` is **in-process, single-instance, lossy, and
    bounded**: events are dropped for a subscriber whose queue stays
    saturated past a 1-second timeout, history is capped (500 events per
    run, oldest of 200 runs evicted first), and nothing is persisted. It
    is excellent for live UIs, dashboards, and forwarding to your own
    OTEL/SIEM pipeline — but it is **not** a complete or tamper-evident
    audit trail on its own. Forward events to durable storage if you need
    retention, and route security decisions through the
    [`AuditTrail`](agentic-ai-security.md) for a tamper-evident record.

This page is the **wire-format contract**. A browser renderer, the
JSON log adapter, and any downstream OTEL bridge consume from it. If you
add a new emission site, list it here.

## How emission works

Every `emit()` call site reads `current_run_id()` from a `ContextVar`
(the `router.*` events and `EventBusHook` instead publish under a run id
they are handed). When no run id is bound (no active `run_context()`,
and not inside a `Router.dispatch`), `emit()` returns immediately —
**zero allocations, zero bus instantiation**. SDK users who never bind a run id pay one contextvar read per call site.

The `run_context()` / `bus.subscribe(rid)` / `close_stream(rid)` shape
is shown once, in
[Observability — In-process SSE](observability.md#in-process-sse-eventbus).
The subscriber's iterator ends only when `close_stream(rid)` is called —
`run_context` does not close the stream for you.

## Event categories

### `agent.*` — ReAct loop

Bridged from the agent's yielded `LocusEvent` stream by
`@_bus_bridge` decorator on `Agent.run` / `_run_from_state`. Fires
for every iteration of the inner loop.

| Event | Payload | Notes |
|---|---|---|
| `agent.think` | `iteration`, `reasoning_preview`, `has_tool_calls`, `tool_call_count` | One per iteration |
| `agent.tool.started` | `tool_name`, `span_id`, `arg_keys` | `span_id` ties to `agent.tool.completed` |
| `agent.tool.completed` | `tool_name`, `span_id`, `success`, `duration_ms`, `output_preview`, `error` | |
| `agent.reflect` | `iteration`, `assessment`, `confidence_delta`, `new_confidence`, `guidance_preview` | Reflexion enabled |
| `agent.grounding` | `score`, `claims_evaluated`, `ungrounded_count`, `requires_replan` | Grounding enabled |
| `agent.model.chunk` | `content_preview`, `done`, `has_tool_calls` | Only with `run(..., stream_tokens=True)` |
| `agent.model.completed` | `content_preview`, `tool_call_count`, `stop_reason` | Not emitted by the `Agent.run` bridge (the loop never yields `ModelCompleteEvent`); `EventBusHook` publishes its own `agent.model.completed` with `stop_reason`, `content_length` |
| `agent.tokens.used` | `prompt_tokens`, `completion_tokens`, `total_tokens` | Not currently emitted: bridged only from `ModelCompleteEvent`, which the loop never yields |
| `agent.interrupt` | `interrupt_id`, `question_preview`, `options` | Human-in-the-loop (HITL) pause |
| `agent.terminate` | `reason`, `iterations_used`, `final_confidence`, `total_tool_calls`, `final_message_preview` | One per dispatch |
| `agent.model.retry` | `attempt`, `max_retries`, `delay_seconds`, `reason` | `ModelRetryHook` only |
| `agent.steering.applied` | `action`, `tool_name`, `reason` | `SteeringHook` only |
| `agent.guardrail.triggered` | `rule_name`, `action`, `location`, `description` | `GuardrailsHook` only |

`EventBusHook(run_id=...)` is a separate, opt-in publisher. It does not
go through `emit()`, and its payloads differ from the rows above:
`agent.invocation.started` (`prompt_preview`, `agent_id`,
`max_iterations`), `agent.invocation.completed` (`agent_id`, `iteration`,
`success`, `tool_calls`, `total_tokens`), `agent.iteration.started`
(`iteration`, `agent_id`), `agent.iteration.completed` (`iteration`,
`agent_id`, `tool_calls_so_far`), `agent.model.started`
(`message_count`, `tool_count`), `agent.model.completed`
(`stop_reason`, `content_length`), `agent.tool.started` (`tool_name`,
`tool_call_id`, `argument_keys`) and `agent.tool.completed`
(`tool_name`, `error`, `result_preview`). If you use it inside a
`run_context()` with the same run id, tool events arrive twice, once in
each shape.

### `multiagent.*` — orchestration shapes

Emitted natively by `Orchestrator`, `Specialist`, `Handoff` and
`StateGraph` — the emit calls sit in the shapes themselves rather than
being bridged off a yielded `LocusEvent` stream as the `agent.*` events
are. Like every `emit()` call site on this page, they are no-ops unless a
run id is bound, either by an active `run_context()` or by
`Router.dispatch` for the length of a dispatch.

| Event | Payload |
|---|---|
| `multiagent.orchestrator.routing` | `orchestrator_id`, `task_preview`, `specialist_count` |
| `multiagent.orchestrator.decision` | `orchestrator_id`, `decision`, `specialists_selected`, `reasoning` |
| `multiagent.orchestrator.specialists_invoked` | `orchestrator_id`, `specialists_invoked`, `specialists_succeeded`, `specialists_failed` |
| `multiagent.orchestrator.summary` | `orchestrator_id`, `summary_length` |
| `multiagent.specialist.started` | `specialist_id`, `specialist_type`, `task_preview` |
| `multiagent.specialist.completed` | `specialist_id`, `specialist_type`, `success`, `confidence`, `duration_ms`, `output_preview`, `error` |
| `multiagent.handoff.initiated` | `source_agent_id`, `target_agent_id`, `reason`, `context_summary` |
| `multiagent.handoff.completed` | `source_agent_id`, `target_agent_id`, `success`, `output_length` |
| `multiagent.graph.node.started` | `graph_id`, `node_id`, `iteration`, `span_id`, `parallel`, `is_resuming` |
| `multiagent.graph.node.completed` | `graph_id`, `node_id`, `span_id`, `status`, `duration_ms`, `parallel` |
| `multiagent.graph.node.routed` | `from_node`, `to_nodes`, `condition_result` |

### `composition.*` — pipelines

| Event | Payload |
|---|---|
| `composition.stage.started` | `pipeline_kind="sequential"`, `stage`, `stage_count` |
| `composition.stage.completed` | `pipeline_kind`, `stage`, `output_length`, `duration_ms`, `success` |
| `composition.fanout.started` | `agent_count`, `merge_strategy` |
| `composition.fanout.completed` | `success_count`, `error_count`, `duration_ms` |
| `composition.loop.iteration.started` | `iteration` |
| `composition.loop.iteration.completed` | `iteration`, `output_length`, `duration_ms` |
| `composition.loop.terminated` | `iterations_run`, `terminated_by` (`"condition"` \| `"max_loops"`) |

### `router.*` — goal routing

Published by `Router.dispatch` (and the compiler it drives) under the
`run_id` you pass it, or a fresh one it generates. These do not need an
enclosing `run_context()`: the dispatch binds its own run id for the
duration of the call, and calls `close_stream` on it when the dispatch
ends, so a subscriber's iterator finishes on its own.

| Event | Payload |
|---|---|
| `router.frame.extracted` | `frame` (the parsed goal frame: `primary_goal`, `secondary_goals`, `domain`, `complexity`, `risk`, `approval_required`, `requires_*` switches, `required_capabilities`, `success_criteria`) |
| `router.frame.failed` | `error` |
| `router.protocol.selected` | `protocol_id`, `protocol_description`, `primary_goal`, `is_canonical`, `cost`, `latency`, `risk_max`, `method` (`"rule_based"` \| `"single_candidate"` \| `"llm_picked"` \| `"rule_based_fallback"`), `rationale` |
| `router.protocol.no_match` | `primary_goal`, `risk`, `error` |
| `router.protocol.picker_fallback` | `primary_goal`, `risk`, `error` |
| `router.policy.verdict` | `protocol_id`, `allow`, `require_approval`, `reason`, `frame_risk` |
| `router.runnable.compiled` | `protocol_id`, `runnable_type`, `capability_count` |
| `router.runnable.executing` | `protocol_id` |
| `router.runnable.executed` | `protocol_id`, `text_length` |
| `router.runnable.failed` | `protocol_id`, `error` |

`router.policy.verdict` carries the gate's decision as two booleans:
`allow: false` is a deny, `allow: true` with `require_approval: true` is
a hold for human approval, and `allow: true` alone is an allow.

### `rag.*` — retrieval

| Event | Payload |
|---|---|
| `rag.query.started` | `query_preview`, `limit`, `store_type`, `threshold` |
| `rag.query.completed` | `hit_count`, `top_score`, `duration_ms`, `store_type` |

### `memory.*` — checkpointing + conversation management

| Event | Payload |
|---|---|
| `memory.checkpoint.saved` | `thread_id`, `iteration`, `backend`, `trigger` (`"every_n_iterations"` \| `"final"` \| `"graph_interrupt"`) |
| `memory.checkpoint.loaded` | `thread_id`, `iteration`, `backend`, `resume_node` (graph only) |
| `memory.conversation.pruned` | `strategy="sliding_window"`, `window_size`, `removed_count` |
| `memory.compactor.triggered` | `strategy="summarizing"`, `messages_before`, `threshold` |
| `memory.compactor.completed` | `strategy`, `messages_before`, `messages_after`, `summarized_count`, `duration_ms` |

### `a2a.*` — Agent-to-Agent protocol

| Event | Payload |
|---|---|
| `a2a.task.received` | `method`, `rpc_id` |
| `a2a.task.processing` | `task_id`, `agent_name` |
| `a2a.task.completed` | `method`, `success`, `error_code` (on error), `duration_ms` |
| `a2a.client.send` | `target_url`, `method` |
| `a2a.client.received` | `target_url`, `method`, `status_code`, `duration_ms`, `content_length` |

### `skills.*` — skill activation

| Event | Payload |
|---|---|
| `skills.activated` | `skill_name`, `has_resources`, `instructions_length` |

### `deepagent.*` — research-shaped agent

| Event | Payload |
|---|---|
| `deepagent.subagent.spawned` | `subagent_type`, `description_preview`, `max_iterations` |
| `deepagent.subagent.completed` | `subagent_type`, `output_length`, `duration_ms`, `success` |
| `deepagent.fs.read` | `path`, `byte_count` |
| `deepagent.fs.write` | `path`, `byte_count` |
| `deepagent.todo.added` | `content`, `status` |
| `deepagent.todo.completed` | `content`, `status` |

### `research.*` — research workflow nodes

Emitted by `create_research_workflow` / individual node primitives from
`locus.deepagent.workflow`. Like the other `emit()` sites, these are no-ops unless a run id is bound (an active `run_context()` or a `Router.dispatch`).

| Event | Payload |
|---|---|
| `research.execute.started` | `prompt_preview`, `replan` (iteration index) |
| `research.execute.completed` | `fact_count` |
| `research.causal.built` | `node_count`, `hypothesis_preview`, `confidence` |
| `research.summarize.completed` | `summary_length`, `has_structured_output` |
| `research.grounding.evaluated` | `score`, `claims_evaluated`, `ungrounded_count`, `requires_replan` |
| `research.regenerate.started` | `ungrounded_count` |
| `research.regenerate.completed` | `regeneration` (attempt index) |
| `research.replan` | `replan` (iteration), `ungrounded_count`, `prompt_preview` |
| `research.completed` | emitted by caller via `close_stream` |

### `tool.*` — sandboxed tool execution

Emitted around each run of a tool declared with `@tool(sandbox=...)`.
`tool.sandbox.denied` comes from `SandboxEnforcerHook` instead: it fires
when a tool carries a label listed in `ControlPolicy.require_sandbox_for`
but has no sandbox, and the hook then cancels that call.

| Event | Payload |
|---|---|
| `tool.sandbox.started` | `tool`, `provider`, `timeout` |
| `tool.sandbox.completed` | `tool`, `ok`, `exit_code`, `timed_out`, `duration_ms` |
| `tool.sandbox.denied` | `tool`, `labels` (the labels that required a sandbox) |

## Span discipline

Started/completed events that share a `span_id` (`agent.tool.*`,
`multiagent.graph.node.*`) let consumers compute durations without
subtracting timestamps and survive interleaved events from concurrent
runs.

## Cost when no one subscribes

| Layer | Cost |
|---|---|
| No run id bound (no `run_context`, no `Router.dispatch`) | One `ContextVar.get()` per emit site; `emit()` never instantiates the bus singleton (constructing an `EventBusHook` does, since it publishes without going through `emit()`). |
| `run_context` active, no subscriber | `bus.publish()` iterates an empty queue list, appends to per-run history (FIFO: a 500-event `deque` per run; oldest of 200 retained runs evicted first by insertion order). Memory bounded. |
| Slow subscriber | Per-event `wait_for(queue.put, timeout=1s)` drops *that* one event for *that* one slow subscriber, increments the bus drop counter (surfaced as `dropped_events_total` in `stats()`), continues for everyone else. |

## Adding a new event

1. Add a constant in `src/locus/observability/emit.py`:

   ```python
   EV_FOO_BAR = "foo.bar"
   ```

2. Emit at the call site:

   <!-- docs: skip -->
   ```python
   from locus.observability.emit import EV_FOO_BAR, emit
   await emit(EV_FOO_BAR, key1=value1, key2=value2)
   ```

3. Add the row to the table in this doc.
4. If the event is tied to a started/completed pair, generate
   `span_id = uuid4().hex[:8]` on `started` and pass it through.
