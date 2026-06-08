# SSE event catalogue

Tulip publishes a single
canonical stream of events on its in-process `EventBus` (see
[Observability](observability.md)). Every event
carries a stable `event_type` string keyed by the component that
produced it (`agent.*`, `multiagent.*`, `composition.*`, `router.*`,
`rag.*`, `memory.*`, `a2a.*`, `skills.*`, `deepagent.*`).

This page is the **wire-format contract**. The workbench renderer,
the JSON log adapter, and any downstream OTEL bridge consume from it.
If you add a new emission site, list it here.

## How emission works

Every emission site reads `current_run_id()` from a `ContextVar`.
When no `run_context()` is active the emit returns immediately —
**zero allocations, zero bus instantiation**. SDK users who don't
subscribe pay one contextvar read per call site.

```python
from tulip.observability import run_context, get_event_bus

async with run_context() as rid:
    # Subscribe — replays history then live events.
    async for event in get_event_bus().subscribe(rid):
        print(event.event_type, event.data)
```

## Event categories

### `agent.*` — ReAct loop

Bridged from the agent's yielded `TulipEvent` stream by
`@_bus_bridge` decorator on `Agent.run` / `_run_from_state`. Fires
for every iteration of the inner loop.

| Event | Payload | Notes |
|---|---|---|
| `agent.think` | `iteration`, `reasoning_preview`, `has_tool_calls`, `tool_call_count` | One per iteration |
| `agent.tool.started` | `tool_name`, `span_id`, `arg_keys` | `span_id` ties to `agent.tool.completed` |
| `agent.tool.completed` | `tool_name`, `span_id`, `success`, `duration_ms`, `output_preview`, `error` | |
| `agent.reflect` | `iteration`, `assessment`, `confidence_delta`, `new_confidence`, `guidance_preview` | Reflexion enabled |
| `agent.grounding` | `score`, `claims_evaluated`, `ungrounded_count`, `requires_replan` | Grounding enabled |
| `agent.model.chunk` | `content_preview`, `done`, `has_tool_calls` | Streaming only |
| `agent.model.completed` | `content_preview`, `tool_call_count`, `stop_reason` | Per LLM call |
| `agent.tokens.used` | `prompt_tokens`, `completion_tokens`, `total_tokens` | Per LLM call (cost dashboards) |
| `agent.interrupt` | `interrupt_id`, `question_preview`, `options` | HITL pause |
| `agent.terminate` | `reason`, `iterations_used`, `final_confidence`, `total_tool_calls`, `final_message_preview` | One per dispatch |
| `agent.model.retry` | `attempt`, `max_retries`, `delay_seconds`, `reason` | `ModelRetryHook` only |
| `agent.steering.applied` | `action`, `tool_name`, `reason` | `SteeringHook` only |
| `agent.guardrail.triggered` | `rule_name`, `action`, `location`, `description` | `GuardrailsHook` only |

### `multiagent.*` — orchestration shapes

Emitted natively by `Orchestrator`, `Specialist`, `Handoff`,
`StateGraph` — telemetry is not opt-in here, every multi-agent run
emits.

| Event | Payload |
|---|---|
| `multiagent.orchestrator.routing` | `orchestrator_id`, `task_preview`, `specialist_count` |
| `multiagent.orchestrator.decision` | `orchestrator_id`, `decision`, `specialists_selected`, `reasoning` |
| `multiagent.orchestrator.specialists_invoked` | `orchestrator_id`, `specialists_invoked`, `specialists_succeeded`, `specialists_failed` |
| `multiagent.orchestrator.summary` | `orchestrator_id`, `summary_length` |
| `multiagent.specialist.started` | `specialist_id`, `specialist_type`, `task_preview` |
| `multiagent.specialist.completed` | `specialist_id`, `specialist_type`, `success`, `confidence`, `duration_ms`, `output_length`, `error` |
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

### `router.*` — PRISM dispatch

| Event | Payload |
|---|---|
| `router.frame.extracted` | `primary_goal`, `secondary_goals`, `domain`, `complexity`, `risk`, `requires_*`, `success_criteria` |
| `router.frame.failed` | `error` |
| `router.protocol.selected` | `protocol_id`, `cost`, `latency`, `reason` |
| `router.protocol.no_match` | `frame_summary` |
| `router.policy.verdict` | `allow`, `require_approval`, `reason` |
| `router.runnable.compiled` | `protocol_id` |
| `router.runnable.executing` | `protocol_id` |
| `router.runnable.executed` | `protocol_id`, `output_length` |
| `router.runnable.failed` | `protocol_id`, `error` |

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
`tulip.deepagent.workflow`. Requires an active `run_context()`.

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

## Span discipline

Started/completed events that share a `span_id` (`agent.tool.*`,
`multiagent.graph.node.*`) let consumers compute durations without
subtracting timestamps and survive interleaved events from concurrent
runs.

## Cost when no one subscribes

| Layer | Cost |
|---|---|
| No `run_context` active | One `ContextVar.get()` per emit site. Bus singleton never instantiated. |
| `run_context` active, no subscriber | `bus.publish()` iterates an empty queue list, appends to per-run history (LRU cap 200 runs × 500 events). Memory bounded. |
| Slow subscriber | Per-event `wait_for(queue.put, timeout=1s)` drops *that* one event for *that* one slow subscriber, increments `bus._dropped_events`, continues for everyone else. |

## Adding a new event

1. Add a constant in `src/tulip/observability/emit.py`:

   ```python
   EV_FOO_BAR = "foo.bar"
   ```

2. Emit at the call site:

   ```python
   from tulip.observability.emit import EV_FOO_BAR, emit
   await emit(EV_FOO_BAR, key1=value1, key2=value2)
   ```

3. Add the row to the table in this doc.
4. If the event is tied to a started/completed pair, generate
   `span_id = uuid4().hex[:8]` on `started` and pass it through.
