# Agent Yield Bridge

Every event an agent yields is republished on the bus automatically as
an `agent.*` event whenever a `run_context` is open. No hook
registration or configuration flag is required. Internally, `Agent.run`
is decorated with `@_bus_bridge`; the bridge is always there and only
fires when telemetry is active. The scenario is a data-privacy review
agent, whose tool calls are the part data protection officers (DPOs)
and auditors care about most: every dataset the reviewer inspects shows
up in the run stream.

The bridge maps each yielded event to one bus event type:

| Yielded `LocusEvent` | Bus `event_type` | Note |
|---|---|---|
| `ThinkEvent` | `agent.think` | |
| `ToolStartEvent` | `agent.tool.started` | shares `span_id` with `agent.tool.completed` |
| `ToolCompleteEvent` | `agent.tool.completed` | same `span_id`; carries `duration_ms` |
| `ReflectEvent` | `agent.reflect` | |
| `GroundingEvent` | `agent.grounding` | |
| `ModelChunkEvent` | `agent.model.chunk` | streaming |
| `ModelCompleteEvent` | `agent.model.completed` | also emits `agent.tokens.used` |
| `InterruptEvent` | `agent.interrupt` | |
| `TerminateEvent` | `agent.terminate` | |

- Tool-call telemetry with `span_id` pairing —
  `agent.tool.started` and `agent.tool.completed` share an id so
  consumers can compute durations without subtracting timestamps. Each
  tool span is one auditable review step (classify a field's
  sensitivity, count PII records in a dataset).
- Token usage from `result.metrics` — the canonical source for cost
  meters and budget enforcers on always-on privacy-review automation.

Run it (OCI Generative AI with an OCI profile, otherwise the bundled mock model; set `LOCUS_MODEL_PROVIDER` to force a provider):

    python examples/notebook_61_agent_yield_bridge.py

Offline:

    LOCUS_MODEL_PROVIDER=mock python examples/notebook_61_agent_yield_bridge.py

## Source

````python
--8<-- "examples/notebook_61_agent_yield_bridge.py"
````
