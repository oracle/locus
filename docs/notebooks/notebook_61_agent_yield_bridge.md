# Agent yield bridge

`Agent.run` is decorated with `@_bus_bridge`, so whenever a `run_context`
is open each yielded event the bridge recognises is also published on the
`EventBus` as an `agent.*` event. No hook registration or configuration
flag is required: the decorator is always present, and `emit()` returns
at once when no run id is bound, so an agent that runs outside
`run_context` publishes nothing through the bridge. The scenario is a
data-privacy review agent, whose tool calls are the part data protection
officers (DPOs) and auditors care about most: each inspection the
reviewer performs appears in the run stream as a tool span.

The bridge recognises ten `LocusEvent` classes and maps each to a
canonical `agent.*` event type (`ModelCompleteEvent` to two). The
stream-only classes in the drawing below (`ToolProgressEvent`,
`SubagentEvent`, `FinalAnswerVerificationEvent`, `ModelRetryEvent`,
`CustomEvent`) have no branch in the bridge and reach only the
`async for` caller:

| Bridged `LocusEvent` | Bus `event_type` | Note |
| --- | --- | --- |
| `ThinkEvent` | `agent.think` | |
| `ToolStartEvent` | `agent.tool.started` | shares `span_id` with `agent.tool.completed` |
| `ToolCompleteEvent` | `agent.tool.completed` | same `span_id`; carries `duration_ms` |
| `ReflectEvent` | `agent.reflect` | |
| `GroundingEvent` | `agent.grounding` | |
| `ModelChunkEvent` | `agent.model.chunk` | only with `run(..., stream_tokens=True)` |
| `ModelCompleteEvent` | `agent.model.completed` | plus `agent.tokens.used` (`scope="call"`) when `usage` is set; never yielded by `Agent.run` |
| `InterruptEvent` | `agent.interrupt` | |
| `CompactionEvent` | `agent.context.compacted` | only when a `ContextCompactor` compacts the history; stage, tokens before and after |
| `TerminateEvent` | `agent.terminate` | plus `agent.tokens.used` (`scope="run"`) with the run's cumulative usage when the provider metered it |

What this notebook covers:

- Tool-call telemetry with `span_id` pairing: `agent.tool.started` and
  `agent.tool.completed` share one `span_id`, so consumers can pair the
  two and read `duration_ms` instead of subtracting timestamps. Each
  tool span is one auditable review step: classify a field's
  sensitivity, or count the records in a dataset that hold personally
  identifiable information (PII).
- Token usage from `result.metrics`, which cost meters and budget
  enforcers on always-on privacy-review automation read after each run.
  The bus carries the same totals once per run as `agent.tokens.used`
  (`scope="run"`), bridged from the terminate event; there are no
  per-call deltas, because `Agent.run` does not yield
  `ModelCompleteEvent`.

The drawing below places this notebook's run in the wider event model: on
the left, the typed event classes, split into the ten the bridge
recognises and the five that stay on the yielded stream; at the top, the
bridge that maps each bridged class to its canonical `agent.*` name and
publishes it through `emit()` only while a run id is bound; on the right,
the bus event types grouped by prefix, of which the `agent.*` lane, where
the reviewer's tool spans land, is the only one the bridge feeds.

{{ locus_diagram event-taxonomy }}

## Key lines

The reviewer runs inside `run_context()` with a bus subscriber attached
before the first event is published, so every `agent.*` event the bridge
republishes is printed from the bus as the run proceeds, with each tool
span tagged by its `span_id`; the consumer returns on `agent.terminate`
and `close_stream(rid)` then closes the run's stream:

{{ locus_source examples/notebook_61_agent_yield_bridge.py::part1_full_lifecycle }}

Token usage comes from `result.metrics` after each `arun` call; the
totals accumulate across the two review prompts, as a budget enforcer
would accumulate them across a sequence of runs:

{{ locus_source examples/notebook_61_agent_yield_bridge.py::part2_token_meter }}

## Run it

```bash
python examples/notebook_61_agent_yield_bridge.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

To run offline regardless of what `~/.oci/config` holds:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_61_agent_yield_bridge.py
```

## See also

- [Observability](../concepts/observability.md#the-agent-yield-bridge) — the
  bridge alongside structured logs, traces and the opt-in `EventBusHook`.
- [Events](../concepts/events.md#modelcompleteevent) — why `Agent.run`
  does not yield `ModelCompleteEvent`, and where the run's cumulative
  token usage sits on the yielded stream (`TerminateEvent.usage`).
- [SSE event catalogue](../concepts/sse-events.md) — the Server-Sent Events
  (SSE) wire names and payload fields in the `agent.*` lane and the other
  bus prefixes.

## Source

??? note "Full source — `examples/notebook_61_agent_yield_bridge.py`"

    ````python
    --8<-- "examples/notebook_61_agent_yield_bridge.py"
    ````
