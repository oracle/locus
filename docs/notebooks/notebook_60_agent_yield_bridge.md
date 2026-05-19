# Agent Yield Bridge

Every `Agent.run` is decorated with `@_bus_bridge` so the nine typed
events it yields get republished on the bus as `agent.*` events when a
`run_context` is open. No hook registration, no config flag — the
bridge is always there; it only fires when telemetry is active.

Event mapping::

    LocusEvent (inner stream)       →  bus event_type
    ─────────────────────────────────────────────────
    ThinkEvent                      →  agent.think
    ToolStartEvent                  →  agent.tool.started   ┐ share span_id
    ToolCompleteEvent               →  agent.tool.completed ┘
    ReflectEvent                    →  agent.reflect
    GroundingEvent                  →  agent.grounding
    ModelChunkEvent                 →  agent.model.chunk      (streaming)
    ModelCompleteEvent              →  agent.model.completed
                                    +  agent.tokens.used      (extra event)
    InterruptEvent                  →  agent.interrupt
    TerminateEvent                  →  agent.terminate

- How nine yielded `LocusEvent` types map to `agent.*` bus events.
- Tool-call telemetry with `span_id` pairing —
  `agent.tool.started` and `agent.tool.completed` share an id so
  consumers can compute durations without subtracting timestamps.
- Token usage from `result.metrics` — the recommended source for cost
  meters and budget enforcers.

Run it (OCI Generative AI is the default; auto-detected from `~/.oci/config`):

    python examples/notebook_60_agent_yield_bridge.py

Offline:

    LOCUS_MODEL_PROVIDER=mock python examples/notebook_60_agent_yield_bridge.py

## Source

```python
--8<-- "examples/notebook_60_agent_yield_bridge.py"
```
