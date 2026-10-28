# Observability Basics

Locus ships an in-process pub/sub `EventBus` that publishes typed
`StreamEvent` objects for every meaningful step of execution: agent thinking,
tool calls, model completions, token usage, multi-agent fan-outs,
checkpoints — all under one canonical `event_type` per component. For a
customer-support bot that is the ticket *timeline*: an ordered,
replayable record of every action the agent took on a case, ready for a
QA review or a customer dispute.

Telemetry is opt-in. Code that never enters a `run_context` pays one
`ContextVar.get()` per emission site — no bus, no events, no
allocations.

Inside `run_context()`, each step of `agent.arun(…)` reaches the
`EventBus` as a canonical event, and `bus.subscribe(rid)` replays the
run's history before streaming live:

{{ locus_diagram run-context-events }}

What the example does:

- Run an Agent with no telemetry (the SDK-default path).
- Wrap the same ticket reply in `run_context()` and subscribe to the bus.
- The canonical events on the agent path: `agent.think`,
  `agent.tool.started/completed`, `agent.terminate`. `agent.tokens.used`
  (the dashed node) is emitted only when a `ModelCompleteEvent` that
  carries usage is bridged.
- Read the per-run history buffer after the fact (replay semantics for
  late subscribers — reconstruct the ticket timeline after the run).

Run it (OCI Generative AI with an OCI profile, otherwise the bundled mock model; set `LOCUS_MODEL_PROVIDER` to force a provider):

    python examples/notebook_60_observability_basics.py

Offline:

    LOCUS_MODEL_PROVIDER=mock python examples/notebook_60_observability_basics.py

## Source

````python
--8<-- "examples/notebook_60_observability_basics.py"
````
