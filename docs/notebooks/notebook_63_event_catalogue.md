# Event catalogue

When a Payment Card Industry Data Security Standard (PCI DSS) assessor
asks "what can your payment agents do, and how would you know they did
it?", the answer is the event catalogue, generated from the code itself
rather than from a spreadsheet that drifts out of date. SDK components
emit typed events under a stable prefix per subsystem, and most of the
names are declared as `EV_*` constants in `locus.observability.emit`, so
changing a constant propagates to every emission site that imports it.
The constants span twelve prefixes: `agent.*`, `multiagent.*`,
`composition.*`, `rag.*`, `memory.*`, `a2a.*`, `skills.*`,
`deepagent.*`, `research.*`, `model.*`, `tool.*` and `harness.*`.

Prefix map, with the component that emits each family:

| Prefix | Emitted by |
| --- | --- |
| `agent.*` | ReAct loop, bridged from the events `Agent.run()` yields (think, tool, reflect, grounding, interrupt, compaction, terminate, model chunks with `stream_tokens=True`), and the built-in retry, steering and guardrail hooks |
| `multiagent.*` | Orchestrator, Specialist, Handoff, StateGraph nodes |
| `composition.*` | SequentialPipeline, ParallelPipeline, LoopAgent |
| `rag.*` | Retriever query lifecycle |
| `memory.*` | Checkpoint saves and loads, conversation pruning and compaction, long-term memory managers |
| `a2a.*` | Agent-to-Agent (A2A) protocol server and client |
| `skills.*` | Skill activation |
| `deepagent.*` | Research-shaped agent (subagents, file system, todos) |
| `research.*` | Research workflow nodes in `locus.deepagent.workflow` |
| `model.*` | `FallbackChain` tier fallback and circuit breaker |
| `tool.*` | Sandboxed `@tool` runs and the `SandboxEnforcerHook` denial |
| `harness.*` | Coding harness (`locus.harness`) command execution records |

Two qualifications matter when the catalogue is handed to an assessor.
The `router.*` events are not in the registry: the helpers in
`locus.observability.router_events` publish them as string literals
under the run id that `Router.dispatch` binds, so they do not appear in
the printed list. And the registry is neither a superset nor a subset of
what the code publishes. Seven names (`multiagent.graph.node.routed`,
`a2a.task.processing`, `rag.embedding.generated`, `rag.store.upsert`,
`rag.store.search`, `memory.conversation.added` and
`research.completed`) are declared but never emitted, and
`agent.tokens.used` is emitted only for a `ModelCompleteEvent` that
`Agent.run()` never yields. In the other direction, `EventBusHook` and
the long-term memory managers publish their names as string literals,
several of them (`agent.invocation.*`, `agent.iteration.*`,
`agent.model.started`, `memory.manager.extract_failed`) with no
constant at all. The catalogue is the naming contract the code
declares, not a record of what a run emitted.

What this notebook covers:

- List every `EV_*` constant and its category prefix, read from the
  installed `locus.observability.emit` module when the script runs.
- Drive a two-stage dispute-triage `SequentialPipeline` that surfaces
  `composition.*` events end-to-end.

The right-hand side of this drawing shows representative names from the
catalogue the script prints in full: the bus `event_type` names grouped
by prefix. The `composition.*` lane that the dispute-triage pipeline
exercises is published by `SequentialPipeline` straight through
`emit()`, not through the bridge that feeds the `agent.*` lane from
`Agent.run()`:

{{ locus_diagram event-taxonomy }}

## Key lines

Part 1 walks `locus.observability.emit` when the script runs, keeps
every `EV_*` string constant and groups the values by their first dotted
segment, so the catalogue is read from the installed code rather than
maintained by hand:

{{ locus_source examples/notebook_63_event_catalogue.py::part1_catalogue_tour }}

Part 2 chains two single-iteration agents in a `SequentialPipeline`,
subscribes to the event bus under the run id from `run_context()` before
the pipeline starts, and collects the `composition.*` events until the
second stage reports completion; the `stage` field is zero-based, so the
second stage is `stage == 1`:

{{ locus_source examples/notebook_63_event_catalogue.py::part2_composition }}

## Run it

```bash
python examples/notebook_63_event_catalogue.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

To run offline regardless of the local OCI configuration:

```bash
LOCUS_MODEL_PROVIDER=mock python examples/notebook_63_event_catalogue.py
```

## Output

Running it offline — no credentials, bundled mock model — prints the canonical `event_type` catalogue grouped by prefix (every `EV_*` constant read from the installed `locus.observability.emit` module when the script runs) and then the `composition.*` events the two-stage dispute-triage pipeline publishes, with no model text in the trail:

```text

--- Part 1: canonical event_type catalogue ---
  a2a.*  (5 events)
    - a2a.client.received
    - a2a.client.send
    - a2a.task.completed
    - a2a.task.processing
    - a2a.task.received
  agent.*  (14 events)
    - agent.context.compacted
    - agent.grounding
    - agent.guardrail.triggered
    - agent.interrupt
    - agent.model.chunk
    - agent.model.completed
    - agent.model.retry
    - agent.reflect
    - agent.steering.applied
    - agent.terminate
    - agent.think
    - agent.tokens.used
    - agent.tool.completed
    - agent.tool.started
  composition.*  (7 events)
    - composition.fanout.completed
    - composition.fanout.started
    - composition.loop.iteration.completed
    - composition.loop.iteration.started
    - composition.loop.terminated
    - composition.stage.completed
    - composition.stage.started
  deepagent.*  (6 events)
    - deepagent.fs.read
    - deepagent.fs.write
    - deepagent.subagent.completed
    - deepagent.subagent.spawned
    - deepagent.todo.added
    - deepagent.todo.completed
  harness.*  (1 events)
    - harness.exec
  memory.*  (8 events)
    - memory.checkpoint.loaded
    - memory.checkpoint.saved
    - memory.compactor.completed
    - memory.compactor.triggered
    - memory.conversation.added
    - memory.conversation.pruned
    - memory.manager.extracted
    - memory.manager.injected
  model.*  (2 events)
    - model.fallback
    - model.fallback.breaker
  multiagent.*  (11 events)
    - multiagent.graph.node.completed
    - multiagent.graph.node.routed
    - multiagent.graph.node.started
    - multiagent.handoff.completed
    - multiagent.handoff.initiated
    - multiagent.orchestrator.decision
    - multiagent.orchestrator.routing
    - multiagent.orchestrator.specialists_invoked
    - multiagent.orchestrator.summary
    - multiagent.specialist.completed
    - multiagent.specialist.started
  rag.*  (5 events)
    - rag.embedding.generated
    - rag.query.completed
    - rag.query.started
    - rag.store.search
    - rag.store.upsert
  research.*  (9 events)
    - research.causal.built
    - research.completed
    - research.execute.completed
    - research.execute.started
    - research.grounding.evaluated
    - research.regenerate.completed
    - research.regenerate.started
    - research.replan
    - research.summarize.completed
  skills.*  (1 events)
    - skills.activated
  tool.*  (3 events)
    - tool.sandbox.completed
    - tool.sandbox.denied
    - tool.sandbox.started

--- Part 2: composition.* events ---
composition events seen so far: ['composition.stage.started', 'composition.stage.completed', 'composition.stage.started', 'composition.stage.completed']
```
<!-- notebook-output:end -->

## See also

- [SSE event catalogue](../concepts/sse-events.md) — the Server-Sent
  Events (SSE) wire-format contract behind these names: payload fields,
  span discipline and the cost when no one subscribes.
- [EventBus subscribers](notebook_62_eventbus_subscribers.md) — the
  per-run and global subscribe shapes behind the consumer in Part 2.

## Source

??? note "Full source — `examples/notebook_63_event_catalogue.py`"

    ````python
    --8<-- "examples/notebook_63_event_catalogue.py"
    ````
