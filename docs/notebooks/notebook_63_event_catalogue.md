# Event Catalogue

When a payment-card compliance (PCI DSS) assessor asks "what can your payment agents do, and how
would you know they did it?", the answer is the event catalogue — and it
is generated from the code itself rather than a spreadsheet that drifts
out of date. Every component in Locus emits typed events under one stable prefix:
`agent.*`, `multiagent.*`, `composition.*`, `router.*`, `rag.*`,
`memory.*`, `a2a.*`, `skills.*`, `deepagent.*`. Most of these names are
defined as `EV_*` constants in `locus.observability.emit`, so changing a
constant propagates to every emission site that imports it. The
`router.*` events are the exception: they are emitted as string literals
from `locus.observability.router_events` rather than `EV_ROUTER_*`
constants, so they are not yet part of the shared `EV_*` registry.

Prefix map::

    agent.*          ReAct loop (think, tool, model, tokens, reflect, …)
    multiagent.*     Orchestrator, Specialist, Handoff, StateGraph nodes
    composition.*    SequentialPipeline, ParallelPipeline, LoopAgent
    rag.*            Retriever query lifecycle
    memory.*         Checkpointing + conversation management
    a2a.*            Agent-to-Agent protocol (server + client)
    skills.*         Skill activation
    deepagent.*      Research-shaped agent (subagents, fs, todos)

- List every `EV_*` constant and its category prefix (read at import
  time from `locus.observability.emit`).
- Drive a two-stage dispute-triage `SequentialPipeline`
  that surfaces `composition.*` events end-to-end.

Run it (defaults to the bundled mock model; set `LOCUS_MODEL_PROVIDER` to `openai` / `anthropic` for a live model):

    python examples/notebook_63_event_catalogue.py

Offline:

    LOCUS_MODEL_PROVIDER=mock python examples/notebook_63_event_catalogue.py

## Source

````python
--8<-- "examples/notebook_63_event_catalogue.py"
````
