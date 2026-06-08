# Event Catalogue

Every component in Tulip emits typed events under one stable prefix:
`agent.*`, `multiagent.*`, `composition.*`, `router.*`, `rag.*`,
`memory.*`, `a2a.*`, `skills.*`, `deepagent.*`. The `EV_*` constants
in `tulip.observability.emit` are the canonical registry — change one
name and it propagates to every emission site and every consumer.

Prefix map::

    agent.*          ReAct loop (think, tool, model, tokens, reflect, …)
    multiagent.*     Orchestrator, Specialist, Handoff, StateGraph nodes
    composition.*    SequentialPipeline, ParallelPipeline, LoopAgent
    router.*         PRISM dispatch (frame → protocol → policy → compiled)
    rag.*            Retriever query lifecycle
    memory.*         Checkpointing + conversation management
    a2a.*            Agent-to-Agent protocol (server + client)
    skills.*         Skill activation
    deepagent.*      Research-shaped agent (subagents, fs, todos)

- List every `EV_*` constant and its category prefix (always in sync
  with the codebase because it's read at import time).
- Drive a `SequentialPipeline` + `LoopAgent` that surfaces
  `composition.*` events end-to-end.

Run it (defaults to the bundled mock model; set `TULIP_MODEL_PROVIDER` to `openai` / `anthropic` for a live model):

    python examples/notebook_62_event_catalogue.py

Offline:

    TULIP_MODEL_PROVIDER=mock python examples/notebook_62_event_catalogue.py

## Source

```python
--8<-- "examples/notebook_62_event_catalogue.py"
```
