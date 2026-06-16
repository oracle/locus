# Event Catalogue

Every component in Locus emits typed events under one stable prefix:
`agent.*`, `multiagent.*`, `composition.*`, `router.*`, `rag.*`,
`memory.*`, `a2a.*`, `skills.*`, `deepagent.*`. The `EV_*` constants
in `locus.observability.emit` are the canonical registry — change one
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

Run it (OCI Generative AI is the default — the notebook harness picks it when `~/.oci/config` exists; set `OCI_PROFILE` to choose a non-default profile):

    python examples/notebook_62_event_catalogue.py

Offline:

    LOCUS_MODEL_PROVIDER=mock python examples/notebook_62_event_catalogue.py

## Source

```python
--8<-- "examples/notebook_62_event_catalogue.py"
```
