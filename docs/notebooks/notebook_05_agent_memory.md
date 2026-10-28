# Agent Memory

Give an agent a checkpointer and every conversation turn is saved under
its `thread_id`. Call the agent again with that `thread_id` — or attach a
new agent to the same store with the same `thread_id` — and the
conversation resumes: messages,
tool history, confidence score and all. With a durable backend, that
holds across a process restart too.

What you'll learn:

- Building a `MemoryCheckpointer` and passing it to the agent.
- Keying conversations with `thread_id` (one per conversation).
- Writing a checkpoint after every iteration
  (`checkpoint_every_n_iterations=1`), not only when a run finishes.
- A second agent attaching to the same store and picking up the same
  `thread_id`.
- Loading the saved `AgentState` and inspecting it field by field.
- Running many independent threads against a single store.

This notebook uses `MemoryCheckpointer`, an in-memory store, so it runs
with no setup; its state lasts only as long as the process. For a
conversation that survives a restart, swap in a durable checkpointer —
`FileCheckpointer`, or `redis_checkpointer()` /
`postgresql_checkpointer()` — see [Checkpointers](../concepts/checkpointers.md).

Run it:

```
python examples/notebook_05_agent_memory.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

## Source

````python
--8<-- "examples/notebook_05_agent_memory.py"
````
