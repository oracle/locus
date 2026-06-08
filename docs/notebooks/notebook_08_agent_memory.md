# Agent Memory

Give an agent a checkpointer and every conversation turn is persisted to
a real store. Restart the process, attach a new agent to the same Redis
and the same `thread_id`, and the conversation resumes — messages,
tool history, confidence score and all.

What you'll learn:

- Building a `RedisBackend` checkpointer.
- Keying conversations with `thread_id`.
- Writing a checkpoint after every iteration so a crash mid-tool-call
  still recovers.
- Loading the saved `AgentState` and inspecting it field by field.
- Running many independent threads against a single Redis.

This notebook does not fall back to in-memory storage — Redis is the
only backend exercised here.

Run it:

```
export REDIS_URL=redis://localhost:6379/0
python examples/notebook_08_agent_memory.py
```

If that env var is unset the script prints a skip banner and exits
0 — convenient for CI. The agent's model goes through whichever
provider you configure via `TULIP_MODEL_PROVIDER` (openai / anthropic / for a live model). For offline runs set `TULIP_MODEL_PROVIDER=mock`.

## Source

```python
--8<-- "examples/notebook_08_agent_memory.py"
```
