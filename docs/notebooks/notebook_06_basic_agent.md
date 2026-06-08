# Basic Agent

The smallest end-to-end Tulip example. Build an agent, send it a prompt
two ways (blocking and streaming), and look at what comes back.

What you'll learn:

- How an `Agent` pairs a model with a system prompt.
- The difference between `agent.run_sync(...)` (one result) and
  `agent.run(...)` (an async stream of events).
- The fields on `AgentResult`: `message`, `success`, `stop_reason`,
  `metrics`.
- Reusing the same agent across multiple prompts.

Run it:

```
.venv/bin/python examples/notebook_06_basic_agent.py
```

The default provider is the bundled deterministic mock model. Set
`TULIP_MODEL_PROVIDER=openai` (or anthropic ) and the matching
credentials to send prompts to a live model.

## Source

```python
--8<-- "examples/notebook_06_basic_agent.py"
```
