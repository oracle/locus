# Basic Agent

The smallest end-to-end Locus example. Build an agent, send it a prompt
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

The default provider is OCI Generative AI — a working `~/.oci/config`
sends prompts to a live OCI model. Without one, set
`LOCUS_MODEL_PROVIDER=mock` to use the bundled deterministic model.
OpenAI, Anthropic, and Ollama are also supported.

## Source

```python
--8<-- "examples/notebook_06_basic_agent.py"
```
