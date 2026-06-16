# Agent with Tools

Plain Python functions, decorated with `@tool`, become things the agent
can call. The model decides when to use them; Locus runs them and feeds
the result back. This is what turns an LLM into an agent.

What you'll learn:

- Turning a Python function into a tool with `@tool`.
- Passing tools to `Agent(tools=[...])`.
- Watching `ToolStartEvent` and `ToolCompleteEvent` in the stream.
- Tools with optional arguments, default values, and structured return
  types.

Run it:

```
.venv/bin/python examples/notebook_07_agent_with_tools.py
```

Uses the OCI Generative AI default provider (canonical model id:
`openai.gpt-4.1` or `meta.llama-3.3-70b-instruct`). Set
`LOCUS_MODEL_PROVIDER=mock` for an offline run. Tool-calling also works
against OpenAI, Anthropic, and Ollama.

Prerequisite: notebook 08.

## Source

```python
--8<-- "examples/notebook_07_agent_with_tools.py"
```
