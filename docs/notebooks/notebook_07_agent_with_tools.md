# Agent with Tools

Plain Python functions, decorated with `@tool`, become things the agent
can call. The model decides when to use them; Tulip runs them and feeds
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

Uses the bundled mock model by default. Set `TULIP_MODEL_PROVIDER` to
openai / anthropic for a live model; keep `TULIP_MODEL_PROVIDER=mock`
for an offline run. Tool-calling also works
against OpenAI, Anthropic.

Prerequisite: notebook 08.

## Source

```python
--8<-- "examples/notebook_07_agent_with_tools.py"
```
