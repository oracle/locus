# Agent Streaming

`agent.run(prompt)` returns an async iterator of events. Watch the agent
think, call tools, and terminate — live, in order. This is what lets you
build progress bars, dashboards, audit logs, and SSE endpoints.

What you'll learn:

- The event types: `ThinkEvent`, `ToolStartEvent`, `ToolCompleteEvent`,
  `TerminateEvent`, plus model chunk events.
- Filtering with `isinstance(event, EventType)`.
- Building a live console UI from the stream.
- Rolling event counts into per-run metrics.
- Drawing a progress bar from `ToolCompleteEvent`.
- A pointer to `StructuredStream` for incremental Pydantic parsing.

Run it:

```
.venv/bin/python examples/notebook_11_agent_streaming.py
```

The default provider is the bundled mock model. Set `TULIP_MODEL_PROVIDER`
to openai / anthropic for a live model. For offline runs keep
`TULIP_MODEL_PROVIDER=mock`.

Prerequisite: notebook 09.

## Source

```python
--8<-- "examples/notebook_11_agent_streaming.py"
```
