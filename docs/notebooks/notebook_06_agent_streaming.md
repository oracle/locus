# Agent Streaming

`agent.run(prompt)` returns an async iterator of events. Watch the agent
think, call tools, and terminate — live, in order. This is what lets you
build progress bars, dashboards, audit logs, and SSE endpoints.

This notebook follows a data-privacy scenario: triaging data-subject
access requests (DSARs) under GDPR Art. 15. The agent extracts PII,
checks consent, classifies requests, and files reports — all on
fabricated subjects (`*.example` domains, 555-01xx phone numbers).

What you'll learn:

- The event types: `ThinkEvent`, `ToolStartEvent`, `ToolCompleteEvent`,
  `TerminateEvent`.
- Event-level streaming by default — the assistant's text arrives in one
  piece when a step completes — and token-level output with
  `stream_tokens=True`, which adds `ModelChunkEvent`s as the model emits
  text (see `example_token_streaming` in the source below).
- Filtering with `isinstance(event, EventType)`.
- Building a live console UI from the stream.
- Rolling event counts into per-run metrics.
- Drawing a progress bar from `ToolCompleteEvent`.
- A pointer to `StructuredStream` for incremental Pydantic parsing.

Run it:

```
.venv/bin/python examples/notebook_06_agent_streaming.py
```

Model choice comes from `examples/config.py`: OCI Generative AI when
`~/.oci/config` has a usable profile, otherwise the bundled mock model, so the
example also runs offline with no credentials. Set `LOCUS_MODEL_PROVIDER`
(`oci`, `openai`, `anthropic` or `mock`) to force one.

Prerequisite: the agent-basics notebook.

## Source

````python
--8<-- "examples/notebook_06_agent_streaming.py"
````
