# SSE Streaming

Server-Sent Events is the simplest way to push live payment-review
updates from a Python backend to a payments operations dashboard. Locus
ships two handlers that turn the agent event stream into SSE wire format:
a buffered `SSEHandler` for tests and a queue-based `AsyncSSEHandler` for
production streaming.

What you'll learn:

- The SSE wire format and `SSEMessage`.
- Buffered vs. async handlers — when to use each.
- The right HTTP response headers
  (`create_sse_response_headers()`).
- Shaping the wire payload with a custom serializer.
- Every event type the handlers render, read from
  `SSEHandler.SUPPORTED_EVENTS`.
- Heartbeats that keep long batch-settlement streams alive.

Run it:

```
.venv/bin/python examples/notebook_64_sse_streaming.py
```

This notebook exercises only the SSE plumbing — no LLM call is made, so
no provider configuration is required. Set `LOCUS_MODEL_PROVIDER=mock`
if you want a uniform offline setup across notebooks.

## Source

````python
--8<-- "examples/notebook_64_sse_streaming.py"
````
