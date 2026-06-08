# SSE Streaming

Server-Sent Events is the simplest way to push live agent updates from a
Python backend to a browser. Tulip ships two handlers that turn the
agent event stream into SSE wire format: a buffered `SSEHandler` for
tests and a queue-based `AsyncSSEHandler` for production streaming.

What you'll learn:

- The SSE wire format and `SSEMessage`.
- Buffered vs. async handlers — when to use each.
- The right HTTP response headers
  (`create_sse_response_headers()`).
- Shaping the wire payload with a custom serializer.
- A drop-in FastAPI `StreamingResponse` example.
- Production checklist: IDs, heartbeats, reconnection, error events.

Run it:

```
.venv/bin/python examples/notebook_13_sse_streaming.py
```

This notebook exercises only the SSE plumbing — no LLM call is made, so
no provider configuration is required. Set `TULIP_MODEL_PROVIDER=mock`
if you want a uniform offline setup across notebooks.

## Source

```python
--8<-- "examples/notebook_13_sse_streaming.py"
```
