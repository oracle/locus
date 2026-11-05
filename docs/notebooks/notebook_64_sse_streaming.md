# SSE streaming

Server-Sent Events (SSE) push live payment-review updates from a Python
backend to a payments operations dashboard over a single HTTP response, one
frame per event. The SDK ships two handlers that turn the agent event stream
into SSE wire format: a buffered `SSEHandler` for short runs and tests, and a
queue-based `AsyncSSEHandler` for production streaming.

What this notebook covers:

- The SSE wire format and `SSEMessage`.
- Buffered and queue-based handlers, and when to use each.
- The HTTP response headers an SSE route needs
  (`create_sse_response_headers()`).
- Shaping the wire payload with a custom serializer.
- The event types `SSEHandler.SUPPORTED_EVENTS` declares, and how that set
  relates to what the handlers render.
- A hand-built `heartbeat` frame for long batch-settlement streams.

The drawing shows where this notebook's two handlers sit. `agent.run(...)`
yields the loop events of a payment review (and one `ModelChunkEvent` per
chunk only with `stream_tokens=True`); the consumer's own `async for` loop
hands each event to a handler through `on_event`, then `on_complete`, or
`on_error` if the run raised. `SSEHandler` or `AsyncSSEHandler` turns every
event into one `SSEMessage` frame whose `format()` is the wire text the
operations dashboard reads; this notebook exercises that last step with
hand-built events in place of a live agent run.

{{ locus_diagram streaming-paths }}

## Key lines

`AsyncSSEHandler` backs the stream with an `asyncio.Queue`, so the producer
and the consumer run concurrently; the producer side is the same pair of
calls any consumer of `agent.run(...)` makes, `on_event` per event and
`on_complete` at the end, here with stand-in events for a payment review:

{{ locus_source examples/notebook_64_sse_streaming.py::main.produce_events }}

The consumer iterates `AsyncSSEHandler.stream()`, an async iterator of
formatted SSE frames that a FastAPI or Starlette `StreamingResponse` can
yield as they arrive; it ends after the `done` frame that `on_complete`
enqueues:

{{ locus_source examples/notebook_64_sse_streaming.py::main.consume_events }}

A `custom_serializer` passed to `SSEHandler` (the only one of the two
handlers that accepts one) decides what the dashboard sees in each `data:`
line; this one trims the payload to the event type, a timestamp and one
content field:

{{ locus_source examples/notebook_64_sse_streaming.py::main.custom_serializer }}

The buffered `SSEHandler` with `format_all()` and `pop_messages()`, the
`error` frame that `on_error` emits, `create_sse_response_headers()` and the
`SSEHandler.SUPPORTED_EVENTS` listing all sit in `main()` in the full source
below. Three details of the handlers matter before you wire one into a
route:

- `on_error` keeps the exception text off the wire. The `error` frame
  carries `"internal error"` and a `correlation_id`; the exception text
  reaches the frame only when `LocusSettings.debug` is true, and is
  otherwise logged server-side under the same id.
- `SUPPORTED_EVENTS` is a declared set, not a filter. `on_event` serializes
  any `LocusEvent` it receives, and the set lists 17 of the 24 event types in
  `locus.core.events`: `tool_progress`, `subagent`, `interrupt`,
  `compaction`, `model_retry`, `final_answer_verification` and `custom` are
  absent. Read the Part 9 listing as the declared subset, not as the
  catalogue.
- `include_timestamp` is accepted and stored by both constructors but never
  consulted by either handler; the `timestamp` in each payload is the event's
  own field, carried through `model_dump()`.

## Run it

```bash
python examples/notebook_64_sse_streaming.py
```

The script does not call a model: it imports nothing from
`examples/config.py`, needs no credentials and makes no network calls, so it
runs the same way with or without a usable `~/.oci/config` profile on the
machine. Part 5 also writes an `SSE stream error` entry to stderr: that is
the server-side log record the generic `error` frame points to.

## See also

- [Streaming](../concepts/streaming.md) — consuming the event stream; its
  [SSE over HTTP](../concepts/streaming.md#sse-over-http-for-browser-uis)
  section describes the frames `AgentServer` writes.
- [Events](../concepts/events.md) — every event class and its fields.
- [Streaming events](notebook_06_agent_streaming.md) — the live
  `agent.run(...)` stream these handlers consume.
- [Agent server](notebook_65_agent_server.md) — the ready-made `/stream`
  route, which writes its own `data:` frames and does not use these
  handlers.

## Source

??? note "Full source — `examples/notebook_64_sse_streaming.py`"

    ````python
    --8<-- "examples/notebook_64_sse_streaming.py"
    ````
