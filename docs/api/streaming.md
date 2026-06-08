# Streaming

Handlers that consume the `TulipEvent` stream from `agent.run(...)`
and emit it somewhere useful — a terminal, an HTTP SSE response, a
log buffer, a downstream collector. All handlers share the
`StreamHandler` Protocol so they can be composed via
`CompositeHandler` and filtered via `FilteringHandler`.

## Base contracts

::: tulip.streaming.handler.StreamHandler
::: tulip.streaming.handler.BaseStreamHandler

## Composition

::: tulip.streaming.handler.CompositeHandler
::: tulip.streaming.handler.FilteringHandler
::: tulip.streaming.handler.BufferingHandler

## Console output

::: tulip.streaming.console.ConsoleHandler
::: tulip.streaming.console.MinimalConsoleHandler

## Server-Sent Events (SSE)

Wire-format helpers for HTTP SSE responses. Use `SSEHandler` with sync
frameworks, `AsyncSSEHandler` with FastAPI / Starlette.

::: tulip.streaming.sse.SSEHandler
::: tulip.streaming.sse.AsyncSSEHandler
::: tulip.streaming.sse.SSEMessage
::: tulip.streaming.sse.create_sse_response_headers

## Structured streaming

Stream the agent's structured output (`AgentConfig.output_schema`) as
incremental JSON tokens so the UI can render partial fields as they
arrive.

::: tulip.streaming.structured.StructuredStream
::: tulip.streaming.structured.stream_structured
