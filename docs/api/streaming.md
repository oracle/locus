# Streaming

Handlers that consume the `LocusEvent` stream from `agent.run(...)`
and emit it somewhere useful — a terminal, an HTTP SSE response, a
log buffer, a downstream collector. All handlers share the
`StreamHandler` Protocol so they can be composed via
`CompositeHandler` and filtered via `FilteringHandler`.

## Base contracts

::: locus.streaming.handler.StreamHandler
::: locus.streaming.handler.BaseStreamHandler

## Composition

::: locus.streaming.handler.CompositeHandler
::: locus.streaming.handler.FilteringHandler
::: locus.streaming.handler.BufferingHandler

## Console output

::: locus.streaming.console.ConsoleHandler
::: locus.streaming.console.MinimalConsoleHandler

## Server-Sent Events (SSE)

Wire-format helpers for HTTP SSE responses. Use `SSEHandler` with sync
frameworks, `AsyncSSEHandler` with FastAPI / Starlette.

::: locus.streaming.sse.SSEHandler
::: locus.streaming.sse.AsyncSSEHandler
::: locus.streaming.sse.SSEMessage
::: locus.streaming.sse.create_sse_response_headers

## Structured streaming

Stream the agent's structured output (`AgentConfig.output_schema`) as
incremental JSON tokens so the UI can render partial fields as they
arrive.

::: locus.streaming.structured.StructuredStream
::: locus.streaming.structured.stream_structured
