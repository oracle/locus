# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0
"""
Notebook 13: stream an investigation to a SOC dashboard with SSE.

SSE is the simplest way to push live investigation updates from a
Python backend to a SOC dashboard — one HTTP response, one event per
line. Tulip ships two handlers that turn the agent event stream into
SSE wire format: a buffered ``SSEHandler`` and a queue-based
``AsyncSSEHandler`` for true streaming.

Key ideas:
- An ``SSEMessage`` is a small dataclass with ``event``, ``data``, and
  ``id`` fields, plus ``.format()`` to produce the wire bytes.
- ``SSEHandler`` collects messages in memory — good for short runs and
  tests.
- ``AsyncSSEHandler.stream()`` is an async iterator you can hand
  straight to FastAPI / Starlette ``StreamingResponse``.
- ``create_sse_response_headers()`` returns the correct
  ``text/event-stream`` headers and disables proxy buffering.
- Pass ``custom_serializer=...`` to either handler to shape what the
  dashboard sees.

Run it:
    .venv/bin/python examples/notebook_13_sse_streaming.py

This notebook does not call an LLM — it only exercises the SSE
plumbing, so no provider configuration is needed. The mock provider is
still a fine default if you set ``TULIP_MODEL_PROVIDER=mock``.
"""

import asyncio
from datetime import UTC, datetime

from tulip.core.events import (
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
    TulipEvent,
)
from tulip.streaming.sse import (
    AsyncSSEHandler,
    SSEHandler,
    SSEMessage,
    create_sse_response_headers,
)


async def main():
    print("=" * 60)
    print("Notebook 13: SOC Dashboard SSE Streaming")
    print("=" * 60)

    # =========================================================================
    # Part 1: the SSE wire format
    # =========================================================================
    print("\n=== Part 1: SSE Message Format ===\n")

    message = SSEMessage(
        event="thinking",
        data='{"content": "Correlating alert SOC-1042 with sign-in history..."}',
        id="1",
    )

    print("SSE Message components:")
    print(f"  event: {message.event}")
    print(f"  data: {message.data}")
    print(f"  id: {message.id}")

    wire_format = message.format()
    print("\nWire format:")
    print("-" * 30)
    print(wire_format)
    print("-" * 30)

    # =========================================================================
    # Part 2: building messages by hand
    # =========================================================================
    print("\n=== Part 2: Creating SSE Messages ===\n")

    messages = [
        SSEMessage(event="start", data='{"investigation_id": "SOC-1042"}'),
        SSEMessage(event="chunk", data="Sender domain registered"),
        SSEMessage(event="chunk", data=" 3 days ago — suspicious."),
        SSEMessage(event="done", data='{"status": "complete"}'),
    ]

    print("Message sequence:")
    for msg in messages:
        print(f"  [{msg.event}] {msg.data}")

    # Multi-line payloads are valid; format() emits one data: line per line.
    # A brute-force detection rule (ATT&CK T1110).
    multiline_msg = SSEMessage(
        event="detection_rule",
        data="rule: repeated_failed_logins\n  when: failed_logins > 5\n  then: page on-call",
    )
    print("\nMulti-line message format:")
    print(multiline_msg.format())

    # =========================================================================
    # Part 3: buffered handler — collect everything, then flush
    # =========================================================================
    print("\n=== Part 3: SSE Handler (Buffered) ===\n")

    handler = SSEHandler(
        include_timestamp=True,
        include_id=True,
        id_prefix="evt_",
    )

    print("Handler config:")
    print(f"  Include timestamp: {handler.include_timestamp}")
    print(f"  Include ID: {handler.include_id}")
    print(f"  ID prefix: {handler.id_prefix}")

    # Stand-in events; in a real app these come from the triage agent's
    # agent.run(...).
    events = [
        ThinkEvent(iteration=1, reasoning="Triaging alert SOC-1042"),
        ToolStartEvent(
            tool_name="lookup_ip", tool_call_id="call_001", arguments={"ip": "198.51.100.7"}
        ),
        ToolCompleteEvent(
            tool_name="lookup_ip",
            tool_call_id="call_001",
            result="198.51.100.7: 2 abuse reports in the last 30 days",
        ),
    ]

    for event in events:
        await handler.on_event(event)

    await handler.on_complete()

    print(f"\nBuffered messages: {len(handler.get_messages())}")
    print(f"Is complete: {handler.is_complete}")

    for msg in handler.get_messages():
        print(f"  [{msg.event}] id={msg.id}")

    # =========================================================================
    # Part 4: format the buffer to wire bytes
    # =========================================================================
    print("\n=== Part 4: Formatted Output ===\n")

    full_output = handler.format_all()
    print("Full SSE output (first 500 chars):")
    print("-" * 40)
    print(full_output[:500] + "..." if len(full_output) > 500 else full_output)
    print("-" * 40)

    # pop_messages drains and returns — get_messages copies and keeps.
    handler.clear()
    await handler.on_event(ThinkEvent(iteration=1, reasoning="New pivot: check sender domain"))
    popped = handler.pop_messages()
    remaining = handler.get_messages()
    print(f"\nAfter pop: got {len(popped)}, remaining {len(remaining)}")

    # =========================================================================
    # Part 5: report errors to the dashboard
    # =========================================================================
    print("\n=== Part 5: Error Handling ===\n")

    handler.clear()

    await handler.on_event(ThinkEvent(iteration=1, reasoning="Starting enrichment..."))
    await handler.on_error(ValueError("Intel feed unavailable"))

    print(f"Has error: {handler.has_error}")
    print(f"Is complete: {handler.is_complete}")

    for msg in handler.get_messages():
        print(f"  [{msg.event}] {msg.data[:50]}...")

    # =========================================================================
    # Part 6: async handler — queue-based, true streaming
    # =========================================================================
    print("\n=== Part 6: Async SSE Handler ===\n")

    # AsyncSSEHandler backs the stream with an asyncio.Queue, so producer
    # and consumer run concurrently — the pattern a live SOC dashboard
    # needs.
    async_handler = AsyncSSEHandler(
        include_timestamp=True,
        include_id=True,
    )

    async def produce_events():
        await async_handler.on_event(ThinkEvent(iteration=1, reasoning="Enriching indicators..."))
        await asyncio.sleep(0.1)
        await async_handler.on_event(
            ToolStartEvent(tool_name="whois_domain", tool_call_id="call_002", arguments={})
        )
        await asyncio.sleep(0.1)
        await async_handler.on_complete()

    async def consume_events():
        count = 0
        async for sse_text in async_handler.stream():
            count += 1
            # A real app would yield sse_text from a StreamingResponse here.
        return count

    producer = asyncio.create_task(produce_events())
    count = await consume_events()
    await producer

    print(f"Streamed {count} SSE messages")

    # =========================================================================
    # Part 7: the right HTTP response headers
    # =========================================================================
    print("\n=== Part 7: HTTP Response Headers ===\n")

    headers = create_sse_response_headers()

    print("SSE Response Headers:")
    for name, value in headers.items():
        print(f"  {name}: {value}")

    # =========================================================================
    # Part 8: shape the wire payload with a custom serializer
    # =========================================================================
    print("\n=== Part 8: Custom Serialization ===\n")

    def custom_serializer(event: TulipEvent) -> dict:
        """Trim the payload to type, timestamp, and one content field."""
        return {
            "type": event.event_type,
            "time": datetime.now(UTC).isoformat(),
            "data": getattr(event, "reasoning", None) or getattr(event, "result", None),
        }

    custom_handler = SSEHandler(custom_serializer=custom_serializer)

    await custom_handler.on_event(
        ThinkEvent(iteration=1, reasoning="Custom serialization for the dashboard")
    )
    msg = custom_handler.get_messages()[0]

    print("Custom serialized event:")
    print(f"  {msg.data}")

    # =========================================================================
    # Part 9: drop it into FastAPI
    # =========================================================================
    print("\n=== Part 9: Web Framework Integration ===\n")

    print("FastAPI Example:")
    print("-" * 40)
    print("""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from tulip.streaming.sse import AsyncSSEHandler, create_sse_response_headers

app = FastAPI()

@app.get("/investigations/stream")
async def stream_events():
    handler = AsyncSSEHandler()

    async def generate():
        # Start the triage agent in the background
        task = asyncio.create_task(run_agent(handler))

        # Stream events to the dashboard
        async for sse_text in handler.stream():
            yield sse_text

        await task

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=create_sse_response_headers(),
    )

async def run_agent(handler):
    # Your investigation logic
    await handler.on_event(ThinkEvent(iteration=1, reasoning="Investigating..."))
    await handler.on_complete()
""")
    print("-" * 40)

    # =========================================================================
    # Part 10: every Tulip event type the handlers know how to render
    # =========================================================================
    print("\n=== Part 10: Supported Event Types ===\n")

    supported_events = [
        # Loop events
        ("think", "Agent thinking/reasoning"),
        ("tool_start", "Tool execution started"),
        ("tool_complete", "Tool execution completed"),
        ("reflect", "Self-reflection result"),
        ("grounding", "Grounding evaluation"),
        ("terminate", "Agent terminated"),
        # Model events
        ("model_chunk", "Streaming model output"),
        ("model_complete", "Model generation complete"),
        # Multi-agent events
        ("specialist_start", "Specialist started"),
        ("specialist_complete", "Specialist completed"),
        ("orchestrator_decision", "Orchestrator routing decision"),
        # Hook events
        ("before_invocation", "Before agent invocation"),
        ("after_invocation", "After agent invocation"),
    ]

    print("Event types a SOC dashboard can subscribe to:")
    for event_type, description in supported_events:
        print(f"  {event_type}: {description}")

    # =========================================================================
    # Part 11: production checklist
    # =========================================================================
    print("\n=== Part 11: Best Practices ===\n")

    print("1. Always set proper SSE headers")
    print("2. Include event IDs so dashboards can reconnect with Last-Event-ID")
    print("3. Send a 'done' event when the investigation terminates")
    print("4. Send error events on failure — never leave the analyst's stream hanging")
    print("5. Use AsyncSSEHandler for real streaming, not the buffered one")
    print("6. Keep individual event payloads small (< 65KB)")
    print("7. Implement client-side reconnection")
    print("8. Send periodic heartbeats during long-running scans")

    heartbeat = SSEMessage(event="heartbeat", data='{"status": "alive"}')
    print(f"\nHeartbeat message:\n{heartbeat.format()}")

    print("\n" + "=" * 60)
    print("Notebook 13 complete.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
