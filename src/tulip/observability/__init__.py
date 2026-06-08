# Copyright (c) 2026 tuliplabs.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://opensource.org/license/UPL

"""Observability — centralised event bus + SSE telemetry for tulip.

Modelled on a common observability-bus pattern: a singleton
:class:`EventBus` that fans events out to multiple consumers (Web SSE,
CLI tail, JSON logs) from a single emission point. Tulip components —
the router, the agent loop's hooks, custom user code — publish
:class:`StreamEvent` instances scoped to a *run id* (one cognitive
dispatch); subscribers consume them filtered by run id, or globally.

Quick start::

    from tulip.observability import StreamEvent, get_event_bus

    bus = get_event_bus()

    # Publisher
    await bus.publish(
        StreamEvent(
            run_id="abc",
            event_type="router.protocol.selected",
            data={"protocol_id": "specialist_fanout"},
        )
    )

    # Consumer
    async for ev in bus.subscribe("abc"):
        print(ev.event_type, ev.data)

The workbench's SSE endpoint at ``/api/events/{run_id}`` is the public
HTTP wrapper around :meth:`EventBus.subscribe`.
"""

from __future__ import annotations

from tulip.observability.bus_hook import EventBusHook
from tulip.observability.context import (
    current_run_id,
    reset_run_id,
    run_context,
    set_run_id,
)
from tulip.observability.emit import emit, emit_sync
from tulip.observability.event_bus import (
    EventBus,
    StreamEvent,
    get_event_bus,
    reset_event_bus,
)


__all__ = [
    "EventBus",
    "EventBusHook",
    "StreamEvent",
    "current_run_id",
    "emit",
    "emit_sync",
    "get_event_bus",
    "reset_event_bus",
    "reset_run_id",
    "run_context",
    "set_run_id",
]
