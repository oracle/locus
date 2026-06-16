# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Observability — centralised event bus + SSE telemetry for locus.

Modelled on the pattern Oracle's optic SDK uses: a singleton
:class:`EventBus` that fans events out to multiple consumers (Web SSE,
CLI tail, JSON logs) from a single emission point. Locus components —
the router, the agent loop's hooks, custom user code — publish
:class:`StreamEvent` instances scoped to a *run id* (one cognitive
dispatch); subscribers consume them filtered by run id, or globally.

Quick start::

    from locus.observability import StreamEvent, get_event_bus

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

from locus.observability.bus_hook import EventBusHook
from locus.observability.context import (
    current_run_id,
    reset_run_id,
    run_context,
    set_run_id,
)
from locus.observability.emit import emit, emit_sync
from locus.observability.event_bus import (
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
