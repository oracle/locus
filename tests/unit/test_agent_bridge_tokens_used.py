# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``agent.tokens.used`` reaches the bus once per run, from the terminate event.

The agent loop never yields ``ModelCompleteEvent``, so the run's cumulative
usage on ``TerminateEvent`` is the one place a run's token counts can be
bridged. An unmetered run (``usage=None``) publishes no counts.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from locus.core.events import ModelCompleteEvent, TerminateEvent
from locus.observability.agent_bridge import bridge_locus_event
from locus.observability.context import run_context
from locus.observability.event_bus import get_event_bus


async def _bridge_and_collect(rid: str, event: Any) -> list[Any]:
    """Bridge ``event`` under ``run_context(rid)`` and return what the bus saw."""
    bus = get_event_bus()
    async with run_context(rid):
        await bridge_locus_event(event)
        await bus.close_stream(rid)
    collected = []
    async for seen in bus.subscribe(rid):
        collected.append(seen)
    return collected


def _terminate(usage: dict[str, int] | None) -> TerminateEvent:
    return TerminateEvent(
        reason="completed",
        iterations_used=2,
        final_confidence=0.9,
        total_tool_calls=1,
        final_message="done",
        usage=usage,
    )


@pytest.mark.asyncio
async def test_terminate_with_usage_publishes_the_run_totals_once() -> None:
    usage = {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}
    seen = await _bridge_and_collect("bridge-tokens-run", _terminate(usage))

    assert [e.event_type for e in seen] == ["agent.terminate", "agent.tokens.used"]
    assert seen[1].data == {
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
        "scope": "run",
    }


@pytest.mark.asyncio
async def test_unmetered_run_publishes_no_counts() -> None:
    seen = await _bridge_and_collect("bridge-tokens-unmetered", _terminate(None))

    assert [e.event_type for e in seen] == ["agent.terminate"]


@pytest.mark.asyncio
async def test_model_complete_shape_is_marked_per_call() -> None:
    event = ModelCompleteEvent(
        content="final",
        tool_calls=[],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        stop_reason="end_turn",
    )
    seen = await _bridge_and_collect("bridge-tokens-call", event)

    tokens = [e for e in seen if e.event_type == "agent.tokens.used"]
    assert len(tokens) == 1
    assert tokens[0].data["scope"] == "call"
    assert tokens[0].data["total_tokens"] == 15


@pytest.mark.asyncio
async def test_a_subscriber_sees_the_totals_live() -> None:
    """The live path, not only the replayed history: a subscriber attached
    before the run ends receives the counts."""
    bus = get_event_bus()
    rid = "bridge-tokens-live"
    received: list[Any] = []

    async def subscriber() -> None:
        async for seen in bus.subscribe(rid):
            received.append(seen)

    async with run_context(rid):
        task = asyncio.create_task(subscriber())
        await asyncio.sleep(0.01)
        await bridge_locus_event(
            _terminate({"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10})
        )
        await bus.close_stream(rid)
        await asyncio.wait_for(task, timeout=2.0)

    totals = next(e for e in received if e.event_type == "agent.tokens.used")
    assert totals.data["total_tokens"] == 10
    assert totals.data["scope"] == "run"
