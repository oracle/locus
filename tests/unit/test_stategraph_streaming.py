# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Unit tests for ``StateGraph.stream()`` real intermediate-event streaming.

Previously ``stream()`` ran ``execute()`` to completion and only yielded
events at the end. These tests guard the new sink-via-asyncio.Queue
bridge so events arrive as nodes complete.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from locus.multiagent.graph import (
    END,
    START,
    Node,
    StateGraph,
    StreamEvent,
    StreamMode,
    emit_custom,
)


def _node(node_id: str, output_key: str, value: Any, *, sleep: float = 0.0) -> Node:
    """Build a node that records ``output_key=value`` after sleeping ``sleep``s."""

    async def executor(state: dict[str, Any]) -> dict[str, Any]:
        if sleep:
            await asyncio.sleep(sleep)
        return {output_key: value}

    return Node(id=node_id, name=node_id, executor=executor)


def _graph_three_nodes() -> StateGraph:
    g = StateGraph(id="three")
    g.add_node(_node("a", "from_a", 1))
    g.add_node(_node("b", "from_b", 2))
    g.add_node(_node("c", "from_c", 3))
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", END)
    return g


# =============================================================================
# Mode coverage
# =============================================================================


class TestStreamModes:
    async def test_updates_yields_one_event_per_node(self):
        g = _graph_three_nodes()
        events: list[StreamEvent] = [ev async for ev in g.stream({}, mode=StreamMode.UPDATES)]
        assert [e.node_id for e in events] == ["a", "b", "c"]
        assert all(e.mode == StreamMode.UPDATES for e in events)
        # Each event carries this node's output dict.
        assert events[0].data == {"from_a": 1}
        assert events[1].data == {"from_b": 2}
        assert events[2].data == {"from_c": 3}

    async def test_nodes_yields_full_node_result(self):
        g = _graph_three_nodes()
        events: list[StreamEvent] = [ev async for ev in g.stream({}, mode=StreamMode.NODES)]
        assert [e.node_id for e in events] == ["a", "b", "c"]
        # Each event's data is the NodeResult, not just the output.
        for ev, expected in zip(events, ["a", "b", "c"], strict=True):
            assert ev.data.node_id == expected
            assert ev.data.success is True

    async def test_values_yields_per_node_then_final(self):
        g = _graph_three_nodes()
        events: list[StreamEvent] = [ev async for ev in g.stream({}, mode=StreamMode.VALUES)]
        # 3 per-node snapshots + 1 terminal final-state event.
        assert len(events) == 4
        assert [e.node_id for e in events[:3]] == ["a", "b", "c"]
        # Last event has no node_id and carries the merged final state.
        assert events[-1].node_id is None
        final = events[-1].data
        assert final["from_a"] == 1
        assert final["from_b"] == 2
        assert final["from_c"] == 3


# =============================================================================
# Real-time delivery: events arrive before the run completes
# =============================================================================


class TestRealTimeDelivery:
    async def test_events_arrive_before_final_node_completes(self):
        """Earlier events should be delivered before the last node sleeps.

        Previously stream() awaited execute() to completion before yielding
        anything. With the queue bridge, the consumer should see the first
        event well before the last node finishes its sleep.
        """
        g = StateGraph(id="paced")
        g.add_node(_node("a", "x", 1))  # fast
        g.add_node(_node("b", "y", 2, sleep=0.25))  # slow
        g.add_edge(START, "a")
        g.add_edge("a", "b")
        g.add_edge("b", END)

        first_event_time: float | None = None
        start = time.perf_counter()
        async for ev in g.stream({}, mode=StreamMode.UPDATES):
            if ev.node_id == "a" and first_event_time is None:
                first_event_time = time.perf_counter() - start
        end = time.perf_counter() - start

        assert first_event_time is not None
        # The first event must arrive WAY before the slow node's 0.25s
        # sleep finishes. If stream() were still buffering until execute()
        # returns, first_event_time ~= end.
        assert first_event_time < end / 2, (
            f"first event at {first_event_time:.3f}s, end at {end:.3f}s — "
            "events not arriving in real time"
        )


# =============================================================================
# Error propagation
# =============================================================================


class TestErrorPropagation:
    async def test_node_error_doesnt_deadlock_consumer(self):
        """A failing node must not leave the consumer waiting on the queue."""

        async def boom(state: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("kaboom")

        g = StateGraph(id="boom")
        g.add_node(Node(id="a", name="a", executor=boom))
        g.add_edge(START, "a")
        g.add_edge("a", END)

        # The Node catches its own exceptions and surfaces them via
        # NodeResult.error, so streaming should yield the failure as an
        # event and the run completes (success=False).
        events = [ev async for ev in g.stream({}, mode=StreamMode.NODES)]
        assert len(events) == 1
        result = events[0].data
        assert result.success is False
        assert "kaboom" in (result.error or "")

    async def test_consumer_break_doesnt_leak_task(self):
        """Breaking out of the iterator early cancels the background driver."""
        g = _graph_three_nodes()
        seen: list[StreamEvent] = []
        async for ev in g.stream({}, mode=StreamMode.UPDATES):
            seen.append(ev)
            break
        # We should have seen exactly one event before bailing; no hangs.
        assert len(seen) == 1
        # Give the cancelled driver a tick to settle before the test
        # tears down, so we don't see "Task was destroyed but it is pending!"
        # warnings in CI logs.
        await asyncio.sleep(0.01)


# =============================================================================
# CUSTOM mode: nodes can emit progress events via emit_custom()
# =============================================================================


class TestCustomEmit:
    async def test_emit_custom_inside_node_reaches_consumer(self):
        """A node can call ``emit_custom(...)`` mid-execution and the consumer
        sees the resulting CUSTOM event in real time, before the node returns.
        """

        async def progress_node(state: dict[str, Any]) -> dict[str, Any]:
            await emit_custom({"step": 1, "total": 3}, node_id="progress_node")
            await emit_custom({"step": 2, "total": 3}, node_id="progress_node")
            await emit_custom({"step": 3, "total": 3}, node_id="progress_node")
            return {"done": True}

        g = StateGraph(id="custom-emit")
        g.add_node(Node(id="p", name="p", executor=progress_node))
        g.add_edge(START, "p")
        g.add_edge("p", END)

        events: list[StreamEvent] = [ev async for ev in g.stream({}, mode=StreamMode.UPDATES)]
        # We should have 3 CUSTOM progress events plus the node's UPDATES event.
        custom = [e for e in events if e.mode == StreamMode.CUSTOM]
        assert len(custom) == 3
        assert [c.data["step"] for c in custom] == [1, 2, 3]
        assert all(c.node_id == "progress_node" for c in custom)
        # The final UPDATES event for the node still appears after.
        updates = [e for e in events if e.mode == StreamMode.UPDATES]
        assert len(updates) == 1
        assert updates[0].data == {"done": True}

    async def test_emit_custom_outside_streaming_is_noop(self):
        """``emit_custom`` outside a ``stream()`` context must not raise.

        Lets node code be written once and run under either ``execute()``
        or ``stream()`` without branching on context.
        """

        async def node_with_emit(state: dict[str, Any]) -> dict[str, Any]:
            await emit_custom({"hello": "world"})
            return {"x": 1}

        g = StateGraph(id="execute-only")
        g.add_node(Node(id="n", name="n", executor=node_with_emit))
        g.add_edge(START, "n")
        g.add_edge("n", END)

        # Calling execute() (not stream()) — emit_custom should be a no-op,
        # not raise.
        result = await g.execute({})
        assert result.success is True
        assert result.final_state.get("x") == 1
