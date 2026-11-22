# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""``multiagent.graph.node.routed`` is emitted for dynamic routing decisions."""

from __future__ import annotations

import asyncio
from typing import Any

from locus.core import Command
from locus.multiagent.graph import END, START, StateGraph
from locus.observability import run_context
from locus.observability.event_bus import get_event_bus


ROUTED = "multiagent.graph.node.routed"
STARTED = "multiagent.graph.node.started"


async def _run(graph: StateGraph, inputs: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Execute ``graph`` under a run id; return the ``multiagent.graph.node.*`` events seen."""
    bus = get_event_bus()
    events: list[tuple[str, dict[str, Any]]] = []
    async with run_context() as rid:

        async def collect() -> None:
            async for event in bus.subscribe(rid):
                if event.event_type.startswith("multiagent.graph.node."):
                    events.append((event.event_type, dict(event.data)))

        collector = asyncio.create_task(collect())
        await asyncio.sleep(0)
        await graph.execute(inputs)
        await bus.close_stream(rid)
        await collector
    return events


async def test_conditional_edge_emits_one_routed_event() -> None:
    graph = StateGraph()

    async def decide(inputs: dict[str, Any]) -> dict[str, Any]:
        return {"seen": True}

    async def fast(inputs: dict[str, Any]) -> dict[str, Any]:
        return {"path": "fast"}

    async def slow(inputs: dict[str, Any]) -> dict[str, Any]:
        return {"path": "slow"}

    graph.add_node("decide", decide)
    graph.add_node("fast", fast)
    graph.add_node("slow", slow)
    graph.add_edge(START, "decide")
    graph.add_conditional_edges(
        "decide",
        lambda state: "fast" if state.get("go_fast") else "slow",
        {"fast": "fast", "slow": "slow"},
    )
    graph.add_edge("fast", END)
    graph.add_edge("slow", END)

    events = await _run(graph, {"go_fast": True})

    routed = [data for name, data in events if name == ROUTED]
    assert len(routed) == 1
    payload = routed[0]
    assert set(payload) >= {"graph_id", "node_id", "targets", "iteration", "source"}
    assert payload["graph_id"] == graph.id
    assert payload["node_id"] == "decide"
    assert payload["source"] == "conditional_edge"
    assert payload["targets"] == ["fast"]
    started = [data["node_id"] for name, data in events if name == STARTED]
    assert "fast" in started
    assert "slow" not in started


async def test_command_goto_emits_routed_event_with_command_source() -> None:
    graph = StateGraph()

    async def router(inputs: dict[str, Any]) -> Command:
        return Command(update={"routed": True}, goto="target")

    async def target(inputs: dict[str, Any]) -> dict[str, Any]:
        return {"done": True}

    graph.add_node("router", router)
    graph.add_node("target", target)
    graph.add_edge(START, "router")
    graph.add_edge("target", END)

    events = await _run(graph, {})

    routed = [data for name, data in events if name == ROUTED]
    assert len(routed) == 1
    assert routed[0]["source"] == "command"
    assert routed[0]["node_id"] == "router"
    assert routed[0]["targets"] == ["target"]


async def test_regular_edges_emit_no_routed_event() -> None:
    graph = StateGraph()

    async def a(inputs: dict[str, Any]) -> dict[str, Any]:
        return {"a": 1}

    async def b(inputs: dict[str, Any]) -> dict[str, Any]:
        return {"b": 1}

    graph.add_node("a", a)
    graph.add_node("b", b)
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)

    events = await _run(graph, {})

    assert any(name == STARTED for name, _ in events)
    assert not [1 for name, _ in events if name == ROUTED]
