# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint history on graphs: every step saved, readable, editable, forkable.

``StateGraph`` used to save a checkpoint only when a node paused, so a run that
finished left nothing behind. These tests pin a checkpoint per completed step,
reading any of them, an edit that a paused graph resumes from, and a fork that
leaves its parent alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tulip.core.command import Command
from tulip.memory.backends.file import FileCheckpointer
from tulip.memory.backends.memory import MemoryCheckpointer
from tulip.multiagent.graph import END, START, GraphConfig, StateGraph


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(params=["memory", "file"])
def checkpointer(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    if request.param == "memory":
        return MemoryCheckpointer()
    return FileCheckpointer(tmp_path / "checkpoints")


def _counting_graph(checkpointer: Any, thread_id: str) -> StateGraph:
    graph = StateGraph()

    async def a(inputs: dict[str, Any]) -> dict[str, Any]:
        return {"count": inputs.get("count", 0) + 1, "path": "a"}

    async def b(inputs: dict[str, Any]) -> dict[str, Any]:
        return {"count": inputs["count"] + 1, "path": "b"}

    async def c(inputs: dict[str, Any]) -> dict[str, Any]:
        return {"count": inputs["count"] + 1, "path": "c"}

    for name, fn in (("a", a), ("b", b), ("c", c)):
        graph.add_node(name, fn)
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.add_edge("c", END)
    graph.compile(checkpointer=checkpointer)
    graph.config.thread_id = thread_id
    return graph


def _gated_graph(checkpointer: Any, thread_id: str) -> StateGraph:
    graph = StateGraph()

    async def gate(inputs: dict[str, Any]) -> dict[str, Any]:
        return {"gated": True}

    async def after(inputs: dict[str, Any]) -> dict[str, Any]:
        return {"done": True, "x_seen": inputs.get("x")}

    graph.add_node("gate", gate)
    graph.add_node("after", after)
    graph.add_edge(START, "gate")
    graph.add_edge("gate", "after")
    graph.add_edge("after", END)
    graph.compile(checkpointer=checkpointer, interrupt_before=["gate"])
    graph.config.thread_id = thread_id
    return graph


@pytest.mark.asyncio
async def test_a_completed_run_leaves_a_checkpoint_per_step(checkpointer: Any) -> None:
    graph = _counting_graph(checkpointer, "g1")

    result = await graph.execute({"count": 0})
    history = await graph.aget_state_history("g1")

    assert result.final_state["count"] == 3
    assert [state["path"] for _, state in history] == ["c", "b", "a"]
    oldest_id, oldest = history[-1]
    assert oldest["count"] == 1
    assert (await graph.aget_state("g1", checkpoint_id=oldest_id))["count"] == 1
    assert (await graph.aget_state({"configurable": {"thread_id": "g1"}}))["count"] == 3


@pytest.mark.asyncio
async def test_a_paused_graph_resumes_from_an_edit(checkpointer: Any) -> None:
    graph = _gated_graph(checkpointer, "g2")
    first = await graph.execute({"x": 1})
    assert first.interrupt is not None
    [(paused_id, paused)] = (await graph.aget_state_history("g2"))[:1]

    edited_id = await graph.aupdate_state("g2", {"x": 10})
    resumed = await graph.execute(Command(resume=True))

    assert edited_id != paused_id
    assert (await graph.aget_state_history("g2", limit=50))[-1][1]["x"] == 1
    assert (await graph.aget_state("g2", checkpoint_id=paused_id))["x"] == paused["x"] == 1
    assert resumed.interrupt is None
    assert resumed.final_state["done"] is True
    assert resumed.final_state["x_seen"] == 10


@pytest.mark.asyncio
async def test_a_fork_starts_from_a_step_and_leaves_the_parent_alone(checkpointer: Any) -> None:
    graph = _counting_graph(checkpointer, "g1")
    await graph.execute({"count": 0})
    oldest_id, _ = (await graph.aget_state_history("g1"))[-1]

    forked = await graph.afork("g1", oldest_id, new_thread_id="g1-alt")
    unnamed = await graph.afork("g1")

    assert forked == "g1-alt"
    assert (await graph.aget_state("g1-alt"))["count"] == 1
    assert (await graph.aget_state("g1"))["count"] == 3
    assert unnamed.startswith("g1-fork-")
    assert (await graph.aget_state(unnamed))["count"] == 3


@pytest.mark.asyncio
async def test_step_checkpoints_can_be_turned_off() -> None:
    cp = MemoryCheckpointer()
    graph = _counting_graph(cp, "g3")

    await graph.execute(
        {"count": 0},
        config=GraphConfig(checkpointer=cp, thread_id="g3", checkpoint_every_step=False),
    )

    assert await cp.list_checkpoints("g3") == []


@pytest.mark.asyncio
async def test_history_needs_a_checkpointer_and_a_real_checkpoint() -> None:
    with pytest.raises(ValueError, match="checkpointer"):
        await StateGraph().aget_state_history("t")

    cp = MemoryCheckpointer()
    graph = _counting_graph(cp, "g4")
    await graph.execute({"count": 0})
    with pytest.raises(LookupError):
        await graph.aupdate_state("g4", {"count": 9}, checkpoint_id="no-such-checkpoint")
    with pytest.raises(LookupError):
        await graph.afork("no-such-thread")
    with pytest.raises(ValueError, match="different"):
        await graph.afork("g4", new_thread_id="g4")
    assert await graph.aget_state("g4", checkpoint_id="no-such-checkpoint") is None
