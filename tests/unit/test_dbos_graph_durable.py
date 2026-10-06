# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable StateGraph runs on DBOS, against a SQLite system database and a file checkpointer.

The local default, with nothing but the machine: a graph pauses on an interrupt, waits
for a value from outside (a person, a job on another machine), and resumes from its
checkpoint, across a restart if it must; a segment cut off halfway is never run again.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

import pytest


pytest.importorskip("dbos")

from dbos import DBOS  # noqa: E402

from tulip.core.interrupt import interrupt  # noqa: E402
from tulip.durable.dbos import (  # noqa: E402
    GraphSegmentOutcome,
    InterruptedSegmentError,
    pending_interrupt,
    register_graphs,
    resume_graph,
    start_graph_run,
)
from tulip.durable.segments import SegmentError, run_graph  # noqa: E402
from tulip.memory.backends.file import FileCheckpointer  # noqa: E402
from tulip.multiagent.graph import END, START, StateGraph  # noqa: E402


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


@pytest.fixture
def system_db(tmp_path: Path) -> Iterator[str]:
    yield f"sqlite:///{tmp_path}/dbos.sqlite"
    DBOS.destroy()


def _launch(url: str) -> None:
    DBOS(config={"name": "tulip-test", "system_database_url": url, "run_admin_server": False})
    DBOS.launch()


def _relaunch(url: str) -> None:
    DBOS.destroy()
    _launch(url)


def _name() -> str:
    return f"g-{uuid.uuid4().hex[:6]}"


def _review_graph(
    root: Path, effects: list[str], *, checkpointer: bool = True
) -> Callable[[], StateGraph]:
    """draft -> wait for a job's result -> wait for a person -> publish."""

    async def draft(state: dict[str, Any]) -> dict[str, Any]:
        effects.append(f"draft {state['request']}")
        return {"drafted": True}

    async def job(state: dict[str, Any]) -> dict[str, Any]:
        result = interrupt({"waiting": "job", "request": state["request"]})
        return {"result": result}

    async def review(state: dict[str, Any]) -> dict[str, Any]:
        decision = interrupt({"waiting": "person", "request": state["request"]})
        return {"approved": decision == "yes"}

    async def publish(state: dict[str, Any]) -> dict[str, Any]:
        if state["approved"]:
            effects.append(f"publish {state['request']}:{state['result']}")
        return {"published": state["approved"]}

    def make() -> StateGraph:
        g = StateGraph()
        for name, fn in (("draft", draft), ("job", job), ("review", review), ("publish", publish)):
            g.add_node(name, fn)
        g.add_edge(START, "draft")
        g.add_edge("draft", "job")
        g.add_edge("job", "review")
        g.add_edge("review", "publish")
        g.add_edge("publish", END)
        if checkpointer:
            g.compile(checkpointer=FileCheckpointer(root / "checkpoints"))
        return g

    return make


async def _wait_for_pause(workflow_id: str, node: str) -> GraphSegmentOutcome:
    for _ in range(200):
        pending = await pending_interrupt(workflow_id)
        if pending is not None and pending.node == node:
            return pending
        await asyncio.sleep(0.05)
    raise AssertionError(f"the run never paused at {node}")


@pytest.mark.asyncio
async def test_a_graph_waits_twice_and_finishes_once(system_db: str, tmp_path: Path) -> None:
    effects: list[str] = []
    name = _name()
    register_graphs({name: _review_graph(tmp_path, effects)})
    _launch(system_db)

    handle = await start_graph_run(graph=name, inputs={"request": 12}, thread_id="t1")
    assert handle.workflow_id == "tulip-graph-t1"
    paused = await _wait_for_pause(handle.workflow_id, "job")
    assert paused.status == "paused"
    assert paused.interrupt == {"waiting": "job", "request": 12}
    assert paused.state["drafted"] is True

    await resume_graph(handle.workflow_id, "plan-7")
    paused = await _wait_for_pause(handle.workflow_id, "review")
    assert paused.interrupt == {"waiting": "person", "request": 12}
    assert paused.state["result"] == "plan-7"
    assert effects == ["draft 12"]

    await resume_graph(handle.workflow_id, "yes")
    outcome = await handle.get_result()
    assert outcome.status == "done"
    assert outcome.state["published"] is True
    assert not any(k.startswith("__") for k in outcome.state)
    assert effects == ["draft 12", "publish 12:plan-7"]
    assert await pending_interrupt(handle.workflow_id) is None


@pytest.mark.asyncio
async def test_a_paused_graph_survives_a_restart(system_db: str, tmp_path: Path) -> None:
    effects: list[str] = []
    name = _name()
    register_graphs({name: _review_graph(tmp_path, effects)})
    _launch(system_db)
    await start_graph_run(graph=name, inputs={"request": 3}, thread_id="t2", workflow_id="w2")
    await _wait_for_pause("w2", "job")

    _relaunch(system_db)  # the process goes away; a new one recovers the wait

    await _wait_for_pause("w2", "job")
    await resume_graph("w2", "plan-1")
    await _wait_for_pause("w2", "review")
    await resume_graph("w2", "no")
    outcome = await (await DBOS.retrieve_workflow_async("w2")).get_result()
    assert outcome.status == "done"
    assert outcome.state["published"] is False
    assert effects == ["draft 3"]


@pytest.mark.asyncio
async def test_two_runs_of_one_graph_keep_their_own_state(system_db: str, tmp_path: Path) -> None:
    effects: list[str] = []
    name = _name()
    register_graphs({name: _review_graph(tmp_path, effects)})
    _launch(system_db)
    a = await start_graph_run(graph=name, inputs={"request": 1}, thread_id="ta")
    b = await start_graph_run(graph=name, inputs={"request": 2}, thread_id="tb")
    await _wait_for_pause(a.workflow_id, "job")
    await _wait_for_pause(b.workflow_id, "job")

    await resume_graph(b.workflow_id, "plan-b")
    await resume_graph(a.workflow_id, "plan-a")
    pa = await _wait_for_pause(a.workflow_id, "review")
    pb = await _wait_for_pause(b.workflow_id, "review")
    assert (pa.state["request"], pa.state["result"]) == (1, "plan-a")
    assert (pb.state["request"], pb.state["result"]) == (2, "plan-b")

    await resume_graph(a.workflow_id, "yes")
    await resume_graph(b.workflow_id, "yes")
    assert (await a.get_result()).state["result"] == "plan-a"
    assert (await b.get_result()).state["result"] == "plan-b"
    assert sorted(effects) == ["draft 1", "draft 2", "publish 1:plan-a", "publish 2:plan-b"]


@pytest.mark.asyncio
async def test_a_failed_node_ends_the_run_failed(system_db: str, tmp_path: Path) -> None:
    name = _name()

    async def broken(state: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("the plan leaves the box")

    def make() -> StateGraph:
        g = StateGraph()
        g.add_node("check", broken)
        g.add_edge(START, "check")
        g.add_edge("check", END)
        g.compile(checkpointer=FileCheckpointer(tmp_path / "checkpoints"))
        return g

    register_graphs({name: make})
    _launch(system_db)
    handle = await start_graph_run(graph=name, inputs={"request": 9}, thread_id="t3")
    outcome = await handle.get_result()
    assert outcome.status == "failed"
    assert outcome.node == "check"
    assert outcome.error is not None
    assert "the plan leaves the box" in outcome.error


@pytest.mark.asyncio
async def test_a_graph_segment_cut_off_halfway_is_not_run_again(
    system_db: str, tmp_path: Path
) -> None:
    started: list[str] = []
    never = asyncio.Event()
    name = _name()

    async def slow(state: dict[str, Any]) -> dict[str, Any]:
        started.append("node")
        await never.wait()  # the process dies while the node is working
        return {}  # pragma: no cover

    def make() -> StateGraph:
        g = StateGraph()
        g.add_node("slow", slow)
        g.add_edge(START, "slow")
        g.add_edge("slow", END)
        g.compile(checkpointer=FileCheckpointer(tmp_path / "checkpoints"))
        return g

    register_graphs({name: make})
    _launch(system_db)
    await start_graph_run(graph=name, inputs={}, thread_id="t5", workflow_id="w5")
    for _ in range(200):
        if started:
            break
        await asyncio.sleep(0.05)
    assert started == ["node"]

    _relaunch(system_db)

    with pytest.raises(InterruptedSegmentError, match="segment 0 of graph run w5 was interrupted"):
        await (await DBOS.retrieve_workflow_async("w5")).get_result()
    assert started == ["node"]


@pytest.mark.asyncio
async def test_graph_segments_refuse_unknown_graphs_and_missing_checkpointers(
    tmp_path: Path,
) -> None:
    with pytest.raises(SegmentError, match="no graph registered as 'nobody'"):
        await run_graph({}, "nobody", thread_id="t")
    graphs = {"bare": _review_graph(tmp_path, [], checkpointer=False)}
    with pytest.raises(SegmentError, match="needs a checkpointer"):
        await run_graph(graphs, "bare", thread_id="t", inputs={"request": 1})


@pytest.mark.asyncio
async def test_a_segment_runs_a_graph_and_resumes_it_without_dbos(tmp_path: Path) -> None:
    effects: list[str] = []
    graphs = {"g": _review_graph(tmp_path, effects)}
    first = await run_graph(graphs, "g", thread_id="t6", inputs={"request": 6})
    assert (first.status, first.node) == ("paused", "job")
    second = await run_graph(graphs, "g", thread_id="t6", resume="plan-6", resuming=True)
    assert (second.status, second.node) == ("paused", "review")
    done = await run_graph(graphs, "g", thread_id="t6", resume="yes", resuming=True)
    assert done.status == "done"
    assert effects == ["draft 6", "publish 6:plan-6"]
