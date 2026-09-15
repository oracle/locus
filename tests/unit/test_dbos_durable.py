# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable agent runs on DBOS, against a SQLite system database.

The cases that matter: a run pauses on a held refund, the process goes away, a
person approves, and a relaunched process finishes the run with the refund made
once; and a segment cut off halfway is never run a second time.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

import pytest


pytest.importorskip("dbos")

from dbos import DBOS  # noqa: E402

from tulip.agent import Agent  # noqa: E402
from tulip.control import Action, ControlPolicy, FileApprovals, gate_tool  # noqa: E402
from tulip.core.messages import Message, Role  # noqa: E402
from tulip.durable.dbos import (  # noqa: E402
    InterruptedSegmentError,
    pending_decision,
    register_agents,
    signal_decided,
    start_agent_run,
)
from tulip.durable.segments import SegmentError, run_agent_segment  # noqa: E402
from tulip.memory.backends.file import FileCheckpointer  # noqa: E402
from tulip.testing import FunctionModel, text, tool_call  # noqa: E402
from tulip.tools.decorator import tool  # noqa: E402


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


def _refund_agent(
    root: Path, refunds: list[float], *, hold: bool = True, checkpointer: bool = True
) -> Callable[[], Agent]:
    @tool
    def issue_refund(order_id: str, amount_usd: float) -> str:
        """Refund an order."""
        refunds.append(amount_usd)
        return f"refunded {amount_usd} on {order_id}"

    def model(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
        if any(m.role == Role.TOOL for m in messages):
            return text("refund issued")
        return tool_call("issue_refund", order_id="o1", amount_usd=250.0)

    def make() -> Agent:
        refund = gate_tool(
            issue_refund,
            policy=ControlPolicy(
                require_verification_score=0.0,
                require_human_for=frozenset({"payment"}) if hold else frozenset(),
            ),
            action=lambda name, args: Action(name=name, asset=args["order_id"], kind="payment"),
            approval=FileApprovals(root / "approvals.json"),
            on_refusal="interrupt",
        )
        return Agent(
            model=FunctionModel(model),
            tools=[refund],
            checkpointer=FileCheckpointer(root / "checkpoints") if checkpointer else None,
            reflexion=False,
            grounding=False,
        )

    return make


async def _wait_for_pause(workflow_id: str) -> Any:
    for _ in range(200):
        pending = await pending_decision(workflow_id)
        if pending is not None:
            return pending
        await asyncio.sleep(0.05)
    raise AssertionError("the run never paused")


def _name() -> str:
    return f"refunds-{uuid.uuid4().hex[:6]}"


@pytest.mark.asyncio
async def test_a_held_refund_waits_for_a_person_and_runs_once(
    system_db: str, tmp_path: Path
) -> None:
    refunds: list[float] = []
    name = _name()
    register_agents({name: _refund_agent(tmp_path, refunds)})
    _launch(system_db)

    handle = await start_agent_run(agent=name, prompt="refund o1", thread_id="t1")
    assert handle.workflow_id == "tulip-t1"
    pending = await _wait_for_pause(handle.workflow_id)
    assert pending.status == "paused"
    assert pending.approval_id
    assert refunds == []

    FileApprovals(tmp_path / "approvals.json").decide(pending.approval_id, "approved", by="alice")
    await signal_decided(handle.workflow_id)
    outcome = await handle.get_result()
    assert outcome.status == "done"
    assert outcome.final_message == "refund issued"
    assert refunds == [250.0]
    assert await pending_decision(handle.workflow_id) is None


@pytest.mark.asyncio
async def test_a_paused_run_survives_a_restart(system_db: str, tmp_path: Path) -> None:
    refunds: list[float] = []
    name = _name()
    register_agents({name: _refund_agent(tmp_path, refunds)})
    _launch(system_db)
    handle = await start_agent_run(agent=name, prompt="refund o1", thread_id="t2", workflow_id="w2")
    pending = await _wait_for_pause("w2")

    _relaunch(system_db)  # the process goes away; a new one recovers the run

    FileApprovals(tmp_path / "approvals.json").decide(pending.approval_id, "approved", by="alice")
    assert (await _wait_for_pause("w2")).approval_id == pending.approval_id
    await signal_decided("w2")
    outcome = await (await DBOS.retrieve_workflow_async("w2")).get_result()
    assert outcome.status == "done"
    assert refunds == [250.0]
    assert handle.workflow_id == "w2"


@pytest.mark.asyncio
async def test_a_run_with_nothing_held_finishes_in_one_segment(
    system_db: str, tmp_path: Path
) -> None:
    refunds: list[float] = []
    name = _name()
    register_agents({name: _refund_agent(tmp_path, refunds, hold=False)})
    _launch(system_db)
    handle = await start_agent_run(agent=name, prompt="refund o1", thread_id="t3")
    outcome = await handle.get_result()
    assert outcome.status == "done"
    assert refunds == [250.0]


@pytest.mark.asyncio
async def test_a_rejected_refund_is_not_performed(system_db: str, tmp_path: Path) -> None:
    refunds: list[float] = []
    name = _name()
    register_agents({name: _refund_agent(tmp_path, refunds)})
    _launch(system_db)
    handle = await start_agent_run(agent=name, prompt="refund o1", thread_id="t4")
    pending = await _wait_for_pause(handle.workflow_id)
    FileApprovals(tmp_path / "approvals.json").decide(pending.approval_id, "denied", by="bob")
    await signal_decided(handle.workflow_id, answer="denied", perform=False)
    outcome = await handle.get_result()
    assert outcome.status == "done"
    assert refunds == []


@pytest.mark.asyncio
async def test_a_segment_cut_off_halfway_is_not_run_again(system_db: str, tmp_path: Path) -> None:
    started: list[str] = []
    never = asyncio.Event()
    name = _name()

    @tool
    async def slow_lookup() -> str:
        """Look something up, slowly."""
        started.append("tool")
        await never.wait()  # the process dies while the tool is working
        return "unreachable"  # pragma: no cover

    def model(messages: list[Message], tools: list[dict[str, Any]]) -> Any:
        return tool_call("slow_lookup")

    def make() -> Agent:
        return Agent(
            model=FunctionModel(model),
            tools=[slow_lookup],
            checkpointer=FileCheckpointer(tmp_path / "checkpoints"),
            reflexion=False,
            grounding=False,
        )

    register_agents({name: make})
    _launch(system_db)
    await start_agent_run(agent=name, prompt="go", thread_id="t5", workflow_id="w5")
    for _ in range(200):
        if started:
            break
        await asyncio.sleep(0.05)
    assert started == ["tool"]

    _relaunch(system_db)

    with pytest.raises(InterruptedSegmentError, match="segment 0 of run w5 was interrupted"):
        await (await DBOS.retrieve_workflow_async("w5")).get_result()
    assert started == ["tool"]


@pytest.mark.asyncio
async def test_segments_refuse_unknown_agents_and_missing_checkpointers(tmp_path: Path) -> None:
    with pytest.raises(SegmentError, match="no agent registered as 'nobody'"):
        await run_agent_segment({}, "nobody", thread_id="t", kind="run")
    agents = {"bare": _refund_agent(tmp_path, [], checkpointer=False)}
    with pytest.raises(SegmentError, match="needs a checkpointer"):
        await run_agent_segment(agents, "bare", thread_id="t", kind="run")
