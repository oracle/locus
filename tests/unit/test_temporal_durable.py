# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable agent runs on Temporal, against a local Temporal dev server.

The case that matters: a run pauses on a held refund, the worker goes away, a
person approves, and a new worker finishes the run with the refund made once.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio


pytest.importorskip("temporalio")

from temporalio.client import WorkflowFailureError  # noqa: E402
from temporalio.testing import WorkflowEnvironment  # noqa: E402
from temporalio.worker import UnsandboxedWorkflowRunner  # noqa: E402

from tulip.agent import Agent  # noqa: E402
from tulip.control import Action, ControlPolicy, FileApprovals, gate_tool  # noqa: E402
from tulip.core.messages import Message, Role  # noqa: E402
from tulip.durable.temporal import (  # noqa: E402
    AgentWorkflow,
    create_worker,
    signal_decided,
    start_agent_run,
)
from tulip.memory.backends.file import FileCheckpointer  # noqa: E402
from tulip.testing import FunctionModel, text, tool_call  # noqa: E402
from tulip.tools.decorator import tool  # noqa: E402


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from pathlib import Path


@pytest_asyncio.fixture
async def env() -> AsyncIterator[WorkflowEnvironment]:
    async with await WorkflowEnvironment.start_local() as environment:
        yield environment


def _refund_agent(root: Path, refunds: list[float], *, hold: bool = True) -> Callable[[], Agent]:
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
            checkpointer=FileCheckpointer(root / "checkpoints"),
            reflexion=False,
            grounding=False,
        )

    return make


async def _wait_for_pause(handle: Any) -> Any:
    for _ in range(200):
        pending = await handle.query(AgentWorkflow.pending)
        if pending is not None:
            return pending
        await asyncio.sleep(0.05)
    raise AssertionError("the run never paused")


def _worker(env: WorkflowEnvironment, queue: str, agents: dict[str, Any]) -> Any:
    return create_worker(
        env.client, task_queue=queue, agents=agents, workflow_runner=UnsandboxedWorkflowRunner()
    )


@pytest.mark.asyncio
async def test_a_held_refund_waits_for_a_person_and_runs_once(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    refunds: list[float] = []
    queue = f"q-{uuid.uuid4().hex[:6]}"
    agents = {"refunds": _refund_agent(tmp_path, refunds)}

    async with _worker(env, queue, agents):
        handle = await start_agent_run(
            env.client, agent="refunds", prompt="refund o1", thread_id="t1", task_queue=queue
        )
        pending = await _wait_for_pause(handle)
        assert pending.status == "paused"
        assert refunds == []

        FileApprovals(tmp_path / "approvals.json").decide(
            pending.approval_id, "approved", by="alice"
        )
        await signal_decided(env.client, handle.id)
        outcome = await handle.result()

    assert (outcome.status, outcome.final_message) == ("done", "refund issued")
    assert refunds == [250.0]


@pytest.mark.asyncio
async def test_a_new_worker_finishes_a_run_paused_under_an_old_one(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    refunds: list[float] = []
    queue = f"q-{uuid.uuid4().hex[:6]}"
    agents = {"refunds": _refund_agent(tmp_path, refunds)}

    async with _worker(env, queue, agents):
        handle = await start_agent_run(
            env.client, agent="refunds", prompt="refund o1", thread_id="t2", task_queue=queue
        )
        pending = await _wait_for_pause(handle)

    FileApprovals(tmp_path / "approvals.json").decide(pending.approval_id, "approved", by="alice")
    await signal_decided(env.client, handle.id)

    async with _worker(env, queue, agents):
        outcome = await handle.result()

    assert outcome.status == "done"
    assert refunds == [250.0]


@pytest.mark.asyncio
async def test_a_run_with_nothing_held_completes(env: WorkflowEnvironment, tmp_path: Path) -> None:
    refunds: list[float] = []
    queue = f"q-{uuid.uuid4().hex[:6]}"

    async with _worker(env, queue, {"refunds": _refund_agent(tmp_path, refunds, hold=False)}):
        handle = await start_agent_run(
            env.client, agent="refunds", prompt="refund o1", thread_id="t3", task_queue=queue
        )
        outcome = await handle.result()

    assert (outcome.status, outcome.stop_reason) == ("done", "complete")
    assert refunds == [250.0]


@pytest.mark.asyncio
async def test_an_unregistered_agent_fails_the_run_clearly(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    queue = f"q-{uuid.uuid4().hex[:6]}"

    async with _worker(env, queue, {}):
        handle = await start_agent_run(
            env.client, agent="nobody", prompt="hi", thread_id="t4", task_queue=queue
        )
        with pytest.raises(WorkflowFailureError) as failure:
            await handle.result()

    assert "no agent registered as 'nobody'" in str(failure.value.cause.cause)


@pytest.mark.asyncio
async def test_an_agent_without_a_checkpointer_is_refused(env: WorkflowEnvironment) -> None:
    queue = f"q-{uuid.uuid4().hex[:6]}"

    def bare() -> Agent:
        return Agent(
            model=FunctionModel(lambda m, t: text("hi")), tools=[], reflexion=False, grounding=False
        )

    async with _worker(env, queue, {"bare": bare}):
        handle = await start_agent_run(
            env.client, agent="bare", prompt="hi", thread_id="t5", task_queue=queue
        )
        with pytest.raises(WorkflowFailureError) as failure:
            await handle.result()

    assert "needs a checkpointer" in str(failure.value.cause.cause)
