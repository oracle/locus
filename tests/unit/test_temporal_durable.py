# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""Durable agent runs on Temporal, against a local Temporal dev server.

The case that matters: a run pauses on a held refund, the worker goes away, a
person approves, and a new worker finishes the run with the refund made once.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio


pytest.importorskip("temporalio")

from temporalio.client import WorkflowFailureError  # noqa: E402
from temporalio.testing import WorkflowEnvironment  # noqa: E402
from temporalio.worker import UnsandboxedWorkflowRunner  # noqa: E402

from locus.agent import Agent  # noqa: E402
from locus.control import Action, ControlPolicy, FileApprovals, gate_tool  # noqa: E402
from locus.core.messages import Message, Role  # noqa: E402
from locus.durable.temporal import (  # noqa: E402
    AgentWorkflow,
    create_worker,
    signal_decided,
    start_agent_run,
)
from locus.memory.backends.file import FileCheckpointer  # noqa: E402
from locus.testing import FunctionModel, text, tool_call  # noqa: E402
from locus.tools.decorator import tool  # noqa: E402


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable


#: Directory the Temporal dev-server binary is downloaded into and reused from.
#: Unset, the SDK uses the system temporary directory; CI points it at a path
#: it caches between runs, so a warm run starts the server without a download.
DEV_SERVER_DIR_ENV = "LOCUS_TEMPORAL_DEV_SERVER_DIR"

#: Why the dev server could not be started, once two attempts have failed.
#: The remaining tests of the module then skip without trying again.
_dev_server_unavailable: str | None = None


async def _start_dev_server() -> WorkflowEnvironment:
    """Start the local dev server; retry once, then skip the module.

    ``start_local`` downloads the Temporal CLI the first time it runs. A
    download that fails or arrives corrupt raises ``RuntimeError`` here,
    before any test body runs; it says nothing about the code under test, so
    it is retried once and then reported as a skip with the reason.
    """
    global _dev_server_unavailable  # noqa: PLW0603
    if _dev_server_unavailable is not None:
        pytest.skip(_dev_server_unavailable)
    download_dir = os.environ.get(DEV_SERVER_DIR_ENV) or None
    if download_dir is not None:
        Path(download_dir).mkdir(parents=True, exist_ok=True)
    try:
        return await WorkflowEnvironment.start_local(download_dest_dir=download_dir)
    except RuntimeError:
        pass
    try:
        return await WorkflowEnvironment.start_local(download_dest_dir=download_dir)
    except RuntimeError as error:
        _dev_server_unavailable = (
            f"the Temporal dev server did not start after two attempts ({error})"
        )
    pytest.skip(_dev_server_unavailable)


@pytest_asyncio.fixture
async def env() -> AsyncIterator[WorkflowEnvironment]:
    async with await _start_dev_server() as environment:
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
