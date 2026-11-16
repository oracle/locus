# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""A held call is written into the thread's checkpoint, and read back after a restart.

Before this, a paused thread's checkpoint said only that a tool call had no
result; which approval it waited on, and in which store, was known to the
process that paused and to nobody else. These tests pin the restart story end
to end: a fresh process with the same checkpointer and the same store finds the
hold, re-checks it against the store, performs the approved call once, and the
checkpoint stops carrying the hold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from locus.agent import Agent
from locus.control import (
    Action,
    ControlPolicy,
    FileApprovals,
    InMemoryApprovals,
    PendingHold,
    gate_tool,
    holds_of,
    pending_holds,
)
from locus.control.holds import HOLDS_KEY, hold_from_interrupt, with_hold, without_hold
from locus.core.errors import ApprovalPendingError
from locus.core.events import InterruptEvent, TerminateEvent
from locus.core.state import AgentState
from locus.memory.backends.file import FileCheckpointer
from locus.testing import ScriptedModel, text, tool_call
from locus.tools.decorator import Tool, tool


if TYPE_CHECKING:
    from pathlib import Path


def _refund_tool(calls: list[dict[str, Any]]) -> Tool:
    @tool
    def issue_refund(order_id: str, amount_usd: float) -> str:
        """Refund an order."""
        calls.append({"order_id": order_id, "amount_usd": amount_usd})
        return f"refunded {amount_usd} on {order_id}"

    return issue_refund


def _gated(calls: list[dict[str, Any]], store: Any) -> Tool:
    return gate_tool(
        _refund_tool(calls),
        policy=ControlPolicy(
            require_verification_score=0.0, require_human_for=frozenset({"production"})
        ),
        action=lambda name, kwargs: Action(
            name=name, asset=str(kwargs["order_id"]), environment="production"
        ),
        approval=store,
        on_refusal="interrupt",
        principal="svc-billing",
    )


def _agent(model: ScriptedModel, gated: Tool, checkpointer: Any = None) -> Agent:
    return Agent(
        model=model,
        tools=[gated],
        checkpointer=checkpointer,
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )


@pytest.mark.asyncio
async def test_the_hold_is_in_the_checkpoint_and_a_fresh_process_resumes_from_it(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []
    approvals = tmp_path / "approvals.json"
    checkpointer = FileCheckpointer(tmp_path / "checkpoints")
    first = _agent(
        ScriptedModel([tool_call("issue_refund", order_id="o1", amount_usd=4_000_000.0)]),
        _gated(calls, FileApprovals(approvals)),
        checkpointer,
    )

    events = [event async for event in first.run("refund o1", thread_id="t1")]
    interrupt = next(e for e in events if isinstance(e, InterruptEvent))

    # The process that paused is gone. Another one opens the checkpointer.
    [hold] = await pending_holds(FileCheckpointer(tmp_path / "checkpoints"), "t1")
    assert hold.approval_id == interrupt.metadata["approval_id"]
    assert hold.tool_call_id == interrupt.interrupt_id
    assert (hold.tool, hold.action, hold.asset, hold.principal) == (
        "issue_refund",
        "issue_refund",
        "o1",
        "svc-billing",
    )
    assert hold.arguments == {"order_id": "o1", "amount_usd": 4_000_000.0}
    assert hold.reason
    assert hold.store == f"file://{approvals}", "the hold names the store it is in"
    assert hold.held_at

    # It re-checks the hold against the store the hold names, then decides.
    from locus.control import approval_store

    store = approval_store(hold.store)
    assert hold.state(store) == "pending"
    store.decide(hold.approval_id, "approved", by="alice@example.com", reason="retention")
    assert hold.state(store) == "approved"

    second = _agent(
        ScriptedModel([text("refund issued")]),
        _gated(calls, FileApprovals(approvals)),
        FileCheckpointer(tmp_path / "checkpoints"),
    )
    resumed = [
        event async for event in second.resume("approved", thread_id="t1", perform_dangling=True)
    ]

    assert calls == [{"order_id": "o1", "amount_usd": 4_000_000.0}]
    assert next(e for e in resumed if isinstance(e, TerminateEvent)).final_message == (
        "refund issued"
    )
    assert await pending_holds(checkpointer, "t1") == [], "the answered call is no longer a hold"
    assert store.state(hold.approval_id) == "consumed"


@pytest.mark.asyncio
async def test_a_hold_stays_while_the_approval_is_still_pending(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    approvals = tmp_path / "approvals.json"
    checkpointer = FileCheckpointer(tmp_path / "checkpoints")
    agent = _agent(
        ScriptedModel([tool_call("issue_refund", order_id="o1", amount_usd=9.0)]),
        _gated(calls, FileApprovals(approvals)),
        checkpointer,
    )
    [_ async for _ in agent.run("refund o1", thread_id="t1")]

    fresh = _agent(
        ScriptedModel([text("never reached")]),
        _gated(calls, FileApprovals(approvals)),
        FileCheckpointer(tmp_path / "checkpoints"),
    )
    with pytest.raises(ApprovalPendingError) as pending:
        [_ async for _ in fresh.resume("go", thread_id="t1", perform_dangling=True)]

    [hold] = await pending_holds(checkpointer, "t1")
    assert pending.value.metadata["approval_id"] == hold.approval_id
    assert calls == []


@pytest.mark.asyncio
async def test_a_text_answer_also_clears_the_hold(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    checkpointer = FileCheckpointer(tmp_path / "checkpoints")
    store = FileApprovals(tmp_path / "approvals.json")
    agent = _agent(
        ScriptedModel(
            [tool_call("issue_refund", order_id="o1", amount_usd=9.0), text("left it alone")]
        ),
        _gated(calls, store),
        checkpointer,
    )
    [_ async for _ in agent.run("refund o1", thread_id="t1")]
    assert len(await pending_holds(checkpointer, "t1")) == 1

    [_ async for _ in agent.resume("do not refund", thread_id="t1")]

    assert await pending_holds(checkpointer, "t1") == []
    assert calls == []
    assert store.pending(), "the store's record is the approver's to close; the thread moved on"


@pytest.mark.asyncio
async def test_an_in_memory_pause_carries_the_hold_too() -> None:
    calls: list[dict[str, Any]] = []
    store = InMemoryApprovals()
    agent = _agent(
        ScriptedModel([tool_call("issue_refund", order_id="o1", amount_usd=9.0), text("done")]),
        _gated(calls, store),
    )
    events = [event async for event in agent.run("refund o1")]
    approval_id = next(e for e in events if isinstance(e, InterruptEvent)).metadata["approval_id"]

    [parked] = agent._interrupts.values()  # noqa: SLF001 - the parked state is the point
    [hold] = holds_of(parked.state)
    assert (hold.approval_id, hold.store) == (approval_id, "memory://")

    store.decide(approval_id, "approved", by="alice")
    [_ async for _ in agent.resume("approved", perform_dangling=True)]
    assert calls == [{"order_id": "o1", "amount_usd": 9.0}]


def test_a_checkpoint_written_before_holds_existed_has_none() -> None:
    state = AgentState(metadata={"tenant": "acme"})
    assert holds_of(state) == []
    assert without_hold(state, "call-1") is state
    assert holds_of(AgentState(metadata={HOLDS_KEY: "not a dict"})) == []


def test_holds_come_and_go_by_tool_call() -> None:
    first = PendingHold(
        tool_call_id="call-1",
        approval_id="appr-1",
        tool="t",
        action="t",
        asset="a",
        principal="p",
        reason="r",
        held_at="2026-01-01T00:00:00+00:00",
    )
    second = PendingHold(
        tool_call_id="call-2",
        approval_id="appr-2",
        tool="t",
        action="t",
        asset="b",
        principal="p",
        reason="r",
        held_at="2026-01-02T00:00:00+00:00",
    )
    state = with_hold(with_hold(AgentState(metadata={"tenant": "acme"}), second), first)

    assert holds_of(state) == [first, second], "oldest first"
    assert state.metadata["tenant"] == "acme"
    one_left = without_hold(state, "call-1")
    assert holds_of(one_left) == [second]
    none_left = without_hold(one_left, "call-2")
    assert HOLDS_KEY not in none_left.metadata
    assert none_left.metadata == {"tenant": "acme"}

    # A newer runtime's hold with a field this one does not know still loads.
    stored = {**first.to_dict(), "later_field": 1}
    assert PendingHold.from_dict(stored) == first


def test_only_an_approval_interrupt_is_a_hold() -> None:
    question = {"__interrupt__": True, "question": "Which region?", "metadata": {}}
    assert hold_from_interrupt("call-1", "ask_user", question) is None
    assert hold_from_interrupt("call-1", "ask_user", {"__interrupt__": True}) is None

    approval = {
        "__interrupt__": True,
        "question": "Approve?",
        "metadata": {"approval_id": "appr-1", "action": "refund", "asset": "o1"},
    }
    hold = hold_from_interrupt("call-1", "issue_refund", approval)
    assert hold is not None
    assert (hold.tool, hold.action, hold.asset, hold.principal, hold.store) == (
        "issue_refund",
        "refund",
        "o1",
        "",
        None,
    )
