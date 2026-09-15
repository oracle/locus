# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""A held action pauses the run, waits for a named person, and survives a restart.

Before this, ``require_human_for`` produced a refusal the model read and the run
carried on: nothing paused, and a pending approval died with the process. These
tests pin the behaviour a governance feature is worthless without — the side
effect does not happen until a person approves, it happens exactly once with
the arguments that were approved, and a denial is final.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from tulip.agent import Agent
from tulip.control import (
    Action,
    AdmissionError,
    AuditTrail,
    ControlPolicy,
    FileApprovals,
    InMemoryApprovals,
    admit,
    call_digest,
    gate_tool,
)
from tulip.core.events import InterruptEvent, TerminateEvent, ToolCompleteEvent
from tulip.memory.backends.file import FileCheckpointer
from tulip.testing import ScriptedModel, text, tool_call
from tulip.tools.decorator import Tool, tool


if TYPE_CHECKING:
    from pathlib import Path


def _refund_tool(calls: list[dict[str, Any]]) -> Tool:
    @tool
    def issue_refund(order_id: str, amount_usd: float) -> str:
        """Refund an order."""
        calls.append({"order_id": order_id, "amount_usd": amount_usd})
        return f"refunded {amount_usd} on {order_id}"

    return issue_refund


def _policy() -> ControlPolicy:
    return ControlPolicy(
        require_verification_score=0.0, require_human_for=frozenset({"production"})
    )


def _production(name: str, kwargs: dict[str, Any]) -> Action:
    return Action(name=name, asset=str(kwargs["order_id"]), environment="production")


def _gated(calls: list[dict[str, Any]], store: Any, trail: AuditTrail | None = None) -> Tool:
    return gate_tool(
        _refund_tool(calls),
        policy=_policy(),
        action=_production,
        approval=store,
        on_refusal="interrupt",
        principal="svc-billing",
        trail=trail,
    )


def _agent(model: ScriptedModel, gated: Tool, checkpoints: Path | None = None) -> Agent:
    return Agent(
        model=model,
        tools=[gated],
        checkpointer=FileCheckpointer(checkpoints) if checkpoints else None,
        max_iterations=10,
        reflexion=False,
        grounding=False,
    )


@pytest.mark.asyncio
async def test_a_hold_pauses_the_run_and_nothing_executes() -> None:
    calls: list[dict[str, Any]] = []
    store = InMemoryApprovals()
    agent = _agent(
        ScriptedModel([tool_call("issue_refund", order_id="o1", amount_usd=4_000_000.0)]),
        _gated(calls, store),
    )

    events = [event async for event in agent.run("refund o1")]

    interrupt = next(e for e in events if isinstance(e, InterruptEvent))
    assert calls == []
    assert interrupt.metadata["action"] == "issue_refund"
    assert interrupt.metadata["arguments"] == {"order_id": "o1", "amount_usd": 4_000_000.0}
    [pending] = store.pending()
    assert pending.approval_id == interrupt.metadata["approval_id"]
    assert pending.principal == "svc-billing"
    assert pending.reason
    assert not any(isinstance(e, TerminateEvent) for e in events), "a paused run has not finished"


@pytest.mark.asyncio
async def test_an_approval_decided_elsewhere_resumes_in_a_fresh_agent(tmp_path: Path) -> None:
    """The process that paused is gone; a new one resumes from disk alone."""
    calls: list[dict[str, Any]] = []
    approvals = tmp_path / "approvals.json"
    checkpoints = tmp_path / "checkpoints"
    trail = AuditTrail()

    first = _agent(
        ScriptedModel([tool_call("issue_refund", order_id="o1", amount_usd=4_000_000.0)]),
        _gated(calls, FileApprovals(approvals), trail),
        checkpoints,
    )
    events = [event async for event in first.run("refund o1", thread_id="t1")]
    approval_id = next(e for e in events if isinstance(e, InterruptEvent)).metadata["approval_id"]
    assert calls == []

    FileApprovals(approvals).decide(approval_id, "approved", by="alice@example.com")

    second = _agent(
        ScriptedModel([text("refund issued")]),
        _gated(calls, FileApprovals(approvals), trail),
        checkpoints,
    )
    resumed = [
        event async for event in second.resume("approved", thread_id="t1", perform_dangling=True)
    ]

    assert calls == [{"order_id": "o1", "amount_usd": 4_000_000.0}], (
        "the approved call runs exactly once, with the arguments that were approved"
    )
    assert next(e for e in resumed if isinstance(e, TerminateEvent)).final_message == (
        "refund issued"
    )
    assert FileApprovals(approvals).state(approval_id) == "consumed"
    assert "alice@example.com" in trail.export_jsonl()
    assert trail.verify()


@pytest.mark.asyncio
async def test_a_denial_never_executes_and_the_model_hears_it() -> None:
    calls: list[dict[str, Any]] = []
    store = InMemoryApprovals()
    agent = _agent(
        ScriptedModel(
            [tool_call("issue_refund", order_id="o1", amount_usd=9.0), text("could not refund")]
        ),
        _gated(calls, store),
    )
    events = [event async for event in agent.run("refund o1")]
    approval_id = next(e for e in events if isinstance(e, InterruptEvent)).metadata["approval_id"]

    store.decide(approval_id, "denied", by="bob@example.com")
    resumed = [event async for event in agent.resume("denied", perform_dangling=True)]

    assert calls == []
    performed = next(e for e in resumed if isinstance(e, ToolCompleteEvent))
    assert json.loads(str(performed.result))["status"] == "denied"
    assert "bob@example.com" in str(performed.result)
    record = store.get(approval_id)
    assert record is not None
    assert (record.status, record.verdict) == ("consumed", "denied")


@pytest.mark.asyncio
async def test_an_approval_is_used_once() -> None:
    calls: list[dict[str, Any]] = []
    store = InMemoryApprovals()
    gated = _gated(calls, store)

    first = json.loads(await gated.execute(order_id="o1", amount_usd=10.0))
    store.decide(first["metadata"]["approval_id"], "approved", by="alice@example.com")
    assert await gated.execute(order_id="o1", amount_usd=10.0) == "refunded 10.0 on o1"

    again = json.loads(await gated.execute(order_id="o1", amount_usd=10.0))

    assert again["__interrupt__"] is True, "a repeated call must hold, not ride the old yes"
    assert again["metadata"]["approval_id"] != first["metadata"]["approval_id"]
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_different_arguments_need_their_own_approval() -> None:
    calls: list[dict[str, Any]] = []
    store = InMemoryApprovals()
    gated = _gated(calls, store)

    ten = json.loads(await gated.execute(order_id="o1", amount_usd=10.0))
    store.decide(ten["metadata"]["approval_id"], "approved", by="alice@example.com")

    eleven = json.loads(await gated.execute(order_id="o1", amount_usd=11.0))

    assert eleven["__interrupt__"] is True
    assert eleven["metadata"]["approval_id"] != ten["metadata"]["approval_id"]
    assert calls == []
    assert await gated.execute(order_id="o1", amount_usd=10.0) == "refunded 10.0 on o1"


@pytest.mark.asyncio
async def test_a_policy_denial_never_pauses() -> None:
    calls: list[dict[str, Any]] = []
    store = InMemoryApprovals()
    gated = gate_tool(
        _refund_tool(calls),
        policy=ControlPolicy(require_verification_score=0.0, deny_for=frozenset({"irreversible"})),
        action=lambda name, kwargs: Action(name=name, asset="o2", tags=frozenset({"irreversible"})),
        approval=store,
        on_refusal="interrupt",
    )

    payload = json.loads(await gated.execute(order_id="o2", amount_usd=1.0))

    assert payload["status"] == "denied"
    assert "__interrupt__" not in payload
    assert store.pending() == []
    assert calls == []


def test_interrupt_mode_needs_a_store() -> None:
    class _BridgeOnly:
        def submit(self, principal: str, tool_name: str, args: dict[str, Any]) -> str:
            return "appr-1"

        def state(self, approval_id: str) -> str:
            return "pending"

    with pytest.raises(TypeError, match="ApprovalStore"):
        gate_tool(_refund_tool([]), policy=_policy(), on_refusal="interrupt")
    with pytest.raises(TypeError, match="ApprovalStore"):
        gate_tool(
            _refund_tool([]), policy=_policy(), on_refusal="interrupt", approval=_BridgeOnly()
        )


def test_a_decision_is_made_once_and_names_who_made_it() -> None:
    store = InMemoryApprovals()
    approval_id = store.submit("svc", "issue_refund", {"order_id": "o1"})

    with pytest.raises(ValueError, match="who"):
        store.decide(approval_id, "approved", by="")
    store.decide(approval_id, "approved", by="alice@example.com")
    with pytest.raises(ValueError, match="already approved"):
        store.decide(approval_id, "denied", by="bob@example.com")
    with pytest.raises(KeyError):
        store.decide("appr-unknown", "approved", by="alice@example.com")


def test_file_approvals_see_a_decision_written_by_another_instance(tmp_path: Path) -> None:
    path = tmp_path / "approvals.json"
    approval_id = FileApprovals(path).submit("svc", "issue_refund", {"order_id": "o1"})

    FileApprovals(path).decide(approval_id, "approved", by="alice@example.com")

    record = FileApprovals(path).get(approval_id)
    assert record is not None
    assert (record.status, record.decided_by) == ("approved", "alice@example.com")
    assert FileApprovals(path).submit("svc", "issue_refund", {"order_id": "o1"}) == approval_id


@pytest.mark.asyncio
async def test_admit_runs_an_approved_hold_but_never_a_denial() -> None:
    trail = AuditTrail()
    held = Action(name="refund", asset="o1", environment="production")

    result = await admit(
        held, _async("done"), policy=_policy(), trail=trail, approved_by="alice@example.com"
    )

    assert result == "done"
    assert "alice@example.com" in trail.export_jsonl()
    with pytest.raises(AdmissionError):
        await admit(
            Action(name="wipe", asset="db", tags=frozenset({"irreversible"})),
            _async("wiped"),
            policy=ControlPolicy(deny_for=frozenset({"irreversible"})),
            approved_by="alice@example.com",
        )


def test_call_digest_ignores_key_order_and_nothing_else() -> None:
    base = call_digest("svc", "issue_refund", {"order_id": "o1", "amount_usd": 10.0})

    assert base == call_digest("svc", "issue_refund", {"amount_usd": 10.0, "order_id": "o1"})
    assert base != call_digest("svc", "issue_refund", {"order_id": "o1", "amount_usd": 10.01})
    assert base != call_digest("other", "issue_refund", {"order_id": "o1", "amount_usd": 10.0})


def _async(value: str) -> Any:
    async def perform() -> str:
        return value

    return perform
