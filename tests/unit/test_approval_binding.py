# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""An approval names one call, in one context, under one policy, and can edit it.

These tests pin the two halves of #6 that #21 left open: a decision made under
one policy version or in one context is never redeemed in another, and an
approver who changes the arguments approves exactly the edited call, which the
policy weighs again.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from tulip.control import (
    Action,
    ApprovalAuthority,
    ApproverRule,
    AuditTrail,
    ControlPolicy,
    InMemoryApprovals,
    call_digest,
    gate_tool,
)
from tulip.tools.decorator import Tool, tool


def _refund(calls: list[float]) -> Tool:
    @tool
    def issue_refund(order_id: str, amount_usd: float) -> str:
        """Refund an order."""
        calls.append(amount_usd)
        return f"refunded {amount_usd} on {order_id}"

    return issue_refund


def _policy(**overrides: Any) -> ControlPolicy:
    return ControlPolicy(
        require_verification_score=0.0, require_human_for=frozenset({"payment"}), **overrides
    )


def _payment(name: str, kwargs: dict[str, Any]) -> Action:
    tags = frozenset({"irreversible"}) if kwargs["amount_usd"] > 1_000 else frozenset()
    return Action(name=name, asset=kwargs["order_id"], kind="payment", tags=tags)


def _gated(calls: list[float], store: InMemoryApprovals, **options: Any) -> Tool:
    options.setdefault("policy", _policy())
    return gate_tool(
        _refund(calls),
        action=_payment,
        approval=store,
        on_refusal="interrupt",
        principal="svc-billing",
        **options,
    )


async def _hold(gated: Tool, amount: float = 250.0) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(await gated.execute(order_id="o1", amount_usd=amount))
    assert payload.get("__interrupt__") is True, payload
    return payload


def test_an_empty_context_hashes_exactly_like_no_context() -> None:
    canonical = json.dumps(
        {"principal": "p", "tool": "t", "arguments": {"a": 1}},
        sort_keys=True,
        separators=(",", ":"),
    )
    before_contexts = hashlib.sha256(canonical.encode()).hexdigest()

    assert call_digest("p", "t", {"a": 1}) == before_contexts
    assert call_digest("p", "t", {"a": 1}, {}) == before_contexts
    assert call_digest("p", "t", {"a": 1}, {"thread": "t1"}) != before_contexts


@pytest.mark.asyncio
async def test_a_decision_is_never_redeemed_under_another_policy_version() -> None:
    calls: list[float] = []
    store = InMemoryApprovals()
    v1 = _gated(calls, store, policy=_policy(version="2026-09-01"))
    v2 = _gated(calls, store, policy=_policy(version="2026-09-15"))

    held = await _hold(v1)
    store.decide(held["metadata"]["approval_id"], "approved", by="alice")
    under_v2 = await _hold(v2)

    assert under_v2["metadata"]["approval_id"] != held["metadata"]["approval_id"]
    assert under_v2["metadata"]["context"] == {"policy_version": "2026-09-15"}
    assert calls == []
    assert await v1.execute(order_id="o1", amount_usd=250.0) == "refunded 250.0 on o1"


@pytest.mark.asyncio
async def test_the_caller_binds_approvals_to_a_thread() -> None:
    calls: list[float] = []
    store = InMemoryApprovals()
    first = _gated(calls, store, approval_context=lambda tool_name, args: {"thread": "t1"})
    second = _gated(calls, store, approval_context={"thread": "t2"})

    held = await _hold(first)
    store.decide(held["metadata"]["approval_id"], "approved", by="alice")
    other = await _hold(second)

    assert held["metadata"]["context"] == {"thread": "t1"}
    assert other["metadata"]["approval_id"] != held["metadata"]["approval_id"]
    assert calls == []


@pytest.mark.asyncio
async def test_an_approver_edits_the_amount_and_only_the_edit_runs() -> None:
    calls: list[float] = []
    store = InMemoryApprovals()
    trail = AuditTrail()
    gated = _gated(calls, store, trail=trail)
    approval_id = (await _hold(gated))["metadata"]["approval_id"]

    record = store.decide(
        approval_id, "approved", by="alice", arguments={"order_id": "o1", "amount_usd": 100.0}
    )
    assert record.approved_arguments == {"order_id": "o1", "amount_usd": 100.0}

    assert await gated.execute(order_id="o1", amount_usd=250.0) == "refunded 100.0 on o1"
    assert calls == [100.0]
    exported = [json.loads(line) for line in trail.export_jsonl().splitlines()]
    edit = next(r for r in exported if r["event_type"] == "approval-edited")
    assert edit["payload"]["requested_arguments"]["amount_usd"] == 250.0
    assert edit["payload"]["approved_arguments"]["amount_usd"] == 100.0
    assert trail.verify()


def test_an_edit_keeps_the_calls_keys_and_only_approves() -> None:
    store = InMemoryApprovals()
    approval_id = store.submit("svc", "issue_refund", {"order_id": "o1", "amount_usd": 250.0})

    with pytest.raises(ValueError, match="keeps the call's arguments"):
        store.decide(approval_id, "approved", by="alice", arguments={"order_id": "o1"})
    with pytest.raises(ValueError, match="keeps the call's arguments"):
        store.decide(
            approval_id,
            "approved",
            by="alice",
            arguments={"order_id": "o1", "amount_usd": 1.0, "currency": "EUR"},
        )
    with pytest.raises(ValueError, match="only an approval"):
        store.decide(
            approval_id, "denied", by="alice", arguments={"order_id": "o1", "amount_usd": 1.0}
        )
    assert store.state(approval_id) == "pending"


def test_approvers_in_a_quorum_must_approve_the_same_arguments() -> None:
    store = InMemoryApprovals(
        ApprovalAuthority(rules=(ApproverRule(approvers=frozenset({"alice", "bob"}), quorum=2),))
    )
    approval_id = store.submit("svc", "issue_refund", {"order_id": "o1", "amount_usd": 250.0})
    edit = {"order_id": "o1", "amount_usd": 100.0}

    store.decide(approval_id, "approved", by="alice", arguments=edit)
    with pytest.raises(ValueError, match="same arguments"):
        store.decide(approval_id, "approved", by="bob", arguments={**edit, "amount_usd": 90.0})
    with pytest.raises(ValueError, match="same arguments"):
        store.decide(approval_id, "approved", by="bob")

    record = store.decide(approval_id, "approved", by="bob", arguments=edit)
    assert (record.status, record.approved_arguments) == ("approved", edit)


@pytest.mark.asyncio
async def test_an_edit_the_policy_denies_does_not_run() -> None:
    calls: list[float] = []
    store = InMemoryApprovals()
    gated = _gated(calls, store, policy=_policy(deny_for=frozenset({"irreversible"})))
    approval_id = (await _hold(gated, amount=500.0))["metadata"]["approval_id"]

    store.decide(
        approval_id, "approved", by="alice", arguments={"order_id": "o1", "amount_usd": 5_000.0}
    )
    payload = json.loads(await gated.execute(order_id="o1", amount_usd=500.0))

    assert payload["status"] == "denied"
    assert calls == []
    assert store.state(approval_id) == "consumed"


@pytest.mark.asyncio
async def test_an_approval_without_an_edit_runs_the_original_call() -> None:
    calls: list[float] = []
    store = InMemoryApprovals()
    gated = _gated(calls, store)
    approval_id = (await _hold(gated))["metadata"]["approval_id"]

    record = store.decide(approval_id, "approved", by="alice")

    assert record.approved_arguments is None
    assert await gated.execute(order_id="o1", amount_usd=250.0) == "refunded 250.0 on o1"
    assert calls == [250.0]
