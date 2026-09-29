# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Who may approve a held call, decided in the SDK at decision time.

Every other agent framework leaves this to the application: whoever holds the
API key can approve anything, including their own request. These tests pin the
rules that make an approval mean something — the approver is authorised for what
they approve, the requester cannot approve itself, dual control takes two
people, a delegation ends when it says it does, and nobody may approve what no
rule covers.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from tulip.control import (
    Action,
    ApprovalAuthority,
    ApprovalAuthorityError,
    ApproverRule,
    AuditTrail,
    ControlPolicy,
    Delegation,
    FileApprovals,
    InMemoryApprovals,
    gate_tool,
)
from tulip.tools.decorator import tool


if TYPE_CHECKING:
    from pathlib import Path


def _held(store: InMemoryApprovals | FileApprovals, labels: tuple[str, ...] = ("payment",)) -> str:
    return store.submit("svc-billing", "issue_refund", {"order_id": "o1"}, labels=labels)


def _people(**roles: set[str]) -> Any:
    return lambda principal: roles.get(principal, set())


def test_an_approver_without_authority_is_refused_and_recorded() -> None:
    store = InMemoryApprovals(
        ApprovalAuthority(
            rules=(ApproverRule(labels=frozenset({"payment"}), approvers=frozenset({"alice"})),)
        )
    )
    approval_id = _held(store)

    with pytest.raises(ApprovalAuthorityError, match="mallory may not decide"):
        store.decide(approval_id, "approved", by="mallory")

    record = store.get(approval_id)
    assert record is not None
    assert record.status == "pending"
    assert [r["by"] for r in record.rejections] == ["mallory"]
    assert store.decide(approval_id, "approved", by="alice").status == "approved"


def test_the_requester_cannot_approve_its_own_action() -> None:
    rule = ApproverRule(approvers=frozenset({"svc-billing", "alice"}))
    store = InMemoryApprovals(ApprovalAuthority(rules=(rule,)))
    approval_id = _held(store)

    with pytest.raises(ApprovalAuthorityError, match="requested this action"):
        store.decide(approval_id, "approved", by="svc-billing")

    lenient = InMemoryApprovals(
        ApprovalAuthority(
            rules=(ApproverRule(approvers=frozenset({"svc-billing"}), allow_self_approval=True),)
        )
    )
    assert lenient.decide(_held(lenient), "approved", by="svc-billing").status == "approved"


def test_a_quorum_needs_distinct_approvers() -> None:
    store = InMemoryApprovals(
        ApprovalAuthority(rules=(ApproverRule(approvers=frozenset({"alice", "bob"}), quorum=2),))
    )
    approval_id = _held(store)

    first = store.decide(approval_id, "approved", by="alice")
    assert (first.status, first.approvers) == ("pending", ["alice"])
    with pytest.raises(ValueError, match="already approved"):
        store.decide(approval_id, "approved", by="alice")

    second = store.decide(approval_id, "approved", by="bob")
    assert (second.status, second.decided_by) == ("approved", "alice, bob")


def test_an_action_under_two_rules_needs_both() -> None:
    authority = ApprovalAuthority(
        rules=(
            ApproverRule(labels=frozenset({"payment"}), roles=frozenset({"finance"})),
            ApproverRule(labels=frozenset({"production"}), roles=frozenset({"ops"})),
        ),
        roles_of=_people(alice={"finance"}, olga={"ops"}),
    )
    store = InMemoryApprovals(authority)
    approval_id = _held(store, labels=("payment", "production"))

    assert store.decide(approval_id, "approved", by="alice").status == "pending"
    assert store.decide(approval_id, "approved", by="olga").status == "approved"


def test_one_authorised_denial_ends_it() -> None:
    store = InMemoryApprovals(
        ApprovalAuthority(rules=(ApproverRule(approvers=frozenset({"alice", "bob"}), quorum=2),))
    )
    approval_id = _held(store)

    store.decide(approval_id, "approved", by="alice")
    denied = store.decide(approval_id, "denied", by="bob")

    assert (denied.status, denied.decided_by) == ("denied", "bob")


def test_a_delegation_works_until_its_deadline() -> None:
    def authority(now: datetime) -> ApprovalAuthority:
        return ApprovalAuthority(
            rules=(ApproverRule(approvers=frozenset({"alice"})),),
            delegations=(
                Delegation(grantor="alice", grantee="dan", expires_at="2026-09-20T00:00:00+00:00"),
            ),
            clock=lambda: now,
        )

    before = InMemoryApprovals(authority(datetime(2026, 9, 18, tzinfo=UTC)))
    record = before.decide(_held(before), "approved", by="dan")
    assert record.status == "approved"
    assert record.approvals[-1]["basis"] == "delegated by alice"

    after = InMemoryApprovals(authority(datetime(2026, 9, 21, tzinfo=UTC)))
    with pytest.raises(ApprovalAuthorityError):
        after.decide(_held(after), "approved", by="dan")


def test_the_requester_cannot_approve_through_a_delegation_it_granted() -> None:
    store = InMemoryApprovals(
        ApprovalAuthority(
            rules=(ApproverRule(approvers=frozenset({"svc-billing"})),),
            delegations=(
                Delegation(grantor="svc-billing", grantee="dan", expires_at="2099-01-01T00:00:00"),
            ),
        )
    )

    with pytest.raises(ApprovalAuthorityError):
        store.decide(_held(store), "approved", by="dan")


def test_nobody_may_approve_what_no_rule_covers() -> None:
    store = InMemoryApprovals(
        ApprovalAuthority(
            rules=(ApproverRule(labels=frozenset({"payment"}), approvers=frozenset({"alice"})),)
        )
    )

    with pytest.raises(ApprovalAuthorityError, match="no approver rule covers"):
        store.decide(_held(store, labels=("infrastructure",)), "approved", by="alice")


def test_a_rule_must_name_someone_and_a_quorum() -> None:
    with pytest.raises(ValueError, match="approvers or roles"):
        ApproverRule(labels=frozenset({"payment"}))
    with pytest.raises(ValueError, match="quorum"):
        ApproverRule(approvers=frozenset({"alice"}), quorum=0)


def test_a_file_store_accumulates_a_quorum_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "approvals.json"
    authority = ApprovalAuthority(
        rules=(ApproverRule(approvers=frozenset({"alice", "bob"}), quorum=2),)
    )
    approval_id = _held(FileApprovals(path, authority))

    FileApprovals(path, authority).decide(approval_id, "approved", by="alice")
    record = FileApprovals(path, authority).decide(approval_id, "approved", by="bob")

    assert record.status == "approved"
    reloaded = FileApprovals(path, authority).get(approval_id)
    assert reloaded is not None
    assert reloaded.approvers == ["alice", "bob"]


@pytest.mark.asyncio
async def test_the_gate_waits_for_the_quorum_and_names_every_approver() -> None:
    calls: list[dict[str, Any]] = []

    @tool
    def issue_refund(order_id: str, amount_usd: float) -> str:
        """Refund an order."""
        calls.append({"order_id": order_id, "amount_usd": amount_usd})
        return f"refunded {amount_usd} on {order_id}"

    trail = AuditTrail()
    store = InMemoryApprovals(
        ApprovalAuthority(
            rules=(
                ApproverRule(
                    labels=frozenset({"payment"}), approvers=frozenset({"alice", "bob"}), quorum=2
                ),
            )
        )
    )
    gated = gate_tool(
        issue_refund,
        policy=ControlPolicy(
            require_verification_score=0.0, require_human_for=frozenset({"payment"})
        ),
        action=lambda name, kwargs: Action(name=name, asset=kwargs["order_id"], kind="payment"),
        approval=store,
        on_refusal="interrupt",
        principal="svc-billing",
        trail=trail,
    )

    held = json.loads(await gated.execute(order_id="o1", amount_usd=500.0))
    approval_id = held["metadata"]["approval_id"]
    # No environment on the action: it carries the fail-safe "unknown" label too.
    assert store.get(approval_id).labels == ["payment", "unknown"]  # type: ignore[union-attr]

    store.decide(approval_id, "approved", by="alice")
    still = json.loads(await gated.execute(order_id="o1", amount_usd=500.0))
    assert still["__interrupt__"] is True
    assert still["metadata"]["approvers"] == ["alice"]
    assert calls == []

    store.decide(approval_id, "approved", by="bob")
    assert await gated.execute(order_id="o1", amount_usd=500.0) == "refunded 500.0 on o1"
    assert len(calls) == 1
    exported = trail.export_jsonl()
    assert "alice, bob" in exported
    assert trail.verify()
