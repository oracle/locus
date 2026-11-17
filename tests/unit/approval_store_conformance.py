# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""What every ``ApprovalStore`` must do, whatever it is backed by.

A test class mixes this in and provides two fixtures: ``make_store(authority)``
builds a store on a fresh backing, and ``reopen(store)`` returns a second store
object on the SAME backing, as another process would open it. The unit suite
runs it over the in-memory and file stores and over the database stores with
their drivers faked; the integration suite runs it over live PostgreSQL, Redis
and Oracle behind the usual service markers.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from locus.control import (
    ApprovalAuthority,
    ApprovalAuthorityError,
    ApprovalStore,
    ApproverRule,
    Delegation,
)


def _held(store: ApprovalStore, **arguments: Any) -> str:
    args = arguments or {"order_id": "o1", "amount_usd": 10.0}
    return store.submit(
        "svc-billing", "issue_refund", args, reason="labels require a human", labels=["payment"]
    )


def _people(**roles: set[str]) -> Any:
    return lambda principal: roles.get(principal, set())


class ApprovalStoreConformance:
    """Mix in; provide ``make_store`` and ``reopen`` fixtures."""

    @pytest.fixture
    def store(self, make_store: Any) -> ApprovalStore:
        return make_store(None)  # type: ignore[no-any-return]

    # -- submitting ----------------------------------------------------------

    def test_it_is_an_approval_store(self, store: ApprovalStore) -> None:
        assert isinstance(store, ApprovalStore)

    def test_the_same_call_gets_the_same_live_record(self, store: ApprovalStore) -> None:
        approval_id = _held(store)

        assert store.submit("svc-billing", "issue_refund", {"amount_usd": 10.0, "order_id": "o1"})
        assert _held(store) == approval_id
        record = store.get(approval_id)
        assert record is not None
        assert (record.status, record.principal, record.tool) == (
            "pending",
            "svc-billing",
            "issue_refund",
        )
        assert record.arguments == {"order_id": "o1", "amount_usd": 10.0}
        assert record.labels == ["payment"]
        assert record.reason == "labels require a human"
        assert record.approval_id.startswith(f"appr-{record.digest[:16]}-")

    def test_different_arguments_are_different_approvals(self, store: ApprovalStore) -> None:
        ten = _held(store, order_id="o1", amount_usd=10.0)
        eleven = _held(store, order_id="o1", amount_usd=11.0)

        assert ten != eleven
        assert [r.approval_id for r in store.pending()] == [ten, eleven]

    def test_a_context_binds_the_call(self, store: ApprovalStore) -> None:
        plain = _held(store)
        bound = store.submit(
            "svc-billing",
            "issue_refund",
            {"order_id": "o1", "amount_usd": 10.0},
            context={"policy_version": "v2"},
        )

        assert bound != plain
        record = store.get(bound)
        assert record is not None
        assert record.context == {"policy_version": "v2"}

    def test_unknown_ids_are_none(self, store: ApprovalStore) -> None:
        assert store.get("appr-nope") is None
        assert store.state("appr-nope") is None
        with pytest.raises(KeyError):
            store.decide("appr-nope", "approved", by="alice")
        with pytest.raises(KeyError):
            store.consume("appr-nope")

    # -- deciding ------------------------------------------------------------

    def test_a_decision_names_who_made_it_and_why(self, store: ApprovalStore) -> None:
        approval_id = _held(store)

        record = store.decide(approval_id, "approved", by="alice", reason="within policy")

        assert (record.status, record.verdict, record.decided_by) == (
            "approved",
            "approved",
            "alice",
        )
        assert record.decided_at is not None
        assert record.approvals[-1]["by"] == "alice"
        assert record.approvals[-1]["reason"] == "within policy"
        assert record.approvers == ["alice"]
        assert store.state(approval_id) == "approved"
        assert store.pending() == []

    def test_a_decision_is_made_once(self, store: ApprovalStore) -> None:
        approval_id = _held(store)
        store.decide(approval_id, "denied", by="bob")

        with pytest.raises(ValueError, match="already denied"):
            store.decide(approval_id, "approved", by="alice")

    def test_a_decision_is_validated_before_anything_is_written(self, store: ApprovalStore) -> None:
        approval_id = _held(store)

        with pytest.raises(ValueError, match="who"):
            store.decide(approval_id, "approved", by="")
        with pytest.raises(ValueError, match="verdict"):
            store.decide(approval_id, "maybe", by="alice")  # type: ignore[arg-type]
        assert store.state(approval_id) == "pending"

    def test_an_approver_can_edit_the_call_but_not_its_shape(self, store: ApprovalStore) -> None:
        approval_id = _held(store)

        with pytest.raises(ValueError, match="keeps the call's arguments"):
            store.decide(approval_id, "approved", by="alice", arguments={"order_id": "o1"})
        with pytest.raises(ValueError, match="only an approval"):
            store.decide(
                approval_id, "denied", by="alice", arguments={"order_id": "o1", "amount_usd": 5.0}
            )
        record = store.decide(
            approval_id, "approved", by="alice", arguments={"order_id": "o1", "amount_usd": 5.0}
        )
        assert record.approved_arguments == {"order_id": "o1", "amount_usd": 5.0}
        assert record.arguments == {"order_id": "o1", "amount_usd": 10.0}

    # -- consuming -----------------------------------------------------------

    def test_an_approval_is_used_once(self, store: ApprovalStore) -> None:
        approval_id = _held(store)
        with pytest.raises(ValueError, match="not decided"):
            store.consume(approval_id)
        store.decide(approval_id, "approved", by="alice")

        consumed = store.consume(approval_id)

        assert (consumed.status, consumed.verdict) == ("consumed", "approved")
        with pytest.raises(ValueError, match="not decided"):
            store.consume(approval_id)
        again = _held(store)
        assert again != approval_id, "a repeated call opens a new approval"
        assert again.endswith("-2")
        assert store.state(again) == "pending"

    def test_racing_consumers_redeem_an_approval_exactly_once(self, store: ApprovalStore) -> None:
        approval_id = _held(store)
        store.decide(approval_id, "approved", by="alice")
        won: list[str] = []
        lost: list[str] = []

        def redeem() -> None:
            try:
                won.append(store.consume(approval_id).status)
            except ValueError as error:
                lost.append(str(error))

        threads = [threading.Thread(target=redeem) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert won == ["consumed"]
        assert len(lost) == 7

    # -- another process -----------------------------------------------------

    def test_a_decision_made_elsewhere_is_seen(
        self, make_store: Any, reopen: Any, store: ApprovalStore
    ) -> None:
        approval_id = _held(store)

        other = reopen(store)
        other.decide(approval_id, "approved", by="alice")

        assert store.state(approval_id) == "approved"
        assert _held(store) == approval_id, "the live record is found, not duplicated"
        assert reopen(store).consume(approval_id).status == "consumed"
        assert store.state(approval_id) == "consumed"

    # -- who may decide ------------------------------------------------------

    def test_an_approver_without_authority_is_refused_and_recorded(self, make_store: Any) -> None:
        store = make_store(
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

    def test_the_requester_cannot_approve_its_own_action(self, make_store: Any) -> None:
        store = make_store(
            ApprovalAuthority(rules=(ApproverRule(approvers=frozenset({"svc-billing", "alice"})),))
        )
        approval_id = _held(store)

        with pytest.raises(ApprovalAuthorityError, match="requested this action"):
            store.decide(approval_id, "approved", by="svc-billing")
        assert store.decide(approval_id, "approved", by="alice").status == "approved"

    def test_a_quorum_takes_distinct_approvers(self, make_store: Any) -> None:
        store = make_store(
            ApprovalAuthority(
                rules=(ApproverRule(roles=frozenset({"finance"}), quorum=2),),
                roles_of=_people(alice={"finance"}, bob={"finance"}),
            )
        )
        approval_id = _held(store)

        first = store.decide(approval_id, "approved", by="alice")
        assert first.status == "pending"
        assert first.approvers == ["alice"]
        with pytest.raises(ValueError, match="already approved"):
            store.decide(approval_id, "approved", by="alice")
        second = store.decide(approval_id, "approved", by="bob")

        assert second.status == "approved"
        assert second.decided_by == "alice, bob"
        assert second.approvals[-1]["basis"] == "bob"

    def test_one_authorised_denial_ends_it(self, make_store: Any) -> None:
        store = make_store(
            ApprovalAuthority(
                rules=(ApproverRule(approvers=frozenset({"alice", "bob"}), quorum=2),)
            )
        )
        approval_id = _held(store)
        store.decide(approval_id, "approved", by="alice")

        denied = store.decide(approval_id, "denied", by="bob")

        assert (denied.status, denied.decided_by) == ("denied", "bob")

    def test_a_delegation_lends_authority_until_it_expires(self, make_store: Any) -> None:
        rule = ApproverRule(labels=frozenset({"payment"}), approvers=frozenset({"olga"}))
        soon = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        store = make_store(
            ApprovalAuthority(
                rules=(rule,),
                delegations=(
                    Delegation(grantor="olga", grantee="dan", expires_at=soon),
                    Delegation(grantor="olga", grantee="eve", expires_at=past),
                ),
            )
        )
        approval_id = _held(store)

        with pytest.raises(ApprovalAuthorityError, match="eve may not decide"):
            store.decide(approval_id, "approved", by="eve")
        record = store.decide(approval_id, "approved", by="dan")

        assert record.status == "approved"
        assert record.approvals[-1]["basis"] == "delegated by olga"

    def test_an_action_no_rule_covers_cannot_be_approved(self, make_store: Any) -> None:
        store = make_store(
            ApprovalAuthority(
                rules=(ApproverRule(labels=frozenset({"infra"}), approvers=frozenset({"alice"})),)
            )
        )
        approval_id = _held(store)

        with pytest.raises(ApprovalAuthorityError, match="nobody may decide"):
            store.decide(approval_id, "approved", by="alice")
