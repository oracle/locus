# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit: a step's resolved approval, enforced by the open-source approval stores.

``authority_from_resolved`` turns "Finance and CFO must both approve" into an
``ApprovalAuthority``: every group gives its count, from distinct people, each approval
counting for one group only; ``only_named`` keeps everyone else out but break-glass.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from tulip.control import (
    ApprovalAuthority,
    ApprovalAuthorityError,
    ApproverRule,
    Delegation,
    FileApprovals,
    InMemoryApprovals,
)
from tulip.playbooks.v2 import ResolvedApproval, StepApproval, authority_from_resolved, pick_rule
from tulip.playbooks.v2.approvals import ApprovalRule


if TYPE_CHECKING:
    from pathlib import Path

    from tulip.control.approvals import ApprovalRecord


ROLES: dict[str, set[str]] = {
    "fiona": {"finance"},
    "frank": {"finance"},
    "carol": {"cfo"},
    "both": {"finance", "cfo"},
    "gina": {"approvers"},
    "ada": {"admin"},
    "svc-ap": {"finance", "cfo"},  # the requester, in both groups
}


def roles_of(principal: str) -> set[str]:
    return ROLES.get(principal, set())


def resolved(*groups: tuple[str, int], only_named: bool = False) -> ResolvedApproval:
    approval = StepApproval(rules=(ApprovalRule(groups=groups),), only_these_approvers=only_named)
    return pick_rule(approval, {}, step="pay")


FINANCE_AND_CFO = resolved(("finance", 1), ("cfo", 1))


def store_for(approval: ResolvedApproval, **kwargs: Any) -> tuple[InMemoryApprovals, str]:
    authorities: dict[str, ApprovalAuthority] = {}
    store = InMemoryApprovals(authority=lambda record: authorities.get(record.approval_id))
    approval_id = store.submit("svc-ap", "pay_vendor", {"vendor": "v-1", "amount": 25000})
    authorities[approval_id] = authority_from_resolved(
        approval, roles_of=kwargs.pop("roles_of", roles_of), **kwargs
    )
    return store, approval_id


def test_all_of_needs_every_group() -> None:
    store, approval_id = store_for(FINANCE_AND_CFO)
    first = store.decide(approval_id, "approved", by="fiona")
    assert first.status == "pending"
    assert first.approvals[0]["rules"] == [0]
    second = store.decide(approval_id, "approved", by="carol")
    assert second.status == "approved"
    assert second.decided_by == "fiona, carol"


def test_one_person_counts_for_one_group_only() -> None:
    store, approval_id = store_for(FINANCE_AND_CFO)
    record = store.decide(approval_id, "approved", by="both")
    assert record.status == "pending"  # counted for Finance, not for CFO too
    assert record.approvals[0]["rules"] == [0]
    with pytest.raises(ValueError, match="already approved"):
        store.decide(approval_id, "approved", by="both")
    assert store.decide(approval_id, "approved", by="carol").status == "approved"


def test_a_group_that_has_counted_refuses_more_while_another_waits() -> None:
    store, approval_id = store_for(FINANCE_AND_CFO)
    store.decide(approval_id, "approved", by="fiona")
    with pytest.raises(
        ApprovalAuthorityError,
        match="Finance already counted; this approval needs someone from CFO",
    ):
        store.decide(approval_id, "approved", by="frank")
    record = store.get(approval_id)
    assert record is not None
    assert record.status == "pending"
    assert record.rejections[0]["by"] == "frank"


def test_someone_in_both_groups_fills_the_one_still_waiting() -> None:
    store, approval_id = store_for(FINANCE_AND_CFO)
    store.decide(approval_id, "approved", by="fiona")
    record = store.decide(approval_id, "approved", by="both")
    assert record.status == "approved"
    assert record.approvals[1]["rules"] == [1]


def test_a_count_needs_that_many_distinct_people() -> None:
    store, approval_id = store_for(resolved(("finance", 2)))
    assert store.decide(approval_id, "approved", by="fiona").status == "pending"
    with pytest.raises(ValueError, match="already approved"):
        store.decide(approval_id, "approved", by="fiona")
    assert store.decide(approval_id, "approved", by="frank").status == "approved"


def test_the_requester_never_approves_its_own_call() -> None:
    store, approval_id = store_for(FINANCE_AND_CFO)
    with pytest.raises(ApprovalAuthorityError, match="requested this action"):
        store.decide(approval_id, "approved", by="svc-ap")


def test_general_approvers_count_unless_only_named() -> None:
    store, approval_id = store_for(FINANCE_AND_CFO, also=("approvers",))
    record = store.decide(approval_id, "approved", by="gina")
    assert record.approvals[0]["rules"] == [0]  # the first group still waiting

    named, named_id = store_for(resolved(("finance", 1), only_named=True), also=("approvers",))
    with pytest.raises(ApprovalAuthorityError, match="may not decide"):
        named.decide(named_id, "approved", by="gina")
    assert named.decide(named_id, "approved", by="fiona").status == "approved"


def test_break_glass_counts_even_when_only_named() -> None:
    store, approval_id = store_for(
        resolved(("finance", 1), ("cfo", 1), only_named=True), break_glass=("admin",)
    )
    record = store.decide(approval_id, "approved", by="ada")
    assert record.approvals[0]["rules"] == [0]
    assert store.decide(approval_id, "approved", by="carol").status == "approved"


def test_members_name_a_group_without_a_directory() -> None:
    store, approval_id = store_for(
        FINANCE_AND_CFO, roles_of=None, members={"finance": ["fiona"], "cfo": ["carol"]}
    )
    with pytest.raises(ApprovalAuthorityError):
        store.decide(approval_id, "approved", by="frank")
    store.decide(approval_id, "approved", by="fiona")
    assert store.decide(approval_id, "approved", by="carol").status == "approved"


def test_one_authorised_denial_ends_it_even_from_a_group_that_counted() -> None:
    store, approval_id = store_for(FINANCE_AND_CFO)
    store.decide(approval_id, "approved", by="fiona")
    record = store.decide(approval_id, "denied", by="frank")
    assert record.status == "denied"


def test_a_delegation_lends_a_group() -> None:
    lent = Delegation(grantor="carol", grantee="dan", expires_at="2026-10-11T00:00:00Z")
    store, approval_id = store_for(
        FINANCE_AND_CFO,
        delegations=(lent,),
        clock=lambda: datetime(2026, 10, 10, tzinfo=UTC),
    )
    store.decide(approval_id, "approved", by="fiona")
    record = store.decide(approval_id, "approved", by="dan")
    assert record.status == "approved"
    assert record.approvals[1]["basis"] == "delegated by carol"


def test_the_adapter_builds_one_rule_per_group() -> None:
    authority = authority_from_resolved(
        resolved(("finance", 2), ("cfo", 1)), roles_of=roles_of, also=("approvers",)
    )
    assert authority.count_once
    assert authority.rules == (
        ApproverRule(roles=frozenset({"finance", "approvers"}), quorum=2, name="Finance"),
        ApproverRule(roles=frozenset({"cfo", "approvers"}), quorum=1, name="CFO"),
    )


def test_one_authority_for_every_record_still_works() -> None:
    authority = authority_from_resolved(FINANCE_AND_CFO, roles_of=roles_of)
    store = InMemoryApprovals(authority)
    assert store.authority_for(_record(store)) is authority


def test_a_file_store_takes_an_authority_per_record(tmp_path: Path) -> None:
    path = tmp_path / "approvals.json"
    authorities: dict[str, ApprovalAuthority] = {}

    def for_record(record: ApprovalRecord) -> ApprovalAuthority | None:
        return authorities.get(record.approval_id)

    approval_id = FileApprovals(path, for_record).submit("svc-ap", "pay_vendor", {"amount": 1})
    authorities[approval_id] = authority_from_resolved(FINANCE_AND_CFO, roles_of=roles_of)
    FileApprovals(path, for_record).decide(approval_id, "approved", by="both")
    record = FileApprovals(path, for_record).decide(approval_id, "approved", by="carol")
    assert record.status == "approved"


def test_without_count_once_one_person_may_fill_two_rules() -> None:
    authority = ApprovalAuthority(
        rules=(
            ApproverRule(roles=frozenset({"finance"})),
            ApproverRule(roles=frozenset({"cfo"})),
        ),
        roles_of=roles_of,
    )
    store = InMemoryApprovals(authority)
    approval_id = store.submit("svc-ap", "pay_vendor", {"amount": 1})
    assert store.decide(approval_id, "approved", by="both").status == "approved"


def test_a_rule_without_a_name_is_named_by_who_it_lets_decide() -> None:
    assert (
        ApproverRule(roles=frozenset({"cfo"}), approvers=frozenset({"ann"})).shown_as == "ann, cfo"
    )


def _record(store: InMemoryApprovals) -> ApprovalRecord:
    approval_id = store.submit("svc-ap", "pay_vendor", {"amount": 2})
    record = store.get(approval_id)
    assert record is not None
    return record


def test_the_refusal_names_every_group_that_counted_and_every_one_that_waits() -> None:
    store, approval_id = store_for(resolved(("finance", 1), ("cfo", 1), ("legal", 1)))
    store.decide(approval_id, "approved", by="fiona")
    store.decide(approval_id, "approved", by="carol")
    with pytest.raises(
        ApprovalAuthorityError,
        match="Finance and CFO already counted; this approval needs someone from Legal",
    ):
        store.decide(approval_id, "approved", by="both")


def test_checking_a_record_every_group_has_filled_counts_for_the_first() -> None:
    authority = authority_from_resolved(FINANCE_AND_CFO, roles_of=roles_of)
    store, approval_id = store_for(FINANCE_AND_CFO)
    store.decide(approval_id, "approved", by="fiona")
    store.decide(approval_id, "approved", by="carol")
    record = store.get(approval_id)
    assert record is not None
    assert authority.check(record, "both").rules == frozenset({0})
    assert authority.check(record, "both", "denied").rules == frozenset({0, 1})
