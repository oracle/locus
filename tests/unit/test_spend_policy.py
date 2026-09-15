# Copyright 2026 The Tulip Authors
# SPDX-License-Identifier: Apache-2.0

"""Spend as a policy input, across runs.

``max_cost_usd`` caps what one run spends on model calls. These tests pin the
other half from #28: an action's own cost is weighed by policy, a person is
required above a per-action cost, and a scope (a customer, a tenant, a month)
is denied once its cumulative spend would cross a limit, even with an approval.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from tulip.control import (
    Action,
    AdmissionError,
    AuditTrail,
    ControlPolicy,
    FileSpendLedger,
    InMemoryApprovals,
    InMemorySpendLedger,
    SpendLedger,
    admit,
    approve,
    gate_tool,
)
from tulip.tools.decorator import Tool, tool


if TYPE_CHECKING:
    from pathlib import Path


def _policy(**limits: Any) -> ControlPolicy:
    return ControlPolicy(require_verification_score=0.0, require_human_for=frozenset(), **limits)


def _refund(amount: float, order: str = "o1") -> Action:
    return Action(name="refund", asset=order, kind="payment", cost_usd=amount)


def _async(value: str) -> Any:
    async def perform() -> str:
        return value

    return perform


def test_a_cost_above_the_per_action_limit_needs_a_person() -> None:
    policy = _policy(require_human_over_usd=500.0)

    assert approve(_refund(499.0), policy=policy).outcome == "allow"
    held = approve(_refund(501.0), policy=policy)
    assert held.outcome == "require_human"
    assert "exceeds the per-action limit" in held.reason


def test_crossing_the_spend_limit_is_denied() -> None:
    policy = _policy(spend_limit_usd=1_000.0)

    assert approve(_refund(400.0), policy=policy, spent_usd=600.0).outcome == "allow"
    denied = approve(_refund(400.01), policy=policy, spent_usd=600.0)
    assert denied.outcome == "deny"
    assert "would exceed the spend limit" in denied.reason


@pytest.mark.asyncio
async def test_admit_records_spend_only_for_actions_that_ran() -> None:
    ledger = InMemorySpendLedger()
    policy = _policy(spend_limit_usd=100.0)

    await admit(_refund(40.0), _async("ok"), policy=policy, ledger=ledger, spend_scope="customer:7")
    await admit(_refund(40.0), _async("ok"), policy=policy, ledger=ledger, spend_scope="customer:7")

    async def fails() -> str:
        raise RuntimeError("payment provider down")

    with pytest.raises(RuntimeError):
        await admit(_refund(10.0), fails, policy=policy, ledger=ledger, spend_scope="customer:7")
    with pytest.raises(AdmissionError):
        await admit(
            _refund(40.0), _async("ok"), policy=policy, ledger=ledger, spend_scope="customer:7"
        )

    assert ledger.spent("customer:7") == 80.0
    assert ledger.spent("customer:8") == 0.0


@pytest.mark.asyncio
async def test_an_approval_cannot_spend_past_the_limit() -> None:
    ledger = InMemorySpendLedger()
    ledger.record("tenant-a", 900.0)

    with pytest.raises(AdmissionError):
        await admit(
            _refund(200.0),
            _async("ok"),
            policy=_policy(spend_limit_usd=1_000.0, require_human_over_usd=100.0),
            ledger=ledger,
            spend_scope="tenant-a",
            approved_by="alice",
        )
    assert ledger.spent("tenant-a") == 900.0


@pytest.mark.asyncio
async def test_the_trail_shows_spend_and_is_unchanged_without_a_ledger() -> None:
    with_ledger, without = AuditTrail(), AuditTrail()

    await admit(
        _refund(25.0),
        _async("ok"),
        policy=_policy(),
        trail=with_ledger,
        ledger=InMemorySpendLedger(),
        spend_scope="s",
    )
    await admit(Action(name="lookup", asset="o1"), _async("ok"), policy=_policy(), trail=without)

    [spent] = [json.loads(line)["payload"] for line in with_ledger.export_jsonl().splitlines()]
    assert (spent["cost_usd"], spent["spent_usd"], spent["spend_scope"]) == (25.0, 0.0, "s")
    [plain] = [json.loads(line)["payload"] for line in without.export_jsonl().splitlines()]
    assert set(plain) == {"action", "asset", "outcome", "reason"}


def test_a_file_ledger_keeps_totals_and_entries_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "spend.json"

    FileSpendLedger(path).record("customer:7", 40.0, action="refund")
    total = FileSpendLedger(path).record("customer:7", 2.5, action="refund")

    assert total == 42.5
    assert FileSpendLedger(path).spent("customer:7") == 42.5
    assert [e["amount_usd"] for e in FileSpendLedger(path).entries("customer:7")] == [40.0, 2.5]
    assert isinstance(FileSpendLedger(path), SpendLedger)
    with pytest.raises(ValueError, match="negative"):
        FileSpendLedger(path).record("customer:7", -1.0)
    with pytest.raises(ValueError, match="negative"):
        InMemorySpendLedger().record("customer:7", -1.0)


def _refund_tool(calls: list[float]) -> Tool:
    @tool
    def issue_refund(customer_id: str, order_id: str, amount_usd: float) -> str:
        """Refund an order."""
        calls.append(amount_usd)
        return f"refunded {amount_usd}"

    return issue_refund


def _costed(name: str, args: dict[str, Any]) -> Action:
    return Action(name=name, asset=args["order_id"], kind="payment", cost_usd=args["amount_usd"])


@pytest.mark.asyncio
async def test_the_gate_caps_each_customer_separately() -> None:
    calls: list[float] = []
    ledger = InMemorySpendLedger()
    gated = gate_tool(
        _refund_tool(calls),
        policy=_policy(spend_limit_usd=100.0),
        action=_costed,
        ledger=ledger,
        spend_scope=lambda name, args: f"customer:{args['customer_id']}",
    )

    assert await gated.execute(customer_id="7", order_id="o1", amount_usd=60.0) == "refunded 60.0"
    refused = json.loads(await gated.execute(customer_id="7", order_id="o2", amount_usd=60.0))
    assert await gated.execute(customer_id="8", order_id="o3", amount_usd=60.0) == "refunded 60.0"

    assert refused["status"] == "denied"
    assert calls == [60.0, 60.0]
    assert (ledger.spent("customer:7"), ledger.spent("customer:8")) == (60.0, 60.0)


@pytest.mark.asyncio
async def test_a_held_action_that_became_unaffordable_does_not_run_after_approval() -> None:
    calls: list[float] = []
    ledger = InMemorySpendLedger()
    store = InMemoryApprovals()
    gated = gate_tool(
        _refund_tool(calls),
        policy=_policy(spend_limit_usd=1_000.0, require_human_over_usd=500.0),
        action=_costed,
        approval=store,
        on_refusal="interrupt",
        ledger=ledger,
        spend_scope="tenant-a",
    )

    held = json.loads(await gated.execute(customer_id="7", order_id="o1", amount_usd=800.0))
    store.decide(held["metadata"]["approval_id"], "approved", by="alice")
    ledger.record("tenant-a", 300.0, action="another refund")

    payload = json.loads(await gated.execute(customer_id="7", order_id="o1", amount_usd=800.0))

    assert payload["status"] == "denied"
    assert "spend limit" in payload["reason"]
    assert calls == []
    assert ledger.spent("tenant-a") == 300.0
