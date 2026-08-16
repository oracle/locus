# Copyright 2026 Tulip Labs
# SPDX-License-Identifier: Apache-2.0

"""``refusal_reason`` — what the *user* hears when the gate refuses.

The policy's own reason names the checks that fired: "blast radius 3 exceeds
the maximum 1", "labels ['large_refund'] are denied by policy". That is the
right detail for an audit record. Handed to a model, it is repeated verbatim
to whoever is on the other end — measured against a live model, which produced
"the blast radius (3) exceeds the maximum 1" in a customer-facing sentence.
This gives an integrator the sentence instead, without shrinking the trail.
"""

from __future__ import annotations

import json

import pytest

from tulip.control import Action, AuditTrail, ControlPolicy
from tulip.control.gate import gate_tool
from tulip.security.verify import VerificationResult
from tulip.tools.decorator import tool


RAN: list[float] = []


@tool
def issue_refund(order_id: str, amount_usd: float) -> str:
    """Issue a refund for an order."""
    RAN.append(amount_usd)
    return f"refunded ${amount_usd}"


def _action(name: str, kwargs: dict) -> Action:
    amount = float(kwargs.get("amount_usd", 0))
    tags = {"large"} if amount > 500 else {"mid"} if amount > 100 else set()
    return Action(
        name=name,
        asset=str(kwargs.get("order_id", "?")),
        kind="write",
        environment="staging",
        blast_radius=1,
        tags=frozenset(tags),
    )


PASSING = VerificationResult(survives=True, confidence=0.95, evidence_quality=0.95)
POLICY = ControlPolicy(require_human_for=frozenset({"mid"}), deny_for=frozenset({"large"}))


def _gate(**kw):
    return gate_tool(issue_refund, policy=POLICY, action=_action, verdict=PASSING, **kw)


@pytest.fixture(autouse=True)
def _clear() -> None:
    RAN.clear()


async def test_default_reason_is_the_policy_reason() -> None:
    """Unchanged behaviour: no ``refusal_reason`` means the policy's own words."""
    payload = json.loads(await _gate().fn(order_id="ord-1", amount_usd=900))
    assert payload["status"] == "denied"
    assert "denied by policy" in payload["reason"]
    assert RAN == []


async def test_a_string_replaces_the_policy_vocabulary() -> None:
    gated = _gate(refusal_reason="A manager needs to sign off on this refund.")
    payload = json.loads(await gated.fn(order_id="ord-1", amount_usd=900))
    assert payload["reason"] == "A manager needs to sign off on this refund."
    assert "blast radius" not in payload["reason"]
    assert "large" not in payload["reason"]


async def test_a_callable_may_vary_by_outcome() -> None:
    gated = _gate(
        refusal_reason=lambda d: (
            "We cannot refund this amount." if d.outcome == "deny" else "Waiting on a manager."
        )
    )
    held = json.loads(await gated.fn(order_id="ord-1", amount_usd=250))
    denied = json.loads(await gated.fn(order_id="ord-1", amount_usd=900))
    assert held["status"] == "held_for_approval"
    assert held["reason"] == "Waiting on a manager."
    assert denied["status"] == "denied"
    assert denied["reason"] == "We cannot refund this amount."


async def test_the_trail_still_records_the_full_policy_reason() -> None:
    """A friendlier user-facing sentence must not shrink the audit record."""
    trail = AuditTrail()
    gated = _gate(refusal_reason="A manager needs to sign off.", trail=trail)
    await gated.fn(order_id="ord-1", amount_usd=900)
    [record] = trail.records()
    assert "denied by policy" in record.payload["reason"]
    assert "sign off" not in record.payload["reason"]
    assert trail.verify() is True


async def test_an_allowed_call_is_untouched() -> None:
    gated = _gate(refusal_reason="never seen")
    assert await gated.fn(order_id="ord-1", amount_usd=50) == "refunded $50"
    assert RAN == [50]


async def test_refusal_keys_are_still_the_shared_shape() -> None:
    """The payload contract is shared with the tulip-frameworks bridges."""
    from tulip.control.gate import _REFUSAL_KEYS

    payload = json.loads(await _gate(refusal_reason="nope").fn(order_id="ord-1", amount_usd=900))
    assert set(_REFUSAL_KEYS) <= set(payload)
